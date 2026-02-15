"""
RAG+GNN pipeline.
Pipeline:
 - YOLOv8n
 - HybridGAT (ResNet18 global + GAT local)
 - FAISS retriever
 - LLM generator
 - Alerting (SMS, Email)
"""

import argparse
import cv2
import os
import sys
import logging
import torch
import numpy as np
import yaml

# Import modules
try:
    from modules.detector import Detector
    from modules.gnn_model import HybridGAT, build_graph, compress, load_checkpoint
    from modules.retriever import Retriever
    from modules.generator import Generator
    from modules.alerting import Alerting
    from modules.tts import tts_announce
except Exception as e:
    print("Failed to import modules/:", e)
    sys.exit(1)


# Utils
def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_alerts", action="store_true")
    parser.add_argument("--no_tts", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    log = logging.getLogger("pipeline")

    # Load config
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    log.info(f"Running on device: {device}")

    # Load YOLO detector
    det_name = cfg["models"].get("detector", "yolov8n.pt")
    detector = Detector(model_name=det_name, device=args.device)
    log.info(f"Detector initialized ({det_name})")

    # Load Retriever + FAISS index
    retr_cfg = cfg["retrieval"]
    retriever = Retriever(model_name=cfg["models"]["text_emb"],
                          index_path=retr_cfg["faiss_index_path"])

    retriever.load_index(retr_cfg["faiss_index_path"],
                         docs_dir=retr_cfg.get("docs_path"))
    log.info("Retriever ready")

    # Load LLM Generator
    generator = Generator(cfg.get("llm", {}))
    log.info("LLM generator initialized")

    # Load Hybrid GNN
    gnn_cfg = cfg["gnn"]
    gnn_weights = gnn_cfg["weights_path"]
    print(gnn_weights)
    model = HybridGAT(in_dim=133, hidden=256, num_classes=3).to(device)

    ckpt = load_checkpoint(gnn_weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    compress.load_state_dict(ckpt["compress_state"])
    model.eval()
    compress.eval()

    log.info(f"Hybrid GNN loaded from {gnn_weights}")

    # Load test image
    if not os.path.exists(args.image):
        log.error(f"Image not found: {args.image}")
        sys.exit(1)

    frame = cv2.imread(args.image)
    if frame is None:
        log.error("Failed to load image.")
        sys.exit(1)

    log.info(f"Loaded image: {args.image}  shape={frame.shape}")

    # YOLO detection
    detections = detector.predict(frame)
    num_det = len(detections.boxes) if detections and detections.boxes is not None else 0
    log.info(f"YOLO detections: {num_det}")

    # Build GNN graph
    graph = build_graph(frame, detections, device=device)
    graph = graph.to(device)

    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    normal_p, fire_p, accident_p = probs
    log.info(f"GNN probabilities = normal={normal_p:.3f}, fire={fire_p:.3f}, accident={accident_p:.3f}")

    # Decide class
    cls_id = int(np.argmax(probs))
    cls_name = ["normal", "fire", "accident"][cls_id]

    # exit if normal: exit immediately
    if cls_name == "normal":
        print("\n===== RESULT =====")
        print("This image appears NORMAL. No hazard detected.")
        print("==================\n")
        return

    # Retrieve relevant safety documents
    query = f"incident involving {cls_name}"
    retrieved = retriever.query(query, k=3)

    retrieved_texts = []
    if retrieved:
        for r in retrieved:
            retrieved_texts.append(r["text"])
    else:
        log.warning("No retrieved documents. Using empty context.")

    # Build prompt for LLM
    prompt = [
        "You are an emergency advisory assistant.",
        f"Detected hazard: {cls_name.upper()}",
        "",
        "Provide:",
        "1. One short announcement.",
        "2. Five essential safety instructions.",
        "3. One common mistake to avoid.",
        "",
        "REFERENCE DOCUMENTS:"
    ]

    for i, txt in enumerate(retrieved_texts):
        prompt.append(f"[DOC {i+1}] {txt[:300].replace('\n',' ')}")

    prompt.append("\nRespond now with ONLY the instructions.")

    prompt = "\n".join(prompt)

    # LLM generation
    llm_out = generator.generate(prompt)
    print("\n=========== SAFETY INSTRUCTIONS ===========")
    print(llm_out)
    print("===========================================\n")

    # TTS
    if not args.no_tts:
        try:
            tts_announce(llm_out)
        except Exception as e:
            log.warning("TTS failed: %s", e)

    # Alerts (SMS/Email)
    if not args.no_alerts:
        alert_cfg = cfg.get("alerts", {})
        
        # Pass FULL alerts config
        alerting = Alerting(alert_cfg)
    
        payload = {
            "hazard": cls_name,
            "probabilities": {
                "normal": float(normal_p),
                "fire": float(fire_p),
                "accident": float(accident_p)
            },
            "message": llm_out
        }
    
        # SMS
        try:
            for num in alert_cfg.get("notify_numbers", []):
                alerting.send_sms(num, llm_out)
                log.info(f"SMS sent to {num}")
        except Exception as e:
            log.warning("SMS failed: %s", e)
    
        # EMAIL
        try:
            for em in alert_cfg.get("notify_emails", []):
                alerting.send_email(
                    em,
                    subject=f"[HAZARD ALERT] {cls_name.upper()} detected",
                    body=llm_out + "\n\nPayload:\n" + str(payload)
                )
                log.info(f"Email sent to {em}")
        except Exception as e:
            log.warning("Email failed: %s", e)
    
    log.info("Pipeline completed.")
    
if __name__ == "__main__":
    main()
