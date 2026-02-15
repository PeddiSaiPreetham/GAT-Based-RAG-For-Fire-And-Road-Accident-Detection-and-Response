# modules/gnn_model.py
"""
Hybrid GNN module:
 - HybridGAT (ResNet-global compress + GAT local nodes)
 - build_graph(image_np, yres, device)  -> torch_geometric.data.Data
 - compress: nn.Linear instance used in training/inference
 - load_checkpoint(path, map_location) -> dict (handles flexible checkpoint shapes)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from PIL import Image
import numpy as np
from torchvision import models, transforms
import logging

log = logging.getLogger("gnn_model")

GLOBAL_DIM = 128    # compressed global feature dim
LOCAL_DIM = 5
IN_DIM = GLOBAL_DIM + LOCAL_DIM

# Pretrained ResNet backbone (frozen)
_resnet = None
_img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
try:
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet18.fc = nn.Identity()
    resnet18.eval()
    for p in resnet18.parameters():
        p.requires_grad = False
    _resnet = resnet18
    log.info("ResNet backbone loaded (frozen).")
except Exception as e:
    log.warning("Failed to init ResNet backbone: %s. Global features disabled.", e)
    _resnet = None

compress = nn.Linear(512, GLOBAL_DIM)

# --------------------------
# Hybrid GAT
# --------------------------
class HybridGAT(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=256, num_classes=3):
        super().__init__()
        # conv1: heads=2 -> out dim = hidden * heads? In PyG GATConv(in_channels, out_channels, heads)
        # we set out_channels=hidden and heads=2 -> output will be hidden*heads
        self.conv1 = GATConv(in_dim, hidden, heads=2)
        # conv2: in channels = hidden*2, out_channels=hidden, heads=1 -> output hidden*1
        self.conv2 = GATConv(hidden * 2, hidden, heads=1)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        # x: (N, in_dim)
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        # graph-level pooling: mean across nodes
        x = x.mean(dim=0, keepdim=True)  # shape (1, hidden)
        return self.fc(x)  # (1, num_classes)


def build_graph(image_np, yres, device="cpu"):
    H, W = image_np.shape[:2]

    # compute global vector via resnet + compress if available
    if _resnet is not None:
        try:
            pil = Image.fromarray(image_np)
            t = _img_transform(pil).unsqueeze(0)  # (1,3,224,224)
            with torch.no_grad():
                feat = _resnet(t)  # (1,512)
            gvec = compress(feat.to(torch.float32)).squeeze(0)  # (GLOBAL_DIM,)
        except Exception as e:
            log.warning("ResNet/compress failed: %s", e)
            gvec = torch.zeros((GLOBAL_DIM,), dtype=torch.float32)
    else:
        gvec = torch.zeros((GLOBAL_DIM,), dtype=torch.float32)

    # handle boxes existence
    boxes = getattr(yres, "boxes", None)
    if boxes is None:
        # dummy node: global_vec + zeros
        local_pad = torch.zeros((LOCAL_DIM,), dtype=torch.float32)
        node = torch.cat([gvec, local_pad], dim=0).unsqueeze(0)  # (1,IN_DIM)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return Data(x=node, edge_index=edge_index)

    try:
        xyxy = boxes.xyxy.cpu().numpy() 
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()
    except Exception:
        # fallback: dummy node
        local_pad = torch.zeros((LOCAL_DIM,), dtype=torch.float32)
        node = torch.cat([gvec, local_pad], dim=0).unsqueeze(0)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return Data(x=node, edge_index=edge_index)

    nodes = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(float)
        cx = float((x1 + x2) / 2.0 / max(1.0, W))
        cy = float((y1 + y2) / 2.0 / max(1.0, H))
        area = float(((x2 - x1) * (y2 - y1)) / (W * H + 1e-9))
        local_feat = torch.tensor([float(cls_ids[i]), float(confs[i]), cx, cy, area], dtype=torch.float32)
        node_vec = torch.cat([gvec.to(torch.float32), local_feat], dim=0)
        nodes.append(node_vec)

    if len(nodes) == 0:
        local_pad = torch.zeros((LOCAL_DIM,), dtype=torch.float32)
        node = torch.cat([gvec, local_pad], dim=0).unsqueeze(0)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        return Data(x=node, edge_index=edge_index)

    x = torch.stack(nodes, dim=0)  # (N, IN_DIM)

    # fully connected directed edges excluding self
    N = x.size(0)
    if N == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edges = [[i, j] for i in range(N) for j in range(N) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def load_checkpoint(path="hybrid_gnn_checkpoint.pt", map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ck = torch.load(path, map_location=map_location)
    if isinstance(ck, dict):
        if "model_state" in ck and "compress_state" in ck:
            return ck
        raw_keys = list(ck.keys())
        if raw_keys and ("conv1.att_src" in raw_keys or "conv1.lin.weight" in raw_keys or any(k.startswith("conv1") for k in raw_keys)):
            return {"model_state": ck, "compress_state": compress.state_dict()}
        if "model_state" in ck:
            ck.setdefault("compress_state", compress.state_dict())
            return ck

    raise RuntimeError("Unrecognized checkpoint format for: " + str(path))
