# modules/detector.py
"""
YOLOv8 detector wrapper (Ultralytics).
Provides Detector(model_name, device) with .predict(image_np) -> result object
The returned 'result' is the same as ultralytics' single result object:
 - result.boxes.xyxy  (tensor Nx4)
 - result.boxes.conf  (tensor N)
 - result.boxes.cls   (tensor N)
If ultralytics isn't available, returns a safe empty object.
"""
import logging
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
    UL_AVAILABLE = True
except Exception:
    UL_AVAILABLE = False

log = logging.getLogger("detector")


class _EmptyBoxes:
    def __init__(self):
        self.xyxy = None
        self.conf = None
        self.cls = None

class _EmptyResult:
    def __init__(self):
        self.boxes = None


class Detector:
    def __init__(self, model_name="yolov8n.pt", device="cpu"):
        """
        model_name: path like 'yolov8n.pt' or model alias
        device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.model = None

        if not UL_AVAILABLE:
            log.warning("Ultralytics YOLO not available. Detector will be a no-op.")
            return

        try:
            # allow passing 'yolov8n' or 'yolov8n.pt'
            self.model = YOLO(self.model_name)
            # don't move to device here; predict accepts device arg
            log.info("Initialized YOLO model: %s", self.model_name)
        except Exception as e:
            log.exception("Failed to init YOLO model: %s", e)
            self.model = None

    def predict(self, image_np, conf_thresh=0.2):
        """
        image_np: HxWx3 uint8 numpy array (BGR or RGB both accepted; we convert via PIL)
        Returns: single result object (or minimal safe object with boxes=None)
        """
        if self.model is None:
            return _EmptyResult()

        # Accept both BGR (cv2) and RGB; convert via PIL (which expects RGB)
        try:
            if isinstance(image_np, np.ndarray):
                # if OpenCV BGR, convert to RGB by swapping channels if mean seems BGR-like
                img = Image.fromarray(image_np[..., ::-1] if image_np.shape[2] == 3 else image_np)
            else:
                img = image_np
        except Exception:
            img = image_np

        # Use model.predict/predict() with PIL so Ultralytics handles preprocessing
        try:
            results = self.model.predict(img, device=self.device, verbose=False)
            if not results:
                return _EmptyResult()
            return results[0]
        except Exception as e:
            log.warning("YOLO predict failed: %s", e)
            return _EmptyResult()
