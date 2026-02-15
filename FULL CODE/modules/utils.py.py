# modules/utils.py
import time
import cv2
import os

def current_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def save_clip(frames, path, fps=10):
    """
    frames: list of numpy HWC uint8 images
    path: path to mp4 file
    """
    if len(frames) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def crop_box(img, bbox):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h,w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    return img[y1:y2, x1:x2]
