import os, random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_video(frames, path, fps=20):
    """
    frames: list of HxW or HxWx3 uint8 arrays
    path: output .mp4 path under results/
    """
    ensure_dir(os.path.dirname(path))
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required for save_video()")

    # Ensure 3-channel
    proc = []
    for f in frames:
        if f.ndim == 2:
            f = np.stack([f]*3, axis=-1)
        proc.append(f)

    h, w, _ = proc[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in proc:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
