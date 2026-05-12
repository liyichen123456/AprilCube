from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 2.x pickles that reference numpy._core to load in NumPy 1.x."""
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def load_recording(pkl_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(pkl_path).expanduser().resolve()
    _install_numpy_pickle_compat()
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload, got {type(payload).__name__}")

    meta = payload.get("meta", {})
    frames = payload.get("frames", None)
    if isinstance(frames, np.ndarray):
        frames = frames.tolist()
    if not isinstance(frames, list):
        raise ValueError("Recording payload must contain a list-like 'frames' field.")
    return dict(meta), frames


def get_single_camera_name(frames: list[dict[str, Any]]) -> str:
    names: set[str] = set()
    for frame in frames:
        cameras = frame.get("cameras", {})
        if isinstance(cameras, dict):
            names.update(str(name) for name in cameras.keys())
    if not names:
        raise ValueError("No camera records found in recording.")
    if len(names) > 1:
        print(f"[INFO] Recording has multiple cameras {sorted(names)}, using the first one.")
    return sorted(names)[0]


def get_camera_record(frame: dict[str, Any], camera_name: str) -> dict[str, Any] | None:
    cameras = frame.get("cameras", {})
    if not isinstance(cameras, dict):
        return None
    record = cameras.get(camera_name, None)
    return record if isinstance(record, dict) else None


def get_detect_frame_bgr(record: dict[str, Any]) -> np.ndarray:
    image = record.get("detect_frame_bgr", None)
    if image is None:
        image = record.get("frame_bgr", None)
    if image is None:
        raise KeyError("Camera record has neither 'detect_frame_bgr' nor 'frame_bgr'.")
    return np.asarray(image, dtype=np.uint8)


def clahe_gray_from_bgr(
    image_bgr: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(int(v) for v in tile_grid_size),
    )
    return clahe.apply(gray)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.uint8)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    return arr


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_bgr, dtype=np.uint8)
    if arr.ndim == 2:
        return gray_to_rgb(arr)
    return arr[:, :, ::-1]
