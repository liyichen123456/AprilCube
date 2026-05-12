from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from aprilcube_pose_benchmark.common_io import gray_to_rgb

try:
    import viser
except ImportError:
    viser = None


def rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    quat_xyzw = R.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    x, y, z, w = quat_xyzw
    return float(w), float(x), float(y), float(z)


def draw_gray_overlay(detector: Any, gray: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    overlay = gray_to_rgb(gray).copy()
    detections = result.get("detections", [])
    if isinstance(detections, list):
        for tag_id, corners in detections:
            pts = np.round(np.asarray(corners, dtype=np.float64).reshape(4, 2)).astype(np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            center = tuple(np.round(np.mean(pts, axis=0)).astype(int))
            cv2.putText(
                overlay,
                f"ID:{int(tag_id)}",
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if result.get("success", False):
        overlay = detector.draw_result(overlay, result)
    return overlay


def play_results_in_viser(
    *,
    algorithm_name: str,
    output_dir: str | Path,
    frames_out: list[dict[str, Any]],
    playback_fps: float,
    host: str,
    port: int,
    loop: bool,
) -> None:
    if viser is None:
        raise ImportError("viser is not installed in this Python environment, but ENABLE_VISER=True.")
    if not frames_out:
        print("[WARNING] No frames to visualize in viser.")
        return
    server = viser.ViserServer(host=host, port=int(port))
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.scene.add_frame(
        "/camera",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.05,
        axes_radius=0.0015,
        origin_radius=0.0,
    )
    cube_handle = server.scene.add_frame(
        "/camera/cube",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.025,
        axes_radius=0.001,
        origin_radius=0.0,
    )
    folder = server.gui.add_folder("AprilCube Benchmark")
    with folder:
        image_handle = server.gui.add_image(
            np.zeros((240, 320, 3), dtype=np.uint8),
            label="CLAHE gray + cube box",
            format="jpeg",
            jpeg_quality=80,
        )
        frame_slider = server.gui.add_slider(
            "frame",
            min=0,
            max=max(0, len(frames_out) - 1),
            step=1,
            initial_value=0,
        )
        status = server.gui.add_text("status", initial_value=algorithm_name)
    print(f"[INFO] Viser: http://{host}:{int(port)}")
    print(f"[INFO] Output directory: {Path(output_dir).resolve()}")

    frame_dt = 1.0 / max(float(playback_fps), 1e-6)
    frame_idx = 0
    last_slider_value = -1
    while True:
        slider_value = int(frame_slider.value)
        if slider_value != last_slider_value:
            frame_idx = slider_value
            last_slider_value = slider_value
        record = frames_out[frame_idx]
        result = record["result"]
        image_handle.image = record["gray_overlay_rgb"]
        if result.get("success", False):
            rot, _ = cv2.Rodrigues(np.asarray(result["rvec"], dtype=np.float64).reshape(3, 1))
            t_m = np.asarray(result["tvec"], dtype=np.float64).reshape(3) / 1000.0
            cube_handle.wxyz = rotation_matrix_to_wxyz(rot)
            cube_handle.position = (float(t_m[0]), float(t_m[1]), float(t_m[2]))
            cube_handle.visible = True
            status.value = (
                f"{algorithm_name} | frame={record['frame_idx']} | "
                f"tags={result.get('n_tags', 0)} faces={len(result.get('visible_faces', []))} "
                f"reproj={float(result.get('reproj_error', float('nan'))):.2f}px"
            )
        else:
            cube_handle.visible = False
            status.value = f"{algorithm_name} | frame={record['frame_idx']} | no pose"
        time.sleep(frame_dt)
        if int(frame_slider.value) == frame_idx:
            frame_idx += 1
            if frame_idx >= len(frames_out):
                if not loop:
                    frame_idx = len(frames_out) - 1
                else:
                    frame_idx = 0
            frame_slider.value = frame_idx
