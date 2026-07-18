#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser

APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PKL = APRILCUBE_ROOT / "recordings" / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
SUPPORTED_FORMATS = {
    "aprilcube_012_offline_pose_vis_stream_v1",
    "aprilcube_012_raw_with_pose_stream_v1",
    "aprilcube_012_raw_with_final_postprocessed_pose_stream_v1",
    "aprilcube_raw_with_020_postprocessed_pose_stream_v1",
    "aprilcube_deeptag_fused_stream_v1",
    "deeptag_012_offline_stream_v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 013 offline pose result pkl with viser.")
    parser.add_argument("pkl_path", nargs="?", default=str(DEFAULT_PKL))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-width", type=int, default=960)
    return parser.parse_args()


def build_stream_index(
    path: Path,
    supported_formats: set[str] | None = None,
) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    allowed_formats = SUPPORTED_FORMATS if supported_formats is None else supported_formats
    with path.open("rb") as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get("format") not in allowed_formats:
            raise ValueError(f"Unsupported pkl format: {header.get('format', None)}")

        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if obj.get("type") == "frame":
                offsets.append(offset)
            elif obj.get("type") == "footer":
                footer = obj
                break
    if not offsets:
        raise ValueError(f"No frame records found in {path}")
    return header, offsets, footer


def load_frame(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("type") != "frame":
        raise ValueError(f"Offset {offset} is not a frame record")
    return obj


def find_raw_image_stream(
    pkl_path: Path,
    header: dict[str, Any],
    offsets: list[int],
) -> tuple[Path, list[int]] | None:
    first_frame = load_frame(pkl_path, offsets[0])
    if isinstance(first_frame.get("image_bgr"), np.ndarray):
        return pkl_path, offsets

    for field in ("source_pkl", "source_raw_pkl"):
        source_value = header.get(field)
        if not source_value:
            continue
        source_path = Path(source_value).expanduser().resolve()
        if not source_path.is_file() or source_path == pkl_path:
            continue
        try:
            _source_header, source_offsets, _source_footer = build_stream_index(
                source_path,
                SUPPORTED_FORMATS
                | {
                    "aprilcube_rs_raw_frame_stream_v1",
                    "aprilcube_raw_frame_stream_v1",
                },
            )
            source_first = load_frame(source_path, source_offsets[0])
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            continue
        if not isinstance(source_first.get("image_bgr"), np.ndarray):
            continue
        mapped_offsets: list[int] = []
        for processed_offset in offsets:
            processed_frame = load_frame(pkl_path, processed_offset)
            source_offset = processed_frame.get(
                "source_offset",
                processed_frame.get("raw_source_offset", None),
            )
            if source_offset is None:
                mapped_offsets = []
                break
            try:
                source_frame = load_frame(source_path, int(source_offset))
            except (OSError, ValueError, EOFError, pickle.UnpicklingError):
                mapped_offsets = []
                break
            if not isinstance(source_frame.get("image_bgr"), np.ndarray):
                mapped_offsets = []
                break
            mapped_offsets.append(int(source_offset))
        if len(mapped_offsets) == len(offsets):
            return source_path, mapped_offsets
        if len(source_offsets) == len(offsets):
            return source_path, source_offsets
    return None


def decode_jpeg_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image")
    return image


def bgr_to_rgb(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    if max_width <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    scale = float(max_width) / float(w)
    return cv2.resize(image_rgb, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return tuple(float(v) for v in quat)


def rvec_to_wxyz(rvec: Any) -> tuple[float, float, float, float]:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rotation_matrix_to_wxyz(rot)


def pose_markdown(frame: dict[str, Any]) -> str:
    pose = frame.get("pose", {})
    lines = [
        f"frame_index: `{frame.get('frame_index', '?')}`",
        f"loop_frame_idx: `{frame.get('loop_frame_idx', '?')}`",
        f"camera: `{frame.get('camera_name', '')}`",
        f"timestamp: `{frame.get('capture_timestamp', None)}`",
        f"success: `{pose.get('success', False)}`",
        f"pose_source: `{pose.get('pose_source', '')}`",
        f"quality_level: `{pose.get('quality_level', '')}`",
        f"quality_reason: `{pose.get('quality_reason', '')}`",
        f"pose_filled: `{pose.get('pose_filled', False)}`",
        f"reproj_error: `{pose.get('reproj_error', None)}`",
        f"n_tags: `{pose.get('n_tags', 0)}`",
        f"visible_faces: `{pose.get('visible_faces', [])}`",
    ]
    tvec = pose.get("tvec", None)
    if tvec is not None:
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        lines.append(f"t_mm: `({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})`")
    if pose.get("fill_original_failure_reason", ""):
        lines.append(f"fill_original_failure_reason: `{pose['fill_original_failure_reason']}`")
    return "\n".join(lines)


def update_cube(cube_handle: Any, frame: dict[str, Any]) -> None:
    pose = frame.get("pose", {})
    if not pose.get("success", False) or pose.get("rvec") is None or pose.get("tvec") is None:
        cube_handle.visible = False
        return
    cube_handle.visible = True
    cube_handle.position = tuple(float(v) for v in (np.asarray(pose["tvec"], dtype=np.float64).reshape(3) / 1000.0))
    cube_handle.wxyz = rvec_to_wxyz(pose["rvec"])


def main() -> None:
    args = parse_args()
    pkl_path = Path(args.pkl_path).expanduser().resolve()
    header, offsets, footer = build_stream_index(pkl_path)
    raw_image_stream = find_raw_image_stream(pkl_path, header, offsets)

    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction("-y")
    server.scene.world_axes.visible = False
    server.gui.set_panel_label("AprilCube Pose PKL")

    server.scene.add_frame(
        "/camera",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.05,
        axes_radius=0.002,
        origin_radius=0.0,
    )
    cube_handle = server.scene.add_frame(
        "/cube",
        axes_length=0.04,
        axes_radius=0.0015,
        origin_radius=0.002,
        visible=False,
    )

    frame_idx = 0
    is_playing = len(offsets) > 1
    loop_playback = True
    last_step_time = time.monotonic()

    with server.gui.add_folder("Replay"):
        play_checkbox = server.gui.add_checkbox("Play", initial_value=is_playing)
        loop_checkbox = server.gui.add_checkbox("Loop", initial_value=loop_playback)
        frame_slider = server.gui.add_slider("Frame", min=0, max=len(offsets) - 1, step=1, initial_value=0)
        status_text = server.gui.add_text("Status", initial_value="", disabled=True)

    with server.gui.add_folder("Images"):
        overlay_handle = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label="Overlay",
            format="jpeg",
            jpeg_quality=80,
        )
        raw_handle = server.gui.add_image(
            np.zeros((120, 160, 3), dtype=np.uint8),
            label="Original image_bgr (no overlay, before undistortion)",
            format="jpeg",
            jpeg_quality=80,
        )

    pose_text = server.gui.add_markdown("")
    server.gui.add_markdown(
        "\n".join(
            [
                f"pkl: `{pkl_path}`",
                f"frames: `{len(offsets)}`",
                f"format: `{header.get('format', '')}`",
                f"raw image source: `{raw_image_stream[0] if raw_image_stream else 'unavailable'}`",
                f"footer: `{footer}`",
            ]
        )
    )

    def clamp_idx(value: int) -> int:
        return max(0, min(int(value), len(offsets) - 1))

    def render(idx: int) -> None:
        frame = load_frame(pkl_path, offsets[idx])
        if raw_image_stream is not None:
            raw_path, raw_offsets = raw_image_stream
            raw_frame = load_frame(raw_path, raw_offsets[idx])
            raw_handle.image = bgr_to_rgb(raw_frame["image_bgr"], int(args.max_width))
        overlay_bgr = decode_jpeg_bgr(frame["overlay_jpeg"])
        overlay_handle.image = bgr_to_rgb(overlay_bgr, int(args.max_width))
        update_cube(cube_handle, frame)
        pose_text.content = pose_markdown(frame)
        pose = frame.get("pose", {})
        status_text.value = (
            f"{idx + 1}/{len(offsets)} "
            f"source={pose.get('pose_source', '')} "
            f"filled={pose.get('pose_filled', False)}"
        )

    @play_checkbox.on_update
    def _on_play(_event: Any) -> None:
        nonlocal is_playing, last_step_time
        is_playing = bool(play_checkbox.value)
        last_step_time = time.monotonic()

    @loop_checkbox.on_update
    def _on_loop(_event: Any) -> None:
        nonlocal loop_playback
        loop_playback = bool(loop_checkbox.value)

    @frame_slider.on_update
    def _on_frame(_event: Any) -> None:
        nonlocal frame_idx, last_step_time
        frame_idx = clamp_idx(int(frame_slider.value))
        last_step_time = time.monotonic()
        render(frame_idx)

    render(frame_idx)
    print(f"[INFO] Loaded {pkl_path} frames={len(offsets)}")
    print(
        "[INFO] Raw image source: "
        f"{raw_image_stream[0] if raw_image_stream else 'unavailable'}"
    )
    print(f"[INFO] Viser server: http://localhost:{args.port}")

    while True:
        if is_playing and len(offsets) > 1:
            now = time.monotonic()
            if now - last_step_time >= 1.0 / max(float(args.fps), 1e-6):
                next_idx = frame_idx + 1
                if next_idx >= len(offsets):
                    if loop_playback:
                        next_idx = 0
                    else:
                        next_idx = len(offsets) - 1
                        is_playing = False
                        play_checkbox.value = False
                frame_idx = next_idx
                frame_slider.value = frame_idx
                render(frame_idx)
                last_step_time = now
        time.sleep(0.005)


if __name__ == "__main__":
    main()
