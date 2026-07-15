#!/usr/bin/env python3
from __future__ import annotations

import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import viser


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = (
    APRILCUBE_ROOT
    / "recordings/021_hand_back_sync_raw_frames_20260712_173546.pkl"
)
EXPECTED_FORMAT = "aprilcube_hand_back_software_synced_raw_v1"

VISER_HOST = "0.0.0.0"
VISER_PORT = 8093
VISER_MAX_IMAGE_WIDTH = 960
VISER_JPEG_QUALITY = 85
AUTO_PLAY_FPS = 5.0

THUMB_WEB_CAMERA = "thumb_web_cam"
MIDDLE_FINGER_CAMERA = "middle_finger_cam"


def print_index_progress(
    completed_bytes: int,
    total_bytes: int,
    *,
    finish_line: bool = False,
) -> None:
    width = 36
    ratio = 1.0 if total_bytes <= 0 else completed_bytes / total_bytes
    ratio = min(max(ratio, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(
        f"\r[INFO] Indexing PKL [{bar}] "
        f"{completed_bytes / (1024**3):.2f}/{total_bytes / (1024**3):.2f} GiB"
    )
    if finish_line:
        sys.stdout.write("\n")
    sys.stdout.flush()


def build_frame_pair_index(
    pkl_path: Path,
) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    file_size = pkl_path.stat().st_size
    frame_pair_offsets: list[int] = []
    footer: dict[str, Any] | None = None
    last_progress_time = 0.0

    with pkl_path.open("rb") as file:
        header = pickle.load(file)
        if not isinstance(header, dict) or header.get("type") != "header":
            raise ValueError(f"Invalid 021 PKL header: {pkl_path}")
        if header.get("format") != EXPECTED_FORMAT:
            raise ValueError(
                f"Unsupported PKL format {header.get('format')!r}; "
                f"expected {EXPECTED_FORMAT!r}"
            )

        while True:
            record_offset = file.tell()
            try:
                record = pickle.load(file)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            record_type = record.get("type")
            if record_type == "frame_pair":
                frame_pair_offsets.append(record_offset)
            elif record_type == "footer":
                footer = record
                break

            now = time.monotonic()
            if now - last_progress_time >= 0.5:
                print_index_progress(file.tell(), file_size)
                last_progress_time = now

    print_index_progress(file_size, file_size, finish_line=True)
    if not frame_pair_offsets:
        raise ValueError(f"No frame_pair records found in {pkl_path}")
    return header, frame_pair_offsets, footer


def load_frame_pair(pkl_path: Path, record_offset: int) -> dict[str, Any]:
    with pkl_path.open("rb") as file:
        file.seek(record_offset)
        record = pickle.load(file)
    if not isinstance(record, dict) or record.get("type") != "frame_pair":
        raise ValueError(f"Offset {record_offset} is not a frame_pair record")

    cameras = record.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError("frame_pair has no cameras mapping")
    for camera_name in (THUMB_WEB_CAMERA, MIDDLE_FINGER_CAMERA):
        camera_record = cameras.get(camera_name)
        if not isinstance(camera_record, dict):
            raise ValueError(f"frame_pair has no {camera_name} record")
        if not isinstance(camera_record.get("image_bgr"), np.ndarray):
            raise ValueError(f"{camera_name} record has no image_bgr ndarray")
    return record


def bgr_to_resized_rgb(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    height, width = image.shape[:2]
    if VISER_MAX_IMAGE_WIDTH > 0 and width > VISER_MAX_IMAGE_WIDTH:
        scale = VISER_MAX_IMAGE_WIDTH / width
        image = cv2.resize(
            image,
            (VISER_MAX_IMAGE_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def camera_status(camera_record: dict[str, Any]) -> str:
    return (
        f"sequence={camera_record.get('sequence', '?')} | "
        f"capture_timestamp={float(camera_record.get('capture_timestamp', 0.0)):.6f} | "
        f"shape={camera_record.get('shape', None)}"
    )


def pair_details_markdown(
    frame_index: int,
    total_frames: int,
    frame_pair: dict[str, Any],
) -> str:
    cameras = frame_pair["cameras"]
    thumb = cameras[THUMB_WEB_CAMERA]
    middle = cameras[MIDDLE_FINGER_CAMERA]
    return "\n".join(
        [
            f"**Frame pair:** `{frame_index}/{total_frames - 1}`",
            f"**Recorded pair index:** `{frame_pair.get('pair_index', '?')}`",
            f"**Capture pair sequence:** `{frame_pair.get('pair_sequence', '?')}`",
            f"**Pair timestamp:** `{float(frame_pair.get('pair_timestamp', 0.0)):.6f}`",
            f"**Thumb - middle skew:** `{float(frame_pair.get('signed_skew_ms', 0.0)):+.3f} ms`",
            "",
            f"**Thumb web:** `{camera_status(thumb)}`",
            f"**Middle finger:** `{camera_status(middle)}`",
        ]
    )


def main() -> None:
    pkl_path = PKL_PATH.expanduser().resolve()
    if not pkl_path.is_file():
        raise FileNotFoundError(f"021 PKL not found: {pkl_path}")

    print(f"[INFO] PKL: {pkl_path}")
    print(f"[INFO] Size: {pkl_path.stat().st_size / (1024**3):.2f} GiB")
    header, frame_pair_offsets, footer = build_frame_pair_index(pkl_path)
    total_frames = len(frame_pair_offsets)
    first_pair = load_frame_pair(pkl_path, frame_pair_offsets[0])
    first_cameras = first_pair["cameras"]

    server = viser.ViserServer(host=VISER_HOST, port=VISER_PORT)
    server.gui.set_panel_label("021 Synchronized Raw Images")

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame pair",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
        )
        auto_play = server.gui.add_checkbox("Auto play", initial_value=False)
        loop_playback = server.gui.add_checkbox("Loop", initial_value=True)
        status_text = server.gui.add_text(
            "Status",
            initial_value=f"1/{total_frames}",
            disabled=True,
        )

    with server.gui.add_folder("Thumb Web Cam"):
        thumb_image = server.gui.add_image(
            bgr_to_resized_rgb(first_cameras[THUMB_WEB_CAMERA]["image_bgr"]),
            label="Raw BGR frame",
            format="jpeg",
            jpeg_quality=VISER_JPEG_QUALITY,
        )

    with server.gui.add_folder("Middle Finger Cam"):
        middle_image = server.gui.add_image(
            bgr_to_resized_rgb(first_cameras[MIDDLE_FINGER_CAMERA]["image_bgr"]),
            label="Raw BGR frame",
            format="jpeg",
            jpeg_quality=VISER_JPEG_QUALITY,
        )

    with server.gui.add_folder("Frame Metadata"):
        frame_metadata = server.gui.add_markdown(
            pair_details_markdown(0, total_frames, first_pair)
        )
        metadata = header.get("metadata", {})
        server.gui.add_markdown(
            "\n".join(
                [
                    f"**PKL:** `{pkl_path}`",
                    f"**Format:** `{header.get('format', '')}`",
                    f"**Frame pairs:** `{total_frames}`",
                    f"**Capture size:** `{metadata.get('capture_size', None)}`",
                    f"**Requested FPS:** `{metadata.get('requested_fps', None)}`",
                    f"**Maximum pair skew:** `{metadata.get('max_pair_skew_ms', None)} ms`",
                    f"**Footer:** `{footer}`",
                ]
            )
        )

    render_lock = threading.Lock()

    def render(frame_index: int) -> None:
        with render_lock:
            frame_pair = load_frame_pair(
                pkl_path,
                frame_pair_offsets[frame_index],
            )
            cameras = frame_pair["cameras"]
            thumb_image.image = bgr_to_resized_rgb(
                cameras[THUMB_WEB_CAMERA]["image_bgr"]
            )
            middle_image.image = bgr_to_resized_rgb(
                cameras[MIDDLE_FINGER_CAMERA]["image_bgr"]
            )
            status_text.value = (
                f"{frame_index + 1}/{total_frames} | "
                f"skew={float(frame_pair.get('signed_skew_ms', 0.0)):+.3f} ms"
            )
            frame_metadata.content = pair_details_markdown(
                frame_index,
                total_frames,
                frame_pair,
            )

    print(f"[INFO] Indexed frame pairs: {total_frames}")
    print(f"[INFO] Viser: http://localhost:{VISER_PORT}")

    rendered_frame = 0
    last_playback_step = time.monotonic()
    while True:
        selected_frame = int(frame_slider.value)
        if selected_frame != rendered_frame:
            render(selected_frame)
            rendered_frame = selected_frame

        if bool(auto_play.value):
            now = time.monotonic()
            if now - last_playback_step >= 1.0 / max(AUTO_PLAY_FPS, 1e-6):
                next_frame = rendered_frame + 1
                if next_frame >= total_frames:
                    if bool(loop_playback.value):
                        next_frame = 0
                    else:
                        next_frame = total_frames - 1
                        auto_play.value = False
                frame_slider.value = next_frame
                last_playback_step = now
        else:
            last_playback_step = time.monotonic()
        time.sleep(0.01)


if __name__ == "__main__":
    main()
