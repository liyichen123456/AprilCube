#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PKL = APRILCUBE_ROOT / "recordings" / "012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_POSE_PKL = APRILCUBE_ROOT / "recordings" / "014_offline_pose_vis_012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings" / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge 012 raw image stream pkl with 014 OpenCV/AprilCube pose visualization pkl."
    )
    parser.add_argument("--raw-pkl", default=str(DEFAULT_RAW_PKL))
    parser.add_argument("--pose-pkl", default=str(DEFAULT_POSE_PKL))
    parser.add_argument("--output-pkl", default=str(DEFAULT_OUTPUT_PKL))
    parser.add_argument(
        "--delete-inputs",
        action="store_true",
        help="Delete raw and pose input pkls after the merged pkl is verified.",
    )
    return parser.parse_args()


def build_stream_index(path: Path, expected_format: str) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open("rb") as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get("format") != expected_format:
            raise ValueError(f"Unsupported pkl format in {path}: {header.get('format', None)}")

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


def load_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("type") != "frame":
        raise ValueError(f"Offset {offset} in {path} is not a frame record")
    return obj


def verify_merged(path: Path, expected_frames: int) -> tuple[dict[str, Any], dict[str, int]]:
    header, offsets, footer = build_stream_index(path, "aprilcube_012_raw_with_pose_stream_v1")
    if len(offsets) != expected_frames:
        raise ValueError(f"Merged frame count mismatch: {len(offsets)} != {expected_frames}")
    if footer is None or int(footer.get("frame_count", -1)) != expected_frames:
        raise ValueError(f"Merged footer frame_count mismatch in {path}")

    pose_sources: dict[str, int] = {}
    success_count = 0
    for offset in offsets:
        record = load_at(path, offset)
        image = record.get("image_bgr", None)
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Merged frame at offset {offset} does not contain raw image_bgr ndarray")
        pose = record.get("pose", {})
        if pose.get("success", False):
            success_count += 1
        source = str(pose.get("pose_source", ""))
        pose_sources[source] = pose_sources.get(source, 0) + 1
    return header, {"frame_count": len(offsets), "success_count": success_count, **pose_sources}


def main() -> None:
    args = parse_args()
    raw_pkl = Path(args.raw_pkl).expanduser().resolve()
    pose_pkl = Path(args.pose_pkl).expanduser().resolve()
    output_pkl = Path(args.output_pkl).expanduser().resolve()

    raw_header, raw_offsets, raw_footer = build_stream_index(raw_pkl, "aprilcube_rs_raw_frame_stream_v1")
    pose_header, pose_offsets, pose_footer = build_stream_index(pose_pkl, "aprilcube_012_offline_pose_vis_stream_v1")
    if len(raw_offsets) != len(pose_offsets):
        raise ValueError(f"Frame count mismatch: raw={len(raw_offsets)} pose={len(pose_offsets)}")

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    total = len(raw_offsets)
    success_count = int(pose_header.get("metadata", {}).get("success_count", 0))
    filled_count = int(pose_header.get("metadata", {}).get("filled_pose_count", 0))
    t0 = time.perf_counter()

    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "aprilcube_012_raw_with_pose_stream_v1",
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_raw_pkl": str(raw_pkl),
                "source_pose_pkl": str(pose_pkl),
                "raw_header": raw_header,
                "raw_footer": raw_footer,
                "pose_header": pose_header,
                "pose_footer": pose_footer,
                "metadata": {
                    "script": str(Path(__file__).resolve()),
                    "method": "OpenCV/AprilCube + reprojection filtering + pose interpolation",
                    "frame_count": int(total),
                    "success_count": int(success_count),
                    "filled_pose_count": int(filled_count),
                    "raw_image_field": "image_bgr",
                    "raw_image_storage": "original numpy ndarray from 012 pkl",
                    "overlay_field": "overlay_jpeg",
                    "pose_field": "pose",
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for idx, (raw_offset, pose_offset) in enumerate(zip(raw_offsets, pose_offsets, strict=True)):
            raw_record = load_at(raw_pkl, raw_offset)
            pose_record = load_at(pose_pkl, pose_offset)

            if int(pose_record.get("source_offset", -1)) != int(raw_offset):
                raise ValueError(
                    f"source_offset mismatch at frame {idx}: "
                    f"pose={pose_record.get('source_offset')} raw={raw_offset}"
                )
            if pose_record.get("capture_timestamp", None) != raw_record.get("capture_timestamp", None):
                raise ValueError(f"capture_timestamp mismatch at frame {idx}")

            image_bgr = raw_record["image_bgr"]
            frame_record = {
                "type": "frame",
                "frame_index": int(idx),
                "raw_source_offset": int(raw_offset),
                "pose_source_offset": int(pose_offset),
                "device_name": str(raw_record.get("device_name", "")),
                "camera_name": str(raw_record.get("camera_name", "")),
                "loop_frame_idx": int(raw_record.get("loop_frame_idx", idx)),
                "capture_timestamp": raw_record.get("capture_timestamp", None),
                "write_monotonic": raw_record.get("write_monotonic", None),
                "shape": tuple(int(v) for v in np.asarray(image_bgr).shape),
                "dtype": str(np.asarray(image_bgr).dtype),
                "image_bgr": image_bgr,
                "overlay_shape": pose_record.get("overlay_shape", None),
                "overlay_format": pose_record.get("overlay_format", "jpeg_bgr"),
                "overlay_jpeg": pose_record["overlay_jpeg"],
                "pose": pose_record["pose"],
            }
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            done = idx + 1
            if done == total or done % 10 == 0:
                elapsed = time.perf_counter() - t0
                fps = done / max(elapsed, 1e-9)
                print(f"\r[INFO] Merging {done}/{total} fps={fps:.1f}", end="", flush=True)

        pickle.dump(
            {
                "type": "footer",
                "frame_count": int(total),
                "success_count": int(success_count),
                "filled_pose_count": int(filled_count),
                "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print()

    _, summary = verify_merged(output_pkl, total)
    print(f"[INFO] Saved merged pkl: {output_pkl}")
    print(f"[INFO] Verified merged pkl: {summary}")

    if args.delete_inputs:
        for path in (raw_pkl, pose_pkl):
            if path == output_pkl:
                raise ValueError(f"Refusing to delete output pkl: {path}")
            path.unlink()
            print(f"[INFO] Deleted input pkl: {path}")


if __name__ == "__main__":
    main()
