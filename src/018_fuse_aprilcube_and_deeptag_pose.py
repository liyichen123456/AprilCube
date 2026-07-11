#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_APRIL_PKL = APRILCUBE_ROOT / "recordings" / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_DEEPTAG_PKL = APRILCUBE_ROOT / "recordings" / "016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings" / "018_fused_aprilcube_deeptag_012_rs_raw_frames_20260710_214336.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse AprilCube and DeepTag robust pose pkls into one per-frame stream.")
    parser.add_argument("--april-pkl", type=Path, default=DEFAULT_APRIL_PKL)
    parser.add_argument("--deeptag-pkl", type=Path, default=DEFAULT_DEEPTAG_PKL)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--deeptag-multiface-max-reproj", type=float, default=6.0)
    parser.add_argument("--deeptag-singleface-max-reproj", type=float, default=3.0)
    parser.add_argument("--april-direct-max-reproj", type=float, default=5.0)
    parser.add_argument("--april-real-max-reproj", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def build_stream_index(path: Path, supported_formats: set[str]) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open("rb") as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get("format") not in supported_formats:
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


def finite_pose(pose: dict[str, Any]) -> bool:
    if not bool(pose.get("success", False)):
        return False
    if pose.get("rvec") is None or pose.get("tvec") is None:
        return False
    try:
        chunks = [
            np.asarray(pose["rvec"], dtype=np.float64).reshape(-1),
            np.asarray(pose["tvec"], dtype=np.float64).reshape(-1),
            np.asarray([float(pose.get("reproj_error", float("inf")))], dtype=np.float64),
        ]
    except (TypeError, ValueError):
        return False
    return all(bool(np.all(np.isfinite(chunk))) for chunk in chunks)


def finite_transform(pose: dict[str, Any]) -> bool:
    if not bool(pose.get("success", False)):
        return False
    if pose.get("rvec") is None or pose.get("tvec") is None:
        return False
    try:
        chunks = [
            np.asarray(pose["rvec"], dtype=np.float64).reshape(-1),
            np.asarray(pose["tvec"], dtype=np.float64).reshape(-1),
        ]
    except (TypeError, ValueError):
        return False
    return all(bool(np.all(np.isfinite(chunk))) for chunk in chunks)


def reproj(pose: dict[str, Any]) -> float:
    try:
        value = float(pose.get("reproj_error", float("inf")))
    except (TypeError, ValueError):
        return float("inf")
    return value if math.isfinite(value) else float("inf")


def visible_face_count(pose: dict[str, Any]) -> int:
    faces = pose.get("visible_faces", []) or []
    return len(set(str(face) for face in faces))


def copy_pose_with_fusion(
    pose: dict[str, Any],
    *,
    source: str,
    quality_level: str,
    quality_reason: str,
) -> dict[str, Any]:
    out = copy.deepcopy(pose)
    out["success"] = bool(out.get("success", False))
    out["pose_source_original"] = str(out.get("pose_source", ""))
    out["pose_source"] = source
    out["quality_level"] = quality_level
    out["quality_reason"] = quality_reason
    out["fused_pose"] = True
    return out


def select_pose(
    april_pose: dict[str, Any],
    deeptag_pose: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    dt_ok = finite_pose(deeptag_pose)
    ap_ok = finite_pose(april_pose)
    ap_temporal_ok = finite_transform(april_pose)
    dt_reproj = reproj(deeptag_pose)
    ap_reproj = reproj(april_pose)
    dt_faces = visible_face_count(deeptag_pose)
    ap_filled = bool(april_pose.get("pose_filled", False))
    ap_source = str(april_pose.get("pose_source", ""))

    if dt_ok and dt_faces >= 2 and dt_reproj <= float(args.deeptag_multiface_max_reproj):
        return (
            copy_pose_with_fusion(
                deeptag_pose,
                source="fused_deeptag_robust_multiface",
                quality_level="A",
                quality_reason=f"deeptag_multiface_reproj:{dt_reproj:.2f}",
            ),
            "deeptag",
        )

    if (
        ap_ok
        and not ap_filled
        and ap_source == "aprilcube_detector"
        and ap_reproj <= float(args.april_direct_max_reproj)
    ):
        return (
            copy_pose_with_fusion(
                april_pose,
                source="fused_aprilcube_direct",
                quality_level="B",
                quality_reason=f"aprilcube_direct_reproj:{ap_reproj:.2f}",
            ),
            "april",
        )

    if dt_ok and dt_reproj <= float(args.deeptag_singleface_max_reproj):
        return (
            copy_pose_with_fusion(
                deeptag_pose,
                source="fused_deeptag_robust_singleface",
                quality_level="C",
                quality_reason=f"deeptag_single_or_low_reproj:{dt_reproj:.2f}",
            ),
            "deeptag",
        )

    if ap_ok and not ap_filled and ap_reproj <= float(args.april_real_max_reproj):
        return (
            copy_pose_with_fusion(
                april_pose,
                source="fused_aprilcube_real_fallback",
                quality_level="D",
                quality_reason=f"aprilcube_real_reproj:{ap_reproj:.2f}",
            ),
            "april",
        )

    if ap_temporal_ok:
        return (
            copy_pose_with_fusion(
                april_pose,
                source="fused_temporal_fill",
                quality_level="E",
                quality_reason="aprilcube_temporal_or_previous_fill",
            ),
            "april",
        )

    if dt_ok:
        return (
            copy_pose_with_fusion(
                deeptag_pose,
                source="fused_deeptag_last_resort",
                quality_level="F",
                quality_reason=f"deeptag_last_resort_reproj:{dt_reproj:.2f}",
            ),
            "deeptag",
        )

    return (
        {
            "success": False,
            "pose_source": "fused_failed",
            "quality_level": "Z",
            "quality_reason": "no_finite_candidate",
            "reproj_error": float("inf"),
            "pose_filled": False,
        },
        "april",
    )


def decode_jpeg_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG")
    return image


def encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image_bgr, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def overlay_with_label(frame: dict[str, Any], label: str, quality: str, quality_reason: str, jpeg_quality: int) -> bytes:
    image = decode_jpeg_bgr(frame["overlay_jpeg"])
    text = f"Fused {quality}: {label}"
    reason = quality_reason[:96]
    cv2.rectangle(image, (8, 8), (860, 62), (0, 0, 0), -1)
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, reason, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return encode_bgr_jpeg(image, jpeg_quality)


def minimal_pose(pose: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "success",
        "failure_reason",
        "n_tags",
        "n_inliers",
        "reproj_error",
        "tag_ids",
        "visible_faces",
        "pose_source",
        "pose_filled",
        "quality_level",
        "quality_reason",
        "robust_cluster_size",
        "robust_candidate_count",
        "single_tag_cfg_pose",
        "single_tag_id",
        "single_tag_face",
        "rvec",
        "tvec",
        "T",
    }
    return {key: copy.deepcopy(value) for key, value in pose.items() if key in keys}


def main() -> None:
    args = parse_args()
    april_pkl = Path(args.april_pkl).expanduser().resolve()
    deeptag_pkl = Path(args.deeptag_pkl).expanduser().resolve()
    output_pkl = Path(args.output_pkl).expanduser().resolve()

    april_header, april_offsets, april_footer = build_stream_index(april_pkl, {"aprilcube_012_raw_with_pose_stream_v1"})
    deeptag_header, deeptag_offsets, deeptag_footer = build_stream_index(deeptag_pkl, {"deeptag_012_offline_stream_v1"})
    if len(april_offsets) != len(deeptag_offsets):
        raise ValueError(f"Frame count mismatch: april={len(april_offsets)} deeptag={len(deeptag_offsets)}")

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    total = len(april_offsets)
    t0 = time.perf_counter()

    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "aprilcube_deeptag_fused_stream_v1",
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_april_pkl": str(april_pkl),
                "source_deeptag_pkl": str(deeptag_pkl),
                "april_header": april_header,
                "april_footer": april_footer,
                "deeptag_header": deeptag_header,
                "deeptag_footer": deeptag_footer,
                "metadata": {
                    "script": str(Path(__file__).resolve()),
                    "method": "quality-tier fusion of AprilCube offline poses and DeepTag robust single-frame poses",
                    "frame_count": int(total),
                    "thresholds": {
                        "deeptag_multiface_max_reproj": float(args.deeptag_multiface_max_reproj),
                        "deeptag_singleface_max_reproj": float(args.deeptag_singleface_max_reproj),
                        "april_direct_max_reproj": float(args.april_direct_max_reproj),
                        "april_real_max_reproj": float(args.april_real_max_reproj),
                    },
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for idx, (april_offset, deeptag_offset) in enumerate(zip(april_offsets, deeptag_offsets, strict=True)):
            april_frame = load_at(april_pkl, april_offset)
            deeptag_frame = load_at(deeptag_pkl, deeptag_offset)
            if int(april_frame.get("frame_index", idx)) != int(deeptag_frame.get("frame_index", idx)):
                raise ValueError(f"Frame index mismatch at {idx}")
            if april_frame.get("capture_timestamp", None) != deeptag_frame.get("capture_timestamp", None):
                raise ValueError(f"Timestamp mismatch at frame {idx}")

            fused_pose, overlay_source = select_pose(april_frame.get("pose", {}), deeptag_frame.get("pose", {}), args)
            quality = str(fused_pose.get("quality_level", "Z"))
            source = str(fused_pose.get("pose_source", ""))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(fused_pose.get("success", False)))

            overlay_frame = deeptag_frame if overlay_source == "deeptag" else april_frame
            overlay_jpeg = overlay_with_label(
                overlay_frame,
                source,
                quality,
                str(fused_pose.get("quality_reason", "")),
                int(args.jpeg_quality),
            )

            image_bgr = april_frame["image_bgr"]
            frame_record = {
                "type": "frame",
                "frame_index": int(idx),
                "camera_name": str(april_frame.get("camera_name", "")),
                "device_name": str(april_frame.get("device_name", "")),
                "loop_frame_idx": int(april_frame.get("loop_frame_idx", idx)),
                "capture_timestamp": april_frame.get("capture_timestamp", None),
                "shape": tuple(int(v) for v in np.asarray(image_bgr).shape),
                "dtype": str(np.asarray(image_bgr).dtype),
                "image_bgr": image_bgr,
                "overlay_shape": april_frame.get("overlay_shape", None),
                "overlay_format": "jpeg_bgr",
                "overlay_jpeg": overlay_jpeg,
                "pose": fused_pose,
                "pose_candidates": {
                    "aprilcube": minimal_pose(april_frame.get("pose", {})),
                    "deeptag_robust": minimal_pose(deeptag_frame.get("pose", {})),
                },
                "selected_overlay_source": overlay_source,
                "april_source_offset": int(april_offset),
                "deeptag_source_offset": int(deeptag_offset),
            }
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            done = idx + 1
            if done == total or done % 10 == 0:
                elapsed = time.perf_counter() - t0
                fps = done / max(elapsed, 1e-9)
                print(f"\r[INFO] Fusing {done}/{total} success={success_count}/{done} fps={fps:.1f}", end="", flush=True)

        pickle.dump(
            {
                "type": "footer",
                "frame_count": int(total),
                "success_count": int(success_count),
                "quality_counts": quality_counts,
                "source_counts": source_counts,
                "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print()
    print(f"[INFO] Saved fused pkl: {output_pkl}")
    print(f"[INFO] success={success_count}/{total} quality_counts={quality_counts}")
    print(f"[INFO] source_counts={source_counts}")


if __name__ == "__main__":
    main()
