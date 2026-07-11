#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial.transform import Rotation, Slerp

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
SCRIPT_022_PATH = THIS_FILE.parent / "022_benchmark_single_frame_recovery_methods.py"
DEFAULT_INPUT_PKL = APRILCUBE_ROOT / "recordings/024_temporal_outline_refine_recovery_conservative_fixed.pkl"
DEFAULT_RAW_PKL = APRILCUBE_ROOT / "recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings/025_global_temporal_filter_fill_final.pkl"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import aprilcube  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill remaining failed frames from the whole-sequence temporal pose trajectory.")
    parser.add_argument("--input-pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--raw-pkl", type=Path, default=DEFAULT_RAW_PKL)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--translation-smooth", type=float, default=2400.0)
    parser.add_argument("--max-bracket-gap", type=int, default=40)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image_bgr, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot


def make_pose_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rvec(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def draw_overlay(
    script012: Any,
    draw_detector: Any,
    detect_frame: np.ndarray,
    pose: dict[str, Any],
    quality: int,
) -> bytes:
    base = script012.make_detector_input_vis(detect_frame)
    result = {
        "success": bool(pose.get("success", False)),
        "detections": [],
        "rvec": np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1) if pose.get("success", False) else None,
        "tvec": np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1) if pose.get("success", False) else None,
        "reproj_error": float(pose.get("reproj_error", float("inf"))),
        "n_tags": int(pose.get("n_tags", 0) or 0),
        "visible_faces": set(pose.get("visible_faces", []) or []),
        "predicted": False,
    }
    vis = draw_detector.draw_result(base.copy(), result)
    cv2.rectangle(vis, (8, 8), (1180, 66), (0, 0, 0), -1)
    cv2.putText(vis, f"Global temporal fill: {pose.get('pose_source', '')}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, str(pose.get("quality_reason", ""))[:120], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return encode_bgr_jpeg(vis, quality)


def main() -> None:
    args = parse_args()
    bm = load_module(SCRIPT_022_PATH, "temporal025_benchmark_helpers")
    script012 = load_module(SCRIPT_012_PATH, "temporal025_script012")

    header, frames, footer = bm.load_pose_records(args.input_pkl)
    raw_header, raw_offsets, raw_footer = bm.build_stream_index(
        args.raw_pkl,
        {"aprilcube_rs_raw_frame_stream_v1", "aprilcube_012_raw_with_pose_stream_v1"},
    )
    metadata: dict[str, Any] = {}
    if raw_header.get("format") == "aprilcube_012_raw_with_pose_stream_v1":
        metadata.update(raw_header.get("raw_header", {}).get("metadata", {}) or {})
    metadata.update(raw_header.get("metadata", {}) or {})
    cube_cfg = Path(metadata["cube_cfg"]).expanduser().resolve()
    config, _face_id_sets = aprilcube.load_cube_config(str(cube_cfg / "config.json" if cube_cfg.is_dir() else cube_cfg))
    calib = script012.load_intrinsics_yaml(metadata.get("intrinsics_yaml"))
    image_size = tuple(int(v) for v in metadata.get("image_size", calib["image_size"]))
    undistort_pack = script012.create_undistort_maps(calib, image_size) if bool(metadata.get("undistort_for_detection", True)) else None
    camera_matrix = np.asarray(metadata.get("detection_camera_matrix", calib["K"]), dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get("detector_dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)
    if undistort_pack is not None:
        camera_matrix = np.asarray(metadata.get("detection_camera_matrix", undistort_pack[2]), dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(metadata.get("detector_dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)
    raw_offset_by_frame = {
        int(bm.load_at(args.raw_pkl, offset).get("frame_index", idx)): int(offset)
        for idx, offset in enumerate(raw_offsets)
    }
    draw_detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=script012.camera_matrix_to_intrinsic_dict(camera_matrix),
        dist_coeffs=dist_coeffs,
        enable_filter=False,
        fast=True,
    )

    indices = sorted(frames)
    valid_indices = [idx for idx in indices if bool(frames[idx].get("pose", {}).get("success", False))]
    failed_indices = [idx for idx in indices if idx not in valid_indices]
    if len(valid_indices) < 4:
        raise RuntimeError("Need at least 4 valid poses for global temporal fill")

    x = np.asarray(valid_indices, dtype=np.float64)
    translations = np.vstack([np.asarray(frames[idx]["pose"]["tvec"], dtype=np.float64).reshape(3) for idx in valid_indices])
    # Smoothing spline for translation; rotations use Slerp across the same whole-sequence valid trajectory.
    splines = [
        UnivariateSpline(x, translations[:, dim], k=3, s=float(args.translation_smooth))
        for dim in range(3)
    ]
    rotations = Rotation.from_matrix(
        np.stack([rotation_from_rvec(frames[idx]["pose"]["rvec"]) for idx in valid_indices], axis=0)
    )
    slerp = Slerp(x, rotations)

    filled: dict[int, dict[str, Any]] = {}
    rejected: dict[int, str] = {}
    for idx in failed_indices:
        prevs = [v for v in valid_indices if v < idx]
        nexts = [v for v in valid_indices if v > idx]
        if not prevs or not nexts:
            rejected[idx] = "no_bracketing_valid_pose"
            continue
        prev_idx, next_idx = prevs[-1], nexts[0]
        gap = int(next_idx - prev_idx)
        if gap > int(args.max_bracket_gap):
            rejected[idx] = f"bracket_gap_too_large:{gap}>{int(args.max_bracket_gap)}"
            continue
        tvec = np.array([spline(float(idx)) for spline in splines], dtype=np.float64).reshape(3, 1)
        rvec = slerp([float(idx)]).as_rotvec()[0].reshape(3, 1).astype(np.float64)
        raw_record = bm.load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
        image = np.asarray(raw_record["image_bgr"], dtype=np.uint8)
        detect_frame = script012.undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        edge_score = bm.edge_alignment_score(
            gray,
            {"success": True, "rvec": rvec, "tvec": tvec},
            config=config,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        filled[idx] = {
            "success": True,
            "pose_source": "stage11_global_temporal_filter_fill",
            "quality_level": "F",
            "quality_reason": f"global_temporal_fill;bracket:{prev_idx}-{next_idx};gap:{gap};edge:{edge_score:.2f}",
            "pose_filled": True,
            "temporal_filter_fill": True,
            "single_frame_only": False,
            "rvec": rvec,
            "tvec": tvec,
            "T": make_pose_transform(rvec, tvec),
            "reproj_error": float("nan"),
            "n_tags": 0,
            "tag_ids": [],
            "visible_faces": [],
            "edge_score": float(edge_score),
            "prev_success_frame": int(prev_idx),
            "next_success_frame": int(next_idx),
            "bracket_gap": int(gap),
            "translation_smooth": float(args.translation_smooth),
        }

    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    with args.output_pkl.open("wb") as f:
        out_header = copy.deepcopy(header)
        out_header["created_wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        out_header["source_input_pkl"] = str(args.input_pkl.resolve())
        out_header["source_raw_pkl"] = str(args.raw_pkl.resolve())
        out_header["raw_footer"] = raw_footer
        out_header["metadata"] = {
            **(out_header.get("metadata", {}) or {}),
            "script": str(THIS_FILE),
            "method": "fill only remaining failed frames from whole-sequence temporal pose trajectory",
            "translation_smooth": float(args.translation_smooth),
            "max_bracket_gap": int(args.max_bracket_gap),
        }
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            if idx in filled:
                raw_record = bm.load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
                image = np.asarray(raw_record["image_bgr"], dtype=np.uint8)
                detect_frame = script012.undistort_frame(image, undistort_pack)
                frame["pose_original"] = copy.deepcopy(frame.get("pose", {}))
                frame["pose"] = filled[idx]
                frame["selected_stage"] = "stage11_global_temporal_filter_fill"
                frame["overlay_jpeg"] = draw_overlay(script012, draw_detector, detect_frame, filled[idx], int(args.jpeg_quality))
                frame["overlay_format"] = "jpeg_bgr"
                frame["overlay_shape"] = tuple(int(v) for v in detect_frame.shape)
            elif idx in rejected:
                frame["pose_original"] = copy.deepcopy(frame.get("pose", {}))
                frame["pose"] = {
                    "success": False,
                    "pose_source": "fused_failed",
                    "quality_level": "Z",
                    "quality_reason": rejected[idx],
                    "failure_reason": rejected[idx],
                    "reproj_error": float("inf"),
                    "pose_filled": False,
                    "single_frame_only": False,
                }
            pose = frame.get("pose", {})
            quality = str(pose.get("quality_level", "Z"))
            source = str(pose.get("pose_source", ""))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(pose.get("success", False)))
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        out_footer = {
            "type": "footer",
            "frame_count": len(indices),
            "success_count": int(success_count),
            "filled_count": len(filled),
            "filled_frames": sorted(int(v) for v in filled),
            "remaining_failed_frames": [int(idx) for idx in indices if idx not in valid_indices and idx not in filled],
            "rejected_temporal_fill_reasons": rejected,
            "quality_counts": quality_counts,
            "source_counts": source_counts,
            "input_footer": footer,
            "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        pickle.dump(out_footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] saved {args.output_pkl}")
    print(f"[INFO] filled={len(filled)} frames={sorted(filled)}")
    print(f"[INFO] success={success_count}/{len(indices)}")
    print(f"[INFO] remaining_failed={out_footer['remaining_failed_frames']}")


if __name__ == "__main__":
    main()
