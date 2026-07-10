#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
DEEPTAG_ROOT = APRILCUBE_ROOT / "thirdparty" / "deeptag-pytorch"
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
DEFAULT_INPUT_PKL = APRILCUBE_ROOT / "recordings" / "012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_MERGED_INPUT_PKL = APRILCUBE_ROOT / "recordings" / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings" / "016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import aprilcube  # noqa: E402
from aprilcube.detect import estimate_pose, estimate_single_tag_cube_pose  # noqa: E402


def load_script012_module() -> Any:
    spec = importlib.util.spec_from_file_location("aprilcube_replay_012_for_deeptag", SCRIPT_012_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load 012 logic from {SCRIPT_012_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["aprilcube_replay_012_for_deeptag"] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepTag AprilTag detector on 012 raw-frame pkl.")
    parser.add_argument(
        "pkl_path",
        nargs="?",
        default=str(DEFAULT_MERGED_INPUT_PKL if DEFAULT_MERGED_INPUT_PKL.exists() else DEFAULT_INPUT_PKL),
    )
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--intrinsics-yaml", type=Path, default=None)
    parser.add_argument("--cube-cfg", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Force DeepTag to run on CPU.")
    parser.add_argument("--detect-scale", type=float, default=-1.0, help="DeepTag detect scale; negative means default.")
    parser.add_argument("--min-center-score", type=float, default=0.2)
    parser.add_argument("--min-corner-score", type=float, default=0.2)
    parser.add_argument("--hamming-dist", type=int, default=8)
    parser.add_argument("--stg2-iter-num", type=int, default=2)
    parser.add_argument("--batch-size-stg2", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--quiet-deeptag", action="store_true", help="Suppress DeepTag per-frame stdout.")
    parser.add_argument(
        "--corner-order",
        choices=("id", "rev", "rot180", "rev_rot180"),
        default="rot180",
        help="Corner order transform used to align DeepTag corners to AprilCube corner convention.",
    )
    parser.add_argument(
        "--pose-mode",
        choices=("robust-cluster", "aprilcube-post"),
        default="robust-cluster",
        help="How to turn DeepTag corners into a cube pose.",
    )
    parser.add_argument("--robust-min-tags", type=int, default=2)
    parser.add_argument("--robust-cluster-trans-mm", type=float, default=70.0)
    parser.add_argument("--robust-cluster-rot-deg", type=float, default=55.0)
    parser.add_argument("--robust-max-reproj", type=float, default=12.0)
    parser.add_argument("--robust-single-tag-max-reproj", type=float, default=4.0)
    return parser.parse_args()


SUPPORTED_INPUT_FORMATS = {
    "aprilcube_rs_raw_frame_stream_v1",
    "aprilcube_012_raw_with_pose_stream_v1",
}


def build_stream_index(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open("rb") as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get("format") not in SUPPORTED_INPUT_FORMATS:
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


def load_frame_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("type") != "frame":
        raise ValueError(f"Offset {offset} is not a frame")
    return obj


def input_metadata(header: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_header = header.get("raw_header", {})
    pose_header = header.get("pose_header", {})
    if isinstance(raw_header, dict):
        metadata.update(raw_header.get("metadata", {}) or {})
    if isinstance(pose_header, dict):
        metadata.update(pose_header.get("metadata", {}) or {})
    metadata.update(header.get("metadata", {}) or {})
    return metadata


def encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.asarray(image_bgr, dtype=np.uint8),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(quality, 100)))],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def load_deeptag_engine(
    *,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_m: float,
    args: argparse.Namespace,
) -> Any:
    if not DEEPTAG_ROOT.exists():
        raise FileNotFoundError(f"DeepTag repo not found: {DEEPTAG_ROOT}")
    if str(DEEPTAG_ROOT) not in sys.path:
        sys.path.insert(0, str(DEEPTAG_ROOT))

    old_cwd = Path.cwd()
    os.chdir(DEEPTAG_ROOT)
    try:
        from deeptag_model_setting import load_deeptag_models
        from marker_dict_setting import load_marker_codebook
        from stag_decode.detection_engine import DetectionEngine

        device = "cpu" if args.cpu else None
        model_detector, model_decoder, device, tag_type, grid_size_cand_list = load_deeptag_models("apriltag", device)
        codebook = load_marker_codebook(str(DEEPTAG_ROOT / "codebook" / "apriltag_codebook.txt"), tag_type)
        engine = DetectionEngine(
            model_detector,
            model_decoder,
            device,
            tag_type,
            grid_size_cand_list,
            stg2_iter_num=int(args.stg2_iter_num),
            min_center_score=float(args.min_center_score),
            min_corner_score=float(args.min_corner_score),
            batch_size_stg2=int(args.batch_size_stg2),
            hamming_dist=int(args.hamming_dist),
            cameraMatrix=np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3),
            distCoeffs=np.asarray(dist_coeffs, dtype=np.float32).reshape(-1),
            codebook=codebook,
            tag_real_size_in_meter_dict={-1: float(tag_size_m)},
        )
        return engine
    finally:
        os.chdir(old_cwd)


def make_runtime(script012: Any, metadata: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    intrinsics_yaml = Path(
        args.intrinsics_yaml or metadata.get("intrinsics_yaml") or script012.DEFAULT_INTRINSICS_YAML
    ).expanduser().resolve()
    cube_cfg = Path(args.cube_cfg or metadata.get("cube_cfg") or script012.DEFAULT_CUBE_CFG).expanduser().resolve()

    calib = script012.load_intrinsics_yaml(intrinsics_yaml)
    image_size = tuple(int(v) for v in metadata.get("image_size", calib["image_size"]))
    raw_camera_matrix = np.asarray(metadata.get("raw_camera_matrix", calib["K"]), dtype=np.float64).reshape(3, 3)
    raw_dist_coeffs = np.asarray(metadata.get("raw_dist_coeffs", calib["dist"]), dtype=np.float64).reshape(-1)
    undistort_pack = None
    detection_camera_matrix = raw_camera_matrix.copy()
    detector_dist_coeffs = raw_dist_coeffs
    if bool(metadata.get("undistort_for_detection", True)) and not args.no_undistort:
        undistort_pack = script012.create_undistort_maps(calib, image_size)
        if undistort_pack is not None:
            detection_camera_matrix = undistort_pack[2]
            detector_dist_coeffs = np.zeros(5, dtype=np.float64)
        if metadata.get("detection_camera_matrix", None) is not None:
            detection_camera_matrix = np.asarray(metadata["detection_camera_matrix"], dtype=np.float64).reshape(3, 3)
        if metadata.get("detector_dist_coeffs", None) is not None:
            detector_dist_coeffs = np.asarray(metadata["detector_dist_coeffs"], dtype=np.float64).reshape(-1)

    cube_config, face_id_sets = aprilcube.load_cube_config(str(cube_cfg / "config.json" if cube_cfg.is_dir() else cube_cfg))
    tag_corner_map = aprilcube.build_tag_corner_map(cube_config)
    april_post_detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=script012.camera_matrix_to_intrinsic_dict(detection_camera_matrix),
        dist_coeffs=detector_dist_coeffs,
        enable_filter=False,
        fast=True,
    )
    # process_detections stores debug_viz via draw_result. Disable that path for
    # post-processing so invalid rejected poses cannot crash visualization code.
    april_post_detector.draw_result = lambda frame, result: frame
    april_draw_detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=script012.camera_matrix_to_intrinsic_dict(detection_camera_matrix),
        dist_coeffs=detector_dist_coeffs,
        enable_filter=False,
        fast=True,
    )
    return {
        "intrinsics_yaml": intrinsics_yaml,
        "cube_cfg": cube_cfg,
        "image_size": image_size,
        "undistort_pack": undistort_pack,
        "detection_camera_matrix": detection_camera_matrix,
        "detector_dist_coeffs": detector_dist_coeffs,
        "cube_config": cube_config,
        "face_id_sets": face_id_sets,
        "tag_corner_map": tag_corner_map,
        "april_post_detector": april_post_detector,
        "april_draw_detector": april_draw_detector,
    }


def detection_frame(script012: Any, runtime: dict[str, Any], image_bgr: np.ndarray) -> np.ndarray:
    target_w, target_h = runtime["image_size"]
    h, w = image_bgr.shape[:2]
    if (w, h) != (target_w, target_h):
        image_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return script012.undistort_frame(image_bgr, runtime["undistort_pack"])


CORNER_ORDER_TRANSFORMS = {
    "id": (0, 1, 2, 3),
    "rot90": (1, 2, 3, 0),
    "rev": (0, 3, 2, 1),
    "rot180": (2, 3, 0, 1),
    "rot270": (3, 0, 1, 2),
    "rev_rot90": (1, 0, 3, 2),
    "rev_rot180": (2, 1, 0, 3),
    "rev_rot270": (3, 2, 1, 0),
}


def quad_quality(corners: np.ndarray) -> float:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    area = float(abs(cv2.contourArea(corners.astype(np.float32))))
    if area <= 0.0:
        return 0.0
    edges = [
        float(np.linalg.norm(corners[(idx + 1) % 4] - corners[idx]))
        for idx in range(4)
    ]
    min_edge = min(edges)
    max_edge = max(edges)
    if max_edge <= 1e-9:
        return 0.0
    return area * (min_edge / max_edge)


def deeptag_detections_to_raw_corners(
    engine: Any,
    decoded_tags: list[dict[str, Any]],
    *,
    valid_ids: set[int],
) -> tuple[list[tuple[int, np.ndarray]], dict[str, int]]:
    best_by_id: dict[int, tuple[float, np.ndarray]] = {}
    rois = getattr(engine, "rois_info", [])
    raw_valid = 0
    invalid_id = 0
    duplicate_id = 0
    for idx, decoded in enumerate(decoded_tags):
        if not decoded.get("is_valid", False):
            continue
        raw_valid += 1
        tag_id = int(decoded.get("tag_id", -1))
        if tag_id < 0 or idx >= len(rois):
            invalid_id += 1
            continue
        if tag_id not in valid_ids:
            invalid_id += 1
            continue
        roi_info = rois[idx]
        roi = roi_info.get("ordered_corners", roi_info) if isinstance(roi_info, dict) else roi_info
        main_idx = int(decoded.get("main_idx", 0))
        ordered = list(roi[main_idx:]) + list(roi[:main_idx])
        corners = np.asarray(ordered, dtype=np.float64).reshape(4, 2)
        quality = quad_quality(corners)
        if tag_id in best_by_id:
            duplicate_id += 1
            if quality <= best_by_id[tag_id][0]:
                continue
        best_by_id[tag_id] = (quality, corners)
    detections = [
        (tag_id, corners)
        for tag_id, (_quality, corners) in sorted(best_by_id.items())
    ]
    stats = {
        "raw_valid_decoded": int(raw_valid),
        "invalid_or_wrong_id": int(invalid_id),
        "duplicate_id": int(duplicate_id),
        "kept": int(len(detections)),
    }
    return detections, stats


def apply_corner_order(
    detections: list[tuple[int, np.ndarray]],
    corner_order: str,
) -> list[tuple[int, np.ndarray]]:
    order = list(CORNER_ORDER_TRANSFORMS[corner_order])
    return [
        (int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2)[order])
        for tag_id, corners in detections
    ]


def _jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonish(item) for item in value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"unit_tag", "H_crop"}:
                continue
            out[str(key)] = _jsonish(item)
        return out
    return str(value)


def sanitize_decoded_tags(decoded_tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_jsonish(tag) for tag in decoded_tags]


def sanitize_pose_result(result: dict[str, Any]) -> dict[str, Any]:
    skip = {"debug_viz"}
    out: dict[str, Any] = {}
    for key, value in result.items():
        if key in skip:
            continue
        if key == "detections":
            detections = []
            for item in value or []:
                if len(item) != 2:
                    continue
                tag_id, corners = item
                detections.append(
                    {
                        "tag_id": int(tag_id),
                        "corners_xy": np.asarray(corners, dtype=np.float64).reshape(4, 2).tolist(),
                    }
                )
            out[key] = detections
            continue
        out[key] = _jsonish(value)
    return out


def reset_aprilcube_single_frame_state(detector: Any) -> None:
    detector.prev_rvec = None
    detector.prev_tvec = None
    detector._prev_gray = None
    detector._prev_corners_2d = None
    detector._prev_corners_3d = None
    if getattr(detector, "pose_filter", None) is not None:
        detector.pose_filter.reset()


def finite_pose_success(result: dict[str, Any]) -> bool:
    if not bool(result.get("success", False)):
        return False
    if result.get("rvec", None) is None or result.get("tvec", None) is None:
        return False
    values = [
        np.asarray(result["rvec"], dtype=np.float64).reshape(-1),
        np.asarray(result["tvec"], dtype=np.float64).reshape(-1),
        np.asarray([float(result.get("reproj_error", float("inf")))], dtype=np.float64),
    ]
    return all(bool(np.all(np.isfinite(chunk))) for chunk in values)


def rvec_to_rot(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot


def rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    ra = rvec_to_rot(rvec_a)
    rb = rvec_to_rot(rvec_b)
    cos_angle = np.clip((np.trace(ra @ rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_delta_mm(tvec_a: Any, tvec_b: Any) -> float:
    ta = np.asarray(tvec_a, dtype=np.float64).reshape(3)
    tb = np.asarray(tvec_b, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(ta - tb))


def visible_faces_for_ids(face_id_sets: dict[str, set[int]], tag_ids: list[int]) -> set[str]:
    visible: set[str] = set()
    for tag_id in tag_ids:
        for face_name, ids in face_id_sets.items():
            if int(tag_id) in ids:
                visible.add(str(face_name))
    return visible


def face_normals_ok(rvec: Any, visible_faces: set[str]) -> bool:
    rot = rvec_to_rot(rvec)
    for visible_face in visible_faces:
        for face_def in aprilcube.FACE_DEFS:
            if face_def[0] != visible_face:
                continue
            normal_obj = np.zeros(3, dtype=np.float64)
            normal_obj[face_def[1]] = face_def[2]
            normal_cam = rot @ normal_obj
            if float(normal_cam[2]) > 0.0:
                return False
            break
    return True


def per_tag_reprojection_errors(
    detections: list[tuple[int, np.ndarray]],
    tag_corner_map: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: Any,
    tvec: Any,
) -> dict[int, float]:
    per_tag: dict[int, float] = {}
    for tag_id, corners_2d in detections:
        corners_3d = tag_corner_map.get(int(tag_id))
        if corners_3d is None:
            continue
        projected, _ = cv2.projectPoints(
            np.asarray(corners_3d, dtype=np.float64).reshape(4, 3),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            camera_matrix,
            dist_coeffs,
        )
        err = np.linalg.norm(
            np.asarray(corners_2d, dtype=np.float64).reshape(4, 2) - projected.reshape(4, 2),
            axis=1,
        )
        per_tag[int(tag_id)] = float(np.mean(err))
    return per_tag


def solve_pose_from_detections(
    detections: list[tuple[int, np.ndarray]],
    runtime: dict[str, Any],
    *,
    seed_rvec: Any | None = None,
    seed_tvec: Any | None = None,
    max_reproj: float,
) -> dict[str, Any]:
    tag_corner_map = runtime["tag_corner_map"]
    object_chunks: list[np.ndarray] = []
    image_chunks: list[np.ndarray] = []
    used: list[tuple[int, np.ndarray]] = []
    for tag_id, corners_2d in detections:
        corners_3d = tag_corner_map.get(int(tag_id))
        if corners_3d is None:
            continue
        object_chunks.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
        image_chunks.append(np.asarray(corners_2d, dtype=np.float64).reshape(4, 2))
        used.append((int(tag_id), np.asarray(corners_2d, dtype=np.float64).reshape(4, 2)))

    if not used:
        return {"success": False, "failure_reason": "no_cluster_detections"}

    object_points = np.vstack(object_chunks).astype(np.float64)
    image_points = np.vstack(image_chunks).astype(np.float64)
    if len(used) == 1:
        ok, rvec, tvec, reproj_err, inliers, meta = estimate_single_tag_cube_pose(
            used,
            tag_corner_map,
            runtime["face_id_sets"],
            runtime["detection_camera_matrix"],
            runtime["detector_dist_coeffs"],
            None if seed_rvec is None else np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1),
            None if seed_tvec is None else np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1),
            allow_corner_rotations=False,
        )
    else:
        ok, rvec, tvec, reproj_err, inliers = estimate_pose(
            object_points,
            image_points,
            runtime["detection_camera_matrix"],
            runtime["detector_dist_coeffs"],
            None if seed_rvec is None else np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1),
            None if seed_tvec is None else np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1),
        )
        meta = {}

    if (
        not ok
        or rvec is None
        or tvec is None
        or not np.all(np.isfinite(rvec))
        or not np.all(np.isfinite(tvec))
        or not np.isfinite(float(reproj_err))
        or float(np.asarray(tvec).reshape(3)[2]) <= 0.0
    ):
        return {
            "success": False,
            "failure_reason": "cluster_pnp_failed",
            "detections": used,
            "n_tags": len(used),
            "tag_ids": [tag_id for tag_id, _ in used],
            "reproj_error": float("inf"),
        }

    visible_faces = visible_faces_for_ids(runtime["face_id_sets"], [tag_id for tag_id, _ in used])
    if not face_normals_ok(rvec, visible_faces):
        return {
            "success": False,
            "failure_reason": "cluster_face_normal_away",
            "detections": used,
            "n_tags": len(used),
            "tag_ids": [tag_id for tag_id, _ in used],
            "reproj_error": float("inf"),
        }

    # Trim tag-level outliers once or twice, then re-solve from the previous pose.
    for _iteration in range(2):
        per_tag = per_tag_reprojection_errors(
            used,
            tag_corner_map,
            runtime["detection_camera_matrix"],
            runtime["detector_dist_coeffs"],
            rvec,
            tvec,
        )
        if len(per_tag) < 3:
            break
        vals = np.asarray(list(per_tag.values()), dtype=np.float64)
        median_err = float(np.median(vals))
        keep_thresh = min(max(median_err * 3.0, 5.0), float(max_reproj))
        keep_ids = {tag_id for tag_id, err in per_tag.items() if err <= keep_thresh}
        if len(keep_ids) == len(used) or len(keep_ids) < 1:
            break
        used = [(tag_id, corners) for tag_id, corners in used if tag_id in keep_ids]
        return solve_pose_from_detections(
            used,
            runtime,
            seed_rvec=rvec,
            seed_tvec=tvec,
            max_reproj=max_reproj,
        )

    per_tag = per_tag_reprojection_errors(
        used,
        tag_corner_map,
        runtime["detection_camera_matrix"],
        runtime["detector_dist_coeffs"],
        rvec,
        tvec,
    )
    if float(reproj_err) > float(max_reproj):
        return {
            "success": False,
            "failure_reason": f"cluster_reproj_too_high:{float(reproj_err):.2f}>{float(max_reproj):.2f}",
            "detections": used,
            "n_tags": len(used),
            "tag_ids": [tag_id for tag_id, _ in used],
            "reproj_error": float("inf"),
            "per_tag_reproj_error": per_tag,
        }

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rvec_to_rot(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    result = {
        "success": True,
        "failure_reason": "",
        "detections": used,
        "n_tags": len(used),
        "tag_ids": [int(tag_id) for tag_id, _ in used],
        "visible_faces": visible_faces,
        "predicted": False,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "T": transform,
        "reproj_error": float(reproj_err),
        "n_inliers": 0 if inliers is None else int(len(inliers)),
        "per_tag_reproj_error": per_tag,
    }
    result.update(meta)
    return result


def robust_cluster_pose(
    raw_detections: list[tuple[int, np.ndarray]],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for tag_id, raw_corners in raw_detections:
        for order_name, order in CORNER_ORDER_TRANSFORMS.items():
            corners = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)[list(order)]
            ok, rvec, tvec, reproj, _inliers, meta = estimate_single_tag_cube_pose(
                [(int(tag_id), corners)],
                runtime["tag_corner_map"],
                runtime["face_id_sets"],
                runtime["detection_camera_matrix"],
                runtime["detector_dist_coeffs"],
                allow_corner_rotations=False,
            )
            if (
                not ok
                or rvec is None
                or tvec is None
                or not np.all(np.isfinite(rvec))
                or not np.all(np.isfinite(tvec))
                or not np.isfinite(float(reproj))
                or float(reproj) > float(args.robust_single_tag_max_reproj)
            ):
                continue
            candidates.append(
                {
                    "tag_id": int(tag_id),
                    "corners": corners,
                    "corner_order": order_name,
                    "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                    "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    "reproj_error": float(reproj),
                    "face": meta.get("single_tag_face", None),
                }
            )

    if not candidates:
        return (
            {
                "success": False,
                "failure_reason": "robust_no_single_tag_candidates",
                "detections": [],
                "n_tags": 0,
                "tag_ids": [],
                "reproj_error": float("inf"),
            },
            [],
            {"candidate_count": 0, "cluster_size": 0},
        )

    best_cluster: list[dict[str, Any]] = []
    best_score: tuple[int, float, float] | None = None
    for seed in candidates:
        by_tag: dict[int, tuple[float, dict[str, Any]]] = {}
        for candidate in candidates:
            trans = translation_delta_mm(seed["tvec"], candidate["tvec"])
            rot = rotation_delta_deg(seed["rvec"], candidate["rvec"])
            if trans > float(args.robust_cluster_trans_mm) or rot > float(args.robust_cluster_rot_deg):
                continue
            score = (
                trans / max(float(args.robust_cluster_trans_mm), 1e-9)
                + rot / max(float(args.robust_cluster_rot_deg), 1e-9)
                + float(candidate["reproj_error"]) / max(float(args.robust_single_tag_max_reproj), 1e-9)
            )
            tag_id = int(candidate["tag_id"])
            if tag_id not in by_tag or score < by_tag[tag_id][0]:
                by_tag[tag_id] = (score, candidate)
        cluster = [item[1] for item in by_tag.values()]
        if not cluster:
            continue
        mean_single_reproj = float(np.mean([item["reproj_error"] for item in cluster]))
        mean_seed_trans = float(np.mean([translation_delta_mm(seed["tvec"], item["tvec"]) for item in cluster]))
        score_key = (len(cluster), -mean_single_reproj, -mean_seed_trans)
        if best_score is None or score_key > best_score:
            best_score = score_key
            best_cluster = cluster

    if len(best_cluster) < int(args.robust_min_tags):
        return (
            {
                "success": False,
                "failure_reason": f"robust_cluster_too_small:{len(best_cluster)}<{int(args.robust_min_tags)}",
                "detections": [(item["tag_id"], item["corners"]) for item in best_cluster],
                "n_tags": len(best_cluster),
                "tag_ids": [int(item["tag_id"]) for item in best_cluster],
                "reproj_error": float("inf"),
            },
            [(item["tag_id"], item["corners"]) for item in best_cluster],
            {"candidate_count": len(candidates), "cluster_size": len(best_cluster)},
        )

    seed = min(best_cluster, key=lambda item: item["reproj_error"])
    cluster_detections = [
        (int(item["tag_id"]), np.asarray(item["corners"], dtype=np.float64).reshape(4, 2))
        for item in sorted(best_cluster, key=lambda item: int(item["tag_id"]))
    ]
    pose = solve_pose_from_detections(
        cluster_detections,
        runtime,
        seed_rvec=seed["rvec"],
        seed_tvec=seed["tvec"],
        max_reproj=float(args.robust_max_reproj),
    )
    pose["pose_source"] = "deeptag_robust_pose_cluster"
    pose["pose_filled"] = False
    pose["robust_candidate_count"] = int(len(candidates))
    pose["robust_cluster_size"] = int(len(best_cluster))
    pose["robust_corner_orders"] = {
        int(item["tag_id"]): str(item["corner_order"])
        for item in best_cluster
    }
    stats = {
        "candidate_count": int(len(candidates)),
        "cluster_size": int(len(best_cluster)),
        "cluster_tag_ids": [int(item["tag_id"]) for item in best_cluster],
        "cluster_corner_orders": {
            int(item["tag_id"]): str(item["corner_order"])
            for item in best_cluster
        },
    }
    selected = pose.get("detections", cluster_detections) or cluster_detections
    return pose, selected, stats


def estimate_cube_pose_from_corners(
    detections: list[tuple[int, np.ndarray]],
    tag_corner_map: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, Any]:
    obj_chunks: list[np.ndarray] = []
    img_chunks: list[np.ndarray] = []
    tag_ids: list[int] = []
    for tag_id, corners_2d in detections:
        corners_3d = tag_corner_map.get(int(tag_id))
        if corners_3d is None:
            continue
        obj_chunks.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
        img_chunks.append(np.asarray(corners_2d, dtype=np.float64).reshape(4, 2))
        tag_ids.append(int(tag_id))
    if not obj_chunks:
        return {"success": False, "tag_ids": [], "n_tags": 0, "reproj_error": float("inf")}

    obj = np.vstack(obj_chunks).astype(np.float64)
    img = np.vstack(img_chunks).astype(np.float64)
    if len(obj) < 4:
        return {"success": False, "tag_ids": tag_ids, "n_tags": len(tag_ids), "reproj_error": float("inf")}

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            camera_matrix,
            dist_coeffs,
            iterationsCount=300,
            reprojectionError=12.0,
            confidence=0.995,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        ok = False
        rvec = None
        tvec = None
        inliers = None
    if not ok or rvec is None or tvec is None or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
        return {"success": False, "tag_ids": tag_ids, "n_tags": len(tag_ids), "reproj_error": float("inf")}

    if inliers is not None and len(inliers) >= 4:
        idx = np.asarray(inliers, dtype=np.int32).reshape(-1)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx], camera_matrix, dist_coeffs, rvec, tvec)
        except cv2.error:
            pass

    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = float(np.mean(np.linalg.norm(img - projected.reshape(-1, 2), axis=1)))
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return {
        "success": True,
        "tag_ids": tag_ids,
        "n_tags": len(tag_ids),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1).tolist(),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1).tolist(),
        "T": transform.tolist(),
        "reproj_error": reproj,
        "n_inliers": 0 if inliers is None else int(len(inliers)),
    }


def draw_overlay(image_bgr: np.ndarray, runtime: dict[str, Any], detections: list[tuple[int, np.ndarray]], pose: dict[str, Any]) -> np.ndarray:
    result = {
        "success": bool(pose.get("success", False)),
        "detections": detections,
        "rvec": np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1) if pose.get("success", False) else None,
        "tvec": np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1) if pose.get("success", False) else None,
        "reproj_error": float(pose.get("reproj_error", float("inf"))),
        "n_tags": int(pose.get("n_tags", 0)),
        "visible_faces": set(pose.get("visible_faces", []) or []),
        "predicted": False,
    }
    vis = runtime["april_draw_detector"].draw_result(image_bgr.copy(), result)
    y = 28
    lines = [
        f"DeepTag tags={pose.get('n_tags', 0)} reproj={float(pose.get('reproj_error', float('inf'))):.2f}px",
        f"ids={pose.get('tag_ids', [])}",
    ]
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return vis


def main() -> None:
    args = parse_args()
    pkl_path = Path(args.pkl_path).expanduser().resolve()
    header, all_offsets, footer = build_stream_index(pkl_path)
    offsets = all_offsets[int(args.start_frame) :: max(int(args.stride), 1)]
    if args.max_frames > 0:
        offsets = offsets[: int(args.max_frames)]

    script012 = load_script012_module()
    metadata = input_metadata(header)
    runtime = make_runtime(script012, metadata, args)
    tag_size_m = float(runtime["cube_config"].tag_size_mm) / 1000.0

    print(f"[INFO] Loading DeepTag models from {DEEPTAG_ROOT}")
    t0 = time.perf_counter()
    engine = load_deeptag_engine(
        camera_matrix=runtime["detection_camera_matrix"],
        dist_coeffs=runtime["detector_dist_coeffs"],
        tag_size_m=tag_size_m,
        args=args,
    )
    print(f"[INFO] DeepTag loaded in {time.perf_counter() - t0:.2f}s")

    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    total_tags = 0

    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "deeptag_012_offline_stream_v1",
                "source_pkl": str(pkl_path),
                "source_footer": footer,
                "metadata": {
                    "script": str(THIS_FILE),
                    "deeptag_root": str(DEEPTAG_ROOT),
                    "cube_cfg": str(runtime["cube_cfg"]),
                    "intrinsics_yaml": str(runtime["intrinsics_yaml"]),
                    "camera_matrix": runtime["detection_camera_matrix"].tolist(),
                    "dist_coeffs": runtime["detector_dist_coeffs"].tolist(),
                    "frame_count": len(offsets),
                    "tag_size_m": tag_size_m,
                    "corner_order": str(args.corner_order),
                    "postprocess": str(args.pose_mode),
                    "robust_min_tags": int(args.robust_min_tags),
                    "robust_cluster_trans_mm": float(args.robust_cluster_trans_mm),
                    "robust_cluster_rot_deg": float(args.robust_cluster_rot_deg),
                    "robust_max_reproj": float(args.robust_max_reproj),
                    "robust_single_tag_max_reproj": float(args.robust_single_tag_max_reproj),
                    "args": vars(args),
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for out_idx, offset in enumerate(offsets):
            record = load_frame_at(pkl_path, offset)
            frame = detection_frame(script012, runtime, np.asarray(record["image_bgr"], dtype=np.uint8))
            t_frame = time.perf_counter()
            stream = io.StringIO()
            ctx = contextlib.redirect_stdout(stream) if args.quiet_deeptag else contextlib.nullcontext()
            with ctx:
                decoded_tags = engine.process(frame, detect_scale=None if args.detect_scale < 0 else float(args.detect_scale))
            elapsed = time.perf_counter() - t_frame

            raw_detections, detection_stats = deeptag_detections_to_raw_corners(
                engine,
                decoded_tags,
                valid_ids=set(int(v) for v in runtime["cube_config"].tag_ids),
            )
            cluster_stats: dict[str, Any] = {}
            if str(args.pose_mode) == "robust-cluster":
                pose_raw, detections, cluster_stats = robust_cluster_pose(raw_detections, runtime, args)
            else:
                detections = apply_corner_order(raw_detections, str(args.corner_order))
                post_detector = runtime["april_post_detector"]
                reset_aprilcube_single_frame_state(post_detector)
                pose_raw = post_detector.process_detections(
                    frame,
                    detections,
                    timestamp=float(record.get("capture_timestamp", out_idx)),
                )
                reset_aprilcube_single_frame_state(post_detector)
                if not finite_pose_success(pose_raw):
                    pose_raw["success"] = False
                    pose_raw["rvec"] = None
                    pose_raw["tvec"] = None
                    pose_raw["T"] = None
                    pose_raw["reproj_error"] = float("inf")
                    if not pose_raw.get("failure_reason", ""):
                        pose_raw["failure_reason"] = "non_finite_or_failed_pose"
                pose_raw["pose_source"] = "deeptag_aprilcube_postprocess"
                pose_raw["pose_filled"] = False
            pose = sanitize_pose_result(pose_raw)
            overlay = draw_overlay(frame, runtime, detections, pose)
            success_count += int(bool(pose.get("success", False)))
            total_tags += int(pose.get("n_tags", 0))
            frame_record = {
                "type": "frame",
                "frame_index": int(out_idx),
                "source_offset": int(offset),
                "loop_frame_idx": int(record.get("loop_frame_idx", out_idx)),
                "capture_timestamp": record.get("capture_timestamp", None),
                "deeptag_elapsed_s": float(elapsed),
                "detection_stats": detection_stats,
                "cluster_stats": _jsonish(cluster_stats),
                "decoded_tags": sanitize_decoded_tags(decoded_tags),
                "raw_detections": [
                    {"tag_id": int(tag_id), "corners_xy": np.asarray(corners).tolist()}
                    for tag_id, corners in raw_detections
                ],
                "detections": [
                    {"tag_id": int(tag_id), "corners_xy": np.asarray(corners).tolist()}
                    for tag_id, corners in detections
                ],
                "pose": pose,
                "overlay_jpeg": encode_bgr_jpeg(overlay, int(args.jpeg_quality)),
                "overlay_format": "jpeg_bgr",
            }
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(
                f"[INFO] frame {out_idx + 1}/{len(offsets)} "
                f"tags={pose.get('n_tags', 0)} success={pose.get('success', False)} "
                f"reproj={float(pose.get('reproj_error', float('inf'))):.2f}px "
                f"time={elapsed:.2f}s"
            )

        pickle.dump(
            {
                "type": "footer",
                "frame_count": len(offsets),
                "success_count": int(success_count),
                "avg_tags": total_tags / max(len(offsets), 1),
                "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"[INFO] Saved DeepTag result pkl: {output_pkl}")
    print(f"[INFO] success={success_count}/{len(offsets)} avg_tags={total_tags / max(len(offsets), 1):.2f}")


if __name__ == "__main__":
    main()
