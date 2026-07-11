#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
SCRIPT_022_PATH = THIS_FILE.parent / "022_benchmark_single_frame_recovery_methods.py"
SCRIPT_020_PATH = THIS_FILE.parent / "020_deeptag_dense_keypoints_pose.py"
DEFAULT_RAW_PKL = APRILCUBE_ROOT / "recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_DEEPTAG_RAW_PKL = APRILCUBE_ROOT / "recordings/016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_DEEPTAG_POSE_PKL = APRILCUBE_ROOT / "recordings/020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336_faceframe_alltags_coverage_mintag2.pkl"
DEFAULT_APRIL_STRICT_PKL = APRILCUBE_ROOT / "recordings/014_offline_pose_vis_012_rs_raw_frames_20260710_214336_aprilcube_style_nofill_notagfix.pkl"
DEFAULT_LOOSE_DEEPTAG_PKL = APRILCUBE_ROOT / "recordings/020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336_faceframe_alltags.pkl"
DEFAULT_OLD_APRIL_PKL = APRILCUBE_ROOT / "recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings/023_fused_all_single_frame_recovery.pkl"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import aprilcube  # noqa: E402
from aprilcube.detect import estimate_single_tag_cube_pose  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse all single-frame AprilCube recovery methods without temporal filtering.")
    parser.add_argument("--raw-pkl", type=Path, default=DEFAULT_RAW_PKL)
    parser.add_argument("--deeptag-raw-pkl", type=Path, default=DEFAULT_DEEPTAG_RAW_PKL)
    parser.add_argument("--deeptag-pose-pkl", type=Path, default=DEFAULT_DEEPTAG_POSE_PKL)
    parser.add_argument("--april-strict-pkl", type=Path, default=DEFAULT_APRIL_STRICT_PKL)
    parser.add_argument("--loose-deeptag-pkl", type=Path, default=DEFAULT_LOOSE_DEEPTAG_PKL)
    parser.add_argument("--old-april-pkl", type=Path, default=DEFAULT_OLD_APRIL_PKL)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-reproj", type=float, default=3.0)
    parser.add_argument("--edge-threshold", type=float, default=0.45)
    parser.add_argument("--single-tag-edge-threshold", type=float, default=0.60)
    parser.add_argument("--single-tag-max-reproj", type=float, default=1.0)
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


def finite_pose(pose: dict[str, Any], *, min_tags: int, max_reproj: float) -> bool:
    if not bool(pose.get("success", False)):
        return False
    if bool(pose.get("pose_filled", False)) or bool(pose.get("predicted", False)):
        return False
    if int(pose.get("n_tags", 0) or 0) < int(min_tags):
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
    if not all(bool(np.all(np.isfinite(chunk))) for chunk in chunks):
        return False
    return float(pose.get("reproj_error", float("inf"))) <= float(max_reproj)


def pkl_pose_candidate_no_temporal(
    bm: Any,
    frame: dict[str, Any],
    method: str,
    frame_index: int,
    min_tags: int,
    max_reproj: float,
) -> Any:
    pose = frame.get("pose", {}) if isinstance(frame, dict) else {}
    if bool(pose.get("pose_filled", False)) or bool(pose.get("predicted", False)):
        return bm.PoseCandidate(False, method, frame_index, {}, [], float("inf"), failure_reason="temporal_or_filled_pose")
    return bm.pkl_pose_candidate(frame, method, frame_index, min_tags, max_reproj)


def copy_pose_with_stage(
    pose: dict[str, Any],
    *,
    source: str,
    quality_level: str,
    quality_reason: str,
    edge_score: float | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(pose)
    out["success"] = True
    out["pose_source_original"] = str(out.get("pose_source", ""))
    out["pose_source"] = source
    out["quality_level"] = quality_level
    out["quality_reason"] = quality_reason
    out["pose_filled"] = False
    out["fused_pose"] = True
    out["single_frame_only"] = True
    if edge_score is not None:
        out["edge_score"] = float(edge_score)
    return out


def failure_pose(reason: str) -> dict[str, Any]:
    return {
        "success": False,
        "pose_source": "fused_failed",
        "quality_level": "Z",
        "quality_reason": reason,
        "reproj_error": float("inf"),
        "pose_filled": False,
        "single_frame_only": True,
    }


def minimal_pose(pose: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "success",
        "failure_reason",
        "n_tags",
        "n_points",
        "n_inliers",
        "reproj_error",
        "tag_ids",
        "visible_faces",
        "pose_source",
        "pose_filled",
        "quality_level",
        "quality_reason",
        "edge_score",
        "rvec",
        "tvec",
        "T",
    }
    return {key: copy.deepcopy(value) for key, value in pose.items() if key in keys}


def draw_overlay(
    bm: Any,
    script012: Any,
    draw_detector: Any,
    detect_frame: np.ndarray,
    pose: dict[str, Any],
    label: str,
    reason: str,
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
    cv2.rectangle(vis, (8, 8), (1100, 66), (0, 0, 0), -1)
    cv2.putText(vis, f"Fused {pose.get('quality_level', 'Z')}: {label}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, reason[:110], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return encode_bgr_jpeg(vis, quality)


def accept_recovery(
    bm: Any,
    candidate: Any,
    gray: np.ndarray,
    *,
    config: Any,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    edge_threshold: float,
) -> bool:
    if not candidate.success:
        return False
    if candidate.edge_score is None:
        candidate.edge_score = bm.edge_alignment_score(
            gray,
            candidate.pose,
            config=config,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
    return float(candidate.edge_score) >= float(edge_threshold)


def tag_center_multiface_pose(
    bm: Any,
    detections: list[tuple[int, np.ndarray]],
    *,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_index: int,
    max_reproj: float,
) -> Any:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    tag_ids: list[int] = []
    for tag_id, corners in detections:
        tag_id = int(tag_id)
        if tag_id not in tag_corner_map:
            continue
        object_points.append(np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3).mean(axis=0))
        image_points.append(np.asarray(corners, dtype=np.float64).reshape(4, 2).mean(axis=0))
        tag_ids.append(tag_id)
    visible_faces = bm.visible_faces_for_ids(face_id_sets, tag_ids)
    if len(tag_ids) < 4 or len(visible_faces) < 2:
        return bm.PoseCandidate(
            False,
            "tag_center_multiface_pnp",
            frame_index,
            {},
            tag_ids,
            float("inf"),
            failure_reason=f"center_tags_or_faces_too_small:{len(tag_ids)}tags/{len(visible_faces)}faces",
        )
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            camera_matrix,
            dist_coeffs,
            iterationsCount=500,
            reprojectionError=5.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        ok, rvec, tvec, inliers = False, None, None, None
    if not ok or rvec is None or tvec is None or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
        return bm.PoseCandidate(False, "tag_center_multiface_pnp", frame_index, {}, tag_ids, float("inf"), failure_reason="center_pnp_failed")
    active = np.ones(len(tag_ids), dtype=bool)
    if inliers is not None and len(inliers) >= 4:
        active[:] = False
        active[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    used_ids = [int(tag_ids[i]) for i in range(len(tag_ids)) if bool(active[i])]
    used_faces = bm.visible_faces_for_ids(face_id_sets, used_ids)
    if len(used_ids) < 4 or len(used_faces) < 2:
        return bm.PoseCandidate(
            False,
            "tag_center_multiface_pnp",
            frame_index,
            {},
            used_ids,
            float("inf"),
            failure_reason=f"center_inliers_too_small:{len(used_ids)}tags/{len(used_faces)}faces",
        )
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj[active], img[active], camera_matrix, dist_coeffs, rvec, tvec)
    except cv2.error:
        pass
    errors = bm.project_errors(obj, img, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = float(np.mean(errors[active]))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return bm.PoseCandidate(False, "tag_center_multiface_pnp", frame_index, {}, used_ids, reproj, failure_reason=f"center_reproj_too_high:{reproj:.2f}")
    if not bm.face_normals_ok(np.asarray(rvec, dtype=np.float64).reshape(3, 1), used_faces):
        return bm.PoseCandidate(False, "tag_center_multiface_pnp", frame_index, {}, used_ids, reproj, failure_reason="center_face_normal_away")
    pose = {
        "success": True,
        "pose_source": "tag_center_multiface_pnp",
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "T": bm.pose_transform(rvec, tvec),
        "reproj_error": reproj,
        "n_tags": len(used_ids),
        "tag_ids": used_ids,
        "visible_faces": sorted(used_faces),
        "pose_filled": False,
        "reproj_metric": "tag_center_mean_px",
    }
    return bm.PoseCandidate(True, "tag_center_multiface_pnp", frame_index, pose, used_ids, reproj)


def apriltag_single_tag_pose(
    bm: Any,
    detections: list[tuple[int, np.ndarray]],
    gray: np.ndarray,
    *,
    config: Any,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_index: int,
    max_reproj: float,
) -> Any:
    best: Any | None = None
    for tag_id, corners in detections:
        try:
            ok, rvec, tvec, reproj, _inliers, meta = estimate_single_tag_cube_pose(
                [(int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2))],
                tag_corner_map,
                face_id_sets,
                camera_matrix,
                dist_coeffs,
                allow_corner_rotations=not bool(config.tag_pattern_mirrored),
            )
        except cv2.error:
            ok, rvec, tvec, reproj, meta = False, None, None, float("inf"), {}
        if not ok or rvec is None or tvec is None or not np.isfinite(reproj) or reproj > float(max_reproj):
            continue
        if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
            continue
        face_name = meta.get("single_tag_face", None)
        tag_id = int(meta.get("single_tag_id", tag_id))
        pose = {
            "success": True,
            "pose_source": "apriltag_single_tag_cfg_pose",
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            "T": bm.pose_transform(rvec, tvec),
            "reproj_error": float(reproj),
            "n_tags": 1,
            "tag_ids": [tag_id],
            "visible_faces": [str(face_name)] if face_name else [],
            "pose_filled": False,
            "single_tag_cfg_pose": True,
            "single_tag_meta": meta,
        }
        edge_score = bm.edge_alignment_score(
            gray,
            pose,
            config=config,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        candidate = bm.PoseCandidate(True, "apriltag_single_tag_cfg_pose", frame_index, pose, [tag_id], float(reproj), edge_score=edge_score)
        if best is None or (candidate.edge_score, -candidate.reproj_error) > (best.edge_score or 0.0, -best.reproj_error):
            best = candidate
    if best is None:
        return bm.PoseCandidate(False, "apriltag_single_tag_cfg_pose", frame_index, {}, [], float("inf"), failure_reason="no_single_tag_cfg_candidate")
    return best


def deeptag_single_tag_dense_pose(
    bm: Any,
    dense020: Any,
    deeptag_frame: dict[str, Any],
    gray: np.ndarray,
    *,
    config: Any,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_index: int,
    max_reproj: float,
) -> Any:
    object_points, image_points, tag_ids, point_counts, dense_stats = dense020.dense_points_for_frame(
        deeptag_frame,
        tag_corner_map=tag_corner_map,
        min_tags=1,
    )
    if object_points.shape[0] < 4:
        return bm.PoseCandidate(
            False,
            "deeptag_single_tag_dense_pose",
            frame_index,
            {},
            tag_ids,
            float("inf"),
            failure_reason=str(dense_stats.get("reason", "dense_single_tag_no_points")),
        )
    pose = dense020.solve_dense_pose(
        object_points,
        image_points,
        tag_ids,
        point_counts,
        cube_config=config,
        face_id_sets=face_id_sets,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        ransac_reproj=4.0,
        max_reproj=float(max_reproj),
        point_reject_px=8.0,
        tag_reject_px=8.0,
        min_tags=1,
        min_inlier_tag_fraction=0.0,
        coverage_check_min_raw_tags=999,
        max_required_inlier_tags=4,
    )
    if not bool(pose.get("success", False)):
        return bm.PoseCandidate(
            False,
            "deeptag_single_tag_dense_pose",
            frame_index,
            {},
            tag_ids,
            float(pose.get("raw_reproj_error", pose.get("reproj_error", float("inf")))),
            failure_reason=str(pose.get("failure_reason", "dense_single_tag_failed")),
        )
    used_ids = [int(v) for v in pose.get("tag_ids", []) or []]
    if len(used_ids) != 1:
        return bm.PoseCandidate(
            False,
            "deeptag_single_tag_dense_pose",
            frame_index,
            {},
            used_ids,
            float(pose.get("reproj_error", float("inf"))),
            failure_reason=f"dense_single_tag_used_count:{len(used_ids)}",
        )
    pose = copy.deepcopy(pose)
    pose["pose_source"] = "deeptag_single_tag_dense_pose"
    pose["dense_stats"] = {
        **dense_stats,
        "raw_tag_ids": tag_ids,
        "raw_point_counts": point_counts,
    }
    edge_score = bm.edge_alignment_score(
        gray,
        pose,
        config=config,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    return bm.PoseCandidate(
        True,
        "deeptag_single_tag_dense_pose",
        frame_index,
        pose,
        used_ids,
        float(pose.get("reproj_error", float("inf"))),
        edge_score=edge_score,
    )


def main() -> None:
    args = parse_args()
    bm = load_module(SCRIPT_022_PATH, "fuse023_benchmark_helpers")
    script012 = load_module(SCRIPT_012_PATH, "fuse023_script012")
    dense020 = load_module(SCRIPT_020_PATH, "fuse023_dense_deeptag_helpers")

    raw_header, raw_offsets, raw_footer = bm.build_stream_index(
        args.raw_pkl,
        {"aprilcube_rs_raw_frame_stream_v1", "aprilcube_012_raw_with_pose_stream_v1"},
    )
    dt_header, dt_frames, dt_footer = bm.load_pose_records(args.deeptag_pose_pkl)
    ap_header, ap_frames, ap_footer = bm.load_pose_records(args.april_strict_pkl)
    deeptag_raw_header, deeptag_raw_offsets, deeptag_raw_footer = bm.build_stream_index(args.deeptag_raw_pkl, {"deeptag_012_offline_stream_v1"})
    loose_header, loose_frames, loose_footer = bm.load_pose_records(args.loose_deeptag_pkl)
    old_header, old_frames, old_footer = bm.load_pose_records(args.old_april_pkl)

    metadata: dict[str, Any] = {}
    if raw_header.get("format") == "aprilcube_012_raw_with_pose_stream_v1":
        metadata.update(raw_header.get("raw_header", {}).get("metadata", {}) or {})
    metadata.update(raw_header.get("metadata", {}) or {})
    cube_cfg = Path(metadata["cube_cfg"]).expanduser().resolve()
    config, face_id_sets = aprilcube.load_cube_config(str(cube_cfg / "config.json" if cube_cfg.is_dir() else cube_cfg))
    tag_corner_map = aprilcube.build_tag_corner_map(config)
    valid_ids = set(int(v) for v in tag_corner_map)

    calib = script012.load_intrinsics_yaml(metadata.get("intrinsics_yaml"))
    image_size = tuple(int(v) for v in metadata.get("image_size", calib["image_size"]))
    undistort_pack = script012.create_undistort_maps(calib, image_size) if bool(metadata.get("undistort_for_detection", True)) else None
    camera_matrix = np.asarray(metadata.get("detection_camera_matrix", calib["K"]), dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get("detector_dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)
    if undistort_pack is not None:
        camera_matrix = np.asarray(metadata.get("detection_camera_matrix", undistort_pack[2]), dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(metadata.get("detector_dist_coeffs", np.zeros(5)), dtype=np.float64).reshape(-1)

    draw_detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=script012.camera_matrix_to_intrinsic_dict(camera_matrix),
        dist_coeffs=dist_coeffs,
        enable_filter=False,
        fast=True,
    )
    raw_offset_by_frame = {
        int(bm.load_at(args.raw_pkl, offset).get("frame_index", idx)): int(offset)
        for idx, offset in enumerate(raw_offsets)
    }
    deeptag_offset_by_frame = {
        int(bm.load_at(args.deeptag_raw_pkl, offset).get("frame_index", idx)): int(offset)
        for idx, offset in enumerate(deeptag_raw_offsets)
    }

    frame_indices = sorted(dt_frames)
    if set(frame_indices) != set(ap_frames):
        raise ValueError("DeepTag pose pkl and April strict pkl have different frame indices")

    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    with args.output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "aprilcube_deeptag_fused_stream_v1",
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_raw_pkl": str(args.raw_pkl.resolve()),
                "source_deeptag_pose_pkl": str(args.deeptag_pose_pkl.resolve()),
                "source_april_strict_pkl": str(args.april_strict_pkl.resolve()),
                "source_deeptag_raw_pkl": str(args.deeptag_raw_pkl.resolve()),
                "source_loose_deeptag_pkl": str(args.loose_deeptag_pkl.resolve()),
                "source_old_april_pkl": str(args.old_april_pkl.resolve()),
                "raw_footer": raw_footer,
                "deeptag_footer": dt_footer,
                "april_footer": ap_footer,
                "deeptag_raw_footer": deeptag_raw_footer,
                "loose_footer": loose_footer,
                "old_footer": old_footer,
                "metadata": {
                    "script": str(THIS_FILE),
                    "method": "single-frame cascade: DeepTag coverage/min-tag2, strict AprilCube, tag-center multiface PnP, face board, AprilTag preprocessing sweep, DeepTag-AprilTag cross validation, edge-checked loose candidates; no temporal filter or fill",
                    "frame_count": len(frame_indices),
                    "min_tags": int(args.min_tags),
                    "max_reproj": float(args.max_reproj),
                    "edge_threshold": float(args.edge_threshold),
                    "single_tag_edge_threshold": float(args.single_tag_edge_threshold),
                    "single_tag_max_reproj": float(args.single_tag_max_reproj),
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for out_idx, frame_index in enumerate(frame_indices):
            dt_frame = dt_frames[frame_index]
            ap_frame = ap_frames[frame_index]
            raw_record = bm.load_at(args.raw_pkl, raw_offset_by_frame[int(frame_index)])
            image = np.asarray(raw_record["image_bgr"], dtype=np.uint8)
            detect_frame = script012.undistort_frame(image, undistort_pack)
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            selected = "failed"
            selected_candidate = None
            pose_candidates: dict[str, Any] = {
                "deeptag_dense": minimal_pose(dt_frame.get("pose", {})),
                "aprilcube_strict": minimal_pose(ap_frame.get("pose", {})),
            }

            dt_pose = dt_frame.get("pose", {})
            ap_pose = ap_frame.get("pose", {})
            if finite_pose(dt_pose, min_tags=int(args.min_tags), max_reproj=float(args.max_reproj)):
                fused_pose = copy_pose_with_stage(
                    dt_pose,
                    source="stage1_deeptag_dense_coverage_mintag2",
                    quality_level="A",
                    quality_reason=f"deeptag_dense_reproj:{float(dt_pose.get('reproj_error', float('inf'))):.2f}",
                )
                selected = "stage1_deeptag"
            elif finite_pose(ap_pose, min_tags=int(args.min_tags), max_reproj=float(args.max_reproj)):
                fused_pose = copy_pose_with_stage(
                    ap_pose,
                    source="stage2_aprilcube_strict_mintag2",
                    quality_level="B",
                    quality_reason=f"aprilcube_strict_reproj:{float(ap_pose.get('reproj_error', float('inf'))):.2f}",
                )
                selected = "stage2_aprilcube_strict"
            else:
                detections = bm.detect_sweep(gray, config=config, valid_ids=valid_ids)
                deeptag_raw = bm.load_at(args.deeptag_raw_pkl, deeptag_offset_by_frame[int(frame_index)])
                stage_candidates = [
                    (
                        "stage3_tag_center_multiface_pnp",
                        "C",
                        tag_center_multiface_pose(
                            bm,
                            detections,
                            tag_corner_map=tag_corner_map,
                            face_id_sets=face_id_sets,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            frame_index=int(frame_index),
                            max_reproj=2.0,
                        ),
                    ),
                    (
                        "stage4_single_face_board",
                        "D",
                        bm.face_board_pose(
                            detections,
                            tag_corner_map=tag_corner_map,
                            face_id_sets=face_id_sets,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            frame_index=int(frame_index),
                            min_tags=int(args.min_tags),
                            max_reproj=float(args.max_reproj),
                        ),
                    ),
                    (
                        "stage5_apriltag_preproc_sweep",
                        "E",
                        bm.solve_pose_from_detections(
                            detections,
                            tag_corner_map=tag_corner_map,
                            face_id_sets=face_id_sets,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            method="apriltag_preproc_sweep",
                            frame_index=int(frame_index),
                            min_tags=int(args.min_tags),
                            max_reproj=float(args.max_reproj),
                        ),
                    ),
                    (
                        "stage6_deeptag_apriltag_cross_validated",
                        "F",
                        bm.deeptag_cross_validated_pose(
                            deeptag_raw,
                            detections,
                            tag_corner_map=tag_corner_map,
                            face_id_sets=face_id_sets,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            frame_index=int(frame_index),
                            min_tags=int(args.min_tags),
                            max_reproj=float(args.max_reproj),
                        ),
                    ),
                ]
                loose_sources = [
                    pkl_pose_candidate_no_temporal(bm, loose_frames.get(frame_index, {}), "loose_deeptag_edge_checked", int(frame_index), int(args.min_tags), float(args.max_reproj)),
                    pkl_pose_candidate_no_temporal(bm, old_frames.get(frame_index, {}), "old_april_edge_checked", int(frame_index), int(args.min_tags), float(args.max_reproj)),
                ]
                best_edge = None
                for cand in loose_sources:
                    if not cand.success:
                        continue
                    cand.edge_score = bm.edge_alignment_score(gray, cand.pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
                    if cand.edge_score >= float(args.edge_threshold) and (
                        best_edge is None
                        or (cand.edge_score, -cand.reproj_error) > (best_edge.edge_score or 0.0, -best_edge.reproj_error)
                    ):
                        best_edge = cand
                stage_candidates.extend(
                    [
                    (
                        "stage7_edge_checked_loose_candidate",
                        "G",
                        best_edge
                        or bm.PoseCandidate(False, "edge_checked_loose_candidates", int(frame_index), {}, [], float("inf"), failure_reason="no_edge_accepted_candidate"),
                    ),
                    (
                        "stage8_apriltag_single_tag_cfg_edge",
                        "H",
                        apriltag_single_tag_pose(
                            bm,
                            detections,
                            gray,
                            config=config,
                            tag_corner_map=tag_corner_map,
                            face_id_sets=face_id_sets,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            frame_index=int(frame_index),
                            max_reproj=float(args.single_tag_max_reproj),
                        ),
                    ),
                    (
                        "stage9_deeptag_single_tag_dense_edge",
                        "I",
                        deeptag_single_tag_dense_pose(
                            bm,
                            dense020,
                            deeptag_raw,
                            gray,
                            config=config,
                            tag_corner_map=tag_corner_map,
                            face_id_sets=face_id_sets,
                            camera_matrix=camera_matrix,
                            dist_coeffs=dist_coeffs,
                            frame_index=int(frame_index),
                            max_reproj=float(args.single_tag_max_reproj),
                        ),
                    )
                    ]
                )

                pose_candidates["recovery_detected_tag_ids"] = [int(v[0]) for v in detections]
                fused_pose = failure_pose("no_single_frame_method_accepted")
                for source, quality, candidate in stage_candidates:
                    pose_candidates[source] = {
                        "success": bool(candidate.success),
                        "failure_reason": candidate.failure_reason,
                        "n_tags": len(candidate.tag_ids),
                        "tag_ids": candidate.tag_ids,
                        "reproj_error": candidate.reproj_error,
                        "edge_score": candidate.edge_score,
                        "pose_source": candidate.pose.get("pose_source", candidate.method) if candidate.pose else candidate.method,
                    }
                    if accept_recovery(
                        bm,
                        candidate,
                        gray,
                        config=config,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                        edge_threshold=(
                            float(args.single_tag_edge_threshold)
                            if source in {
                                "stage8_apriltag_single_tag_cfg_edge",
                                "stage9_deeptag_single_tag_dense_edge",
                            }
                            else float(args.edge_threshold)
                        ),
                    ):
                        fused_pose = copy_pose_with_stage(
                            candidate.pose,
                            source=source,
                            quality_level=quality,
                            quality_reason=(
                                f"{source}_reproj:{candidate.reproj_error:.2f};"
                                f"edge:{float(candidate.edge_score):.2f}"
                            ),
                            edge_score=candidate.edge_score,
                        )
                        selected = source
                        selected_candidate = candidate
                        break

            quality = str(fused_pose.get("quality_level", "Z"))
            source = str(fused_pose.get("pose_source", ""))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(fused_pose.get("success", False)))
            overlay_jpeg = draw_overlay(
                bm,
                script012,
                draw_detector,
                detect_frame,
                fused_pose,
                source,
                str(fused_pose.get("quality_reason", "")),
                int(args.jpeg_quality),
            )
            frame_record = {
                "type": "frame",
                "frame_index": int(frame_index),
                "source_offset": int(raw_offset_by_frame[int(frame_index)]),
                "loop_frame_idx": int(raw_record.get("loop_frame_idx", frame_index)),
                "capture_timestamp": raw_record.get("capture_timestamp", None),
                "overlay_shape": tuple(int(v) for v in detect_frame.shape),
                "overlay_format": "jpeg_bgr",
                "overlay_jpeg": overlay_jpeg,
                "pose": fused_pose,
                "pose_candidates": pose_candidates,
                "selected_stage": selected,
                "selected_candidate_reproj": None if selected_candidate is None else float(selected_candidate.reproj_error),
                "selected_candidate_edge_score": None if selected_candidate is None else float(selected_candidate.edge_score or 0.0),
            }
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)

            done = out_idx + 1
            if done == len(frame_indices) or done % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"\r[INFO] fused all {done}/{len(frame_indices)} "
                    f"success={success_count} fps={done / max(elapsed, 1e-9):.1f}",
                    end="",
                    flush=True,
                )

        footer = {
            "type": "footer",
            "frame_count": len(frame_indices),
            "success_count": int(success_count),
            "quality_counts": quality_counts,
            "source_counts": source_counts,
            "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        pickle.dump(footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()
    print(f"[INFO] saved {args.output_pkl}")
    print(f"[INFO] success={success_count}/{len(frame_indices)} quality_counts={quality_counts}")
    print(f"[INFO] source_counts={source_counts}")


if __name__ == "__main__":
    main()
