#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
SCRIPT_020_PATH = THIS_FILE.parent / "020_deeptag_dense_keypoints_pose.py"
DEFAULT_RAW_PKL = APRILCUBE_ROOT / "recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_DEEPTAG_PKL = APRILCUBE_ROOT / "recordings/016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl"
DEFAULT_FUSED_PKL = APRILCUBE_ROOT / "recordings/021_fused_deeptag_dense_coverage_mintag2_aprilcube_strict_notagfix_mintag2.pkl"
DEFAULT_LOOSE_PKL = APRILCUBE_ROOT / "recordings/020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336_faceframe_alltags.pkl"
DEFAULT_APRIL_OLD_PKL = APRILCUBE_ROOT / "recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings/022_recovery_method_benchmark.pkl"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import aprilcube  # noqa: E402
from aprilcube.detect import (  # noqa: E402
    _gamma_correct,
    _linear_contrast,
    _preprocess,
    _preprocess_clahe,
    _quad_quality,
    _sharpen,
    create_detector,
    create_fallback_detector,
)


@dataclass
class PoseCandidate:
    success: bool
    method: str
    frame_index: int
    pose: dict[str, Any]
    tag_ids: list[int]
    reproj_error: float
    edge_score: float | None = None
    failure_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-frame recovery methods on fused-failed frames.")
    parser.add_argument("--raw-pkl", type=Path, default=DEFAULT_RAW_PKL)
    parser.add_argument("--deeptag-pkl", type=Path, default=DEFAULT_DEEPTAG_PKL)
    parser.add_argument("--failed-reference-pkl", type=Path, default=DEFAULT_FUSED_PKL)
    parser.add_argument("--loose-candidate-pkl", type=Path, default=DEFAULT_LOOSE_PKL)
    parser.add_argument("--april-old-pkl", type=Path, default=DEFAULT_APRIL_OLD_PKL)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--max-reproj", type=float, default=3.0)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--edge-threshold", type=float, default=0.34)
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_stream_index(path: Path, formats: set[str] | None = None) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer = None
    with path.open("rb") as f:
        header = pickle.load(f)
        if formats is not None and header.get("format") not in formats:
            raise ValueError(f"Unsupported format {header.get('format')} for {path}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, dict) and obj.get("type") == "frame":
                offsets.append(offset)
            elif isinstance(obj, dict) and obj.get("type") == "footer":
                footer = obj
                break
    return header, offsets, footer


def load_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open("rb") as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("type") != "frame":
        raise ValueError(f"Offset {offset} in {path} is not a frame")
    return obj


def load_pose_records(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any] | None]:
    header, offsets, footer = build_stream_index(path, None)
    frames: dict[int, dict[str, Any]] = {}
    for offset in offsets:
        frame = load_at(path, offset)
        frames[int(frame["frame_index"])] = frame
    return header, frames, footer


def rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot


def pose_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rvec(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def visible_faces_for_ids(face_id_sets: dict[str, set[int]], tag_ids: list[int]) -> set[str]:
    visible: set[str] = set()
    for tag_id in tag_ids:
        for face, ids in face_id_sets.items():
            if int(tag_id) in ids:
                visible.add(str(face))
    return visible


def face_normals_ok(rvec: np.ndarray, visible_faces: set[str]) -> bool:
    rot = rotation_from_rvec(rvec)
    for face_name in visible_faces:
        for face_def in aprilcube.FACE_DEFS:
            if str(face_def[0]) != str(face_name):
                continue
            normal = np.zeros(3, dtype=np.float64)
            normal[int(face_def[1])] = float(face_def[2])
            if float((rot @ normal)[2]) > 0.0:
                return False
    return True


def project_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return np.linalg.norm(np.asarray(image_points, dtype=np.float64).reshape(-1, 2) - projected.reshape(-1, 2), axis=1)


def detections_to_points(
    detections: list[tuple[int, np.ndarray]],
    tag_corner_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    obj: list[np.ndarray] = []
    img: list[np.ndarray] = []
    ids: list[int] = []
    for tag_id, corners in detections:
        if int(tag_id) not in tag_corner_map:
            continue
        obj.append(np.asarray(tag_corner_map[int(tag_id)], dtype=np.float64).reshape(4, 3))
        img.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
        ids.append(int(tag_id))
    if not obj:
        return np.empty((0, 3)), np.empty((0, 2)), []
    return np.vstack(obj), np.vstack(img), ids


def solve_pose_from_detections(
    detections: list[tuple[int, np.ndarray]],
    *,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    method: str,
    frame_index: int,
    min_tags: int,
    max_reproj: float,
) -> PoseCandidate:
    object_points, image_points, tag_ids = detections_to_points(detections, tag_corner_map)
    if len(tag_ids) < int(min_tags) or object_points.shape[0] < 8:
        return PoseCandidate(False, method, frame_index, {}, tag_ids, float("inf"), failure_reason=f"tags_too_small:{len(tag_ids)}")
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=300,
            reprojectionError=3.0,
            confidence=0.995,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        ok, rvec, tvec, inliers = False, None, None, None
    if not ok or rvec is None or tvec is None or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
        return PoseCandidate(False, method, frame_index, {}, tag_ids, float("inf"), failure_reason="pnp_failed")
    active = np.ones(object_points.shape[0], dtype=bool)
    if inliers is not None and len(inliers) >= 8:
        active[:] = False
        active[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    used_tags: list[int] = []
    for idx, tag_id in enumerate(tag_ids):
        if int(active[idx * 4 : idx * 4 + 4].sum()) >= 3:
            used_tags.append(int(tag_id))
    if len(used_tags) < int(min_tags):
        return PoseCandidate(False, method, frame_index, {}, used_tags, float("inf"), failure_reason=f"inlier_tags_too_small:{len(used_tags)}")
    try:
        rvec, tvec = cv2.solvePnPRefineLM(object_points[active], image_points[active], camera_matrix, dist_coeffs, rvec, tvec)
    except cv2.error:
        pass
    errors = project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = float(np.mean(errors[active]))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return PoseCandidate(False, method, frame_index, {}, used_tags, reproj, failure_reason=f"reproj_too_high:{reproj:.2f}")
    faces = visible_faces_for_ids(face_id_sets, used_tags)
    if not face_normals_ok(rvec, faces):
        return PoseCandidate(False, method, frame_index, {}, used_tags, reproj, failure_reason="face_normal_away")
    pose = {
        "success": True,
        "pose_source": method,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "T": pose_transform(rvec, tvec),
        "reproj_error": reproj,
        "n_tags": len(used_tags),
        "tag_ids": used_tags,
        "visible_faces": sorted(faces),
        "pose_filled": False,
    }
    return PoseCandidate(True, method, frame_index, pose, used_tags, reproj)


def make_variants(gray: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    variants: list[tuple[str, np.ndarray, float]] = [
        ("gray", gray, 1.0),
        ("preprocess", _preprocess(gray), 1.0),
        ("clahe", _preprocess_clahe(gray, clip_limit=2.5, tile_grid_size=(8, 8)), 1.0),
        ("sharpen", _sharpen(gray), 1.0),
        ("gamma07", _gamma_correct(gray, 0.7), 1.0),
        ("gamma13", _gamma_correct(gray, 1.3), 1.0),
        ("contrast", _linear_contrast(gray, 1.35, -18.0), 1.0),
    ]
    big = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    variants.append(("scale15_preprocess", _preprocess(big), 1.5))
    return variants


def detect_sweep(
    gray: np.ndarray,
    *,
    config: Any,
    valid_ids: set[int],
) -> list[tuple[int, np.ndarray]]:
    detectors = [create_detector(config.dict_id, fast=False), create_fallback_detector(config.dict_id)]
    best: dict[int, tuple[float, np.ndarray]] = {}
    for _name, image, scale in make_variants(gray):
        for detector in detectors:
            try:
                corners, ids, _rejected = detector.detectMarkers(image)
            except cv2.error:
                continue
            if ids is None:
                continue
            for idx in range(len(ids)):
                tag_id = int(ids[idx][0])
                if tag_id not in valid_ids:
                    continue
                pts = np.asarray(corners[idx], dtype=np.float64).reshape(4, 2) / float(scale)
                quality = float(_quad_quality(pts))
                if quality <= 0.15:
                    continue
                if tag_id not in best or quality > best[tag_id][0]:
                    best[tag_id] = (quality, pts)
    return [(tag_id, pts) for tag_id, (_q, pts) in sorted(best.items())]


def face_board_pose(
    detections: list[tuple[int, np.ndarray]],
    *,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_index: int,
    min_tags: int,
    max_reproj: float,
) -> PoseCandidate:
    best: PoseCandidate | None = None
    for face_name, ids in face_id_sets.items():
        face_dets = [(tag_id, corners) for tag_id, corners in detections if int(tag_id) in ids]
        candidate = solve_pose_from_detections(
            face_dets,
            tag_corner_map=tag_corner_map,
            face_id_sets=face_id_sets,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            method=f"face_board_{face_name}",
            frame_index=frame_index,
            min_tags=min_tags,
            max_reproj=max_reproj,
        )
        if candidate.success and (best is None or candidate.reproj_error < best.reproj_error):
            best = candidate
    if best is None:
        return PoseCandidate(False, "face_board", frame_index, {}, [], float("inf"), failure_reason="no_face_board_pose")
    best.method = "face_board"
    best.pose["pose_source"] = "face_board"
    return best


def deeptag_cross_validated_pose(
    deeptag_frame: dict[str, Any],
    april_detections: list[tuple[int, np.ndarray]],
    *,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_index: int,
    min_tags: int,
    max_reproj: float,
) -> PoseCandidate:
    april_by_id = {int(tag_id): np.asarray(corners, dtype=np.float64).reshape(4, 2) for tag_id, corners in april_detections}
    detections: list[tuple[int, np.ndarray]] = []
    for decoded in deeptag_frame.get("decoded_tags", []) or []:
        if not decoded.get("is_valid", False):
            continue
        tag_id = int(decoded.get("tag_id", -1))
        if tag_id not in april_by_id:
            continue
        pts = np.asarray(decoded.get("keypoints_in_images", []), dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        center_dist = float(np.linalg.norm(pts.mean(axis=0) - april_by_id[tag_id].mean(axis=0)))
        if center_dist > 18.0:
            continue
        detections.append((tag_id, april_by_id[tag_id]))
    return solve_pose_from_detections(
        detections,
        tag_corner_map=tag_corner_map,
        face_id_sets=face_id_sets,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        method="deeptag_apriltag_cross_validated",
        frame_index=frame_index,
        min_tags=min_tags,
        max_reproj=max_reproj,
    )


def cube_corners(config: Any) -> np.ndarray:
    x, y, z = [float(v) / 2.0 for v in config.box_dims]
    return np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ],
        dtype=np.float64,
    )


CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def edge_alignment_score(
    gray: np.ndarray,
    pose: dict[str, Any],
    *,
    config: Any,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    if not pose.get("success", False):
        return 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1)
    corners_2d, _ = cv2.projectPoints(cube_corners(config), rvec, tvec, camera_matrix, dist_coeffs)
    corners_2d = corners_2d.reshape(-1, 2)
    h, w = gray.shape[:2]
    hits = 0
    total = 0
    for a, b in CUBE_EDGES:
        p0, p1 = corners_2d[a], corners_2d[b]
        length = float(np.linalg.norm(p1 - p0))
        samples = max(4, min(40, int(length / 4.0)))
        for t in np.linspace(0.05, 0.95, samples):
            p = p0 * (1.0 - t) + p1 * t
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if float(dist[y, x]) <= 2.5:
                    hits += 1
    return float(hits / max(total, 1))


def pkl_pose_candidate(frame: dict[str, Any], method: str, frame_index: int, min_tags: int, max_reproj: float) -> PoseCandidate:
    pose = frame.get("pose", {})
    n_tags = int(pose.get("n_tags", 0) or 0)
    try:
        reproj = float(pose.get("reproj_error", float("inf")))
    except (TypeError, ValueError):
        reproj = float("inf")
    if (
        not pose.get("success", False)
        or n_tags < int(min_tags)
        or not np.isfinite(reproj)
        or reproj > float(max_reproj)
        or pose.get("rvec") is None
        or pose.get("tvec") is None
    ):
        return PoseCandidate(False, method, frame_index, {}, [], reproj, failure_reason="candidate_not_usable")
    return PoseCandidate(True, method, frame_index, dict(pose), [int(v) for v in pose.get("tag_ids", []) or []], reproj)


def main() -> None:
    args = parse_args()
    script012 = load_module(SCRIPT_012_PATH, "benchmark_012")
    raw_header, raw_offsets, _raw_footer = build_stream_index(
        args.raw_pkl,
        {"aprilcube_rs_raw_frame_stream_v1", "aprilcube_012_raw_with_pose_stream_v1"},
    )
    failed_header, failed_frames, failed_footer = load_pose_records(args.failed_reference_pkl)
    deeptag_header, deeptag_offsets, deeptag_footer = build_stream_index(args.deeptag_pkl, {"deeptag_012_offline_stream_v1"})
    loose_header, loose_frames, loose_footer = load_pose_records(args.loose_candidate_pkl)
    old_header, old_frames, old_footer = load_pose_records(args.april_old_pkl)

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

    failed_indices = [
        int(idx)
        for idx, frame in sorted(failed_frames.items())
        if not bool(frame.get("pose", {}).get("success", False))
    ]
    if args.max_frames > 0:
        failed_indices = failed_indices[: int(args.max_frames)]
    raw_offset_by_frame = {int(load_at(args.raw_pkl, offset).get("frame_index", idx)): int(offset) for idx, offset in enumerate(raw_offsets)}
    deeptag_offset_by_frame = {int(load_at(args.deeptag_pkl, offset).get("frame_index", idx)): int(offset) for idx, offset in enumerate(deeptag_offsets)}

    results: dict[int, dict[str, Any]] = {}
    method_success: dict[str, set[int]] = {
        "apriltag_preproc_sweep": set(),
        "single_face_board": set(),
        "deeptag_apriltag_cross_validated": set(),
        "edge_checked_loose_candidates": set(),
    }

    for n, frame_index in enumerate(failed_indices, start=1):
        raw = load_at(args.raw_pkl, raw_offset_by_frame[frame_index])
        image = np.asarray(raw["image_bgr"], dtype=np.uint8)
        image = script012.undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detect_sweep(gray, config=config, valid_ids=valid_ids)
        deeptag = load_at(args.deeptag_pkl, deeptag_offset_by_frame[frame_index])

        candidates: dict[str, PoseCandidate] = {}
        candidates["apriltag_preproc_sweep"] = solve_pose_from_detections(
            detections,
            tag_corner_map=tag_corner_map,
            face_id_sets=face_id_sets,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            method="apriltag_preproc_sweep",
            frame_index=frame_index,
            min_tags=int(args.min_tags),
            max_reproj=float(args.max_reproj),
        )
        candidates["single_face_board"] = face_board_pose(
            detections,
            tag_corner_map=tag_corner_map,
            face_id_sets=face_id_sets,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            frame_index=frame_index,
            min_tags=int(args.min_tags),
            max_reproj=float(args.max_reproj),
        )
        candidates["deeptag_apriltag_cross_validated"] = deeptag_cross_validated_pose(
            deeptag,
            detections,
            tag_corner_map=tag_corner_map,
            face_id_sets=face_id_sets,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            frame_index=frame_index,
            min_tags=int(args.min_tags),
            max_reproj=float(args.max_reproj),
        )

        edge_sources = [
            pkl_pose_candidate(loose_frames.get(frame_index, {}), "loose_deeptag_edge_checked", frame_index, int(args.min_tags), float(args.max_reproj)),
            pkl_pose_candidate(old_frames.get(frame_index, {}), "old_april_edge_checked", frame_index, int(args.min_tags), float(args.max_reproj)),
        ]
        best_edge: PoseCandidate | None = None
        for cand in edge_sources:
            if not cand.success:
                continue
            cand.edge_score = edge_alignment_score(gray, cand.pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
            if cand.edge_score >= float(args.edge_threshold) and (best_edge is None or (cand.edge_score, -cand.reproj_error) > (best_edge.edge_score or 0.0, -best_edge.reproj_error)):
                best_edge = cand
        candidates["edge_checked_loose_candidates"] = best_edge or PoseCandidate(False, "edge_checked_loose_candidates", frame_index, {}, [], float("inf"), failure_reason="no_edge_accepted_candidate")

        frame_result: dict[str, Any] = {"frame_index": frame_index, "detected_tag_ids": [int(v[0]) for v in detections], "methods": {}}
        for method, cand in candidates.items():
            if cand.success and cand.edge_score is None:
                cand.edge_score = edge_alignment_score(
                    gray,
                    cand.pose,
                    config=config,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
            if cand.success:
                method_success[method].add(frame_index)
            frame_result["methods"][method] = {
                "success": bool(cand.success),
                "failure_reason": cand.failure_reason,
                "tag_ids": cand.tag_ids,
                "n_tags": len(cand.tag_ids),
                "reproj_error": cand.reproj_error,
                "edge_score": cand.edge_score,
                "pose_source": cand.pose.get("pose_source", cand.method) if cand.pose else cand.method,
            }
        results[frame_index] = frame_result
        if n % 25 == 0 or n == len(failed_indices):
            print(f"[INFO] processed failed frames {n}/{len(failed_indices)}")

    summary: dict[str, Any] = {
        "failed_reference_pkl": str(args.failed_reference_pkl),
        "failed_reference_footer": failed_footer,
        "failed_frame_count": len(failed_indices),
        "method_counts": {method: len(indices) for method, indices in method_success.items()},
        "method_frames": {method: sorted(indices) for method, indices in method_success.items()},
        "union_count": len(set().union(*method_success.values())) if method_success else 0,
        "union_frames": sorted(set().union(*method_success.values())) if method_success else [],
        "params": {
            "max_reproj": float(args.max_reproj),
            "min_tags": int(args.min_tags),
            "edge_threshold": float(args.edge_threshold),
        },
    }
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": "aprilcube_recovery_method_benchmark_v1",
                "summary": summary,
                "raw_header": raw_header,
                "deeptag_header": deeptag_header,
                "loose_footer": loose_footer,
                "old_footer": old_footer,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for frame_index in sorted(results):
            pickle.dump({"type": "frame", **results[frame_index]}, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump({"type": "footer", **summary}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[RESULT] failed_frame_count", len(failed_indices))
    for method, indices in method_success.items():
        print("[RESULT]", method, len(indices))
    print("[RESULT] union", summary["union_count"])
    print(f"[INFO] saved {args.output_pkl}")


if __name__ == "__main__":
    main()
