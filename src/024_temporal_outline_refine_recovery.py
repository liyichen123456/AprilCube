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
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

THIS_FILE = Path(__file__).resolve()
APRILCUBE_ROOT = THIS_FILE.parent.parent
SCRIPT_012_PATH = THIS_FILE.parent / "012_rs_aprilcube_detect.py"
SCRIPT_022_PATH = THIS_FILE.parent / "022_benchmark_single_frame_recovery_methods.py"
DEFAULT_INPUT_PKL = APRILCUBE_ROOT / "recordings/023_fused_all_single_frame_recovery_edge045_centerpnp_singletag.pkl"
DEFAULT_RAW_PKL = APRILCUBE_ROOT / "recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_OUTPUT_PKL = APRILCUBE_ROOT / "recordings/024_temporal_outline_refine_recovery.pkl"

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

import aprilcube  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover failed frames using temporal interpolation plus RGB outline refinement.")
    parser.add_argument("--input-pkl", type=Path, default=DEFAULT_INPUT_PKL)
    parser.add_argument("--raw-pkl", type=Path, default=DEFAULT_RAW_PKL)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    parser.add_argument("--max-gap", type=int, default=25)
    parser.add_argument("--accept-edge", type=float, default=0.58)
    parser.add_argument("--tag-anchor-accept-edge", type=float, default=0.52)
    parser.add_argument("--tag-anchor-max-reproj", type=float, default=4.0)
    parser.add_argument("--tag-anchor-weight", type=float, default=1.8)
    parser.add_argument("--use-interp-if-edge", type=float, default=0.64)
    parser.add_argument("--min-improvement", type=float, default=0.03)
    parser.add_argument("--max-translation-delta-mm", type=float, default=35.0)
    parser.add_argument("--max-rotation-delta-deg", type=float, default=12.0)
    parser.add_argument("--reject-loose-input", action=argparse.BooleanOptionalAction, default=True)
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


def rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    ra = Rotation.from_matrix(rotation_from_rvec(rvec_a))
    rb = Rotation.from_matrix(rotation_from_rvec(rvec_b))
    return float(np.degrees((rb * ra.inv()).magnitude()))


def interpolate_pose(prev_pose: dict[str, Any], next_pose: dict[str, Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    t0 = np.asarray(prev_pose["tvec"], dtype=np.float64).reshape(3, 1)
    t1 = np.asarray(next_pose["tvec"], dtype=np.float64).reshape(3, 1)
    tvec = (1.0 - alpha) * t0 + alpha * t1
    r0 = Rotation.from_matrix(rotation_from_rvec(prev_pose["rvec"]))
    r1 = Rotation.from_matrix(rotation_from_rvec(next_pose["rvec"]))
    r = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))([alpha])[0]
    rvec = r.as_rotvec().reshape(3, 1)
    return rvec.astype(np.float64), tvec.astype(np.float64)


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


def edge_distance_cost(
    dist: np.ndarray,
    corners_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    if float(np.asarray(tvec, dtype=np.float64).reshape(3)[2]) <= 0.0:
        return 1e4
    projected, _ = cv2.projectPoints(corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = projected.reshape(-1, 2)
    h, w = dist.shape[:2]
    values: list[float] = []
    outside = 0
    for a, b in CUBE_EDGES:
        p0, p1 = pts[a], pts[b]
        length = float(np.linalg.norm(p1 - p0))
        samples = max(8, min(70, int(length / 3.0)))
        for t in np.linspace(0.05, 0.95, samples):
            p = p0 * (1.0 - t) + p1 * t
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                values.append(min(float(dist[y, x]), 12.0))
            else:
                outside += 1
                values.append(12.0)
    if not values:
        return 1e4
    return float(np.mean(values) + 0.05 * outside)


def detected_tag_points(
    bm: Any,
    gray: np.ndarray,
    *,
    config: Any,
    tag_corner_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    detections = bm.detect_sweep(gray, config=config, valid_ids=set(int(v) for v in tag_corner_map))
    obj_chunks: list[np.ndarray] = []
    img_chunks: list[np.ndarray] = []
    tag_ids: list[int] = []
    for tag_id, corners in detections:
        tag_id = int(tag_id)
        if tag_id not in tag_corner_map:
            continue
        obj_chunks.append(np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3))
        img_chunks.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
        tag_ids.append(tag_id)
    if not obj_chunks:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), []
    return np.vstack(obj_chunks), np.vstack(img_chunks), tag_ids


def tag_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float]:
    if object_points.shape[0] == 0:
        return float("inf"), float("inf")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)
    return float(np.mean(errors)), float(np.max(errors))


def refine_pose_from_outline(
    gray: np.ndarray,
    init_rvec: np.ndarray,
    init_tvec: np.ndarray,
    *,
    config: Any,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_object_points: np.ndarray | None = None,
    tag_image_points: np.ndarray | None = None,
    tag_anchor_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    corners_3d = cube_corners(config)
    init_cost = edge_distance_cost(dist, corners_3d, init_rvec, init_tvec, camera_matrix, dist_coeffs)
    init_t = np.asarray(init_tvec, dtype=np.float64).reshape(3)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_rot = Rotation.from_rotvec(np.asarray(x[:3], dtype=np.float64))
        init_rot = Rotation.from_matrix(rotation_from_rvec(init_rvec))
        rot = delta_rot * init_rot
        rvec = rot.as_rotvec().reshape(3, 1)
        tvec = (init_t + np.asarray(x[3:6], dtype=np.float64)).reshape(3, 1)
        return rvec, tvec

    def objective(x: np.ndarray) -> float:
        rvec, tvec = unpack(x)
        data_cost = edge_distance_cost(dist, corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
        rot_reg = float(np.linalg.norm(x[:3]) / 0.22) ** 2
        t_reg = float(np.linalg.norm(x[3:6] / np.array([22.0, 22.0, 35.0], dtype=np.float64))) ** 2
        tag_cost = 0.0
        if (
            tag_anchor_weight > 0.0
            and tag_object_points is not None
            and tag_image_points is not None
            and tag_object_points.shape[0] > 0
        ):
            tag_mean, _tag_max = tag_reprojection_error(
                tag_object_points,
                tag_image_points,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            tag_cost = min(float(tag_mean), 50.0) / 5.0
        return data_cost + float(tag_anchor_weight) * tag_cost + 0.25 * rot_reg + 0.20 * t_reg

    best = np.zeros(6, dtype=np.float64)
    seeds = [
        np.zeros(6, dtype=np.float64),
        np.array([0.0, 0.0, 0.0, -8.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, -8.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 8.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -12.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 12.0]),
    ]
    bounds = [(-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32), (-35.0, 35.0), (-35.0, 35.0), (-45.0, 45.0)]
    best_value = objective(best)
    for seed in seeds:
        result = minimize(
            objective,
            seed,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 90, "xtol": 1e-3, "ftol": 1e-3, "disp": False},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best = np.asarray(result.x, dtype=np.float64)
    rvec, tvec = unpack(best)
    return rvec, tvec, init_cost, edge_distance_cost(dist, corners_3d, rvec, tvec, camera_matrix, dist_coeffs)


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
    cv2.putText(vis, f"Temporal outline: {pose.get('pose_source', '')}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, str(pose.get("quality_reason", ""))[:120], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return encode_bgr_jpeg(vis, quality)


def input_pose_usable(frame: dict[str, Any], *, reject_loose_input: bool) -> bool:
    pose = frame.get("pose", {})
    if not bool(pose.get("success", False)):
        return False
    if bool(reject_loose_input) and str(pose.get("pose_source", "")) == "stage7_edge_checked_loose_candidate":
        return False
    return True


def main() -> None:
    args = parse_args()
    bm = load_module(SCRIPT_022_PATH, "temporal024_benchmark_helpers")
    script012 = load_module(SCRIPT_012_PATH, "temporal024_script012")

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
    tag_corner_map = aprilcube.build_tag_corner_map(config)
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
    success_indices = [
        idx
        for idx in indices
        if input_pose_usable(frames[idx], reject_loose_input=bool(args.reject_loose_input))
    ]
    recovered: dict[int, dict[str, Any]] = {}
    rejected: dict[int, str] = {}

    for idx in indices:
        if input_pose_usable(frames[idx], reject_loose_input=bool(args.reject_loose_input)):
            continue
        prevs = [v for v in success_indices if v < idx]
        nexts = [v for v in success_indices if v > idx]
        if not prevs or not nexts:
            rejected[idx] = "no_bracketing_success_pose"
            continue
        prev_idx, next_idx = prevs[-1], nexts[0]
        gap = int(next_idx - prev_idx)
        if gap > int(args.max_gap):
            rejected[idx] = f"bracket_gap_too_large:{gap}>{int(args.max_gap)}"
            continue
        alpha = float(idx - prev_idx) / float(max(gap, 1))
        init_rvec, init_tvec = interpolate_pose(frames[prev_idx]["pose"], frames[next_idx]["pose"], alpha)
        raw_record = bm.load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
        image = np.asarray(raw_record["image_bgr"], dtype=np.uint8)
        detect_frame = script012.undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        init_pose = {
            "success": True,
            "rvec": init_rvec,
            "tvec": init_tvec,
            "pose_filled": True,
        }
        init_edge = bm.edge_alignment_score(gray, init_pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        tag_object_points, tag_image_points, detected_tag_ids = detected_tag_points(
            bm,
            gray,
            config=config,
            tag_corner_map=tag_corner_map,
        )
        use_tag_anchor = len(detected_tag_ids) == 1
        if not use_tag_anchor and init_edge >= float(args.use_interp_if_edge):
            opt_rvec, opt_tvec = init_rvec, init_tvec
            init_cost = edge_distance_cost(
                cv2.distanceTransform(
                    255 - cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140),
                    cv2.DIST_L2,
                    3,
                ),
                cube_corners(config),
                init_rvec,
                init_tvec,
                camera_matrix,
                dist_coeffs,
            )
            opt_cost = init_cost
            used_interp_direct = True
        else:
            used_interp_direct = False
            opt_rvec, opt_tvec, init_cost, opt_cost = refine_pose_from_outline(
                gray,
                init_rvec,
                init_tvec,
                config=config,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                tag_object_points=tag_object_points if use_tag_anchor else None,
                tag_image_points=tag_image_points if use_tag_anchor else None,
                tag_anchor_weight=float(args.tag_anchor_weight) if use_tag_anchor else 0.0,
            )
        opt_pose = {
            "success": True,
            "rvec": opt_rvec,
            "tvec": opt_tvec,
            "pose_filled": True,
        }
        opt_edge = bm.edge_alignment_score(gray, opt_pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        tag_mean_reproj, tag_max_reproj = tag_reprojection_error(
            tag_object_points,
            tag_image_points,
            opt_rvec,
            opt_tvec,
            camera_matrix,
            dist_coeffs,
        ) if len(detected_tag_ids) > 0 else (float("inf"), float("inf"))
        trans_delta = float(np.linalg.norm(np.asarray(opt_tvec).reshape(3) - np.asarray(init_tvec).reshape(3)))
        rot_delta = rotation_delta_deg(init_rvec, opt_rvec)
        accept_edge = float(args.tag_anchor_accept_edge) if use_tag_anchor else float(args.accept_edge)
        if opt_edge < accept_edge:
            rejected[idx] = f"edge_too_low:{opt_edge:.3f}<{accept_edge:.3f}"
            continue
        if use_tag_anchor and tag_mean_reproj > float(args.tag_anchor_max_reproj):
            rejected[idx] = f"tag_anchor_reproj_too_high:{tag_mean_reproj:.2f}>{float(args.tag_anchor_max_reproj):.2f}"
            continue
        if (
            not use_tag_anchor
            and opt_edge < init_edge + float(args.min_improvement)
            and init_edge < float(args.accept_edge)
        ):
            rejected[idx] = f"edge_improvement_too_small:{init_edge:.3f}->{opt_edge:.3f}"
            continue
        if trans_delta > float(args.max_translation_delta_mm):
            rejected[idx] = f"translation_delta_too_large:{trans_delta:.1f}"
            continue
        if rot_delta > float(args.max_rotation_delta_deg):
            rejected[idx] = f"rotation_delta_too_large:{rot_delta:.1f}"
            continue
        pose = {
            "success": True,
            "pose_source": (
                "stage10_temporal_tag_outline_refine"
                if use_tag_anchor
                else ("stage10_temporal_interp" if used_interp_direct else "stage10_temporal_outline_refine")
            ),
            "quality_level": "T",
            "quality_reason": (
                f"bracket:{prev_idx}-{next_idx};edge:{init_edge:.2f}->{opt_edge:.2f};"
                f"cost:{init_cost:.2f}->{opt_cost:.2f};dt:{trans_delta:.1f}mm;dr:{rot_delta:.1f}deg;"
                f"tag_anchor:{detected_tag_ids if use_tag_anchor else []};tag_reproj:{tag_mean_reproj:.2f};"
                f"interp_direct:{used_interp_direct}"
            ),
            "pose_filled": True,
            "temporal_recovery": True,
            "single_frame_only": False,
            "rvec": opt_rvec,
            "tvec": opt_tvec,
            "T": bm.pose_transform(opt_rvec, opt_tvec),
            "reproj_error": float("nan"),
            "n_tags": 0,
            "tag_ids": [],
            "visible_faces": [],
            "edge_score": float(opt_edge),
            "init_edge_score": float(init_edge),
            "outline_cost": float(opt_cost),
            "init_outline_cost": float(init_cost),
            "prev_success_frame": int(prev_idx),
            "next_success_frame": int(next_idx),
            "interpolation_alpha": float(alpha),
            "temporal_init_rvec": init_rvec,
            "temporal_init_tvec": init_tvec,
            "temporal_delta_t_mm": float(trans_delta),
            "temporal_delta_r_deg": float(rot_delta),
            "detected_tag_ids": detected_tag_ids,
            "tag_anchor_used": bool(use_tag_anchor),
            "tag_anchor_reproj_error": float(tag_mean_reproj),
            "tag_anchor_max_reproj_error": float(tag_max_reproj),
            "interp_direct": bool(used_interp_direct),
        }
        recovered[idx] = pose

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
        out_header.setdefault("metadata", {})
        out_header["metadata"] = {
            **(out_header.get("metadata", {}) or {}),
            "script": str(THIS_FILE),
            "method": "temporal interpolation followed by current-frame RGB cube-outline edge refinement",
            "max_gap": int(args.max_gap),
            "accept_edge": float(args.accept_edge),
            "min_improvement": float(args.min_improvement),
        }
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            if idx in recovered:
                raw_record = bm.load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
                image = np.asarray(raw_record["image_bgr"], dtype=np.uint8)
                detect_frame = script012.undistort_frame(image, undistort_pack)
                frame["pose_original"] = copy.deepcopy(frame.get("pose", {}))
                frame["pose"] = recovered[idx]
                frame["selected_stage"] = "stage10_temporal_outline_refine"
                frame["overlay_jpeg"] = draw_overlay(script012, draw_detector, detect_frame, recovered[idx], int(args.jpeg_quality))
                frame["overlay_format"] = "jpeg_bgr"
                frame["overlay_shape"] = tuple(int(v) for v in detect_frame.shape)
            elif not input_pose_usable(frame, reject_loose_input=bool(args.reject_loose_input)):
                original_pose = copy.deepcopy(frame.get("pose", {}))
                frame["pose_original"] = original_pose
                frame["pose"] = {
                    "success": False,
                    "pose_source": "fused_failed",
                    "quality_level": "Z",
                    "quality_reason": rejected.get(idx, "input_pose_rejected"),
                    "failure_reason": rejected.get(idx, "input_pose_rejected"),
                    "reproj_error": float("inf"),
                    "pose_filled": False,
                    "single_frame_only": False,
                }
                frame.setdefault("pose_candidates", {})
                frame["pose_candidates"]["stage10_temporal_outline_refine"] = {
                    "success": False,
                    "failure_reason": rejected.get(idx, "not_attempted"),
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
            "recovered_count": len(recovered),
            "recovered_frames": sorted(int(v) for v in recovered),
            "remaining_failed_frames": [int(idx) for idx in indices if not bool((recovered.get(idx) or frames[idx].get("pose", {})).get("success", False))],
            "rejected_temporal_reasons": rejected,
            "quality_counts": quality_counts,
            "source_counts": source_counts,
            "input_footer": footer,
            "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        pickle.dump(out_footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] saved {args.output_pkl}")
    print(f"[INFO] recovered={len(recovered)} frames={sorted(recovered)}")
    print(f"[INFO] success={success_count}/{len(indices)}")
    print(f"[INFO] remaining_failed={out_footer['remaining_failed_frames']}")


if __name__ == "__main__":
    main()
