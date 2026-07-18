#!/usr/bin/env python3
"""One-file offline AprilCube pose postprocess pipeline.

The readable pipeline lives at the top of this file:

1. Validate a synchronized multi-camera recording and extract one internal
   image stream for each wrist/index/thumb/middle target.
2. Estimate strict AprilCube poses.
3. Estimate DeepTag dense-keypoint poses.
4. Fuse single-frame candidates using reprojection and edge gates.
5. Reject temporally inconsistent single-frame poses before they become anchors.
6. Recover short gaps with embedded bidirectional RGB flow and tag-corner PnP.
7. Recover remaining hard frames with conservative RGB outline refinement.
8. Fill only short bracketed gaps and apply final timestamp-domain SE(3) smoothing.
9. Retarget the three fingertip cubes to the Wuji left hand with a 4-point
   per-finger objective and joint velocity/acceleration/jerk regularization.
10. Solve xArm7 + Wuji full-body IK, enforce hardware trajectory limits, and
    atomically write one ``<raw-stem>_post_progress.pkl`` containing poses,
    raw images, overlays, and all qpos fields.

The copied stage implementations are private details kept in this file so the
single supported CLI can run without launching or importing numbered scripts.
The command-line interface intentionally accepts only raw recordings produced
by ``scripts/drafts/020_visualize_multi_av_cv2_cameras.py``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import io
import os
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
import yaml
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

try:
    import viser
except ImportError:
    # Batch-only callers do not need the optional Viser UI dependency.
    viser = None

APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = APRILCUBE_ROOT / "src"
RECORDINGS_DIR = APRILCUBE_ROOT / "recordings"
PROJECT_ROOT = APRILCUBE_ROOT.parent.parent
RECORDER_UTILS_DIR = PROJECT_ROOT / "scripts" / "utils"
DEEPTAG_ROOT = APRILCUBE_ROOT / "thirdparty" / "deeptag-pytorch"

for import_path in (SRC_DIR, RECORDER_UTILS_DIR, DEEPTAG_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import aprilcube
from aprilcube import detect as aprilcube_detect
from aprilcube.detect import (
    _gamma_correct,
    _linear_contrast,
    _preprocess,
    _preprocess_clahe,
    _quad_quality,
    _sharpen,
    create_detector,
    create_fallback_detector,
    estimate_pose,
    estimate_single_tag_cube_pose,
)
from fiducial_marker.unit_arucotag import UnitArucoTag
from stag_decode.pose_estimator import get_fine_grid_points_anno

try:
    from recorder_cv2_cam import CV2CameraManager
except ImportError:
    # Camera capture/replay is not used by offline 012 batch processing.
    CV2CameraManager = None

preprocess_tag_image = _preprocess

# Retained only by the embedded 008 implementation; the public CLI below does
# not accept 008 recordings. It remains executable code because several strict
# pose primitives share its detector implementation.
_LEGACY_PROCESS_008_CUBE_NAMES: tuple[str, ...] | None = (
    "cube_april_36h11_6_11_1x1x1_15mm",
)

RAW_008_PKL_FORMAT = "aprilcube_raw_frame_stream_v1"
RAW_012_PKL_FORMAT = "aprilcube_rs_raw_frame_stream_v1"
RAW_012_WITH_POSE_PKL_FORMAT = "aprilcube_012_raw_with_pose_stream_v1"
_INTERNAL_TARGET_STREAM_FORMATS = {
    RAW_012_PKL_FORMAT,
    RAW_012_WITH_POSE_PKL_FORMAT,
}
POSTPROCESSED_PKL_FORMAT = "aprilcube_raw_with_020_postprocessed_pose_stream_v1"
LEGACY_POSTPROCESSED_PKL_FORMAT = (
    "aprilcube_012_raw_with_final_postprocessed_pose_stream_v1"
)
PROCESSING_CACHE_IDENTITY_VERSION = 1
HASH_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class Replay008ViewerConfig:
    pkl_path: str
    host: str
    port: int
    max_width: int
    jpeg_quality: int = 85
    cameras: str | None = None
    cube_dirs: str | None = None
    slow: bool = True
    no_filter: bool = False
    with_filter: bool = False
    no_undistort: bool = False
    shared_detect_tags: bool = True
    precompute_only: bool = False


@dataclass(frozen=True)
class StrictAprilCubeEstimationConfig:
    pkl_path: str
    output_pkl: Path
    intrinsics_yaml: Path | None = None
    cube_cfg: Path | None = None
    host: str = "0.0.0.0"
    port: int = 8094
    fps: float = 15.0
    max_width: int = 960
    max_frames: int = 0
    no_undistort: bool = False
    slow: bool = True
    no_filter: bool = True
    fallback_layout: str = "cfg"
    fallback_max_reproj: float = 5.0
    fallback_ransac_reproj: float = 3.0
    no_fill_missing_pose: bool = True
    precompute_only: bool = True
    show_viser: bool = False
    jpeg_quality: int = 90
    save_raw_jpeg: bool = False


@dataclass(frozen=True)
class PoseVisualizationConfig:
    pkl_path: str
    host: str
    port: int
    fps: float = 15.0
    max_width: int = 960


@dataclass(frozen=True)
class DeepTagDetectionConfig:
    pkl_path: str
    output_pkl: Path
    intrinsics_yaml: Path | None = None
    cube_cfg: Path | None = None
    max_frames: int = 0
    start_frame: int = 0
    stride: int = 1
    cpu: bool = False
    detect_scale: float = -1.0
    min_center_score: float = 0.2
    min_corner_score: float = 0.2
    hamming_dist: int = 8
    stg2_iter_num: int = 2
    batch_size_stg2: int = 4
    jpeg_quality: int = 90
    no_undistort: bool = False
    quiet_deeptag: bool = True
    corner_order: str = "rot180"
    pose_mode: str = "robust-cluster"
    robust_min_tags: int = 2
    robust_cluster_trans_mm: float = 70.0
    robust_cluster_rot_deg: float = 55.0
    robust_max_reproj: float = 12.0
    robust_single_tag_max_reproj: float = 4.0


@dataclass(frozen=True)
class RawPoseMergeConfig:
    raw_pkl: str
    pose_pkl: str
    output_pkl: str
    delete_inputs: bool = False


@dataclass(frozen=True)
class DenseDeepTagPoseConfig:
    deeptag_pkl: str
    output_pkl: Path
    max_frames: int = 0
    start_frame: int = 0
    stride: int = 1
    min_tags: int = 2
    ransac_reproj: float = 4.0
    max_reproj: float = 6.0
    point_reject_px: float = 8.0
    tag_reject_px: float = 8.0
    min_inlier_tag_fraction: float = 0.5
    coverage_check_min_raw_tags: int = 3
    max_required_inlier_tags: int = 4
    require_validated_corner_order: bool = True
    min_point_inlier_fraction: float = 0.25
    min_per_tag_inlier_fraction: float = 0.25
    min_per_tag_inlier_points: int = 12
    jpeg_quality: int = 90
    no_source_overlay: bool = False


@dataclass(frozen=True)
class SingleFrameFusionConfig:
    raw_pkl: Path
    deeptag_raw_pkl: Path
    deeptag_pose_pkl: Path
    april_strict_pkl: Path
    loose_deeptag_pkl: Path
    old_april_pkl: Path
    output_pkl: Path
    min_tags: int = 2
    max_reproj: float = 3.0
    edge_threshold: float = 0.45
    single_tag_edge_threshold: float = 0.60
    single_tag_max_reproj: float = 1.0
    preferred_single_tag_id: int | None = None
    prefer_deeptag_single_tag: bool = False
    jpeg_quality: int = 90


@dataclass(frozen=True)
class OutlinePoseRecoveryConfig:
    input_pkl: Path
    raw_pkl: Path
    output_pkl: Path
    max_gap: int = 25
    accept_edge: float = 0.58
    tag_anchor_accept_edge: float = 0.52
    tag_anchor_max_reproj: float = 4.0
    tag_anchor_weight: float = 1.8
    use_interp_if_edge: float = 0.64
    min_improvement: float = 0.03
    max_translation_delta_mm: float = 18.0
    max_rotation_delta_deg: float = 12.0
    reject_loose_input: bool = True
    jpeg_quality: int = 90


@dataclass(frozen=True)
class TemporalOutlierRejectionConfig:
    input_pkl: Path
    output_pkl: Path
    max_neighbor_gap: int = 12
    rotation_residual_deg: float = 25.0
    translation_residual_mm: float = 18.0
    min_two_sided_rotation_jump_deg: float = 35.0
    min_two_sided_translation_jump_mm: float = 25.0


@dataclass(frozen=True)
class AdjacentRgbFlowRecoveryConfig:
    """Conservative adjacent-frame RGB recovery embedded in the 020 pipeline.

    This replaces the former middle-leading and wrist-bidirectional repair
    scripts.  It scans every short failed run automatically, propagates from
    both reliable boundaries, and accepts a pose only after forward/backward
    LK, homography, PnP, motion, and RGB cube-edge gates pass.
    """

    input_pkl: Path
    raw_pkl: Path
    output_pkl: Path
    target_name: str = "cube_Q"
    max_gap_frames: int = 5
    max_features: int = 500
    feature_quality: float = 0.005
    feature_min_distance: float = 4.0
    min_features: int = 40
    lk_window: int = 41
    lk_levels: int = 5
    max_fb_error: float = 1.5
    max_fb_median_px: float = 0.75
    min_good_tracks: int = 30
    homography_ransac_px: float = 2.5
    min_homography_inliers: int = 20
    min_homography_inlier_ratio: float = 0.20
    max_homography_median_px: float = 1.5
    max_current_tag_agreement_px: float = 12.0
    max_flow_corner_reproj_px: float = 4.0
    max_translation_delta_mm: float = 8.0
    max_rotation_delta_deg: float = 12.0
    min_edge_score: float = 0.04
    allow_missing_current_tag: bool = True
    min_tag_corners_inside: int = 4
    feature_mask_scale: float = 1.0


@dataclass(frozen=True)
class TemporalPoseCompletionConfig:
    input_pkl: Path
    raw_pkl: Path
    output_pkl: Path
    # Only fill an isolated/very short miss bracketed by reliable detections.
    # A gap of 3 covers two missing frames between two measured anchors.
    max_bracket_gap: int = 3
    jpeg_quality: int = 90


@dataclass(frozen=True)
class TemporalPoseSmoothingConfig:
    input_pkl: Path
    raw_pkl: Path
    output_pkl: Path
    window_radius: int = 4
    sigma_frames: float = 1.0
    window_seconds: float = 0.18
    sigma_seconds: float = 0.075
    max_measured_translation_delta_mm: float = 4.0
    max_measured_rotation_delta_deg: float = 4.0
    max_filled_translation_delta_mm: float = 10.0
    max_filled_rotation_delta_deg: float = 8.0
    max_edge_score_drop: float = 0.04
    jpeg_quality: int = 90


def inspect_pkl_format(path: Path) -> str:
    header = load_pkl_header(path)
    fmt = str(header.get("format", ""))
    if not fmt:
        raise ValueError(f"PKL has no header format: {path}")
    return fmt


def load_pkl_header(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("rb") as f:
        header = pickle.load(f)
    if not isinstance(header, dict):
        raise ValueError(f"Unsupported pkl header in {path}: {type(header).__name__}")
    return header


def estimate_strict_aprilcube_poses(raw_pkl: Path, output_pkl: Path) -> None:
    print("[STAGE] strict AprilCube pose estimation", flush=True)
    strict_aprilcube_main(StrictAprilCubeEstimationConfig(
        pkl_path=str(raw_pkl),
        output_pkl=output_pkl,
        slow=False,
        no_undistort=False,
        fallback_layout="cfg",
    ))


def attach_strict_poses_to_raw_frames(
    raw_pkl: Path,
    pose_pkl: Path,
    output_pkl: Path,
) -> None:
    print("[STAGE] merge strict AprilCube poses with raw frames", flush=True)
    strict_pose_merge_main(RawPoseMergeConfig(
        raw_pkl=str(raw_pkl),
        pose_pkl=str(pose_pkl),
        output_pkl=str(output_pkl),
    ))


def detect_deeptag_keypoints(input_pkl: Path, output_pkl: Path) -> None:
    print("[STAGE] DeepTag keypoint detection", flush=True)
    deeptag_detection_main(DeepTagDetectionConfig(
        pkl_path=str(input_pkl),
        output_pkl=output_pkl,
    ))


def estimate_deeptag_dense_poses(
    deeptag_pkl: Path,
    output_pkl: Path,
    *,
    min_tags: int,
    max_reproj: float,
    point_reject_px: float,
    tag_reject_px: float,
    min_inlier_tag_fraction: float,
    coverage_check_min_raw_tags: int,
    max_required_inlier_tags: int,
    require_validated_corner_order: bool = True,
    min_point_inlier_fraction: float = 0.25,
    min_per_tag_inlier_fraction: float = 0.25,
    min_per_tag_inlier_points: int = 12,
) -> None:
    print(f"[STAGE] dense DeepTag pose estimation min_tags={min_tags}", flush=True)
    dense_deeptag_main(DenseDeepTagPoseConfig(
        deeptag_pkl=str(deeptag_pkl),
        output_pkl=output_pkl,
        min_tags=min_tags,
        max_reproj=max_reproj,
        point_reject_px=point_reject_px,
        tag_reject_px=tag_reject_px,
        min_inlier_tag_fraction=min_inlier_tag_fraction,
        coverage_check_min_raw_tags=coverage_check_min_raw_tags,
        max_required_inlier_tags=max_required_inlier_tags,
        require_validated_corner_order=require_validated_corner_order,
        min_point_inlier_fraction=min_point_inlier_fraction,
        min_per_tag_inlier_fraction=min_per_tag_inlier_fraction,
        min_per_tag_inlier_points=min_per_tag_inlier_points,
    ))


def fuse_single_frame_pose_candidates(
    raw_pkl: Path,
    deeptag_raw_pkl: Path,
    deeptag_pose_pkl: Path,
    april_strict_pkl: Path,
    loose_deeptag_pkl: Path,
    old_april_pkl: Path,
    output_pkl: Path,
    *,
    single_tag_edge_threshold: float = 0.60,
    single_tag_max_reproj: float = 1.0,
    preferred_single_tag_id: int | None = None,
    prefer_deeptag_single_tag: bool = False,
) -> None:
    print("[STAGE] fuse single-frame pose candidates", flush=True)
    single_frame_fusion_main(SingleFrameFusionConfig(
        raw_pkl=raw_pkl,
        deeptag_raw_pkl=deeptag_raw_pkl,
        deeptag_pose_pkl=deeptag_pose_pkl,
        april_strict_pkl=april_strict_pkl,
        loose_deeptag_pkl=loose_deeptag_pkl,
        old_april_pkl=old_april_pkl,
        output_pkl=output_pkl,
        single_tag_edge_threshold=float(single_tag_edge_threshold),
        single_tag_max_reproj=float(single_tag_max_reproj),
        preferred_single_tag_id=preferred_single_tag_id,
        prefer_deeptag_single_tag=bool(prefer_deeptag_single_tag),
    ))


def recover_poses_from_outlines(
    input_pkl: Path,
    raw_pkl: Path,
    output_pkl: Path,
) -> None:
    print("[STAGE] temporal outline recovery", flush=True)
    outline_recovery_main(OutlinePoseRecoveryConfig(
        input_pkl=input_pkl,
        raw_pkl=raw_pkl,
        output_pkl=output_pkl,
    ))


def reject_temporal_pose_outliers(input_pkl: Path, output_pkl: Path) -> None:
    print("[STAGE] reject temporal pose outliers", flush=True)
    temporal_outlier_rejection_main(TemporalOutlierRejectionConfig(
        input_pkl=input_pkl,
        output_pkl=output_pkl,
    ))


def recover_poses_with_adjacent_rgb_flow(
    input_pkl: Path,
    raw_pkl: Path,
    output_pkl: Path,
    *,
    target_name: str | None,
) -> None:
    """Recover only short failed runs using adjacent RGB optical flow.

    The embedded stage subsumes the former one-frame middle recovery and
    fixed-anchor wrist recovery utilities.  No frame numbers or recording
    paths are hard-coded; every short gap is discovered from the pose stream.
    """

    target = str(target_name or "cube_Q")
    # The middle-finger shell supplies fewer stable edge features, while the
    # larger wrist cube tolerates a slightly larger adjacent-frame motion.
    if target == "middle_Q":
        policy = dict(
            min_features=60,
            min_good_tracks=40,
            max_translation_delta_mm=6.0,
            max_rotation_delta_deg=10.0,
        )
    elif target == "wrist_Q":
        policy = dict(
            min_features=20,
            min_good_tracks=12,
            max_translation_delta_mm=15.0,
            max_rotation_delta_deg=20.0,
            min_edge_score=0.0,
        )
    else:
        policy = dict(
            min_features=40,
            min_good_tracks=30,
            max_translation_delta_mm=8.0,
            max_rotation_delta_deg=12.0,
        )
    print(f"[STAGE] adjacent bidirectional RGB-flow recovery target={target}", flush=True)
    adjacent_rgb_flow_recovery_main(
        AdjacentRgbFlowRecoveryConfig(
            input_pkl=input_pkl,
            raw_pkl=raw_pkl,
            output_pkl=output_pkl,
            target_name=target,
            **policy,
        )
    )


def fill_remaining_poses_from_trajectory(
    input_pkl: Path,
    raw_pkl: Path,
    output_pkl: Path,
) -> None:
    print("[STAGE] short local bracket interpolation", flush=True)
    temporal_completion_main(TemporalPoseCompletionConfig(
        input_pkl=input_pkl,
        raw_pkl=raw_pkl,
        output_pkl=output_pkl,
    ))


def smooth_completed_pose_trajectory(
    input_pkl: Path,
    raw_pkl: Path,
    output_pkl: Path,
) -> None:
    print("[STAGE] constrained temporal pose smoothing", flush=True)
    temporal_pose_smoothing_main(TemporalPoseSmoothingConfig(
        input_pkl=input_pkl,
        raw_pkl=raw_pkl,
        output_pkl=output_pkl,
    ))


def _process_extracted_target_stream(
    *,
    raw_pkl: Path,
    output_pkl: Path,
    work_dir: Path,
    merge_final_raw: bool = True,
    single_tag_edge_threshold: float = 0.60,
    single_tag_max_reproj: float = 1.0,
    preferred_single_tag_id: int | None = None,
    prefer_deeptag_single_tag: bool = False,
    enable_temporal_outline_recovery: bool = True,
    enable_adjacent_rgb_flow_recovery: bool = True,
    target_name: str | None = None,
) -> Path:
    raw_pkl = raw_pkl.expanduser().resolve()
    if target_name == "middle_Q":
        # Former middle_Q repair behavior is now an intrinsic target policy:
        # prefer the physical tag-2 observation, use the DeepTag single-face
        # candidate before AprilTag stage8, and forbid long-range outline fill.
        if preferred_single_tag_id is None:
            preferred_single_tag_id = 2
        prefer_deeptag_single_tag = True
        enable_temporal_outline_recovery = False
    fmt = inspect_pkl_format(raw_pkl)
    if fmt not in _INTERNAL_TARGET_STREAM_FORMATS:
        raise ValueError(
            "The internal target pipeline requires an extracted raw-image stream "
            f"(format={_INTERNAL_TARGET_STREAM_FORMATS}), got {fmt}: {raw_pkl}"
        )

    output_pkl = output_pkl.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if merge_final_raw and processed_output_matches_input(output_pkl, raw_pkl):
        print(f"[INFO] Existing 020 output matches input; skip pose recompute: {output_pkl}")
        summarize_pose_stream(output_pkl, "pose")
        return output_pkl

    april_strict_pkl = work_dir / f"strict_aprilcube_pose_{raw_pkl.stem}.pkl"
    april_merged_pkl = work_dir / f"raw_with_strict_aprilcube_pose_{raw_pkl.stem}.pkl"
    deeptag_raw_pkl = work_dir / f"deeptag_keypoints_{raw_pkl.stem}.pkl"
    deeptag_dense_strict_pkl = (
        work_dir / f"strict_deeptag_dense_pose_{raw_pkl.stem}.pkl"
    )
    deeptag_dense_loose_pkl = work_dir / f"loose_deeptag_dense_pose_{raw_pkl.stem}.pkl"
    fused_single_frame_pkl = work_dir / f"fused_single_frame_pose_{raw_pkl.stem}.pkl"
    temporal_cleaned_pkl = work_dir / f"temporal_outlier_rejected_pose_{raw_pkl.stem}.pkl"
    rgb_flow_recovered_pkl = work_dir / f"adjacent_rgb_flow_recovered_pose_{raw_pkl.stem}.pkl"
    outline_refine_pkl = work_dir / f"outline_recovered_pose_{raw_pkl.stem}.pkl"
    temporally_completed_pkl = work_dir / f"temporally_completed_pose_{raw_pkl.stem}.pkl"
    final_pose_pkl = work_dir / f"temporally_smoothed_pose_{raw_pkl.stem}.pkl"

    work_dir.mkdir(parents=True, exist_ok=True)

    estimate_strict_aprilcube_poses(raw_pkl, april_strict_pkl)

    if fmt == RAW_012_PKL_FORMAT:
        attach_strict_poses_to_raw_frames(
            raw_pkl,
            april_strict_pkl,
            april_merged_pkl,
        )
    else:
        april_merged_pkl = raw_pkl
        print(
            "[INFO] Input already has raw images plus a pose field; "
            "using it as the raw/old-April stream for downstream stages."
        )

    detect_deeptag_keypoints(
        april_merged_pkl,
        deeptag_raw_pkl,
    )
    estimate_deeptag_dense_poses(
        deeptag_raw_pkl,
        deeptag_dense_strict_pkl,
        min_tags=2,
        max_reproj=6.0,
        point_reject_px=8.0,
        tag_reject_px=8.0,
        min_inlier_tag_fraction=0.5,
        coverage_check_min_raw_tags=3,
        max_required_inlier_tags=4,
    )
    estimate_deeptag_dense_poses(
        deeptag_raw_pkl,
        deeptag_dense_loose_pkl,
        min_tags=1,
        max_reproj=12.0,
        point_reject_px=12.0,
        tag_reject_px=12.0,
        min_inlier_tag_fraction=0.0,
        coverage_check_min_raw_tags=1_000_000,
        max_required_inlier_tags=1_000_000,
    )

    fuse_single_frame_pose_candidates(
        raw_pkl=april_merged_pkl,
        deeptag_raw_pkl=deeptag_raw_pkl,
        deeptag_pose_pkl=deeptag_dense_strict_pkl,
        april_strict_pkl=april_strict_pkl,
        loose_deeptag_pkl=deeptag_dense_loose_pkl,
        old_april_pkl=april_merged_pkl,
        output_pkl=fused_single_frame_pkl,
        single_tag_edge_threshold=float(single_tag_edge_threshold),
        single_tag_max_reproj=float(single_tag_max_reproj),
        preferred_single_tag_id=preferred_single_tag_id,
        prefer_deeptag_single_tag=bool(prefer_deeptag_single_tag),
    )
    reject_temporal_pose_outliers(
        input_pkl=fused_single_frame_pkl,
        output_pkl=temporal_cleaned_pkl,
    )
    if enable_adjacent_rgb_flow_recovery:
        recover_poses_with_adjacent_rgb_flow(
            input_pkl=temporal_cleaned_pkl,
            raw_pkl=april_merged_pkl,
            output_pkl=rgb_flow_recovered_pkl,
            target_name=target_name,
        )
    else:
        print("[STAGE] skip adjacent RGB-flow recovery", flush=True)
        shutil.copy2(temporal_cleaned_pkl, rgb_flow_recovered_pkl)
    if enable_temporal_outline_recovery:
        recover_poses_from_outlines(
            input_pkl=rgb_flow_recovered_pkl,
            raw_pkl=april_merged_pkl,
            output_pkl=outline_refine_pkl,
        )
    else:
        print("[STAGE] skip long-range temporal outline recovery", flush=True)
        shutil.copy2(rgb_flow_recovered_pkl, outline_refine_pkl)
    fill_remaining_poses_from_trajectory(
        input_pkl=outline_refine_pkl,
        raw_pkl=april_merged_pkl,
        output_pkl=temporally_completed_pkl,
    )
    smooth_completed_pose_trajectory(
        input_pkl=temporally_completed_pkl,
        raw_pkl=april_merged_pkl,
        output_pkl=final_pose_pkl,
    )

    if not merge_final_raw:
        summarize_pose_stream(final_pose_pkl, "pose")
        return final_pose_pkl

    merge_final_pose_stream(
        raw_pkl=raw_pkl,
        final_pose_pkl=final_pose_pkl,
        output_pkl=output_pkl,
        timestamp_tolerance=1e-6,
        keep_original_pose=True,
        keep_pose_candidates=True,
    )
    summarize_pose_stream(output_pkl, "pose")
    return output_pkl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def build_processing_cache_identity(raw_pkl: Path) -> dict[str, Any]:
    raw_pkl = raw_pkl.expanduser().resolve()
    script_path = Path(__file__).resolve()
    stat_before = raw_pkl.stat()
    raw_sha256 = sha256_file(raw_pkl)
    stat_after = raw_pkl.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError(f"Input PKL changed while hashing: {raw_pkl}")
    return {
        "version": PROCESSING_CACHE_IDENTITY_VERSION,
        "raw_file": {
            "path": str(raw_pkl),
            "size": int(stat_after.st_size),
            "mtime_ns": int(stat_after.st_mtime_ns),
            "sha256": raw_sha256,
        },
        "pipeline_script": {
            "path": str(script_path),
            "sha256": sha256_file(script_path),
        },
    }


def processed_output_matches_input(output_pkl: Path, raw_pkl: Path) -> bool:
    if not output_pkl.exists():
        return False
    try:
        header = load_pkl_header(output_pkl)
    except Exception:
        return False
    if header.get("format") not in {
        POSTPROCESSED_PKL_FORMAT,
        LEGACY_POSTPROCESSED_PKL_FORMAT,
    }:
        return False
    stored_identity = header.get("processing_cache_identity")
    if not isinstance(stored_identity, dict):
        return False
    if stored_identity.get("version") != PROCESSING_CACHE_IDENTITY_VERSION:
        return False
    try:
        raw_pkl = raw_pkl.expanduser().resolve()
        raw_stat = raw_pkl.stat()
        stored_raw = stored_identity["raw_file"]
        if not isinstance(stored_raw, dict):
            return False
        if (
            stored_raw.get("path") != str(raw_pkl)
            or stored_raw.get("size") != int(raw_stat.st_size)
            or stored_raw.get("mtime_ns") != int(raw_stat.st_mtime_ns)
        ):
            return False
        script_path = Path(__file__).resolve()
        stored_script = stored_identity["pipeline_script"]
        if not isinstance(stored_script, dict):
            return False
        if stored_script.get("path") != str(script_path):
            return False
        raw_sha256 = sha256_file(raw_pkl)
        raw_stat_after_hash = raw_pkl.stat()
        if (
            raw_stat_after_hash.st_size != raw_stat.st_size
            or raw_stat_after_hash.st_mtime_ns != raw_stat.st_mtime_ns
        ):
            return False
        return (
            stored_raw.get("sha256") == raw_sha256
            and stored_script.get("sha256") == sha256_file(script_path)
        )
    except (KeyError, OSError):
        return False


def identify_008_camera(header: dict[str, Any], offsets: list[int], pkl_path: Path) -> str:
    metadata = header.get("metadata", {}) or {}
    opened = metadata.get("opened_cameras", []) or []
    if len(opened) == 1:
        return str(opened[0])
    seen: set[str] = set()
    for offset in offsets[: min(len(offsets), 20)]:
        frame = load_at(pkl_path, offset)
        camera_name = frame.get("camera_name", None)
        if camera_name is not None:
            seen.add(str(camera_name))
    if len(seen) == 1:
        return next(iter(seen))
    raise ValueError(
        "020 008-to-012 conversion expects one camera stream per pkl; "
        f"opened_cameras={opened}, sampled_frame_cameras={sorted(seen)}"
    )


def cube_name_from_path(path: Path) -> str:
    return path.name if path.name != "config.json" else path.parent.name


def split_008_recording_into_cube_streams(raw_008_pkl: Path, work_dir: Path) -> list[tuple[str, Path]]:
    header, offsets, footer = build_stream_index(raw_008_pkl)
    if header.get("format") != RAW_008_PKL_FORMAT:
        raise ValueError(f"Expected 008 raw pkl format, got {header.get('format')}: {raw_008_pkl}")
    if not offsets:
        raise ValueError(f"No frame records in {raw_008_pkl}")

    metadata = header.get("metadata", {}) or {}
    cube_paths = [Path(str(v)).expanduser().resolve() for v in metadata.get("cube_paths", []) or []]
    if not cube_paths:
        raise ValueError(f"008 pkl header has no metadata.cube_paths: {raw_008_pkl}")
    if _LEGACY_PROCESS_008_CUBE_NAMES is not None:
        requested_cube_names = set(_LEGACY_PROCESS_008_CUBE_NAMES)
        cube_path_by_name = {cube_name_from_path(path): path for path in cube_paths}
        missing_cube_names = requested_cube_names - cube_path_by_name.keys()
        if missing_cube_names:
            raise ValueError(
                "Requested 008 cubes are missing from metadata.cube_paths: "
                f"{sorted(missing_cube_names)}"
            )
        cube_paths = [
            cube_path_by_name[name]
            for name in _LEGACY_PROCESS_008_CUBE_NAMES
        ]
        print(
            "[INFO] 008 cube filter: "
            f"{[cube_name_from_path(path) for path in cube_paths]}"
        )
    camera_name = identify_008_camera(header, offsets, raw_008_pkl)
    intrinsics_by_camera = metadata.get("intrinsics_yaml", {}) or {}
    if not isinstance(intrinsics_by_camera, dict) or camera_name not in intrinsics_by_camera:
        raise ValueError(f"Missing intrinsics_yaml for camera {camera_name} in {raw_008_pkl}")

    intrinsics_yaml = Path(str(intrinsics_by_camera[camera_name])).expanduser().resolve()
    calib = realsense_load_intrinsics_yaml(intrinsics_yaml)
    first_frame = load_at(raw_008_pkl, offsets[0])
    image_shape = tuple(int(v) for v in first_frame["image_bgr"].shape)
    image_size = tuple(
        int(v)
        for v in (
            metadata.get("detect_img_size")
            or metadata.get("capture_size")
            or (image_shape[1], image_shape[0])
        )
    )

    streams: list[tuple[str, Path]] = []
    for cube_path in cube_paths:
        cube_name = cube_name_from_path(cube_path)
        out_pkl = work_dir / f"008_as_012_raw_{raw_008_pkl.stem}_{cube_name}.pkl"
        streams.append((cube_name, out_pkl))
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with out_pkl.open("wb") as f:
            pickle.dump(
                {
                    "type": "header",
                    "format": RAW_012_PKL_FORMAT,
                    "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_008_raw_pkl": str(raw_008_pkl),
                    "source_008_header": header,
                    "source_008_footer": footer,
                    "metadata": {
                        "script": str(Path(__file__).resolve()),
                        "method": "converted from 008 raw image stream for 020 multistage pose estimation",
                        "source_format": RAW_008_PKL_FORMAT,
                        "source_camera_name": camera_name,
                        "intrinsics_yaml": str(intrinsics_yaml),
                        "cube_cfg": str(cube_path),
                        "image_size": image_size,
                        "fps": int(metadata.get("fps", 0) or 0),
                        "undistort_for_detection": bool(metadata.get("undistort_before_detection", True)),
                        "raw_camera_matrix": calib["K"].tolist(),
                        "raw_dist_coeffs": calib["dist"].tolist(),
                        "raw_image_field": "image_bgr",
                        "raw_image_storage": "original numpy ndarray from 008 pkl",
                    },
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            for idx, offset in enumerate(offsets):
                frame = load_at(raw_008_pkl, offset)
                out_frame = {
                    "type": "frame",
                    "frame_index": int(idx),
                    "raw_source_offset": int(offset),
                    "device_name": str(frame.get("device_name", frame.get("camera_name", ""))),
                    "camera_name": str(frame.get("camera_name", "")),
                    "loop_frame_idx": int(frame.get("loop_frame_idx", idx)),
                    "capture_timestamp": frame.get("capture_timestamp", None),
                    "write_monotonic": frame.get("write_monotonic", None),
                    "shape": tuple(int(v) for v in frame.get("shape", frame["image_bgr"].shape)),
                    "dtype": str(frame.get("dtype", frame["image_bgr"].dtype)),
                    "image_bgr": frame["image_bgr"],
                }
                pickle.dump(out_frame, f, protocol=pickle.HIGHEST_PROTOCOL)
                done = idx + 1
                if done == len(offsets) or done % 25 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"\r[INFO] 008->012 {cube_name} {done}/{len(offsets)} "
                        f"fps={done / max(elapsed, 1e-9):.1f}",
                        end="",
                        flush=True,
                    )
            pickle.dump(
                {
                    "type": "footer",
                    "frame_count": len(offsets),
                    "source_008_footer": footer,
                    "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print()
        print(f"[INFO] saved 012-style raw for 008 cube={cube_name}: {out_pkl}")
    return streams


def load_pose_frames_by_index(path: Path) -> dict[int, dict[str, Any]]:
    _header, offsets, _footer = build_stream_index(path)
    frames: dict[int, dict[str, Any]] = {}
    for idx, offset in enumerate(offsets):
        frame = load_at(path, offset)
        frames[int(frame.get("frame_index", idx))] = frame
    return frames


def build_008_pose_cache_key(raw_008_pkl: Path, final_pose_by_cube: dict[str, Path]) -> dict[str, Any]:
    return {
        "format": "aprilcube_020_multistage_008_pose_v1",
        "source_raw_pkl": str(raw_008_pkl.resolve()),
        "cube_pose_pkls": {name: str(path.resolve()) for name, path in sorted(final_pose_by_cube.items())},
    }


def merge_cube_pose_streams_into_008(
    *,
    raw_008_pkl: Path,
    final_pose_by_cube: dict[str, Path],
) -> None:
    header, offsets, footer = build_stream_index(raw_008_pkl)
    if header.get("format") != RAW_008_PKL_FORMAT:
        raise ValueError(f"Expected 008 raw pkl format, got {header.get('format')}: {raw_008_pkl}")
    pose_frames_by_cube = {
        cube_name: load_pose_frames_by_index(pose_pkl)
        for cube_name, pose_pkl in final_pose_by_cube.items()
    }
    cache_key = build_008_pose_cache_key(raw_008_pkl, final_pose_by_cube)
    pose_cache: list[dict[str, Any]] = []
    success_count = 0
    cube_slots = 0
    for idx, offset in enumerate(offsets):
        raw_frame = load_at(raw_008_pkl, offset)
        cube_results: list[dict[str, Any]] = []
        status_lines: list[str] = []
        for cube_name, frames_by_index in sorted(pose_frames_by_cube.items()):
            pose_frame = frames_by_index.get(idx)
            if pose_frame is None:
                pose = {
                    "success": False,
                    "pose_source": "020_multistage_missing_frame",
                    "failure_reason": "missing_final_pose_frame",
                }
                selected_stage = ""
            else:
                pose = pose_frame.get("pose", {}) or {}
                selected_stage = str(pose_frame.get("selected_stage", ""))
            cube_slots += 1
            success_count += int(bool(pose.get("success", False)))
            status_lines.append(
                f"[{raw_frame.get('camera_name', '')}][{cube_name}] "
                f"success={bool(pose.get('success', False))} "
                f"source={pose.get('pose_source', '')} "
                f"reproj={pose.get('reproj_error', '')}"
            )
            cube_results.append(
                {
                    "cube_name": cube_name,
                    "cube_path": "",
                    "result": pose,
                    "selected_stage": selected_stage,
                    "pose_pipeline": "020_multistage_012_path",
                }
            )
        pose_cache.append(
            {
                "camera_name": str(raw_frame.get("camera_name", "")),
                "timestamp": raw_frame.get("capture_timestamp", None),
                "status_lines": status_lines,
                "cube_results": cube_results,
                "decoded_tag_count": 0,
                "pose_pipeline": "020_multistage_012_path",
            }
        )

    tmp_path = raw_008_pkl.with_name(f".{raw_008_pkl.name}.020-rewrite-{time.time_ns()}.tmp")
    frame_idx = 0
    try:
        with raw_008_pkl.open("rb") as src, tmp_path.open("wb") as dst:
            while True:
                try:
                    record = pickle.load(src)
                except EOFError:
                    break
                if isinstance(record, dict) and record.get("type") == "pose_cache":
                    continue
                if isinstance(record, dict) and record.get("type") == "frame":
                    record["offline_pose_frame"] = pose_cache[frame_idx]
                    record["offline_pose_cache_key"] = cache_key
                    frame_idx += 1
                elif isinstance(record, dict) and record.get("type") == "footer":
                    record = dict(record)
                    record["020_multistage_pose_success_count"] = int(success_count)
                    record["020_multistage_pose_cube_slots"] = int(cube_slots)
                    record["020_multistage_pose_cache_key"] = cache_key
                pickle.dump(record, dst, protocol=pickle.HIGHEST_PROTOCOL)
        if frame_idx != len(pose_cache):
            raise ValueError(f"008 frame count mismatch while writing poses: {frame_idx} != {len(pose_cache)}")
        tmp_path.replace(raw_008_pkl)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"[INFO] wrote 020 multistage poses into 008 pkl: {raw_008_pkl}")
    print(f"[INFO] 008 multistage cube success={success_count}/{cube_slots}")


def build_stream_index(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer = None
    with path.open("rb") as f:
        header = pickle.load(f)
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
        raise ValueError(f"{path}:{offset} is not a frame record")
    return obj


def build_timestamp_index(path: Path, offsets: list[int]) -> dict[float, int]:
    index: dict[float, int] = {}
    for offset in offsets:
        frame = load_at(path, offset)
        timestamp = frame.get("capture_timestamp", None)
        if timestamp is None:
            raise ValueError(f"Frame {frame.get('frame_index')} in {path} has no capture_timestamp")
        key = float(timestamp)
        if key in index:
            raise ValueError(f"Duplicate capture_timestamp {key} in {path}")
        index[key] = int(offset)
    return index


def nearest_timestamp(timestamp: float, index: dict[float, int], tolerance: float) -> tuple[float, int]:
    if timestamp in index:
        return timestamp, index[timestamp]
    best = min(index, key=lambda value: abs(float(value) - float(timestamp)))
    delta = abs(float(best) - float(timestamp))
    if delta > float(tolerance):
        raise ValueError(f"No pose timestamp within {tolerance} for raw timestamp {timestamp}; nearest delta={delta}")
    return best, index[best]


def frame_indices_match(raw_frame: dict[str, Any], pose_frame: dict[str, Any]) -> bool:
    raw_idx = raw_frame.get("frame_index", None)
    pose_idx = pose_frame.get("frame_index", None)
    if raw_idx is None or pose_idx is None:
        return True
    return int(raw_idx) == int(pose_idx)


def merge_final_pose_stream(
    *,
    raw_pkl: Path,
    final_pose_pkl: Path,
    output_pkl: Path,
    timestamp_tolerance: float,
    keep_original_pose: bool,
    keep_pose_candidates: bool,
) -> Path:
    processing_cache_identity = build_processing_cache_identity(raw_pkl)
    raw_header, raw_offsets, raw_footer = build_stream_index(raw_pkl)
    final_header, final_offsets, final_footer = build_stream_index(final_pose_pkl)
    if len(raw_offsets) != len(final_offsets):
        raise ValueError(f"Frame count mismatch: raw={len(raw_offsets)} final_pose={len(final_offsets)}")
    final_by_timestamp = build_timestamp_index(final_pose_pkl, final_offsets)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    source_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    t0 = time.perf_counter()

    with output_pkl.open("wb") as f:
        pickle.dump(
            {
                "type": "header",
                "format": POSTPROCESSED_PKL_FORMAT,
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_raw_pkl": str(raw_pkl),
                "source_final_pose_pkl": str(final_pose_pkl),
                "processing_cache_identity": processing_cache_identity,
                "raw_header": raw_header,
                "raw_footer": raw_footer,
                "final_pose_header": final_header,
                "final_pose_footer": final_footer,
                "metadata": {
                    "merge_key": "capture_timestamp",
                    "timestamp_tolerance": float(timestamp_tolerance),
                    "keep_original_pose": bool(keep_original_pose),
                    "keep_pose_candidates": bool(keep_pose_candidates),
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for out_idx, raw_offset in enumerate(raw_offsets):
            raw_frame = load_at(raw_pkl, raw_offset)
            raw_ts = float(raw_frame["capture_timestamp"])
            pose_ts, pose_offset = nearest_timestamp(raw_ts, final_by_timestamp, float(timestamp_tolerance))
            pose_frame = load_at(final_pose_pkl, pose_offset)
            if not frame_indices_match(raw_frame, pose_frame):
                raise ValueError(
                    f"Frame index mismatch at timestamp {raw_ts}: raw={raw_frame.get('frame_index')} "
                    f"pose={pose_frame.get('frame_index')}"
                )

            out_frame = dict(raw_frame)
            if keep_original_pose:
                out_frame["pose_original_raw"] = raw_frame.get("pose", {})
            out_frame["pose"] = pose_frame.get("pose", {})
            out_frame["pose_postprocessed"] = True
            out_frame["postprocessed_pose_source_offset"] = int(pose_offset)
            out_frame["postprocessed_pose_timestamp"] = float(pose_ts)
            out_frame["postprocessed_pose_timestamp_delta"] = float(pose_ts - raw_ts)
            out_frame["selected_stage"] = pose_frame.get("selected_stage", "")
            out_frame["overlay_shape"] = pose_frame.get("overlay_shape", raw_frame.get("overlay_shape"))
            out_frame["overlay_format"] = pose_frame.get("overlay_format", raw_frame.get("overlay_format"))
            out_frame["overlay_jpeg"] = pose_frame.get("overlay_jpeg", raw_frame.get("overlay_jpeg"))
            if keep_pose_candidates and "pose_candidates" in pose_frame:
                out_frame["pose_candidates"] = pose_frame["pose_candidates"]

            pose = out_frame.get("pose", {})
            success_count += int(bool(pose.get("success", False)))
            source = str(pose.get("pose_source", ""))
            quality = str(pose.get("quality_level", ""))
            source_counts[source] = source_counts.get(source, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            pickle.dump(out_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

            done = out_idx + 1
            if done == len(raw_offsets) or done % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f"\r[INFO] merged {done}/{len(raw_offsets)} fps={done / max(elapsed, 1e-9):.1f}", end="", flush=True)

        pickle.dump(
            {
                "type": "footer",
                "frame_count": len(raw_offsets),
                "success_count": int(success_count),
                "source_counts": source_counts,
                "quality_counts": quality_counts,
                "raw_footer": raw_footer,
                "final_pose_footer": final_footer,
                "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print()
    print(f"[INFO] saved {output_pkl}")
    print(f"[INFO] success={success_count}/{len(raw_offsets)}")
    return output_pkl


def summarize_008_pose_cache(path: Path) -> None:
    frames = 0
    pose_frames = 0
    cube_slots = 0
    success = 0
    with path.open("rb") as f:
        pickle.load(f)
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict) or obj.get("type") != "frame":
                continue
            frames += 1
            pose_frame = obj.get("offline_pose_frame", None)
            if not isinstance(pose_frame, dict):
                continue
            pose_frames += 1
            for cube in pose_frame.get("cube_results", []) or []:
                cube_slots += 1
                success += int(bool((cube.get("result") or {}).get("success", False)))
    print(
        "[INFO] 008 summary: "
        f"frames={frames} offline_pose_frame={pose_frames} cube_success={success}/{cube_slots}"
    )


def summarize_pose_stream(path: Path, pose_field: str) -> None:
    frames = 0
    success = 0
    source_counts: dict[str, int] = {}
    with path.open("rb") as f:
        pickle.load(f)
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict) or obj.get("type") != "frame":
                continue
            frames += 1
            pose = obj.get(pose_field, {}) or {}
            success += int(bool(pose.get("success", False)))
            source = str(pose.get("pose_source", ""))
            source_counts[source] = source_counts.get(source, 0) + 1
    print(f"[INFO] pose summary: success={success}/{frames} sources={source_counts}")




# ============================================================
# Inlined stage implementations
# ============================================================

# ---------------------------------------------------------------------------
# Copied internal implementations, flattened into prefixed functions
# ---------------------------------------------------------------------------


# ---- Camera calibration and AprilCube detector helpers ----
CV2_CAPTURE_THIS_FILE = Path(__file__).resolve()
CV2_CAPTURE_THIRDPARTY_DIR = CV2_CAPTURE_THIS_FILE.parent.parent.parent
CV2_CAPTURE_PROJECT_ROOT = CV2_CAPTURE_THIRDPARTY_DIR.parent
CV2_CAPTURE_RECORDER_UTILS_DIR = CV2_CAPTURE_PROJECT_ROOT / 'scripts' / 'utils'

def cv2_capture_load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    dist = data.get('dist', data.get('D', None))
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    return {'path': str(yaml_path), 'camera_model': str(data.get('camera_model', '')), 'distortion_model': str(data.get('distortion_model', '')), 'image_size': tuple((int(v) for v in data['image_size'])), 'K': np.asarray(data['K'], dtype=np.float64).reshape(3, 3), 'dist': np.asarray(dist, dtype=np.float64).reshape(-1)}
CV2_CAPTURE_CAMERA_TO_PORT: dict[str, str] = {'cam1': '3-9:1.0'}
CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML: dict[str, str] = {'cam1': '/home/ps/RobotCamCalib1/outputs/intrinsics_cam0_fisheye_2592x1944_0703_230535.yaml'}
CV2_CAPTURE_ACTIVE_CAMERA_NAMES: list[str] = ['cam1']
CV2_CAPTURE_FPS = 120
CV2_CAPTURE_FOURCC = 'MJPG'
CV2_CAPTURE_WINDOW_PREFIX = 'CV2 Native AprilCube'
CV2_CAPTURE_PRINT_EVERY_N_FRAMES = 5
CV2_CAPTURE_TIMING_PRINT_EVERY_N_FRAMES = 30
CV2_CAPTURE_UNDISTORT_BEFORE_DETECTION = True
CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG: float | None = None
CV2_CAPTURE_PINHOLE_UNDISTORT_ALPHA = 0.0
CV2_CAPTURE_RECORD_OUTPUT_DIR = CV2_CAPTURE_THIS_FILE.parent.parent / 'recordings'
CV2_CAPTURE_ADAPTIVE_CLAHE_DETECTION = True
CV2_CAPTURE_CUBE_CFG_DIRS: list[Path] = [CV2_CAPTURE_THIRDPARTY_DIR / 'aprilcube' / 'cubes' / 'cube_april_36h11_6_11_1x1x1_15mm', CV2_CAPTURE_THIRDPARTY_DIR / 'aprilcube' / 'cubes' / 'cube_april_36h11_12_17_1x1x1_15mm']
CV2_CAPTURE_ENABLE_FILTER = True
CV2_CAPTURE_FAST_DETECTOR = True
CV2_CAPTURE_ASSETS_DIR = CV2_CAPTURE_THIS_FILE.parent.parent / 'assets'
CV2_CAPTURE_DRAW_OBJ_OVERLAY = True
CV2_CAPTURE_OBJ_OVERLAY_MAX_EDGES = 2500
CV2_CAPTURE_CUBE_CFG_NAME_TO_OBJ_NAME: dict[str, str] = {'cube_april_36h11_0_5_1x1x1_15mm': 'middle', 'cube_april_36h11_6_11_1x1x1_15mm': 'index', 'cube_april_36h11_12_17_1x1x1_15mm': 'thumb'}
CV2_CAPTURE_OBJ_OVERLAY_COLORS: dict[str, tuple[int, int, int]] = {'index': (0, 165, 255), 'middle': (255, 180, 80), 'thumb': (120, 220, 120)}

@dataclass(frozen=True)
class Cv2ObjectOverlay:
    name: str
    path: Path
    vertices_mm: np.ndarray
    edges: np.ndarray
    color_bgr: tuple[int, int, int]

def cv2_capture_camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    return {'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2])}

def cv2_capture_validate_cube_path(cube_path: Path) -> Path:
    cube_path = cube_path.expanduser().resolve()
    if cube_path.is_dir() and (cube_path / 'config.json').is_file():
        return cube_path
    if cube_path.is_file() and cube_path.name == 'config.json':
        return cube_path
    raise FileNotFoundError(f'Invalid AprilCube cfg path: {cube_path}')

def cv2_capture_resolve_common_image_size(calib_by_camera: dict[str, dict[str, Any]]) -> tuple[int, int]:
    image_sizes = {camera_name: tuple((int(v) for v in calib['image_size'])) for camera_name, calib in calib_by_camera.items()}
    unique_sizes = set(image_sizes.values())
    if len(unique_sizes) != 1:
        raise ValueError(f'CV2CameraManager accepts one capture size for this script, but active cameras use different YAML image_size values: {image_sizes}')
    return next(iter(unique_sizes))

def cv2_capture_is_fisheye_calib(calib: dict[str, Any]) -> bool:
    camera_model = str(calib.get('camera_model', '')).lower()
    distortion_model = str(calib.get('distortion_model', '')).lower()
    return camera_model == 'fisheye' or distortion_model == 'opencv_fisheye'

def cv2_capture_make_centered_pinhole_camera_matrix(image_size: tuple[int, int], horizontal_fov_deg: float) -> np.ndarray:
    width, height = image_size
    half_fov_rad = np.radians(horizontal_fov_deg) / 2.0
    if not 0.0 < half_fov_rad < np.pi / 2.0:
        raise ValueError(f'horizontal_fov_deg must be in (0, 180), got {horizontal_fov_deg}.')
    focal = width / (2.0 * np.tan(half_fov_rad))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

def cv2_capture_horizontal_fov_from_camera_matrix(camera_matrix: np.ndarray, image_size: tuple[int, int]) -> float:
    width, _height = image_size
    fx = float(camera_matrix[0, 0])
    if fx <= 0.0:
        raise ValueError(f'camera_matrix fx must be positive, got {fx}.')
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * fx))))

def cv2_capture_resolved_fisheye_rectified_horizontal_fov_deg(calib: dict[str, Any], image_size: tuple[int, int]) -> float:
    if CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG is not None:
        return float(CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG)
    camera_matrix = np.asarray(calib['K'], dtype=np.float64).reshape(3, 3)
    return cv2_capture_horizontal_fov_from_camera_matrix(camera_matrix, image_size)

def cv2_capture_compute_detection_camera_matrix(calib: dict[str, Any], image_size: tuple[int, int], *, undistort_before_detection: bool) -> np.ndarray:
    camera_matrix = np.asarray(calib['K'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get('dist', np.zeros(5)), dtype=np.float64).reshape(-1)
    if not undistort_before_detection or dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return camera_matrix.copy()
    if cv2_capture_is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f'OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.')
        horizontal_fov_deg = cv2_capture_resolved_fisheye_rectified_horizontal_fov_deg(calib, image_size)
        return cv2_capture_make_centered_pinhole_camera_matrix(image_size, horizontal_fov_deg)
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, CV2_CAPTURE_PINHOLE_UNDISTORT_ALPHA, image_size)
    return np.asarray(new_camera_matrix, dtype=np.float64).reshape(3, 3)

def cv2_capture_create_undistort_maps(calib: dict[str, Any], image_size: tuple[int, int], detection_camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    camera_matrix = np.asarray(calib['K'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get('dist', np.zeros(5)), dtype=np.float64).reshape(-1)
    if dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return None
    detection_camera_matrix = np.asarray(detection_camera_matrix, dtype=np.float64).reshape(3, 3)
    if cv2_capture_is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(f'OpenCV fisheye calibration expects 4 coeffs, got {dist_coeffs.size}.')
        return cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs.reshape(4, 1), np.eye(3, dtype=np.float64), detection_camera_matrix, image_size, cv2.CV_16SC2)
    return cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3, dtype=np.float64), detection_camera_matrix, image_size, cv2.CV_16SC2)

def cv2_capture_create_detector_for_camera(cube_path: Path, camera_name: str, calib_by_camera: dict[str, dict[str, Any]], detection_camera_matrix_by_camera: dict[str, np.ndarray], *, enable_filter: bool, fast: bool, undistort_before_detection: bool) -> Any:
    if camera_name not in calib_by_camera:
        raise KeyError(f"Missing intrinsics for camera '{camera_name}'.")
    calib = calib_by_camera[camera_name]
    detection_camera_matrix = detection_camera_matrix_by_camera[camera_name]
    intrinsic_cfg = cv2_capture_camera_matrix_to_intrinsic_dict(detection_camera_matrix)
    dist_coeffs = calib.get('dist', None)
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)
    detector_dist_coeffs = dist_coeffs
    if undistort_before_detection:
        detector_dist_coeffs = np.zeros(5, dtype=np.float64)
    return aprilcube.detector(cube_path, intrinsic_cfg=intrinsic_cfg, dist_coeffs=detector_dist_coeffs, enable_filter=enable_filter, fast=fast)

def cv2_capture_undistort_frame(frame: np.ndarray, undistort_maps: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if undistort_maps is None:
        return frame
    map1, map2 = undistort_maps
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

def cv2_capture_make_tag_detection_vis_image(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    enhanced = preprocess_tag_image(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def cv2_capture_rotation_matrix_to_euler_xyz_deg(rot_mat: np.ndarray) -> np.ndarray:
    r = np.asarray(rot_mat, dtype=np.float64)
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-06
    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z], dtype=np.float64))

def cv2_capture_result_to_text(camera_name: str, cube_name: str, result: dict[str, Any] | None) -> str:
    prefix = f'[{camera_name}][{cube_name}]'
    if not result:
        return f'{prefix} no result'
    if not result.get('success', False):
        n_tags = int(result.get('n_tags', 0))
        reason = str(result.get('failure_reason', 'unknown'))
        tag_ids = result.get('tag_ids', [])
        faces = result.get('visible_faces', None)
        text = f'{prefix} cube not detected tags={n_tags}'
        if tag_ids:
            text += f' ids={list(tag_ids)}'
        if faces:
            text += f' faces={sorted(list(faces))}'
        text += f' reason={reason}'
        if 'tag_corner_rotation_fallback_reject' in result:
            text += f" rot_try={result.get('tag_corner_rotation_fallback_reject')} best_reproj={float(result.get('tag_corner_rotation_fallback_best_reproj', float('inf'))):.2f}"
        per_tag_err = result.get('per_tag_reproj_error', None)
        if per_tag_err:
            compact_err = {int(tag_id): round(float(err), 1) for tag_id, err in dict(per_tag_err).items()}
            text += f' tag_err={compact_err}'
        return text
    tvec = np.asarray(result['tvec'], dtype=np.float64).reshape(-1)
    text = f'{prefix} t=({tvec[0]:.1f},{tvec[1]:.1f},{tvec[2]:.1f})'
    if result.get('rvec', None) is not None:
        rot_mat, _ = cv2.Rodrigues(np.asarray(result['rvec'], dtype=np.float64).reshape(3, 1))
        euler = cv2_capture_rotation_matrix_to_euler_xyz_deg(rot_mat)
        text += f' rot=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})'
    error = result.get('reproj_error', None)
    if error is not None:
        text += f' reproj={float(error):.2f}px'
    text += f" tags={int(result.get('n_tags', 0))}"
    faces = result.get('visible_faces', None)
    if faces is not None:
        text += f' faces={sorted(list(faces))}'
    if result.get('predicted', False):
        text += ' predicted'
    if result.get('single_tag_cfg_pose', False):
        rot_deg = int(result.get('single_tag_corner_rotation_deg', 0))
        text += f" single_tag_cfg_pose(id={result.get('single_tag_id', '?')},face={result.get('single_tag_face', '?')},rot={rot_deg})"
    if result.get('tag_corner_rotation_fallback', False):
        text += f" corner_rot={result.get('tag_corner_rotations_deg', {})}"
    if result.get('face_assignment_fallback', False):
        text += f" face_assign={result.get('tag_face_assignment', {})}"
    per_tag_err = result.get('per_tag_reproj_error', None)
    if per_tag_err:
        compact_err = {int(tag_id): round(float(err), 1) for tag_id, err in dict(per_tag_err).items()}
        text += f' tag_err={compact_err}'
    return text

def cv2_capture_draw_text_panel(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        y += 24
    return out

def cv2_capture_cube_cfg_name_from_path(cube_path: Path) -> str:
    return cube_path.name if cube_path.is_dir() else cube_path.parent.name

def cv2_capture_load_obj_overlay(obj_name: str, *, max_edges: int=CV2_CAPTURE_OBJ_OVERLAY_MAX_EDGES) -> Cv2ObjectOverlay:
    obj_path = CV2_CAPTURE_ASSETS_DIR / f'{obj_name}.obj'
    if not obj_path.is_file():
        raise FileNotFoundError(f'OBJ overlay file not found: {obj_path}')
    loaded = trimesh.load(obj_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    edges = np.asarray(mesh.edges_unique, dtype=np.int32)
    if edges.size == 0 and len(getattr(mesh, 'faces', [])) > 0:
        faces = np.asarray(mesh.faces, dtype=np.int32)
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
    if max_edges > 0 and len(edges) > max_edges:
        keep = np.linspace(0, len(edges) - 1, max_edges, dtype=np.int64)
        edges = edges[keep]
    used_vertex_ids = np.unique(edges.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used_vertex_ids] = np.arange(len(used_vertex_ids), dtype=np.int32)
    vertices = vertices[used_vertex_ids]
    edges = remap[edges]
    return Cv2ObjectOverlay(name=obj_name, path=obj_path, vertices_mm=vertices, edges=edges, color_bgr=CV2_CAPTURE_OBJ_OVERLAY_COLORS.get(obj_name, (180, 180, 180)))

def cv2_capture_draw_obj_overlay(image: np.ndarray, result: dict[str, Any], detector: Any, overlay: Cv2ObjectOverlay | None) -> np.ndarray:
    if overlay is None or not result.get('success', False):
        return image
    if result.get('rvec', None) is None or result.get('tvec', None) is None:
        return image
    rvec = np.asarray(result['rvec'], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(result['tvec'], dtype=np.float64).reshape(3, 1)
    vertices = np.asarray(overlay.vertices_mm, dtype=np.float64).reshape(-1, 3)
    rot_mat, _ = cv2.Rodrigues(rvec)
    vertices_cam = vertices @ rot_mat.T + tvec.reshape(1, 3)
    projected, _ = cv2.projectPoints(vertices, rvec, tvec, detector.camera_matrix, detector.dist_coeffs)
    pts = projected.reshape(-1, 2)
    h, w = image.shape[:2]
    margin = 200
    for i, j in overlay.edges:
        if vertices_cam[i, 2] <= 1.0 or vertices_cam[j, 2] <= 1.0:
            continue
        p0 = pts[i]
        p1 = pts[j]
        if max(p0[0], p1[0]) < -margin or min(p0[0], p1[0]) > w + margin or max(p0[1], p1[1]) < -margin or (min(p0[1], p1[1]) > h + margin):
            continue
        cv2.line(image, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), overlay.color_bgr, 1, cv2.LINE_AA)
    return image

def cv2_capture_count_adaptive_new_tag_ids(shared_tags: dict[str, Any]) -> int:
    attempts = shared_tags.get('adaptive_attempts', [])
    new_ids: set[int] = set()
    for attempt in attempts:
        if attempt.get('base', False):
            continue
        for tag_id in attempt.get('new_ids', []):
            new_ids.add(int(tag_id))
    return len(new_ids)

class Cv2RawFrameRecorder:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.path: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._frames: list[dict[str, Any]] = []
        self.started_wall_time: str | None = None
        self.started_monotonic: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._metadata is not None

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def buffered_bytes(self) -> int:
        return int(sum((frame['image_bgr'].nbytes for frame in self._frames)))

    def start(self, metadata: dict[str, Any]) -> None:
        if self.is_recording:
            print(f'[INFO] Recording already active: {self.path}')
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.path = self.output_dir / f'008_raw_frames_{stamp}.pkl'
        self.started_wall_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.started_monotonic = time.perf_counter()
        self._frames = []
        self._metadata = dict(metadata)
        print(f'[INFO] Started raw-frame memory buffering: {self.path}')

    def write(self, *, camera_name: str, loop_frame_idx: int, image_bgr: np.ndarray | None, capture_timestamp: float | None) -> None:
        if not self.is_recording or image_bgr is None:
            return
        self._frames.append({'type': 'frame', 'camera_name': camera_name, 'loop_frame_idx': int(loop_frame_idx), 'capture_timestamp': None if capture_timestamp is None else float(capture_timestamp), 'write_monotonic': float(time.perf_counter()), 'shape': tuple((int(v) for v in image_bgr.shape)), 'dtype': str(image_bgr.dtype), 'image_bgr': image_bgr})

    def _print_save_progress(self, done: int, total: int) -> None:
        width = 36
        ratio = 1.0 if total <= 0 else done / total
        filled = int(round(width * ratio))
        bar = '#' * filled + '-' * (width - filled)
        sys.stdout.write(f'\r[INFO] Saving PKL [{bar}] {done}/{total} frames')
        sys.stdout.flush()
        if done >= total:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def stop(self, reason: str='user_stop') -> None:
        if not self.is_recording:
            print('[INFO] Recording is not active.')
            return
        path = self.path
        assert path is not None
        assert self._metadata is not None
        total_frames = self.frame_count
        buffered_gb = self.buffered_bytes / 1024 ** 3
        elapsed = time.perf_counter() - self.started_monotonic if self.started_monotonic is not None else 0.0
        print(f'[INFO] Stopped raw-frame buffering: frames={total_frames} buffered={buffered_gb:.2f} GiB duration={elapsed:.2f}s')
        print(f'[INFO] Writing PKL: {path}')
        with path.open('wb') as f:
            pickle.dump({'type': 'header', 'format': 'aprilcube_raw_frame_stream_v1', 'created_wall_time': self.started_wall_time, 'metadata': self._metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._print_save_progress(0, total_frames)
            for idx, frame_record in enumerate(self._frames, start=1):
                pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
                if idx == total_frames or idx % 10 == 0:
                    self._print_save_progress(idx, total_frames)
            pickle.dump({'type': 'footer', 'reason': reason, 'frame_count': int(total_frames), 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._frames = []
        self._metadata = None
        self.started_monotonic = None
        print(f'[INFO] Saved raw-frame PKL recording: {path} frames={total_frames}')

def cv2_capture_main() -> None:
    parser = argparse.ArgumentParser(description='Detect multiple AprilCube cfgs using one shared AprilTag detection pass per CV2 frame.')
    parser.add_argument('--cameras', type=str, default=','.join(CV2_CAPTURE_ACTIVE_CAMERA_NAMES), help='Comma-separated logical camera names.')
    parser.add_argument('--cube-dirs', type=str, default=','.join((str(path) for path in CV2_CAPTURE_CUBE_CFG_DIRS)), help='Comma-separated AprilCube cfg directories or config.json files.')
    parser.add_argument('--slow', action='store_true', help='Use native AprilCube slow/high-accuracy detector parameters.')
    parser.add_argument('--no-filter', action='store_true', help='Disable native AprilCube temporal pose filter.')
    parser.add_argument('--no-undistort', action='store_true', help='Do not undistort images before native AprilCube detection.')
    parser.add_argument('--record-dir', type=str, default=str(CV2_CAPTURE_RECORD_OUTPUT_DIR), help='Directory for raw-frame PKL recordings triggered by s/p.')
    args = parser.parse_args()
    active_camera_names = [x.strip() for x in args.cameras.split(',') if x.strip()]
    cube_paths = [cv2_capture_validate_cube_path(Path(x.strip())) for x in args.cube_dirs.split(',') if x.strip()]
    if not active_camera_names:
        print('[ERROR] No active camera names specified.')
        sys.exit(1)
    if not cube_paths:
        print('[ERROR] No cube cfg paths specified.')
        sys.exit(1)
    missing_camera_cfg = [name for name in active_camera_names if name not in CV2_CAPTURE_CAMERA_TO_PORT]
    if missing_camera_cfg:
        print(f'[ERROR] Missing CAMERA_TO_PORT entries for: {missing_camera_cfg}')
        sys.exit(1)
    missing_intrinsics_cfg = [name for name in active_camera_names if name not in CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML]
    if missing_intrinsics_cfg:
        print(f'[ERROR] Missing CAMERA_TO_INTRINSICS_YAML entries for: {missing_intrinsics_cfg}')
        sys.exit(1)
    calib_by_camera = {name: cv2_capture_load_intrinsics_yaml(CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[name]) for name in active_camera_names}
    image_size = cv2_capture_resolve_common_image_size(calib_by_camera)
    capture_size = image_size
    detect_img_size = image_size
    vis_img_size = (max(1, detect_img_size[0] // 2), max(1, detect_img_size[1] // 2))
    use_undistort = CV2_CAPTURE_UNDISTORT_BEFORE_DETECTION and (not args.no_undistort)
    detection_camera_matrix_by_camera = {camera_name: cv2_capture_compute_detection_camera_matrix(calib, detect_img_size, undistort_before_detection=use_undistort) for camera_name, calib in calib_by_camera.items()}
    undistort_maps_by_camera = {camera_name: cv2_capture_create_undistort_maps(calib, detect_img_size, detection_camera_matrix_by_camera[camera_name]) if use_undistort else None for camera_name, calib in calib_by_camera.items()}
    for camera_name in active_camera_names:
        calib = calib_by_camera[camera_name]
        raw_k = np.asarray(calib['K'], dtype=np.float64).reshape(3, 3)
        detect_k = detection_camera_matrix_by_camera[camera_name]
        print(f"[INFO] [{camera_name}] intrinsics_yaml={calib['path']} image_size={calib['image_size']} camera_model={calib['camera_model'] or 'unknown'} distortion_model={calib['distortion_model'] or 'unknown'} undistort={use_undistort}")
        print(f'[INFO] [{camera_name}] raw_K=fx={raw_k[0, 0]:.3f} fy={raw_k[1, 1]:.3f} cx={raw_k[0, 2]:.3f} cy={raw_k[1, 2]:.3f}')
        print(f'[INFO] [{camera_name}] detection_K=fx={detect_k[0, 0]:.3f} fy={detect_k[1, 1]:.3f} cx={detect_k[0, 2]:.3f} cy={detect_k[1, 2]:.3f}')
        if use_undistort and cv2_capture_is_fisheye_calib(calib):
            hfov_source = 'yaml_fx' if CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG is None else 'FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG'
            hfov_deg = cv2_capture_resolved_fisheye_rectified_horizontal_fov_deg(calib, detect_img_size)
            print(f'[INFO] [{camera_name}] fisheye_rectified_hfov={hfov_deg:.3f}deg source={hfov_source}')
    print(f'[INFO] capture_size={capture_size} detect_img_size={detect_img_size} vis_img_size={vis_img_size}')
    detector_entries_by_camera: dict[str, list[dict[str, Any]]] = {name: [] for name in active_camera_names}
    obj_overlay_by_name: dict[str, Cv2ObjectOverlay] = {}
    if CV2_CAPTURE_DRAW_OBJ_OVERLAY:
        for obj_name in sorted(set(CV2_CAPTURE_CUBE_CFG_NAME_TO_OBJ_NAME.values())):
            try:
                obj_overlay_by_name[obj_name] = cv2_capture_load_obj_overlay(obj_name)
                overlay = obj_overlay_by_name[obj_name]
                print(f'[INFO] Loaded OBJ overlay: {obj_name} path={overlay.path} vertices={len(overlay.vertices_mm)} edges={len(overlay.edges)}')
            except Exception as exc:
                print(f"[WARNING] Failed to load OBJ overlay '{obj_name}': {type(exc).__name__}: {exc}")
    for cube_path in cube_paths:
        cube_name = cv2_capture_cube_cfg_name_from_path(cube_path)
        obj_name = CV2_CAPTURE_CUBE_CFG_NAME_TO_OBJ_NAME.get(cube_name, '')
        obj_overlay = obj_overlay_by_name.get(obj_name)
        if CV2_CAPTURE_DRAW_OBJ_OVERLAY:
            if obj_overlay is None:
                print(f'[INFO] Cube cfg has no OBJ overlay: {cube_name}')
            else:
                print(f'[INFO] Cube cfg -> OBJ overlay: {cube_name} -> {obj_name}')
        for camera_name in active_camera_names:
            detector = cv2_capture_create_detector_for_camera(cube_path, camera_name, calib_by_camera, detection_camera_matrix_by_camera, enable_filter=not args.no_filter, fast=not args.slow, undistort_before_detection=use_undistort)
            detector_entries_by_camera[camera_name].append({'cube_name': cube_name, 'obj_name': obj_name, 'obj_overlay': obj_overlay, 'detector': detector})
            print(f'[INFO] Loaded native AprilCube detector for {camera_name}: {cube_name}')
    camera_manager = CV2CameraManager(camera_to_port={name: CV2_CAPTURE_CAMERA_TO_PORT[name] for name in active_camera_names}, capture_size=capture_size, fps=CV2_CAPTURE_FPS, fourcc=CV2_CAPTURE_FOURCC)
    recorder = Cv2RawFrameRecorder(Path(args.record_dir))
    try:
        opened = camera_manager.open_all_cameras()
        if opened == 0:
            print('[ERROR] No CV2 camera opened.')
            sys.exit(1)
        opened_names = camera_manager.get_active_camera_names()
        print(f'[INFO] Opened CV2 cameras: {opened_names}')
        print('[INFO] Native detection path: shared detect_tags(frame) + per-cube process_detections().')
        print(f'[INFO] Adaptive CLAHE tag recovery: {CV2_CAPTURE_ADAPTIVE_CLAHE_DETECTION}')
        print("[INFO] Press 's' to start raw-frame PKL recording, 'p' to stop, 'q' or ESC to quit.")
        recording_metadata = {'script': str(CV2_CAPTURE_THIS_FILE), 'recorded_image': 'origin_frame_raw_bgr', 'camera_to_port': {name: CV2_CAPTURE_CAMERA_TO_PORT[name] for name in active_camera_names}, 'intrinsics_yaml': {name: str(Path(CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[name]).expanduser().resolve()) for name in active_camera_names}, 'opened_cameras': list(opened_names), 'capture_size': tuple((int(v) for v in capture_size)), 'detect_img_size': tuple((int(v) for v in detect_img_size)), 'fps': int(CV2_CAPTURE_FPS), 'fourcc': str(CV2_CAPTURE_FOURCC), 'undistort_before_detection': bool(use_undistort), 'fisheye_rectified_horizontal_fov_deg_setting': None if CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG is None else float(CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG), 'fisheye_rectified_horizontal_fov_deg_by_camera': {name: cv2_capture_resolved_fisheye_rectified_horizontal_fov_deg(calib_by_camera[name], detect_img_size) for name in active_camera_names if use_undistort and cv2_capture_is_fisheye_calib(calib_by_camera[name])}, 'cube_paths': [str(Path(path).expanduser().resolve()) for path in cube_paths]}

        def handle_key(key: int) -> bool:
            if key == 27 or key == ord('q'):
                return False
            if key == ord('s'):
                recorder.start(recording_metadata)
            elif key == ord('p'):
                recorder.stop('user_stop')
            return True
        frame_idx = 0
        last_no_frame_print_time = 0.0
        while True:
            loop_t0 = time.perf_counter()
            frame_idx += 1
            frames, _origin_frames, _timestamps = camera_manager.get_frames(camera_names=opened_names, img_size=detect_img_size)
            if not frames:
                now = time.time()
                if now - last_no_frame_print_time > 1.0:
                    print('[INFO] No frames received yet.')
                    last_no_frame_print_time = now
                key = cv2.waitKey(1)
                if not handle_key(key):
                    break
                continue
            for camera_name, frame in frames.items():
                camera_t0 = time.perf_counter()
                origin_frame = _origin_frames.get(camera_name)
                capture_ts = _timestamps.get(camera_name)
                capture_age_ms = (camera_t0 - capture_ts) * 1000.0 if capture_ts is not None else None
                recorder.write(camera_name=camera_name, loop_frame_idx=frame_idx, image_bgr=origin_frame, capture_timestamp=capture_ts)
                if origin_frame is not None and frame_idx % CV2_CAPTURE_PRINT_EVERY_N_FRAMES == 0:
                    origin_h, origin_w = origin_frame.shape[:2]
                    detect_h, detect_w = frame.shape[:2]
                    print(f'[{camera_name}] origin_size=({origin_w}, {origin_h}) detect_frame_size=({detect_w}, {detect_h})')
                detector_entries = detector_entries_by_camera[camera_name]
                detect_frame = frame
                undistort_ms = 0.0
                if use_undistort:
                    undistort_t0 = time.perf_counter()
                    detect_frame = cv2_capture_undistort_frame(frame, undistort_maps_by_camera[camera_name])
                    undistort_ms = (time.perf_counter() - undistort_t0) * 1000.0
                fps_text = camera_manager.get_latest_fps(camera_name)
                shared_timestamp = time.monotonic()
                detect_t0 = time.perf_counter()
                shared_tags = detector_entries[0]['detector'].detect_tags(detect_frame, adaptive_clahe=CV2_CAPTURE_ADAPTIVE_CLAHE_DETECTION)
                detect_ms = (time.perf_counter() - detect_t0) * 1000.0
                vis = cv2.cvtColor(shared_tags['enhanced'], cv2.COLOR_GRAY2BGR)
                adaptive_new_tags = cv2_capture_count_adaptive_new_tag_ids(shared_tags)
                status_lines = [f'[{camera_name}] native_aprilcube cubes={len(detector_entries)} detect_size={detect_img_size} vis_size={vis_img_size} capture_size={capture_size} fps={fps_text:.1f}' if fps_text is not None else f'[{camera_name}] native_aprilcube cubes={len(detector_entries)} detect_size={detect_img_size} vis_size={vis_img_size} capture_size={capture_size}']
                status_lines.append(f"tags_decoded={len(shared_tags['detections'])} adaptive_clahe={CV2_CAPTURE_ADAPTIVE_CLAHE_DETECTION} clahe_extra_tags={adaptive_new_tags}")
                if recorder.is_recording:
                    buffered_gb = recorder.buffered_bytes / 1024 ** 3
                    status_lines.append(f'REC buffering frames={recorder.frame_count} mem={buffered_gb:.2f}GiB')
                else:
                    status_lines.append('REC off: press s to start, p to stop')
                process_draw_t0 = time.perf_counter()
                for entry in detector_entries:
                    cube_name = entry['cube_name']
                    obj_overlay = entry.get('obj_overlay', None)
                    detector = entry['detector']
                    result = detector.process_detections(detect_frame, shared_tags['detections'], rejected_quads=shared_tags['rejected'], gray=shared_tags['gray'], enhanced=shared_tags['enhanced'], timestamp=shared_timestamp)
                    try:
                        vis = detector.draw_result(vis, result)
                        vis = cv2_capture_draw_obj_overlay(vis, result, detector, obj_overlay)
                    except Exception as exc:
                        print(f'[WARNING] draw_result failed for {camera_name}/{cube_name}: {type(exc).__name__}: {exc}')
                    line = cv2_capture_result_to_text(camera_name, cube_name, result)
                    status_lines.append(line)
                    if frame_idx % CV2_CAPTURE_PRINT_EVERY_N_FRAMES == 0:
                        print(line)
                process_draw_ms = (time.perf_counter() - process_draw_t0) * 1000.0
                visualize_t0 = time.perf_counter()
                status_lines.append('press s start rec, p stop rec, q or ESC quit')
                vis = cv2_capture_draw_text_panel(vis, status_lines)
                vis = cv2.resize(vis, vis_img_size, interpolation=cv2.INTER_AREA)
                cv2.imshow(f'{CV2_CAPTURE_WINDOW_PREFIX}: {camera_name}', vis)
                visualize_ms = (time.perf_counter() - visualize_t0) * 1000.0
                if frame_idx % CV2_CAPTURE_TIMING_PRINT_EVERY_N_FRAMES == 0:
                    total_ms = (time.perf_counter() - camera_t0) * 1000.0
                    loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                    capture_age_text = f'{capture_age_ms:.1f}ms' if capture_age_ms is not None else 'unknown'
                    print(f'[TIMING] [{camera_name}] capture_age={capture_age_text} undistort={undistort_ms:.1f}ms detect_tags={detect_ms:.1f}ms process_draw={process_draw_ms:.1f}ms visualize={visualize_ms:.1f}ms camera_total={total_ms:.1f}ms loop_total={loop_ms:.1f}ms')
            key = cv2.waitKey(1)
            if not handle_key(key):
                break
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted by user.')
    finally:
        if recorder.is_recording:
            recorder.stop('shutdown')
        camera_manager.release_all()
        cv2.destroyAllWindows()


# ---- 008 recording replay and visualization ----
REPLAY_008_THIS_FILE = Path(__file__).resolve()
REPLAY_008_DEFAULT_RECORDING_DIR = REPLAY_008_THIS_FILE.parent.parent / 'recordings'
REPLAY_008_VISER_HOST = '0.0.0.0'
REPLAY_008_VISER_PORT = 8091
REPLAY_008_ASSETS_DIR = REPLAY_008_THIS_FILE.parent.parent / 'assets'
REPLAY_008_OBJ_MESH_SCALE = 0.001
REPLAY_008_POSE_CACHE_FORMAT = 'aprilcube_008_pose_cache_v1'
REPLAY_008_POSE_CACHE_FORMAT_020_MULTISTAGE = 'aprilcube_020_multistage_008_pose_v1'
REPLAY_008_INLINE_POSE_FRAME_FIELD = 'offline_pose_frame'
REPLAY_008_INLINE_POSE_CACHE_KEY_FIELD = 'offline_pose_cache_key'
REPLAY_008_IMAGE_RECOVERY_VERSION = 9
REPLAY_008_SINGLE_TAG_CONTINUITY_GATE_ENABLED = True
REPLAY_008_SINGLE_TAG_CONTINUITY_MAX_ROTATION_DEG = 45.0
REPLAY_008_SINGLE_TAG_CONTINUITY_MIN_FACE_OBSERVATIONS = 2
REPLAY_008_SINGLE_TAG_CONTINUITY_MAX_OBSERVATION_GAP = 8
REPLAY_008_SINGLE_TAG_CONTINUITY_VERSION = 2
REPLAY_008_TEMPORAL_OUTLIER_GATE_ENABLED = True
REPLAY_008_TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES = 6
REPLAY_008_TEMPORAL_OUTLIER_NEIGHBOR_MAX_ROTATION_DEG = 35.0
REPLAY_008_TEMPORAL_OUTLIER_NEIGHBOR_MAX_TRANSLATION_MM = 35.0
REPLAY_008_TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG = 90.0
REPLAY_008_TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM = 70.0
REPLAY_008_TEMPORAL_OUTLIER_VERSION = 1
REPLAY_008_TEMPORAL_FILL_MAX_GAP_FRAMES = 30
REPLAY_008_TEMPORAL_FILL_MAX_ROTATION_DEG = 45.0
REPLAY_008_TEMPORAL_FILL_VERSION = 5
REPLAY_008_TEMPORAL_SMOOTHING_ENABLED = True
REPLAY_008_TEMPORAL_SMOOTHING_WINDOW_RADIUS = 2
REPLAY_008_TEMPORAL_SMOOTHING_SIGMA_FRAMES = 1.2
REPLAY_008_TEMPORAL_SMOOTHING_MAX_ROTATION_DEG = 15.0
REPLAY_008_TEMPORAL_SMOOTHING_MAX_DISPLAY_REPROJ_PX = 12.0
REPLAY_008_TEMPORAL_SMOOTHING_MAX_REPROJ_RATIO = 2.5
REPLAY_008_TEMPORAL_SMOOTHING_VERSION = 5
REPLAY_008_TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED = True
REPLAY_008_TEMPORAL_ROTATION_JUMP_MAX_DEG = 20.0
REPLAY_008_TEMPORAL_ROTATION_JUMP_HOLD_DEG = 60.0
REPLAY_008_TEMPORAL_ROTATION_JUMP_LIMIT_VERSION = 2

def replay_008_install_numpy_pickle_compat() -> None:
    """Allow NumPy 2.x pickles to load in NumPy 1.x environments."""
    try:
        numpy_core = importlib.import_module('numpy.core')
    except Exception:
        return
    sys.modules.setdefault('numpy._core', numpy_core)
    for module_name in ('multiarray', 'numeric', 'numerictypes', 'overrides', 'fromnumeric', 'shape_base', 'umath', '_multiarray_umath'):
        try:
            module = importlib.import_module(f'numpy.core.{module_name}')
        except Exception:
            continue
        sys.modules.setdefault(f'numpy._core.{module_name}', module)
replay_008_install_numpy_pickle_compat()

def replay_008_resolve_pkl_path(path_str: str | None) -> Path:
    if path_str is None:
        candidates = sorted(REPLAY_008_DEFAULT_RECORDING_DIR.glob('008_raw_frames_*.pkl'))
        if not candidates:
            raise FileNotFoundError(f'No 008_raw_frames_*.pkl found in {REPLAY_008_DEFAULT_RECORDING_DIR}')
        return candidates[-1].resolve()
    path = Path(path_str).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob('008_raw_frames_*.pkl'))
        if not candidates:
            raise FileNotFoundError(f'No 008_raw_frames_*.pkl found in {path}')
        return candidates[-1].resolve()
    if not path.is_file():
        raise FileNotFoundError(f'PKL file does not exist: {path}')
    return path

def replay_008_print_index_progress(done_bytes: int, total_bytes: int, *, force_newline: bool=False) -> None:
    width = 36
    ratio = 1.0 if total_bytes <= 0 else min(max(done_bytes / total_bytes, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(f'\r[INFO] Indexing PKL [{bar}] {done_bytes / 1024 ** 2:.1f}/{total_bytes / 1024 ** 2:.1f} MiB')
    sys.stdout.flush()
    if force_newline:
        sys.stdout.write('\n')
        sys.stdout.flush()

def replay_008_build_frame_index(path: Path) -> tuple[dict[str, Any] | None, list[int], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    pose_cache_record: dict[str, Any] | None = None
    inline_pose_cache_key: dict[str, Any] | None = None
    inline_pose_cache: list[dict[str, Any] | None] = []
    inline_pose_cache_complete = True
    inline_pose_cache_keys_match = True
    frame_offsets: list[int] = []
    file_size = path.stat().st_size
    last_print = time.monotonic()
    with path.open('rb') as f:
        while True:
            offset = f.tell()
            try:
                record = pickle.load(f)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            record_type = record.get('type', None)
            if record_type == 'header':
                header = record
            elif record_type == 'frame':
                frame_offsets.append(offset)
                inline_pose_frame = record.get(REPLAY_008_INLINE_POSE_FRAME_FIELD, None)
                inline_key = record.get(REPLAY_008_INLINE_POSE_CACHE_KEY_FIELD, None)
                if isinstance(inline_pose_frame, dict) and isinstance(inline_key, dict):
                    inline_pose_cache.append(inline_pose_frame)
                    if inline_pose_cache_key is None:
                        inline_pose_cache_key = inline_key
                    elif inline_pose_cache_key != inline_key:
                        inline_pose_cache_keys_match = False
                else:
                    inline_pose_cache.append(None)
                    inline_pose_cache_complete = False
            elif record_type == 'footer':
                footer = record
            elif record_type == 'pose_cache':
                pose_cache_record = record
            now = time.monotonic()
            if now - last_print > 0.5:
                replay_008_print_index_progress(f.tell(), file_size)
                last_print = now
    replay_008_print_index_progress(file_size, file_size, force_newline=True)
    inline_pose_cache_record = None
    if inline_pose_cache_complete and inline_pose_cache_keys_match and (inline_pose_cache_key is not None) and (len(inline_pose_cache) == len(frame_offsets)):
        inline_pose_cache_record = {'type': 'pose_cache', 'format': REPLAY_008_POSE_CACHE_FORMAT, 'key': inline_pose_cache_key, 'pose_cache': inline_pose_cache}
    return (header, frame_offsets, footer, pose_cache_record, inline_pose_cache_record)

def replay_008_load_frame_at_offset(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(offset)
        record = pickle.load(f)
    if not isinstance(record, dict) or record.get('type') != 'frame':
        raise ValueError(f'Offset {offset} does not point to a frame record.')
    image = record.get('image_bgr', None)
    if not isinstance(image, np.ndarray):
        raise ValueError(f'Frame at offset {offset} has no ndarray image_bgr.')
    return record

def replay_008_resize_for_display(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / max(w, 1)
    target_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    pil_image = Image.fromarray(image)
    return np.asarray(pil_image.resize(target_size, Image.Resampling.BILINEAR))

def replay_008_bgr_to_rgb_for_viser(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    image = replay_008_resize_for_display(image_bgr, max_width)
    return image[..., ::-1]

def replay_008_record_summary(record: dict[str, Any], frame_idx: int, total_frames: int) -> str:
    camera_name = record.get('camera_name', 'unknown')
    loop_idx = record.get('loop_frame_idx', 'unknown')
    capture_ts = record.get('capture_timestamp', None)
    shape = record.get('shape', None)
    dtype = record.get('dtype', None)
    return f'frame {frame_idx + 1}/{total_frames} | camera={camera_name} | loop_idx={loop_idx} | shape={shape} | dtype={dtype} | capture_ts={capture_ts}'

def replay_008_print_pose_progress(done: int, total: int, *, force_newline: bool=False) -> None:
    width = 36
    ratio = 1.0 if total <= 0 else min(max(done / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(f'\r[INFO] Estimating poses [{bar}] {done}/{total} frames')
    sys.stdout.flush()
    if force_newline:
        sys.stdout.write('\n')
        sys.stdout.flush()

def replay_008_result_copy_for_replay(result: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key in ('success', 'rvec', 'tvec', 'T', 'reproj_error', 'n_tags', 'n_inliers', 'detections', 'tag_ids', 'visible_faces', 'predicted', 'direct_all_point_pnp', 'single_tag_cfg_pose', 'single_tag_id', 'single_tag_face', 'single_tag_candidate_count', 'temporal_filled', 'temporal_fill_source', 'temporal_fill_alpha', 'temporal_smoothed', 'temporal_smoothing_source_count'):
        value = result.get(key, None)
        if key == 'detections':
            copied[key] = [(int(tag_id), np.asarray(corners, dtype=np.float64).copy()) for tag_id, corners in value or []]
        elif key == 'visible_faces':
            copied[key] = set(value or [])
        elif isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied

def replay_008_clone_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    return None if value is None else value.copy()

def replay_008_snapshot_detector_tracking_state(detector: Any) -> dict[str, Any]:
    return {'prev_rvec': replay_008_clone_optional_array(detector.prev_rvec), 'prev_tvec': replay_008_clone_optional_array(detector.prev_tvec), 'pose_filter': copy.deepcopy(detector.pose_filter), '_prev_gray': replay_008_clone_optional_array(detector._prev_gray), '_prev_corners_2d': replay_008_clone_optional_array(detector._prev_corners_2d), '_prev_corners_3d': replay_008_clone_optional_array(detector._prev_corners_3d)}

def replay_008_restore_detector_tracking_state(detector: Any, state: dict[str, Any]) -> None:
    detector.prev_rvec = replay_008_clone_optional_array(state['prev_rvec'])
    detector.prev_tvec = replay_008_clone_optional_array(state['prev_tvec'])
    detector.pose_filter = copy.deepcopy(state['pose_filter'])
    detector._prev_gray = replay_008_clone_optional_array(state['_prev_gray'])
    detector._prev_corners_2d = replay_008_clone_optional_array(state['_prev_corners_2d'])
    detector._prev_corners_3d = replay_008_clone_optional_array(state['_prev_corners_3d'])

def replay_008_is_measured_pose(result: dict[str, Any]) -> bool:
    return bool(result.get('success', False)) and (not bool(result.get('predicted', False)))

def replay_008_rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = r / angle
    half = angle * 0.5
    return np.array([np.cos(half), *np.sin(half) * axis], dtype=np.float64)

def replay_008_normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return q / max(float(np.linalg.norm(q)), 1e-12)

def replay_008_align_quat_to_reference(quat: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = replay_008_normalize_quat(quat)
    ref = replay_008_normalize_quat(reference)
    if float(np.dot(ref, q)) < 0.0:
        return -q
    return q

def replay_008_quat_short_arc_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0n = replay_008_normalize_quat(q0)
    q1n = replay_008_align_quat_to_reference(q1, q0n)
    dot = abs(float(np.dot(q0n, q1n)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))

def replay_008_quat_to_rvec(quat: np.ndarray) -> np.ndarray:
    q = replay_008_normalize_quat(quat)
    if q[0] < 0:
        q = -q
    sin_half = float(np.linalg.norm(q[1:]))
    if sin_half < 1e-12:
        return np.zeros((3, 1), dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    axis = q[1:] / sin_half
    return (angle * axis).reshape(3, 1)

def replay_008_slerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = replay_008_normalize_quat(q0)
    q1 = replay_008_normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

def replay_008_limit_quat_rotation(source: np.ndarray, target: np.ndarray, max_rotation_deg: float) -> tuple[np.ndarray, float, bool]:
    source_q = replay_008_normalize_quat(source)
    target_q = replay_008_align_quat_to_reference(target, source_q)
    angle_deg = replay_008_quat_short_arc_angle_deg(source_q, target_q)
    if angle_deg <= max_rotation_deg:
        return (target_q, angle_deg, False)
    alpha = max(float(max_rotation_deg), 0.0) / max(angle_deg, 1e-12)
    return (replay_008_normalize_quat(replay_008_slerp_quat(source_q, target_q, alpha)), angle_deg, True)

def replay_008_pose_transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform

class Replay008PoseEstimator:

    def __init__(self, demo008: Any, *, active_camera_names: list[str], cube_paths: list[Path], use_undistort: bool, adaptive_clahe: bool, shared_tag_detection: bool, enable_filter: bool, fast: bool) -> None:
        _unused_demo008 = demo008
        self.active_camera_names = active_camera_names
        self.cube_paths = cube_paths
        self.use_undistort = use_undistort
        self.adaptive_clahe = adaptive_clahe
        self.shared_tag_detection = shared_tag_detection
        self.calib_by_camera = {name: cv2_capture_load_intrinsics_yaml(CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[name]) for name in active_camera_names}
        self.image_size = cv2_capture_resolve_common_image_size(self.calib_by_camera)
        self.detect_img_size = self.image_size
        self.detection_camera_matrix_by_camera = {camera_name: cv2_capture_compute_detection_camera_matrix(calib, self.detect_img_size, undistort_before_detection=use_undistort) for camera_name, calib in self.calib_by_camera.items()}
        self.undistort_maps_by_camera = {camera_name: cv2_capture_create_undistort_maps(calib, self.detect_img_size, self.detection_camera_matrix_by_camera[camera_name]) if use_undistort else None for camera_name, calib in self.calib_by_camera.items()}
        self.detector_entries_by_camera: dict[str, list[dict[str, Any]]] = {name: [] for name in active_camera_names}
        self.detector_by_camera_cube: dict[tuple[str, str], Any] = {}
        for cube_path in cube_paths:
            cube_name = cube_path.name if cube_path.is_dir() else cube_path.parent.name
            for camera_name in active_camera_names:
                detector = cv2_capture_create_detector_for_camera(cube_path, camera_name, self.calib_by_camera, self.detection_camera_matrix_by_camera, enable_filter=enable_filter, fast=fast, undistort_before_detection=use_undistort)
                self.detector_entries_by_camera[camera_name].append({'cube_name': cube_name, 'detector': detector})
                self.detector_by_camera_cube[camera_name, cube_name] = detector

    def prepare_detect_frame(self, image_bgr: np.ndarray, camera_name: str) -> np.ndarray:
        frame = image_bgr
        h, w = frame.shape[:2]
        if (w, h) != self.detect_img_size:
            frame = cv2.resize(frame, self.detect_img_size, interpolation=cv2.INTER_AREA)
        if self.use_undistort:
            frame = cv2_capture_undistort_frame(frame, self.undistort_maps_by_camera[camera_name])
        return frame

    @staticmethod
    def timestamp_for_record(record: dict[str, Any], frame_idx: int, metadata: dict[str, Any]) -> float:
        capture_ts = record.get('capture_timestamp', None)
        if isinstance(capture_ts, (int, float)):
            return float(capture_ts)
        fps = metadata.get('fps', 30) if isinstance(metadata, dict) else 30
        try:
            fps_f = float(fps)
        except (TypeError, ValueError):
            fps_f = 30.0
        return frame_idx / max(fps_f, 1.0)

    def estimate_record(self, record: dict[str, Any], frame_idx: int, metadata: dict[str, Any]) -> dict[str, Any]:
        camera_name = str(record.get('camera_name', self.active_camera_names[0]))
        image_bgr = record['image_bgr']
        if camera_name not in self.detector_entries_by_camera:
            return {'camera_name': camera_name, 'status_lines': [f'[{camera_name}] skipped: no detector config'], 'cube_results': [], 'decoded_tag_count': 0}
        detector_entries = self.detector_entries_by_camera[camera_name]
        detect_frame = self.prepare_detect_frame(image_bgr, camera_name)
        timestamp = self.timestamp_for_record(record, frame_idx, metadata)
        shared_tags = None
        decoded_tag_ids: set[int] = set()
        adaptive_new_tag_ids: set[int] = set()
        if self.shared_tag_detection:
            shared_tags = detector_entries[0]['detector'].detect_tags(detect_frame, adaptive_clahe=self.adaptive_clahe)
            decoded_tag_ids.update((int(tag_id) for tag_id, _ in shared_tags['detections']))
        status_lines = [f"[{camera_name}] 008 replay cubes={len(detector_entries)} detect_size={self.detect_img_size} tag_detect_mode={('shared' if self.shared_tag_detection else 'per_cube')} adaptive_clahe={self.adaptive_clahe}"]
        cube_results: list[dict[str, Any]] = []
        for entry in detector_entries:
            cube_name = entry['cube_name']
            detector = entry['detector']
            if self.shared_tag_detection:
                cube_tags = shared_tags
                assert cube_tags is not None
                result = detector.process_detections(detect_frame, cube_tags['detections'], rejected_quads=cube_tags['rejected'], gray=cube_tags['gray'], enhanced=cube_tags['enhanced'], timestamp=timestamp)
                recovery_mode = 'shared_adaptive' if self.adaptive_clahe else 'shared_base'
            else:
                result, cube_tags, recovery_mode = self.estimate_cube_with_clahe_recovery(detector, detect_frame, timestamp)
            decoded_tag_ids.update((int(tag_id) for tag_id, _ in cube_tags['detections']))
            for attempt in cube_tags.get('adaptive_attempts', []):
                if attempt.get('base', False):
                    continue
                adaptive_new_tag_ids.update((int(tag_id) for tag_id in attempt.get('new_ids', [])))
            result = replay_008_result_copy_for_replay(result)
            result['decoded_tags_this_cube_pass'] = len(cube_tags['detections'])
            result['clahe_recovery_mode'] = recovery_mode
            status_lines.append(cv2_capture_result_to_text(camera_name, cube_name, result))
            cube_results.append({'cube_name': cube_name, 'result': result})
        status_lines[0] += f' decoded_tags={len(decoded_tag_ids)} clahe_extra_tags={len(adaptive_new_tag_ids)}'
        return {'camera_name': camera_name, 'status_lines': status_lines, 'cube_results': cube_results, 'decoded_tag_count': len(decoded_tag_ids), 'adaptive_clahe': self.adaptive_clahe, 'adaptive_new_tags': len(adaptive_new_tag_ids), 'tag_detect_mode': 'shared' if self.shared_tag_detection else 'per_cube'}

    def estimate_cube_with_clahe_recovery(self, detector: Any, detect_frame: np.ndarray, timestamp: float) -> tuple[dict[str, Any], dict[str, Any], str]:
        state_before = replay_008_snapshot_detector_tracking_state(detector)
        base_tags = detector.detect_tags(detect_frame, adaptive_clahe=False)
        base_result = detector.process_detections(detect_frame, base_tags['detections'], rejected_quads=base_tags['rejected'], gray=base_tags['gray'], enhanced=base_tags['enhanced'], timestamp=timestamp)
        base_state_after = replay_008_snapshot_detector_tracking_state(detector)
        if replay_008_is_measured_pose(base_result) or not self.adaptive_clahe:
            return (base_result, base_tags, 'base')
        variants = getattr(aprilcube_detect, '_adaptive_image_enhancement_variants', ())
        if not variants:
            variants = tuple(({'name': f'adaptive clip={float(clip_limit):.1f} tile={tuple(tile_grid_size)}', 'clahe': (float(clip_limit), tuple(tile_grid_size))} for clip_limit, tile_grid_size in getattr(aprilcube_detect, '_adaptive_clahe_variants', ())))
        for variant in variants:
            replay_008_restore_detector_tracking_state(detector, state_before)
            candidate_tags = detector.detect_tags(detect_frame, adaptive_clahe=True, enhancement_variants=(dict(variant),))
            candidate_result = detector.process_detections(detect_frame, candidate_tags['detections'], rejected_quads=candidate_tags['rejected'], gray=candidate_tags['gray'], enhanced=candidate_tags['enhanced'], timestamp=timestamp)
            if replay_008_is_measured_pose(candidate_result):
                return (candidate_result, candidate_tags, str(variant.get('name', 'adaptive enhancement')))
        replay_008_restore_detector_tracking_state(detector, base_state_after)
        return (base_result, base_tags, 'base_failed_enhancement_rejected')

    @staticmethod
    def detector_input_mode_for_pose_frame(pose_frame: dict[str, Any]) -> str:
        for cube in pose_frame.get('cube_results', []):
            result = cube.get('result', {})
            mode = str(result.get('clahe_recovery_mode', 'base'))
            if result.get('success', False) and mode != 'temporal_fill':
                return mode
        for cube in pose_frame.get('cube_results', []):
            result = cube.get('result', {})
            mode = str(result.get('clahe_recovery_mode', 'base'))
            if mode != 'temporal_fill':
                return mode
        return 'base'

    @staticmethod
    def detector_input_gray_for_mode(gray: np.ndarray, mode: str) -> np.ndarray:
        if mode in ('base', 'shared_base', 'base_failed_enhancement_rejected', 'temporal_fill'):
            return aprilcube_detect._preprocess(gray)
        variants = getattr(aprilcube_detect, '_adaptive_image_enhancement_variants', ())
        for variant in variants:
            if str(variant.get('name', '')) == mode:
                return aprilcube_detect._preprocess_enhancement_variant(gray, dict(variant))
        return aprilcube_detect._preprocess(gray)

    def draw_detector_input_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame['camera_name']
        detect_frame = self.prepare_detect_frame(record['image_bgr'], camera_name)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY) if len(detect_frame.shape) == 3 else detect_frame
        mode = self.detector_input_mode_for_pose_frame(pose_frame)
        enhanced = self.detector_input_gray_for_mode(gray, mode)
        vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        mode_text = f'Detector input: {mode}'
        if mode == 'temporal_fill':
            mode_text += ' (pose came from temporal fill; showing base detector input)'
        cv2.putText(vis, mode_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return vis

    def draw_pose_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame['camera_name']
        detect_frame = self.prepare_detect_frame(record['image_bgr'], camera_name)
        vis = cv2_capture_make_tag_detection_vis_image(detect_frame)
        return self.draw_pose_over_base_frame(vis, camera_name, pose_frame)

    def draw_detector_input_pose_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame['camera_name']
        vis = self.draw_detector_input_frame(record, pose_frame)
        return self.draw_pose_over_base_frame(vis, camera_name, pose_frame)

    def draw_pose_over_base_frame(self, base_frame: np.ndarray, camera_name: str, pose_frame: dict[str, Any]) -> np.ndarray:
        vis = base_frame.copy()
        for cube in pose_frame['cube_results']:
            detector = self.detector_by_camera_cube[camera_name, cube['cube_name']]
            result = self.normalize_result_for_draw(cube.get('result', {}))
            vis = detector.draw_result(vis, result)
        vis = cv2_capture_draw_text_panel(vis, pose_frame['status_lines'])
        temporal_cubes = self.temporal_filled_cube_names(pose_frame)
        if temporal_cubes:
            return self.draw_red_alert_box(vis, 'TEMPORAL FILLED CUBE POSE', ', '.join(temporal_cubes[:3]) + (f', +{len(temporal_cubes) - 3}' if len(temporal_cubes) > 3 else ''))
        if not self.pose_frame_has_all_cube_pose(camera_name, pose_frame):
            return self.draw_red_alert_box(vis, 'INCOMPLETE CUBE POSE')
        return vis

    @staticmethod
    def normalize_result_for_draw(result: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(result or {})
        normalized.setdefault('success', False)
        normalized.setdefault('detections', [])
        normalized.setdefault('visible_faces', set())
        normalized.setdefault('n_tags', 0)
        normalized.setdefault('reproj_error', float('inf'))
        for key in ('rvec', 'tvec'):
            if normalized.get(key) is not None and (not isinstance(normalized[key], np.ndarray)):
                normalized[key] = np.asarray(normalized[key], dtype=np.float64).reshape(3, 1)
        if normalized.get('T') is not None and (not isinstance(normalized['T'], np.ndarray)):
            normalized['T'] = np.asarray(normalized['T'], dtype=np.float64).reshape(4, 4)
        return normalized

    @staticmethod
    def draw_red_alert_box(vis: np.ndarray, label: str, detail: str | None=None) -> np.ndarray:
        h, w = vis.shape[:2]
        border = max(6, min(w, h) // 120)
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), border)
        cv2.putText(vis, label, (20, max(42, border + 28)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
        if detail:
            cv2.putText(vis, detail, (20, max(84, border + 68)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    @staticmethod
    def temporal_filled_cube_names(pose_frame: dict[str, Any]) -> list[str]:
        return [str(cube.get('cube_name', 'unknown')) for cube in pose_frame.get('cube_results', []) if bool(cube.get('result', {}).get('temporal_filled', False))]

    def pose_frame_has_all_cube_pose(self, camera_name: str, pose_frame: dict[str, Any]) -> bool:
        expected_cubes = {entry['cube_name'] for entry in self.detector_entries_by_camera.get(camera_name, [])}
        result_cubes = {cube['cube_name'] for cube in pose_frame.get('cube_results', [])}
        if result_cubes != expected_cubes:
            return False
        return all((bool(cube.get('result', {}).get('success', False)) for cube in pose_frame.get('cube_results', [])))

    def missing_cube_names_for_pose_frame(self, camera_name: str, pose_frame: dict[str, Any]) -> list[str]:
        expected_cubes = {entry['cube_name'] for entry in self.detector_entries_by_camera.get(camera_name, [])}
        result_by_cube = {cube['cube_name']: cube.get('result', {}) for cube in pose_frame.get('cube_results', [])}
        return [cube_name for cube_name in sorted(expected_cubes) if not bool(result_by_cube.get(cube_name, {}).get('success', False))]

    def draw_undistorted_debug_frame(self, record: dict[str, Any], pose_frame: dict[str, Any]) -> np.ndarray:
        camera_name = pose_frame['camera_name']
        vis = self.prepare_detect_frame(record['image_bgr'], camera_name).copy()
        missing = self.missing_cube_names_for_pose_frame(camera_name, pose_frame)
        temporal_cubes = self.temporal_filled_cube_names(pose_frame)
        if temporal_cubes:
            return self.draw_red_alert_box(vis, 'TEMPORAL FILLED CUBE POSE', ', '.join(temporal_cubes[:3]) + (f', +{len(temporal_cubes) - 3}' if len(temporal_cubes) > 3 else ''))
        if self.pose_frame_has_all_cube_pose(camera_name, pose_frame):
            return vis
        missing_text = ', '.join(missing[:3])
        if len(missing) > 3:
            missing_text += f', +{len(missing) - 3}'
        return self.draw_red_alert_box(vis, f"MISSING CUBE POSE: {len(missing)}/{len(pose_frame.get('cube_results', []))}", missing_text)

def replay_008_pose_markdown(pose_frame: dict[str, Any]) -> str:
    lines = [f"**camera**: `{pose_frame.get('camera_name', 'unknown')}`", f"**tag detect mode**: `{pose_frame.get('tag_detect_mode', 'unknown')}`", f"**decoded tags**: `{pose_frame.get('decoded_tag_count', 0)}`", f"**adaptive CLAHE**: `{pose_frame.get('adaptive_clahe', False)}`", f"**CLAHE extra tags**: `{pose_frame.get('adaptive_new_tags', 0)}`", '']
    for cube in pose_frame.get('cube_results', []):
        result = cube['result']
        cube_name = cube['cube_name']
        if not result.get('success', False):
            lines.append(f"- `{cube_name}`: no pose, tags={int(result.get('n_tags', 0))}, mode={result.get('clahe_recovery_mode', 'unknown')}")
            continue
        tvec = np.asarray(result['tvec'], dtype=np.float64).reshape(-1)
        faces = sorted(list(result.get('visible_faces', set())))
        predicted = ' predicted' if result.get('predicted', False) else ''
        temporal_fill = ''
        if result.get('temporal_filled', False):
            source = result.get('temporal_fill_source', {})
            temporal_fill = f", temporal_fill={source.get('before_frame', '?')}->{source.get('after_frame', '?')}"
        temporal_smooth = ''
        if result.get('temporal_smoothed', False):
            temporal_smooth = f", smooth_n={int(result.get('temporal_smoothing_source_count', 0))}"
        single_tag_cfg = ''
        if result.get('single_tag_cfg_pose', False):
            single_tag_cfg = f", single_tag_cfg_pose=id{result.get('single_tag_id', '?')}/{result.get('single_tag_face', '?')}"
        lines.append(f"- `{cube_name}`: t=({tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f}) mm, reproj={float(result.get('reproj_error', float('inf'))):.2f}px, tags={int(result.get('n_tags', 0))}, faces={faces}{predicted}, mode={result.get('clahe_recovery_mode', 'unknown')}{single_tag_cfg}{temporal_fill}{temporal_smooth}")
    return '\n'.join(lines)

def replay_008_cube_scene_node_name(cube_name: str) -> str:
    safe = ''.join((ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in cube_name))
    return f'/world_thumb_web_camera/{safe}'

def replay_008_load_obj_mesh_for_viser(obj_name: str, color: tuple[int, int, int]) -> tuple[Any, Path]:
    obj_path = REPLAY_008_ASSETS_DIR / f'{obj_name}.obj'
    if not obj_path.is_file():
        raise FileNotFoundError(f'OBJ mesh not found: {obj_path}')
    loaded = trimesh.load(obj_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    rgba = np.asarray([color[0], color[1], color[2], 210], dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))
    return (mesh, obj_path)

def replay_008_cube_pose_tracks(pose_cache: list[dict[str, Any]]) -> dict[str, list[tuple[int, np.ndarray]]]:
    tracks: dict[str, list[tuple[int, np.ndarray]]] = {}
    for frame_idx, pose_frame in enumerate(pose_cache):
        for cube in pose_frame.get('cube_results', []):
            cube_name = str(cube.get('cube_name', ''))
            result = cube.get('result', {})
            if not cube_name or not bool(result.get('success', False)):
                continue
            tvec = result.get('tvec', None)
            if tvec is None:
                continue
            tracks.setdefault(cube_name, []).append((frame_idx, np.asarray(tvec, dtype=np.float64).reshape(3) / 1000.0))
    return tracks

def replay_008_make_track_segments(track: list[tuple[int, np.ndarray]]) -> np.ndarray:
    if len(track) < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray([[track[i][1], track[i + 1][1]] for i in range(len(track) - 1)], dtype=np.float32)

def replay_008_create_3d_scene_handles(server: viser.ViserServer, estimator: Replay008PoseEstimator, pose_cache: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    server.scene.set_up_direction('-y')
    server.scene.world_axes.visible = False
    server.scene.add_frame('/world_thumb_web_camera', axes_length=0.06, axes_radius=0.002, origin_radius=0.004)
    grid_lines = []
    grid_half = 0.2
    grid_step = 0.05
    n = int(round(grid_half / grid_step))
    for i in range(-n, n + 1):
        x = i * grid_step
        z = i * grid_step
        grid_lines.append([[x, 0.0, -grid_half], [x, 0.0, grid_half]])
        grid_lines.append([[-grid_half, 0.0, z], [grid_half, 0.0, z]])
    grid_handle = server.scene.add_line_segments('/world_thumb_web_camera/xz_grid_y0', points=np.asarray(grid_lines, dtype=np.float32), colors=(80, 80, 80), line_width=1.0, visible=False)
    aspect = estimator.detect_img_size[0] / max(estimator.detect_img_size[1], 1)
    first_camera = estimator.active_camera_names[0]
    camera_matrix = estimator.detection_camera_matrix_by_camera[first_camera]
    fy = float(camera_matrix[1, 1])
    fov_y = float(2.0 * np.arctan(estimator.detect_img_size[1] / max(2.0 * fy, 1e-12)))
    camera_frustum = server.scene.add_camera_frustum('/world_thumb_web_camera/frustum', fov=fov_y, aspect=aspect, scale=0.08, line_width=1.5, color=(180, 180, 180), visible=True)
    palette = [(255, 150, 40), (80, 180, 255), (120, 220, 120), (220, 120, 255), (255, 220, 80), (180, 180, 180)]
    handles: dict[str, dict[str, Any]] = {'__scene__': {'grid': grid_handle, 'camera_frustum': camera_frustum}}
    tracks = replay_008_cube_pose_tracks(pose_cache)
    obj_mesh_cache: dict[str, tuple[Any, Path]] = {}
    cfg_to_obj = CV2_CAPTURE_CUBE_CFG_NAME_TO_OBJ_NAME
    color_idx = 0
    for camera_name in estimator.active_camera_names:
        for entry in estimator.detector_entries_by_camera.get(camera_name, []):
            cube_name = entry['cube_name']
            detector = entry['detector']
            node = replay_008_cube_scene_node_name(cube_name)
            safe = node.rsplit('/', 1)[-1]
            track_node = f'/world_thumb_web_camera/pose_tracks/{safe}'
            dims_m = tuple((float(v) / 1000.0 for v in detector.config.box_dims))
            color = palette[color_idx % len(palette)]
            color_idx += 1
            frame_handle = server.scene.add_frame(node, axes_length=max(dims_m) * 0.8, axes_radius=max(dims_m) * 0.035, origin_radius=0.0, visible=False)
            box_handle = server.scene.add_box(f'{node}/box', dimensions=dims_m, color=color, opacity=0.35, side='double', visible=False)
            obj_mesh_handle = None
            obj_name = str(cfg_to_obj.get(cube_name, ''))
            if obj_name:
                try:
                    if obj_name not in obj_mesh_cache:
                        obj_mesh_cache[obj_name] = replay_008_load_obj_mesh_for_viser(obj_name, color)
                    mesh, obj_path = obj_mesh_cache[obj_name]
                    obj_mesh_handle = server.scene.add_mesh_trimesh(f'{node}/finger_obj', mesh.copy(), scale=REPLAY_008_OBJ_MESH_SCALE, visible=False, cast_shadow=False, receive_shadow=False)
                    print(f'[INFO] 3D OBJ mesh: {cube_name} -> {obj_name} path={obj_path}')
                except Exception as exc:
                    print(f'[WARNING] Failed to add 3D OBJ mesh for {cube_name} -> {obj_name}: {type(exc).__name__}: {exc}')
            track = tracks.get(cube_name, [])
            track_segments = replay_008_make_track_segments(track)
            trajectory_handle = server.scene.add_line_segments(f'{track_node}/trajectory', points=track_segments, colors=np.asarray(color, dtype=np.uint8), line_width=2.0, visible=track_segments.shape[0] > 0)
            if track:
                sample_points = np.asarray([pos for _idx, pos in track], dtype=np.float32)
                sample_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(track), 1))
            else:
                sample_points = np.zeros((0, 3), dtype=np.float32)
                sample_colors = np.zeros((0, 3), dtype=np.uint8)
            samples_handle = server.scene.add_point_cloud(f'{track_node}/trajectory_samples', points=sample_points, colors=sample_colors, point_size=0.004, point_shape='circle', visible=sample_points.shape[0] > 0)
            marker_radius = max(max(dims_m) * 0.08, 0.0015)
            current_handle = server.scene.add_icosphere(f'{track_node}/current_position', radius=marker_radius, color=(255, 255, 255), subdivisions=2, visible=False)
            start_handle = None
            end_handle = None
            if track:
                _start_idx, start_pos = track[0]
                _end_idx, end_pos = track[-1]
                start_handle = server.scene.add_icosphere(f'{track_node}/track_start', radius=marker_radius, color=(40, 220, 80), subdivisions=2, position=start_pos, visible=True)
                end_handle = server.scene.add_icosphere(f'{track_node}/track_end', radius=marker_radius, color=(240, 80, 80), subdivisions=2, position=end_pos, visible=True)
            handles[cube_name] = {'frame': frame_handle, 'box': box_handle, 'obj_mesh': obj_mesh_handle, 'base_color': color, 'trajectory': trajectory_handle, 'samples': samples_handle, 'current': current_handle, 'start': start_handle, 'end': end_handle}
    return handles

def replay_008_update_3d_scene(scene_handles: dict[str, dict[str, Any]], pose_frame: dict[str, Any]) -> None:
    seen: set[str] = set()
    for cube in pose_frame.get('cube_results', []):
        cube_name = str(cube.get('cube_name', ''))
        if cube_name.startswith('__'):
            continue
        result = cube.get('result', {})
        handles = scene_handles.get(cube_name)
        if handles is None:
            continue
        seen.add(cube_name)
        success = bool(result.get('success', False))
        handles['pose_visible'] = success
        for key in ('frame', 'box', 'obj_mesh', 'current'):
            handle = handles.get(key)
            if handle is not None:
                handle.visible = success
        if not success:
            continue
        rvec = np.asarray(result['rvec'], dtype=np.float64).reshape(3, 1)
        tvec_m = np.asarray(result['tvec'], dtype=np.float64).reshape(3) / 1000.0
        wxyz = replay_008_rvec_to_quat(rvec)
        handles['frame'].position = tvec_m
        handles['frame'].wxyz = wxyz
        handles['current'].position = tvec_m
        handles['box'].color = (255, 0, 0) if bool(result.get('temporal_filled', False)) else handles['base_color']
    for cube_name, handles in scene_handles.items():
        if cube_name.startswith('__'):
            continue
        if cube_name in seen:
            continue
        handles['pose_visible'] = False
        for key in ('frame', 'box', 'obj_mesh', 'current'):
            handle = handles.get(key)
            if handle is not None:
                handle.visible = False

def replay_008_set_optional_visible(handle: Any, visible: bool) -> None:
    if handle is not None:
        handle.visible = bool(visible)

def replay_008_apply_3d_visibility(scene_handles: dict[str, dict[str, Any]], *, show_box: bool, show_obj: bool, show_axes: bool, show_trajectory: bool, show_samples: bool, show_endpoints: bool, show_grid: bool, show_camera: bool) -> None:
    scene = scene_handles.get('__scene__', {})
    replay_008_set_optional_visible(scene.get('grid'), show_grid)
    replay_008_set_optional_visible(scene.get('camera_frustum'), show_camera)
    for cube_name, handles in scene_handles.items():
        if cube_name.startswith('__'):
            continue
        pose_visible = bool(handles.get('pose_visible', False))
        if 'box' in handles:
            handles['box'].visible = bool(show_box) and pose_visible
        if 'obj_mesh' in handles and handles['obj_mesh'] is not None:
            handles['obj_mesh'].visible = bool(show_obj) and pose_visible
        if 'frame' in handles:
            handles['frame'].visible = bool(show_axes) and pose_visible
        replay_008_set_optional_visible(handles.get('current'), show_trajectory and pose_visible)
        replay_008_set_optional_visible(handles.get('trajectory'), show_trajectory)
        replay_008_set_optional_visible(handles.get('samples'), show_samples)
        replay_008_set_optional_visible(handles.get('start'), show_endpoints)
        replay_008_set_optional_visible(handles.get('end'), show_endpoints)

def replay_008_precompute_pose_cache(pkl_path: Path, frame_offsets: list[int], metadata: dict[str, Any], estimator: Replay008PoseEstimator) -> list[dict[str, Any]]:
    pose_cache: list[dict[str, Any]] = []
    total = len(frame_offsets)
    last_print = time.monotonic()
    for idx, offset in enumerate(frame_offsets):
        record = replay_008_load_frame_at_offset(pkl_path, offset)
        pose_cache.append(estimator.estimate_record(record, idx, metadata))
        now = time.monotonic()
        if now - last_print > 0.5:
            replay_008_print_pose_progress(idx + 1, total)
            last_print = now
    replay_008_print_pose_progress(total, total, force_newline=True)
    return pose_cache

def replay_008_cube_result_by_name(pose_frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {cube['cube_name']: cube for cube in pose_frame.get('cube_results', []) if isinstance(cube, dict) and 'cube_name' in cube}

def replay_008_is_temporal_anchor(result: dict[str, Any]) -> bool:
    return bool(result.get('success', False)) and (not bool(result.get('predicted', False))) and (not bool(result.get('temporal_filled', False)))

def replay_008_interpolate_pose_result(before_idx: int, before_result: dict[str, Any], after_idx: int, after_result: dict[str, Any], target_idx: int) -> dict[str, Any]:
    alpha = (target_idx - before_idx) / max(after_idx - before_idx, 1)
    before_t = np.asarray(before_result['tvec'], dtype=np.float64).reshape(3, 1)
    after_t = np.asarray(after_result['tvec'], dtype=np.float64).reshape(3, 1)
    tvec = (1.0 - alpha) * before_t + alpha * after_t
    q0 = replay_008_rvec_to_quat(before_result['rvec'])
    q1 = replay_008_rvec_to_quat(after_result['rvec'])
    anchor_rotation_deg = replay_008_quat_short_arc_angle_deg(q0, q1)
    q_interp = replay_008_slerp_quat(q0, q1, alpha)
    rotation_mode = 'slerp_large_anchor_rotation' if anchor_rotation_deg > REPLAY_008_TEMPORAL_FILL_MAX_ROTATION_DEG else 'slerp_short_arc'
    rvec = replay_008_quat_to_rvec(q_interp)
    before_faces = set(before_result.get('visible_faces', set()) or [])
    after_faces = set(after_result.get('visible_faces', set()) or [])
    before_reproj = float(before_result.get('reproj_error', 0.0))
    after_reproj = float(after_result.get('reproj_error', 0.0))
    return {'success': True, 'rvec': rvec, 'tvec': tvec, 'T': replay_008_pose_transform_from_rvec_tvec(rvec, tvec), 'reproj_error': (1.0 - alpha) * before_reproj + alpha * after_reproj, 'n_tags': 0, 'n_inliers': 0, 'detections': [], 'tag_ids': [], 'visible_faces': before_faces | after_faces, 'predicted': False, 'temporal_filled': True, 'temporal_fill_source': {'before_frame': int(before_idx), 'after_frame': int(after_idx)}, 'temporal_fill_alpha': float(alpha), 'temporal_fill_rotation_deg': float(anchor_rotation_deg), 'temporal_fill_rotation_mode': rotation_mode, 'decoded_tags_this_cube_pass': 0, 'clahe_recovery_mode': 'temporal_fill'}

def replay_008_rebuild_pose_frame_status_lines(estimator: Replay008PoseEstimator, pose_frame: dict[str, Any]) -> None:
    camera_name = pose_frame.get('camera_name', estimator.active_camera_names[0])
    cube_results = pose_frame.get('cube_results', [])
    header = f"[{camera_name}] 008 replay cubes={len(cube_results)} detect_size={estimator.detect_img_size} tag_detect_mode={pose_frame.get('tag_detect_mode', 'unknown')} adaptive_clahe={pose_frame.get('adaptive_clahe', False)} decoded_tags={pose_frame.get('decoded_tag_count', 0)} clahe_extra_tags={pose_frame.get('adaptive_new_tags', 0)} continuity_rejected={pose_frame.get('continuity_rejected_count', 0)} temporal_outlier_rejected={pose_frame.get('temporal_outlier_rejected_count', 0)} temporal_filled={pose_frame.get('temporal_filled_count', 0)} rotation_limited={pose_frame.get('temporal_rotation_jump_limited_count', 0)} smoothing={pose_frame.get('temporal_smoothing_enabled', False)}"
    lines = [header]
    for cube in cube_results:
        lines.append(cv2_capture_result_to_text(str(camera_name), str(cube['cube_name']), cube.get('result', {})))
    pose_frame['status_lines'] = lines

def replay_008_is_postprocess_temporal_result(result: dict[str, Any]) -> bool:
    return bool(result.get('temporal_filled', False)) or result.get('clahe_recovery_mode') == 'temporal_fill'

def replay_008_reject_pose_result_for_temporal_fill(result: dict[str, Any], reason: str, *, previous_face: str | None=None, rotation_jump_deg: float | None=None, previous_frame: int | None=None, next_frame: int | None=None, next_rotation_jump_deg: float | None=None, previous_translation_jump_mm: float | None=None, next_translation_jump_mm: float | None=None) -> dict[str, Any]:
    rejected = copy.deepcopy(result)
    rejected['success'] = False
    rejected['rvec'] = None
    rejected['tvec'] = None
    rejected['T'] = None
    rejected['reproj_error'] = float('inf')
    rejected['continuity_rejected'] = True
    rejected['continuity_reject_reason'] = reason
    if previous_face is not None:
        rejected['continuity_previous_face'] = previous_face
    if rotation_jump_deg is not None:
        rejected['continuity_rotation_jump_deg'] = float(rotation_jump_deg)
    if previous_frame is not None:
        rejected['continuity_previous_frame'] = int(previous_frame)
    if next_frame is not None:
        rejected['continuity_next_frame'] = int(next_frame)
    if next_rotation_jump_deg is not None:
        rejected['continuity_next_rotation_jump_deg'] = float(next_rotation_jump_deg)
    if previous_translation_jump_mm is not None:
        rejected['continuity_previous_translation_jump_mm'] = float(previous_translation_jump_mm)
    if next_translation_jump_mm is not None:
        rejected['continuity_next_translation_jump_mm'] = float(next_translation_jump_mm)
    return rejected

def replay_008_single_face_name(result: dict[str, Any]) -> str | None:
    faces = sorted(list(result.get('visible_faces', set()) or []))
    if len(faces) != 1:
        return None
    return str(faces[0])

def replay_008_reset_temporal_postprocess_outputs(pose_cache: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    reset = copy.deepcopy(pose_cache)
    reset_count = 0
    for pose_frame in reset:
        pose_frame['temporal_filled_count'] = 0
        pose_frame['continuity_rejected_count'] = 0
        pose_frame['temporal_outlier_rejected_count'] = 0
        pose_frame['temporal_rotation_jump_limited_count'] = 0
        for cube in pose_frame.get('cube_results', []):
            result = cube.get('result', {})
            for key in ('temporal_smoothed', 'temporal_smoothing_source_count', 'temporal_smoothing_window_radius', 'temporal_smoothing_rotation_delta_deg', 'temporal_smoothing_rotation_limited', 'temporal_rotation_jump_limited', 'temporal_rotation_jump_held', 'temporal_rotation_jump_original_delta_deg', 'temporal_rotation_jump_max_deg', 'temporal_rotation_jump_hold_deg'):
                result.pop(key, None)
            if replay_008_is_postprocess_temporal_result(result):
                cube['result'] = replay_008_reject_pose_result_for_temporal_fill(result, 'reset_previous_temporal_fill')
                reset_count += 1
    return (reset, reset_count)

def replay_008_gate_single_tag_pose_cache(pose_cache: list[dict[str, Any]], estimator: Replay008PoseEstimator, *, max_rotation_deg: float=REPLAY_008_SINGLE_TAG_CONTINUITY_MAX_ROTATION_DEG) -> tuple[list[dict[str, Any]], int]:
    if not REPLAY_008_SINGLE_TAG_CONTINUITY_GATE_ENABLED:
        return (pose_cache, 0)
    gated = copy.deepcopy(pose_cache)
    rejected_count = 0
    for camera_name in estimator.active_camera_names:
        cube_names = [entry['cube_name'] for entry in estimator.detector_entries_by_camera.get(camera_name, [])]
        frame_indices = [idx for idx, pose_frame in enumerate(gated) if pose_frame.get('camera_name') == camera_name]
        for cube_name in cube_names:
            single_face_observations: list[tuple[int, str, dict[str, Any]]] = []
            for idx in frame_indices:
                pose_frame = gated[idx]
                cube = replay_008_cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                result = cube.get('result', {})
                n_tags = int(result.get('n_tags', 0) or 0)
                face = replay_008_single_face_name(result)
                if bool(result.get('success', False)) and (not bool(result.get('predicted', False))) and (not replay_008_is_postprocess_temporal_result(result)) and (n_tags == 1) and (face is not None):
                    single_face_observations.append((idx, face, result))
            trusted_single_tag_indices: set[int] = set()
            current_run: list[tuple[int, str, dict[str, Any]]] = []

            def commit_run(run: list[tuple[int, str, dict[str, Any]]]) -> None:
                if len(run) < int(REPLAY_008_SINGLE_TAG_CONTINUITY_MIN_FACE_OBSERVATIONS):
                    return
                trusted_single_tag_indices.update((idx for idx, _face, _result in run))
            for observation in single_face_observations:
                idx, face, result = observation
                if not current_run:
                    current_run = [observation]
                    continue
                prev_idx, prev_face, _prev_result = current_run[-1]
                if face == prev_face and idx - prev_idx <= int(REPLAY_008_SINGLE_TAG_CONTINUITY_MAX_OBSERVATION_GAP):
                    current_run.append(observation)
                    continue
                commit_run(current_run)
                current_run = [observation]
            commit_run(current_run)
            last_trusted_by_face: dict[str, dict[str, Any]] = {}
            for idx in frame_indices:
                pose_frame = gated[idx]
                cube = replay_008_cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                result = cube.get('result', {})
                if not bool(result.get('success', False)):
                    continue
                if bool(result.get('predicted', False)):
                    continue
                if replay_008_is_postprocess_temporal_result(result):
                    continue
                n_tags = int(result.get('n_tags', 0) or 0)
                face = replay_008_single_face_name(result)
                reject_reason: str | None = None
                rotation_jump_deg: float | None = None
                previous_face: str | None = None
                if n_tags <= 0:
                    reject_reason = 'no_decoded_tag_success_pose'
                elif n_tags == 1:
                    if idx not in trusted_single_tag_indices:
                        reject_reason = 'single_tag_isolated_face_observation'
                    elif face is not None and face in last_trusted_by_face:
                        previous_face = face
                        rotation_jump_deg = replay_008_quat_short_arc_angle_deg(replay_008_rvec_to_quat(last_trusted_by_face[face]['rvec']), replay_008_rvec_to_quat(result['rvec']))
                        if rotation_jump_deg > max_rotation_deg:
                            reject_reason = 'single_tag_same_face_rotation_jump'
                if reject_reason is not None:
                    cube['result'] = replay_008_reject_pose_result_for_temporal_fill(result, reject_reason, previous_face=previous_face, rotation_jump_deg=rotation_jump_deg)
                    pose_frame['continuity_rejected_count'] = int(pose_frame.get('continuity_rejected_count', 0)) + 1
                    rejected_count += 1
                    continue
                if n_tags > 0 and face is not None:
                    last_trusted_by_face[face] = result
    for pose_frame in gated:
        pose_frame['single_tag_continuity_gate_enabled'] = bool(REPLAY_008_SINGLE_TAG_CONTINUITY_GATE_ENABLED)
        replay_008_rebuild_pose_frame_status_lines(estimator, pose_frame)
    return (gated, rejected_count)

def replay_008_pose_translation_jump_mm(a: dict[str, Any], b: dict[str, Any]) -> float:
    at = np.asarray(a['tvec'], dtype=np.float64).reshape(3)
    bt = np.asarray(b['tvec'], dtype=np.float64).reshape(3)
    return float(np.linalg.norm(at - bt))

def replay_008_pose_rotation_jump_deg(a: dict[str, Any], b: dict[str, Any]) -> float:
    return replay_008_quat_short_arc_angle_deg(replay_008_rvec_to_quat(a['rvec']), replay_008_rvec_to_quat(b['rvec']))

def replay_008_gate_temporal_outlier_pose_cache(pose_cache: list[dict[str, Any]], estimator: Replay008PoseEstimator) -> tuple[list[dict[str, Any]], int]:
    if not REPLAY_008_TEMPORAL_OUTLIER_GATE_ENABLED:
        return (pose_cache, 0)
    gated = copy.deepcopy(pose_cache)
    rejected_count = 0
    for camera_name in estimator.active_camera_names:
        cube_names = [entry['cube_name'] for entry in estimator.detector_entries_by_camera.get(camera_name, [])]
        frame_indices = [idx for idx, pose_frame in enumerate(gated) if pose_frame.get('camera_name') == camera_name]
        for cube_name in cube_names:
            anchors: list[tuple[int, dict[str, Any]]] = []
            for idx in frame_indices:
                cube = replay_008_cube_result_by_name(gated[idx]).get(cube_name)
                if cube is None:
                    continue
                result = cube.get('result', {})
                if replay_008_is_temporal_anchor(result):
                    anchors.append((idx, result))
            if len(anchors) < 3:
                continue
            for anchor_pos in range(1, len(anchors) - 1):
                prev_idx, prev_result = anchors[anchor_pos - 1]
                idx, result = anchors[anchor_pos]
                next_idx, next_result = anchors[anchor_pos + 1]
                if idx - prev_idx > REPLAY_008_TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES:
                    continue
                if next_idx - idx > REPLAY_008_TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES:
                    continue
                neighbor_rotation_deg = replay_008_pose_rotation_jump_deg(prev_result, next_result)
                neighbor_translation_mm = replay_008_pose_translation_jump_mm(prev_result, next_result)
                if neighbor_rotation_deg > REPLAY_008_TEMPORAL_OUTLIER_NEIGHBOR_MAX_ROTATION_DEG:
                    continue
                if neighbor_translation_mm > REPLAY_008_TEMPORAL_OUTLIER_NEIGHBOR_MAX_TRANSLATION_MM:
                    continue
                prev_rotation_deg = replay_008_pose_rotation_jump_deg(prev_result, result)
                next_rotation_deg = replay_008_pose_rotation_jump_deg(result, next_result)
                prev_translation_mm = replay_008_pose_translation_jump_mm(prev_result, result)
                next_translation_mm = replay_008_pose_translation_jump_mm(result, next_result)
                rotation_flip = prev_rotation_deg >= REPLAY_008_TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG and next_rotation_deg >= REPLAY_008_TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG
                translation_spike = prev_translation_mm >= REPLAY_008_TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM and next_translation_mm >= REPLAY_008_TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM
                if not (rotation_flip or translation_spike):
                    continue
                pose_frame = gated[idx]
                cube = replay_008_cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                cube['result'] = replay_008_reject_pose_result_for_temporal_fill(result, 'temporal_pose_outlier_between_consistent_neighbors', previous_frame=prev_idx, next_frame=next_idx, rotation_jump_deg=prev_rotation_deg, next_rotation_jump_deg=next_rotation_deg, previous_translation_jump_mm=prev_translation_mm, next_translation_jump_mm=next_translation_mm)
                cube['result']['temporal_outlier_rejected'] = True
                cube['result']['temporal_outlier_neighbor_rotation_deg'] = float(neighbor_rotation_deg)
                cube['result']['temporal_outlier_neighbor_translation_mm'] = float(neighbor_translation_mm)
                pose_frame['continuity_rejected_count'] = int(pose_frame.get('continuity_rejected_count', 0)) + 1
                pose_frame['temporal_outlier_rejected_count'] = int(pose_frame.get('temporal_outlier_rejected_count', 0)) + 1
                rejected_count += 1
    for pose_frame in gated:
        pose_frame['temporal_outlier_gate_enabled'] = bool(REPLAY_008_TEMPORAL_OUTLIER_GATE_ENABLED)
        replay_008_rebuild_pose_frame_status_lines(estimator, pose_frame)
    return (gated, rejected_count)

def replay_008_complete_pose_cache_temporally(pose_cache: list[dict[str, Any]], estimator: Replay008PoseEstimator, *, max_gap_frames: int=REPLAY_008_TEMPORAL_FILL_MAX_GAP_FRAMES) -> tuple[list[dict[str, Any]], int]:
    completed = copy.deepcopy(pose_cache)
    filled_count = 0
    for camera_name in estimator.active_camera_names:
        cube_names = [entry['cube_name'] for entry in estimator.detector_entries_by_camera.get(camera_name, [])]
        frame_indices = [idx for idx, pose_frame in enumerate(completed) if pose_frame.get('camera_name') == camera_name]
        for cube_name in cube_names:
            anchors: list[tuple[int, dict[str, Any]]] = []
            for idx in frame_indices:
                cube = replay_008_cube_result_by_name(completed[idx]).get(cube_name)
                if cube is None:
                    continue
                result = cube.get('result', {})
                if replay_008_is_temporal_anchor(result):
                    anchors.append((idx, result))
            for (before_idx, before_result), (after_idx, after_result) in zip(anchors, anchors[1:]):
                if after_idx - before_idx - 1 <= 0:
                    continue
                if after_idx - before_idx - 1 > max_gap_frames:
                    continue
                for target_idx in range(before_idx + 1, after_idx):
                    pose_frame = completed[target_idx]
                    cube_map = replay_008_cube_result_by_name(pose_frame)
                    cube = cube_map.get(cube_name)
                    if cube is not None and bool(cube.get('result', {}).get('success', False)):
                        continue
                    filled_result = replay_008_interpolate_pose_result(before_idx, before_result, after_idx, after_result, target_idx)
                    old_result = {} if cube is None else cube.get('result', {})
                    if bool(old_result.get('continuity_rejected', False)):
                        filled_result['temporal_fill_replaced_rejection'] = old_result.get('continuity_reject_reason', 'continuity_rejected')
                    if cube is None:
                        pose_frame.setdefault('cube_results', []).append({'cube_name': cube_name, 'result': filled_result})
                    else:
                        cube['result'] = filled_result
                    pose_frame['temporal_filled_count'] = int(pose_frame.get('temporal_filled_count', 0)) + 1
                    filled_count += 1
    for pose_frame in completed:
        pose_frame['temporal_fill_enabled'] = True
        pose_frame['temporal_fill_max_gap_frames'] = int(max_gap_frames)
        replay_008_rebuild_pose_frame_status_lines(estimator, pose_frame)
    return (completed, filled_count)

def replay_008_pose_result_smoothing_weight(result: dict[str, Any], frame_distance: int) -> float:
    sigma = max(float(REPLAY_008_TEMPORAL_SMOOTHING_SIGMA_FRAMES), 1e-06)
    time_weight = float(np.exp(-0.5 * (float(frame_distance) / sigma) ** 2))
    if bool(result.get('predicted', False)):
        quality_weight = 0.35
    elif bool(result.get('temporal_filled', False)):
        quality_weight = 0.65
    else:
        quality_weight = 1.0
    reproj = result.get('reproj_error', None)
    if reproj is not None and np.isfinite(float(reproj)):
        quality_weight *= 1.0 / (1.0 + max(float(reproj), 0.0) / 5.0)
    return time_weight * quality_weight

def replay_008_pose_reprojection_errors_for_result(result: dict[str, Any], detector: Any, rvec: np.ndarray, tvec: np.ndarray) -> tuple[float, dict[int, float]] | None:
    detections = result.get('detections', [])
    if not detections:
        return None
    object_chunks = []
    image_chunks = []
    tag_ids = []
    for tag_id, corners_2d in detections:
        corners_3d = detector.tag_corner_map.get(int(tag_id))
        if corners_3d is None:
            continue
        object_chunks.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
        image_chunks.append(np.asarray(corners_2d, dtype=np.float64).reshape(4, 2))
        tag_ids.append(int(tag_id))
    if not object_chunks:
        return None
    object_points = np.vstack(object_chunks).astype(np.float64)
    image_points = np.vstack(image_chunks).astype(np.float64)
    projected, _ = cv2.projectPoints(object_points, np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1), detector.camera_matrix, detector.dist_coeffs)
    projected = projected.reshape(-1, 2)
    per_tag: dict[int, float] = {}
    for k, tag_id in enumerate(tag_ids):
        start = k * 4
        end = start + 4
        per_tag[tag_id] = float(np.mean(np.linalg.norm(image_points[start:end] - projected[start:end], axis=1)))
    return (float(np.mean(list(per_tag.values()))), per_tag)

def replay_008_weighted_average_quats(quats: list[np.ndarray], weights: list[float], reference: np.ndarray | None=None) -> np.ndarray:
    if not quats:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ref = replay_008_normalize_quat(reference) if reference is not None else replay_008_normalize_quat(quats[len(quats) // 2])
    accum = np.zeros(4, dtype=np.float64)
    for quat, weight in zip(quats, weights):
        q = replay_008_align_quat_to_reference(quat, ref)
        accum += float(weight) * q
    return accum / max(float(np.linalg.norm(accum)), 1e-12)

def replay_008_smooth_pose_cache_temporally(pose_cache: list[dict[str, Any]], estimator: Replay008PoseEstimator, *, window_radius: int=REPLAY_008_TEMPORAL_SMOOTHING_WINDOW_RADIUS) -> tuple[list[dict[str, Any]], int]:
    if window_radius <= 0:
        return (pose_cache, 0)
    source = pose_cache
    smoothed = copy.deepcopy(pose_cache)
    smoothed_count = 0
    for camera_name in estimator.active_camera_names:
        cube_names = [entry['cube_name'] for entry in estimator.detector_entries_by_camera.get(camera_name, [])]
        frame_indices = [idx for idx, pose_frame in enumerate(source) if pose_frame.get('camera_name') == camera_name]
        for cube_name in cube_names:
            for target_idx in frame_indices:
                cube = replay_008_cube_result_by_name(smoothed[target_idx]).get(cube_name)
                if cube is None:
                    continue
                source_cube = replay_008_cube_result_by_name(source[target_idx]).get(cube_name)
                source_result = {} if source_cube is None else source_cube.get('result', {})
                if not bool(source_result.get('success', False)):
                    continue
                samples: list[tuple[int, dict[str, Any], float]] = []
                for neighbor_idx in frame_indices:
                    distance = abs(neighbor_idx - target_idx)
                    if distance > window_radius:
                        continue
                    neighbor_cube = replay_008_cube_result_by_name(source[neighbor_idx]).get(cube_name)
                    if neighbor_cube is None:
                        continue
                    neighbor_result = neighbor_cube.get('result', {})
                    if not bool(neighbor_result.get('success', False)):
                        continue
                    weight = replay_008_pose_result_smoothing_weight(neighbor_result, distance)
                    if weight <= 0.0:
                        continue
                    samples.append((neighbor_idx, neighbor_result, weight))
                if len(samples) <= 1:
                    continue
                weights = np.asarray([sample[2] for sample in samples], dtype=np.float64)
                weights = weights / max(float(np.sum(weights)), 1e-12)
                t_stack = np.stack([np.asarray(sample[1]['tvec'], dtype=np.float64).reshape(3) for sample in samples], axis=0)
                tvec = np.sum(t_stack * weights[:, None], axis=0).reshape(3, 1)
                q_target = replay_008_rvec_to_quat(source_result['rvec'])
                q_avg = replay_008_weighted_average_quats([replay_008_rvec_to_quat(sample[1]['rvec']) for sample in samples], [float(w) for w in weights], reference=q_target)
                q_limited, rotation_delta_deg, rotation_limited = replay_008_limit_quat_rotation(q_target, q_avg, REPLAY_008_TEMPORAL_SMOOTHING_MAX_ROTATION_DEG)
                rvec = replay_008_quat_to_rvec(q_limited)
                target_result = cube.get('result', {})
                detector = estimator.detector_by_camera_cube.get((camera_name, cube_name))
                reproj_eval = None if detector is None else replay_008_pose_reprojection_errors_for_result(source_result, detector, rvec, tvec)
                if reproj_eval is not None:
                    smoothed_reproj, _smoothed_per_tag = reproj_eval
                    source_reproj = float(source_result.get('reproj_error', smoothed_reproj))
                    max_allowed_reproj = max(REPLAY_008_TEMPORAL_SMOOTHING_MAX_DISPLAY_REPROJ_PX, source_reproj * REPLAY_008_TEMPORAL_SMOOTHING_MAX_REPROJ_RATIO)
                    if smoothed_reproj > max_allowed_reproj:
                        target_result['temporal_smoothing_rejected'] = True
                        target_result['temporal_smoothing_reject_reason'] = 'display_reprojection_too_high'
                        target_result['temporal_smoothing_candidate_reproj_error'] = float(smoothed_reproj)
                        target_result['temporal_smoothing_max_allowed_reproj_error'] = float(max_allowed_reproj)
                        continue
                target_result['tvec'] = tvec
                target_result['rvec'] = rvec
                target_result['T'] = replay_008_pose_transform_from_rvec_tvec(rvec, tvec)
                if reproj_eval is not None:
                    smoothed_reproj, smoothed_per_tag = reproj_eval
                    if 'reproj_error_before_smoothing' not in target_result:
                        target_result['reproj_error_before_smoothing'] = target_result.get('reproj_error', None)
                    if 'per_tag_reproj_error_before_smoothing' not in target_result:
                        target_result['per_tag_reproj_error_before_smoothing'] = target_result.get('per_tag_reproj_error', None)
                    target_result['reproj_error'] = float(smoothed_reproj)
                    target_result['per_tag_reproj_error'] = smoothed_per_tag
                target_result['temporal_smoothed'] = True
                target_result['temporal_smoothing_source_count'] = int(len(samples))
                target_result['temporal_smoothing_window_radius'] = int(window_radius)
                target_result['temporal_smoothing_rotation_delta_deg'] = float(rotation_delta_deg)
                target_result['temporal_smoothing_rotation_limited'] = bool(rotation_limited)
                smoothed_count += 1
    for pose_frame in smoothed:
        pose_frame['temporal_smoothing_enabled'] = bool(REPLAY_008_TEMPORAL_SMOOTHING_ENABLED)
        pose_frame['temporal_smoothing_window_radius'] = int(window_radius)
        replay_008_rebuild_pose_frame_status_lines(estimator, pose_frame)
    return (smoothed, smoothed_count)

def replay_008_limit_pose_cache_rotation_jumps(pose_cache: list[dict[str, Any]], estimator: Replay008PoseEstimator, *, max_rotation_deg: float=REPLAY_008_TEMPORAL_ROTATION_JUMP_MAX_DEG, hold_rotation_deg: float=REPLAY_008_TEMPORAL_ROTATION_JUMP_HOLD_DEG) -> tuple[list[dict[str, Any]], int]:
    if not REPLAY_008_TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED:
        return (pose_cache, 0)
    limited = copy.deepcopy(pose_cache)
    limited_count = 0
    for camera_name in estimator.active_camera_names:
        cube_names = [entry['cube_name'] for entry in estimator.detector_entries_by_camera.get(camera_name, [])]
        frame_indices = [idx for idx, pose_frame in enumerate(limited) if pose_frame.get('camera_name') == camera_name]
        for cube_name in cube_names:
            previous_quat: np.ndarray | None = None
            for idx in frame_indices:
                pose_frame = limited[idx]
                cube = replay_008_cube_result_by_name(pose_frame).get(cube_name)
                if cube is None:
                    continue
                result = cube.get('result', {})
                if not bool(result.get('success', False)):
                    previous_quat = None
                    continue
                current_quat = replay_008_rvec_to_quat(result['rvec'])
                if previous_quat is None:
                    previous_quat = current_quat
                    continue
                limited_quat, rotation_delta_deg, was_limited = replay_008_limit_quat_rotation(previous_quat, current_quat, max_rotation_deg)
                if was_limited:
                    if rotation_delta_deg > hold_rotation_deg:
                        output_quat = previous_quat
                        result['temporal_rotation_jump_held'] = True
                    else:
                        output_quat = limited_quat
                    rvec = replay_008_quat_to_rvec(output_quat)
                    tvec = np.asarray(result['tvec'], dtype=np.float64).reshape(3, 1)
                    result['rvec'] = rvec
                    result['T'] = replay_008_pose_transform_from_rvec_tvec(rvec, tvec)
                    result['temporal_rotation_jump_limited'] = True
                    result['temporal_rotation_jump_original_delta_deg'] = float(rotation_delta_deg)
                    result['temporal_rotation_jump_max_deg'] = float(max_rotation_deg)
                    result['temporal_rotation_jump_hold_deg'] = float(hold_rotation_deg)
                    pose_frame['temporal_rotation_jump_limited_count'] = int(pose_frame.get('temporal_rotation_jump_limited_count', 0)) + 1
                    limited_count += 1
                    previous_quat = output_quat
                else:
                    previous_quat = current_quat
    for pose_frame in limited:
        pose_frame['temporal_rotation_jump_limit_enabled'] = bool(REPLAY_008_TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED)
        pose_frame['temporal_rotation_jump_max_deg'] = float(max_rotation_deg)
        pose_frame['temporal_rotation_jump_hold_deg'] = float(hold_rotation_deg)
        replay_008_rebuild_pose_frame_status_lines(estimator, pose_frame)
    return (limited, limited_count)

def replay_008_complete_and_smooth_pose_cache(pose_cache: list[dict[str, Any]], estimator: Replay008PoseEstimator) -> tuple[list[dict[str, Any]], int, int, int, int]:
    reset_pose_cache, reset_count = replay_008_reset_temporal_postprocess_outputs(pose_cache)
    gated_pose_cache, rejected_count = replay_008_gate_single_tag_pose_cache(reset_pose_cache, estimator)
    outlier_gated_pose_cache, outlier_rejected_count = replay_008_gate_temporal_outlier_pose_cache(gated_pose_cache, estimator)
    rejected_count += outlier_rejected_count
    completed, filled_count = replay_008_complete_pose_cache_temporally(outlier_gated_pose_cache, estimator)
    if not REPLAY_008_TEMPORAL_SMOOTHING_ENABLED:
        return (completed, filled_count, 0, rejected_count, reset_count)
    smoothed, smoothed_count = replay_008_smooth_pose_cache_temporally(completed, estimator)
    limited, limited_count = replay_008_limit_pose_cache_rotation_jumps(smoothed, estimator)
    return (limited, filled_count, smoothed_count + limited_count, rejected_count, reset_count)

def replay_008_make_pose_cache_key(*, frame_offsets: list[int], active_camera_names: list[str], cube_paths: list[Path], use_undistort: bool, adaptive_clahe: bool, shared_tag_detection: bool, enable_filter: bool, fast: bool, demo008: Any) -> dict[str, Any]:
    return {'format': REPLAY_008_POSE_CACHE_FORMAT, 'frame_count': len(frame_offsets), 'active_camera_names': list(active_camera_names), 'cube_paths': [str(path) for path in cube_paths], 'intrinsics_yaml': {name: CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[name] for name in active_camera_names}, 'use_undistort': bool(use_undistort), 'adaptive_clahe': bool(adaptive_clahe), 'image_recovery_version': int(REPLAY_008_IMAGE_RECOVERY_VERSION), 'shared_tag_detection': bool(shared_tag_detection), 'enable_filter': bool(enable_filter), 'fast': bool(fast), 'single_tag_continuity_gate_enabled': bool(REPLAY_008_SINGLE_TAG_CONTINUITY_GATE_ENABLED), 'single_tag_continuity_max_rotation_deg': float(REPLAY_008_SINGLE_TAG_CONTINUITY_MAX_ROTATION_DEG), 'single_tag_continuity_version': int(REPLAY_008_SINGLE_TAG_CONTINUITY_VERSION), 'temporal_outlier_gate_enabled': bool(REPLAY_008_TEMPORAL_OUTLIER_GATE_ENABLED), 'temporal_outlier_max_neighbor_gap_frames': int(REPLAY_008_TEMPORAL_OUTLIER_MAX_NEIGHBOR_GAP_FRAMES), 'temporal_outlier_neighbor_max_rotation_deg': float(REPLAY_008_TEMPORAL_OUTLIER_NEIGHBOR_MAX_ROTATION_DEG), 'temporal_outlier_neighbor_max_translation_mm': float(REPLAY_008_TEMPORAL_OUTLIER_NEIGHBOR_MAX_TRANSLATION_MM), 'temporal_outlier_min_rotation_jump_deg': float(REPLAY_008_TEMPORAL_OUTLIER_MIN_ROTATION_JUMP_DEG), 'temporal_outlier_min_translation_jump_mm': float(REPLAY_008_TEMPORAL_OUTLIER_MIN_TRANSLATION_JUMP_MM), 'temporal_outlier_version': int(REPLAY_008_TEMPORAL_OUTLIER_VERSION), 'temporal_fill_enabled': True, 'temporal_fill_max_gap_frames': int(REPLAY_008_TEMPORAL_FILL_MAX_GAP_FRAMES), 'temporal_fill_max_rotation_deg': float(REPLAY_008_TEMPORAL_FILL_MAX_ROTATION_DEG), 'temporal_fill_version': int(REPLAY_008_TEMPORAL_FILL_VERSION), 'temporal_smoothing_enabled': bool(REPLAY_008_TEMPORAL_SMOOTHING_ENABLED), 'temporal_smoothing_window_radius': int(REPLAY_008_TEMPORAL_SMOOTHING_WINDOW_RADIUS), 'temporal_smoothing_sigma_frames': float(REPLAY_008_TEMPORAL_SMOOTHING_SIGMA_FRAMES), 'temporal_smoothing_max_rotation_deg': float(REPLAY_008_TEMPORAL_SMOOTHING_MAX_ROTATION_DEG), 'temporal_smoothing_version': int(REPLAY_008_TEMPORAL_SMOOTHING_VERSION), 'temporal_rotation_jump_limit_enabled': bool(REPLAY_008_TEMPORAL_ROTATION_JUMP_LIMIT_ENABLED), 'temporal_rotation_jump_max_deg': float(REPLAY_008_TEMPORAL_ROTATION_JUMP_MAX_DEG), 'temporal_rotation_jump_hold_deg': float(REPLAY_008_TEMPORAL_ROTATION_JUMP_HOLD_DEG), 'temporal_rotation_jump_limit_version': int(REPLAY_008_TEMPORAL_ROTATION_JUMP_LIMIT_VERSION), 'fisheye_rectified_horizontal_fov_deg': None if CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG is None else float(CV2_CAPTURE_FISHEYE_RECTIFIED_HORIZONTAL_FOV_DEG)}

def replay_008_load_cached_pose_cache(pose_cache_record: dict[str, Any] | None, expected_key: dict[str, Any]) -> tuple[list[dict[str, Any]], bool] | None:
    if not isinstance(pose_cache_record, dict):
        return None
    if pose_cache_record.get('format') != REPLAY_008_POSE_CACHE_FORMAT:
        return None
    record_key = pose_cache_record.get('key')
    if isinstance(record_key, dict) and record_key.get('format') == REPLAY_008_POSE_CACHE_FORMAT_020_MULTISTAGE:
        pose_cache = pose_cache_record.get('pose_cache', None)
        if isinstance(pose_cache, list) and len(pose_cache) == int(expected_key['frame_count']):
            return (pose_cache, True)
        return None
    exact_match = record_key == expected_key
    if not exact_match and isinstance(record_key, dict):
        stable_record_key = {key: value for key, value in record_key.items() if key != 'frame_offsets'}
        stable_expected_key = {key: value for key, value in expected_key.items() if key != 'frame_offsets'}
        exact_match = stable_record_key == stable_expected_key
    compatible_without_temporal = False
    if not exact_match and isinstance(record_key, dict):
        temporal_keys = {'frame_offsets', 'single_tag_continuity_gate_enabled', 'single_tag_continuity_max_rotation_deg', 'single_tag_continuity_version', 'temporal_outlier_gate_enabled', 'temporal_outlier_max_neighbor_gap_frames', 'temporal_outlier_neighbor_max_rotation_deg', 'temporal_outlier_neighbor_max_translation_mm', 'temporal_outlier_min_rotation_jump_deg', 'temporal_outlier_min_translation_jump_mm', 'temporal_outlier_version', 'temporal_fill_enabled', 'temporal_fill_max_gap_frames', 'temporal_fill_max_rotation_deg', 'temporal_fill_version', 'temporal_smoothing_enabled', 'temporal_smoothing_window_radius', 'temporal_smoothing_sigma_frames', 'temporal_smoothing_max_rotation_deg', 'temporal_smoothing_version', 'temporal_rotation_jump_limit_enabled', 'temporal_rotation_jump_max_deg', 'temporal_rotation_jump_hold_deg', 'temporal_rotation_jump_limit_version'}
        stripped_record_key = {key: value for key, value in record_key.items() if key not in temporal_keys}
        stripped_expected_key = {key: value for key, value in expected_key.items() if key not in temporal_keys}
        compatible_without_temporal = stripped_record_key == stripped_expected_key
    if not exact_match and (not compatible_without_temporal):
        return None
    pose_cache = pose_cache_record.get('pose_cache', None)
    if not isinstance(pose_cache, list):
        return None
    if len(pose_cache) != int(expected_key['frame_count']):
        return None
    return (pose_cache, exact_match)

def replay_008_write_pose_cache_into_pkl_frames(pkl_path: Path, cache_key: dict[str, Any], pose_cache: list[dict[str, Any]]) -> None:
    tmp_path = pkl_path.with_name(f'.{pkl_path.name}.rewrite-{time.time_ns()}.tmp')
    frame_idx = 0
    try:
        with pkl_path.open('rb') as src, tmp_path.open('wb') as dst:
            while True:
                try:
                    record = pickle.load(src)
                except EOFError:
                    break
                if isinstance(record, dict) and record.get('type') == 'pose_cache':
                    continue
                if isinstance(record, dict) and record.get('type') == 'frame':
                    if frame_idx >= len(pose_cache):
                        raise ValueError(f'PKL has more frame records than pose cache entries: >{len(pose_cache)}')
                    record[REPLAY_008_INLINE_POSE_FRAME_FIELD] = pose_cache[frame_idx]
                    record[REPLAY_008_INLINE_POSE_CACHE_KEY_FIELD] = cache_key
                    frame_idx += 1
                pickle.dump(record, dst, protocol=pickle.HIGHEST_PROTOCOL)
        if frame_idx != len(pose_cache):
            raise ValueError(f'PKL frame count {frame_idx} does not match pose cache count {len(pose_cache)}')
        tmp_path.replace(pkl_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

def replay_008_main(args: Replay008ViewerConfig) -> None:
    demo008 = None
    pkl_path = replay_008_resolve_pkl_path(args.pkl_path)
    print(f'[INFO] PKL: {pkl_path}')
    print('[INFO] Building lightweight frame index. This scans the file once without retaining images.')
    header, frame_offsets, footer, pose_cache_record, inline_pose_cache_record = replay_008_build_frame_index(pkl_path)
    if not frame_offsets:
        raise ValueError(f'No frame records found in {pkl_path}')
    total_frames = len(frame_offsets)
    metadata = header.get('metadata', {}) if isinstance(header, dict) else {}
    first_record = replay_008_load_frame_at_offset(pkl_path, frame_offsets[0])
    first_record_camera_name = str(first_record.get('camera_name', ''))
    print(f'[INFO] Indexed frames: {total_frames}')
    if footer is not None:
        print(f"[INFO] Footer frame_count={footer.get('frame_count')} reason={footer.get('reason')}")
    if args.cameras:
        active_camera_names = [x.strip() for x in args.cameras.split(',') if x.strip()]
    else:
        active_camera_names = list(CV2_CAPTURE_ACTIVE_CAMERA_NAMES)
        if first_record_camera_name and len(active_camera_names) == 1 and (first_record_camera_name != active_camera_names[0]):
            config_camera_name = active_camera_names[0]
            active_camera_names = [first_record_camera_name]
            CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[first_record_camera_name] = CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[config_camera_name]
            print(f"[INFO] Historical PKL camera alias: recorded camera '{first_record_camera_name}' uses current 008 config '{config_camera_name}'.")
    missing_camera_configs = [name for name in active_camera_names if name not in CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML]
    if missing_camera_configs and len(CV2_CAPTURE_ACTIVE_CAMERA_NAMES) == 1:
        config_camera_name = CV2_CAPTURE_ACTIVE_CAMERA_NAMES[0]
        for camera_name in missing_camera_configs:
            CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[camera_name] = CV2_CAPTURE_CAMERA_TO_INTRINSICS_YAML[config_camera_name]
        print(f"[INFO] Historical PKL camera alias: {missing_camera_configs} use current 008 config '{config_camera_name}'.")
    cube_paths = [cv2_capture_validate_cube_path(Path(x.strip())) for x in args.cube_dirs.split(',') if x.strip()] if args.cube_dirs else [cv2_capture_validate_cube_path(Path(path)) for path in metadata.get('cube_paths') or CV2_CAPTURE_CUBE_CFG_DIRS]
    use_undistort = bool(CV2_CAPTURE_UNDISTORT_BEFORE_DETECTION) and (not args.no_undistort)
    adaptive_clahe = bool(CV2_CAPTURE_ADAPTIVE_CLAHE_DETECTION)
    enable_filter = bool(args.with_filter) and (not args.no_filter)
    fast = not args.slow
    estimator = Replay008PoseEstimator(demo008, active_camera_names=active_camera_names, cube_paths=cube_paths, use_undistort=use_undistort, adaptive_clahe=adaptive_clahe, shared_tag_detection=bool(args.shared_detect_tags), enable_filter=enable_filter, fast=fast)
    pose_cache_key = replay_008_make_pose_cache_key(frame_offsets=frame_offsets, active_camera_names=active_camera_names, cube_paths=cube_paths, use_undistort=use_undistort, adaptive_clahe=adaptive_clahe, shared_tag_detection=bool(args.shared_detect_tags), enable_filter=enable_filter, fast=fast, demo008=demo008)
    print(f"[INFO] 008 replay detection path: {('shared' if args.shared_detect_tags else 'per-cube')} detect_tags(frame) + per-cube process_detections(), sequential over PKL frames.")
    inline_cached_pose = replay_008_load_cached_pose_cache(inline_pose_cache_record, pose_cache_key)
    appended_cached_pose = replay_008_load_cached_pose_cache(pose_cache_record, pose_cache_key)
    cached_pose = inline_cached_pose if inline_cached_pose is not None else appended_cached_pose
    pose_cache_needs_write = inline_cached_pose is None
    if cached_pose is not None:
        pose_cache, cache_exact_match = cached_pose
        cache_source = 'inline frame records' if inline_cached_pose is not None else 'appended PKL cache'
        if cache_exact_match:
            print(f'[INFO] Loaded cached temporal-completed smoothed pose estimation from {cache_source}: frames={len(pose_cache)}')
        else:
            pose_cache, filled_count, smoothed_count, rejected_count, reset_count = replay_008_complete_and_smooth_pose_cache(pose_cache, estimator)
            pose_cache_needs_write = True
            print(f'[INFO] Loaded cached pose estimation from {cache_source} and applied single-tag gate + temporal completion+smoothing: frames={len(pose_cache)} reset={reset_count} rejected={rejected_count} filled={filled_count} smoothed={smoothed_count}')
    else:
        pose_cache = replay_008_precompute_pose_cache(pkl_path, frame_offsets, metadata, estimator)
        pose_cache, filled_count, smoothed_count, rejected_count, reset_count = replay_008_complete_and_smooth_pose_cache(pose_cache, estimator)
        pose_cache_needs_write = True
        print(f'[INFO] Applied single-tag gate + temporal completion+smoothing: reset={reset_count} rejected={rejected_count} filled={filled_count} smoothed={smoothed_count}')
    if pose_cache_needs_write:
        replay_008_write_pose_cache_into_pkl_frames(pkl_path, pose_cache_key, pose_cache)
        print(f'[INFO] Wrote temporal-completed smoothed pose estimation into ordered PKL frame records: frames={len(pose_cache)}')
    if args.precompute_only:
        print('[INFO] Precompute-only mode finished; exiting before starting Viser.')
        return
    first_raw_rgb = replay_008_bgr_to_rgb_for_viser(first_record['image_bgr'], int(args.max_width))
    first_detector_tagpose_bgr = estimator.draw_detector_input_pose_frame(first_record, pose_cache[0])
    first_detector_tagpose_rgb = replay_008_bgr_to_rgb_for_viser(first_detector_tagpose_bgr, int(args.max_width))
    first_undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(first_record, pose_cache[0])
    first_undistorted_debug_rgb = replay_008_bgr_to_rgb_for_viser(first_undistorted_debug_bgr, int(args.max_width))
    server = viser.ViserServer(host=args.host, port=int(args.port))
    scene_handles = replay_008_create_3d_scene_handles(server, estimator, pose_cache)
    replay_008_update_3d_scene(scene_handles, pose_cache[0])
    with server.gui.add_folder('Detector Input TagPose'):
        detector_tagpose_handle = server.gui.add_image(first_detector_tagpose_rgb, label='', format='jpeg', jpeg_quality=int(args.jpeg_quality))
        frame_slider = server.gui.add_slider('Frame', min=0, max=total_frames - 1, step=1, initial_value=0)
        auto_play_checkbox = server.gui.add_checkbox('Auto play', initial_value=False)
        status_text = server.gui.add_text('Status', initial_value=replay_008_record_summary(first_record, 0, total_frames), disabled=True)
        pose_text = server.gui.add_markdown(replay_008_pose_markdown(pose_cache[0]))
    with server.gui.add_folder('Undistorted Debug Image'):
        undistorted_debug_handle = server.gui.add_image(first_undistorted_debug_rgb, label='undistorted frame red-box on missing pose', format='jpeg', jpeg_quality=int(args.jpeg_quality))
    with server.gui.add_folder('Raw Image'):
        raw_image_handle = server.gui.add_image(first_raw_rgb, label='raw origin_frame_bgr', format='jpeg', jpeg_quality=int(args.jpeg_quality))
    with server.gui.add_folder('3D View'):
        show_box_checkbox = server.gui.add_checkbox('Cube box', initial_value=True)
        show_obj_checkbox = server.gui.add_checkbox('Finger OBJ', initial_value=True)
        show_axes_checkbox = server.gui.add_checkbox('Cube axes', initial_value=True)
        show_trajectory_checkbox = server.gui.add_checkbox('Trajectory', initial_value=True)
        show_samples_checkbox = server.gui.add_checkbox('Pose samples', initial_value=True)
        show_endpoints_checkbox = server.gui.add_checkbox('Start/end points', initial_value=True)
        show_camera_checkbox = server.gui.add_checkbox('Camera frustum', initial_value=True)
    with server.gui.add_folder('Replay Metadata'):
        server.gui.add_text('PKL', initial_value=str(pkl_path), disabled=True)
        if isinstance(metadata, dict):
            server.gui.add_markdown('\n'.join([f"`recorded_image`: `{metadata.get('recorded_image', 'unknown')}`", f"`capture_size`: `{metadata.get('capture_size', 'unknown')}`", f"`fps`: `{metadata.get('fps', 'unknown')}`", f"`fourcc`: `{metadata.get('fourcc', 'unknown')}`"]))
    print(f'[INFO] Viser: http://{args.host}:{int(args.port)}')
    print('[INFO] Use the sidebar folders: Detector Input TagPose, Undistorted Debug Image, Raw Image, Replay Metadata.')
    current_idx = -1
    last_auto_play_step = time.monotonic()
    while True:
        replay_008_apply_3d_visibility(scene_handles, show_box=bool(show_box_checkbox.value), show_obj=bool(show_obj_checkbox.value), show_axes=bool(show_axes_checkbox.value), show_trajectory=bool(show_trajectory_checkbox.value), show_samples=bool(show_samples_checkbox.value), show_endpoints=bool(show_endpoints_checkbox.value), show_grid=False, show_camera=bool(show_camera_checkbox.value))
        if bool(auto_play_checkbox.value):
            now = time.monotonic()
            if now - last_auto_play_step >= 0.1:
                frame_slider.value = (int(frame_slider.value) + 1) % total_frames
                last_auto_play_step = now
        else:
            last_auto_play_step = time.monotonic()
        slider_idx = int(frame_slider.value)
        if slider_idx != current_idx:
            try:
                record = replay_008_load_frame_at_offset(pkl_path, frame_offsets[slider_idx])
                detector_tagpose_bgr = estimator.draw_detector_input_pose_frame(record, pose_cache[slider_idx])
                detector_tagpose_handle.image = replay_008_bgr_to_rgb_for_viser(detector_tagpose_bgr, int(args.max_width))
                undistorted_debug_bgr = estimator.draw_undistorted_debug_frame(record, pose_cache[slider_idx])
                undistorted_debug_handle.image = replay_008_bgr_to_rgb_for_viser(undistorted_debug_bgr, int(args.max_width))
                raw_image_handle.image = replay_008_bgr_to_rgb_for_viser(record['image_bgr'], int(args.max_width))
                status_text.value = replay_008_record_summary(record, slider_idx, total_frames)
                pose_text.content = replay_008_pose_markdown(pose_cache[slider_idx])
                replay_008_update_3d_scene(scene_handles, pose_cache[slider_idx])
                replay_008_apply_3d_visibility(scene_handles, show_box=bool(show_box_checkbox.value), show_obj=bool(show_obj_checkbox.value), show_axes=bool(show_axes_checkbox.value), show_trajectory=bool(show_trajectory_checkbox.value), show_samples=bool(show_samples_checkbox.value), show_endpoints=bool(show_endpoints_checkbox.value), show_grid=False, show_camera=bool(show_camera_checkbox.value))
                current_idx = slider_idx
            except Exception as exc:
                status_text.value = f'Failed to load frame {slider_idx}: {type(exc).__name__}: {exc}'
                print(f'[WARNING] {status_text.value}')
                current_idx = slider_idx
        time.sleep(0.03)


# ---- RealSense recording and calibration helpers ----
# D435 color stream (SDK S/N 244222070135), calibrated at the native
# 1920x1080 resolution used by multi_cam_record_0716_180451.pkl.
REALSENSE_DEFAULT_INTRINSICS_YAML = Path('/home/ps/RobotCamCalib1/outputs/intrinsics_d435_color_charuco_1920x1080_0716_130910_offline_filtered.yaml')
REALSENSE_DEFAULT_CUBE_CFG = Path('/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/cubes/cube_april_36h11_100_123_2x2x2_outer62p5mm')
REALSENSE_PINHOLE_UNDISTORT_ALPHA = 0.0

def realsense_load_intrinsics_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().resolve()
    with yaml_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    dist = data.get('dist', data.get('D', None))
    if dist is None:
        dist = np.zeros(5, dtype=np.float64)
    return {'path': str(yaml_path), 'camera_model': str(data.get('camera_model', 'pinhole')), 'distortion_model': str(data.get('distortion_model', '')), 'image_size': tuple((int(v) for v in data['image_size'])), 'K': np.asarray(data['K'], dtype=np.float64).reshape(3, 3), 'dist': np.asarray(dist, dtype=np.float64).reshape(-1)}

def realsense_camera_matrix_to_intrinsic_dict(k: np.ndarray) -> dict[str, float]:
    return {'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2])}

def realsense_create_undistort_maps(calib: dict[str, Any], image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    camera_matrix = np.asarray(calib['K'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(calib.get('dist', np.zeros(5)), dtype=np.float64).reshape(-1)
    if dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return None
    if cv2_capture_is_fisheye_calib(calib):
        if dist_coeffs.size != 4:
            raise ValueError(
                f"Fisheye calibration requires 4 distortion coefficients, got {dist_coeffs.size}"
            )
        detection_camera_matrix = cv2_capture_compute_detection_camera_matrix(
            calib,
            image_size,
            undistort_before_detection=True,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs.reshape(4, 1),
            np.eye(3, dtype=np.float64),
            detection_camera_matrix,
            image_size,
            cv2.CV_16SC2,
        )
        return (map1, map2, detection_camera_matrix)
    detection_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, REALSENSE_PINHOLE_UNDISTORT_ALPHA, image_size)
    detection_camera_matrix = np.asarray(detection_camera_matrix, dtype=np.float64).reshape(3, 3)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3, dtype=np.float64), detection_camera_matrix, image_size, cv2.CV_16SC2)
    return (map1, map2, detection_camera_matrix)

def realsense_undistort_frame(frame: np.ndarray, undistort_pack: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
    if undistort_pack is None:
        return frame
    map1, map2, _new_camera_matrix = undistort_pack
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

def realsense_make_detector_input_vis(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    enhanced = preprocess_tag_image(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# ---- Strict AprilCube offline pose estimation ----
STRICT_APRILCUBE_THIS_FILE = Path(__file__).resolve()
STRICT_APRILCUBE_APRILCUBE_ROOT = STRICT_APRILCUBE_THIS_FILE.parent.parent
STRICT_APRILCUBE_DEFAULT_RECORDING_DIR = STRICT_APRILCUBE_APRILCUBE_ROOT / 'recordings'
STRICT_APRILCUBE_DEFAULT_PORT = 8094
STRICT_APRILCUBE_PLAYBACK_FPS = 15.0

def strict_aprilcube_resolve_pkl_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob('012_rs_raw_frames_*.pkl'))
        if not candidates:
            raise FileNotFoundError(f'No 012_rs_raw_frames_*.pkl found in directory: {path}')
        return candidates[-1]
    raise FileNotFoundError(f'Invalid pkl path: {path}')

def strict_aprilcube_build_stream_index(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    frame_offsets: list[int] = []
    footer: dict[str, Any] | None = None
    supported_formats = {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'}
    with path.open('rb') as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get('format') not in supported_formats:
            raise ValueError(f'Unsupported pkl format in {path}')
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get('type')
            if obj_type == 'frame':
                frame_offsets.append(offset)
            elif obj_type == 'footer':
                footer = obj
                break
    if not frame_offsets:
        raise ValueError(f'No frame records found in {path}')
    return (header, frame_offsets, footer)

def strict_aprilcube_load_frame_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(int(offset))
        record = pickle.load(f)
    if not isinstance(record, dict) or record.get('type') != 'frame':
        raise ValueError(f'Offset {offset} in {path} is not a frame record')
    return record

def strict_aprilcube_resize_bgr_if_needed(frame: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = image_size
    h, w = frame.shape[:2]
    if (w, h) == (target_w, target_h):
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

def strict_aprilcube_scale_for_gui(image_rgb: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    scale = float(max_width) / float(w)
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (max_width, out_h), interpolation=cv2.INTER_AREA)

def strict_aprilcube_bgr_to_rgb(image_bgr: np.ndarray, max_width: int=0) -> np.ndarray:
    image_bgr = np.asarray(image_bgr, dtype=np.uint8)
    if image_bgr.ndim == 2:
        image_rgb = np.repeat(image_bgr[:, :, None], 3, axis=2)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return strict_aprilcube_scale_for_gui(image_rgb, max_width)

def strict_aprilcube_rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
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
    return tuple((float(v) for v in quat))

def strict_aprilcube_rvec_to_wxyz(rvec: Any) -> tuple[float, float, float, float]:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return strict_aprilcube_rotation_matrix_to_wxyz(rot)

def strict_aprilcube_wxyz_to_rvec(wxyz: Any) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64).reshape(4)
    n = max(float(np.linalg.norm([w, x, y, z])), 1e-12)
    w, x, y, z = (w / n, x / n, y / n, z / n)
    rot = np.array([[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rot)
    return np.asarray(rvec, dtype=np.float64).reshape(3, 1)

def strict_aprilcube_slerp_wxyz(q0: Any, q1: Any, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 /= max(float(np.linalg.norm(q0)), 1e-12)
    q1 /= max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + float(alpha) * (q1 - q0)
        out /= max(float(np.linalg.norm(out)), 1e-12)
        return out
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * float(alpha)
    sin_theta = float(np.sin(theta))
    sin_theta_0 = float(np.sin(theta_0))
    s0 = float(np.cos(theta)) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

def strict_aprilcube_ndarray_to_list(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).tolist()

def strict_aprilcube_scalar_or_none(value: Any) -> float | int | bool | str | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)

def strict_aprilcube_sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    detections = []
    for item in result.get('detections', []) or []:
        if len(item) != 2:
            continue
        tag_id, corners = item
        detections.append({'tag_id': int(tag_id), 'corners_xy': strict_aprilcube_ndarray_to_list(np.asarray(corners, dtype=np.float64).reshape(4, 2))})
    per_tag = result.get('per_tag_reproj_error', {})
    if isinstance(per_tag, dict):
        per_tag_reproj_error = {int(k): float(v) for k, v in per_tag.items()}
    else:
        per_tag_reproj_error = {}
    return {'success': bool(result.get('success', False)), 'failure_reason': str(result.get('failure_reason', '')), 'n_tags': int(result.get('n_tags', 0)), 'n_inliers': int(result.get('n_inliers', 0)), 'reproj_error': float(result.get('reproj_error', float('inf'))), 'tag_ids': [int(v) for v in result.get('tag_ids', [])], 'visible_faces': sorted((str(v) for v in result.get('visible_faces', set()))), 'predicted': bool(result.get('predicted', False)), 'pose_source': str(result.get('pose_source', 'aprilcube_detector')), 'pose_filled': bool(result.get('pose_filled', False)), 'fill_original_failure_reason': str(result.get('fill_original_failure_reason', '')), 'fallback_original_failure_reason': str(result.get('fallback_original_failure_reason', '')), 'fallback_layout': str(result.get('fallback_layout', '')), 'single_tag_cfg_pose': bool(result.get('single_tag_cfg_pose', False)), 'single_tag_id': strict_aprilcube_scalar_or_none(result.get('single_tag_id', None)), 'single_tag_face': strict_aprilcube_scalar_or_none(result.get('single_tag_face', None)), 'rvec': strict_aprilcube_ndarray_to_list(result.get('rvec', None)), 'tvec': strict_aprilcube_ndarray_to_list(result.get('tvec', None)), 'T': strict_aprilcube_ndarray_to_list(result.get('T', None)), 'detections': detections, 'per_tag_reproj_error': per_tag_reproj_error, 'fallback_outlier_rejected_ids': [int(v) for v in result.get('fallback_outlier_rejected_ids', []) or []]}

def strict_aprilcube_encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    quality = int(max(1, min(int(quality), 100)))
    ok, encoded = cv2.imencode('.jpg', np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('cv2.imencode(.jpg) failed')
    return encoded.tobytes()

def strict_aprilcube_result_to_markdown(record: dict[str, Any], result: dict[str, Any], slider_idx: int) -> str:
    lines = [f'frame_index: `{slider_idx}`', f"loop_frame_idx: `{record.get('loop_frame_idx', '?')}`", f"camera: `{record.get('camera_name', record.get('device_name', '?'))}`", f"timestamp: `{record.get('capture_timestamp', None)}`"]
    if not result.get('success', False):
        tag_ids = result.get('tag_ids', [])
        lines.append(f"pose: `not detected`, tags=`{int(result.get('n_tags', 0))}`")
        lines.append(f"failure_reason: `{result.get('failure_reason', '')}`")
        lines.append(f'tag_ids: `{tag_ids}`')
        return '\n'.join(lines)
    tvec = np.asarray(result['tvec'], dtype=np.float64).reshape(3)
    faces = sorted(list(result.get('visible_faces', set())))
    lines.extend(['pose: `detected`', f't_mm: `({tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f})`', f"reproj_px: `{float(result.get('reproj_error', float('nan'))):.3f}`", f"tags: `{int(result.get('n_tags', 0))}`", f'faces: `{faces}`'])
    if result.get('predicted', False):
        lines.append('predicted: `true`')
    if result.get('single_tag_cfg_pose', False):
        lines.append(f"single_tag_cfg_pose: `id={result.get('single_tag_id', '?')}, face={result.get('single_tag_face', '?')}`")
    return '\n'.join(lines)

class StrictAprilCubeEstimator:

    def __init__(self, script012: Any, metadata: dict[str, Any], args: StrictAprilCubeEstimationConfig) -> None:
        self.script012 = script012
        self.metadata = metadata
        self.fallback_layout = str(args.fallback_layout)
        self.fallback_max_reproj = float(args.fallback_max_reproj)
        self.fallback_ransac_reproj = float(args.fallback_ransac_reproj)
        self.intrinsics_yaml = Path(args.intrinsics_yaml or metadata.get('intrinsics_yaml') or REALSENSE_DEFAULT_INTRINSICS_YAML).expanduser().resolve()
        self.cube_cfg = Path(args.cube_cfg or metadata.get('cube_cfg') or REALSENSE_DEFAULT_CUBE_CFG).expanduser().resolve()
        calib = realsense_load_intrinsics_yaml(self.intrinsics_yaml)
        self.image_size = tuple((int(v) for v in calib['image_size']))
        self.raw_camera_matrix = np.asarray(calib['K'], dtype=np.float64).reshape(3, 3)
        self.raw_dist_coeffs = np.asarray(calib['dist'], dtype=np.float64).reshape(-1)
        if args.intrinsics_yaml is None:
            if metadata.get('image_size', None) is not None:
                self.image_size = tuple((int(v) for v in metadata['image_size']))
            if metadata.get('raw_camera_matrix', None) is not None:
                self.raw_camera_matrix = np.asarray(metadata['raw_camera_matrix'], dtype=np.float64).reshape(3, 3)
            if metadata.get('raw_dist_coeffs', None) is not None:
                self.raw_dist_coeffs = np.asarray(metadata['raw_dist_coeffs'], dtype=np.float64).reshape(-1)
        self.undistort_pack = None
        self.detection_camera_matrix = self.raw_camera_matrix.copy()
        self.detector_dist_coeffs = self.raw_dist_coeffs
        should_undistort = bool(metadata.get('undistort_for_detection', True)) and (not bool(args.no_undistort))
        if should_undistort:
            self.undistort_pack = realsense_create_undistort_maps(calib, self.image_size)
            if self.undistort_pack is not None:
                self.detection_camera_matrix = self.undistort_pack[2]
                self.detector_dist_coeffs = np.zeros(5, dtype=np.float64)
            if args.intrinsics_yaml is None and metadata.get('detection_camera_matrix', None) is not None:
                self.detection_camera_matrix = np.asarray(metadata['detection_camera_matrix'], dtype=np.float64).reshape(3, 3)
            if args.intrinsics_yaml is None and metadata.get('detector_dist_coeffs', None) is not None:
                self.detector_dist_coeffs = np.asarray(metadata['detector_dist_coeffs'], dtype=np.float64).reshape(-1)
        self.detector = aprilcube.detector(self.cube_cfg, intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(self.detection_camera_matrix), dist_coeffs=self.detector_dist_coeffs, enable_filter=not bool(args.no_filter), fast=not bool(args.slow))
        self.fallback_tag_corner_map, self.fallback_face_id_sets = self._build_fallback_geometry()

    @property
    def cube_name(self) -> str:
        return self.cube_cfg.name if self.cube_cfg.is_dir() else self.cube_cfg.parent.name

    def _build_fallback_geometry(self) -> tuple[dict[int, np.ndarray], dict[str, set[int]]]:
        if self.fallback_layout == 'off':
            return ({}, {})
        cfg_path = self.cube_cfg / 'config.json' if self.cube_cfg.is_dir() else self.cube_cfg
        config, face_id_sets = aprilcube.load_cube_config(str(cfg_path))
        if self.fallback_layout == 'printed-pdf':
            all_ids = sorted({int(tag_id) for ids in face_id_sets.values() for tag_id in ids})
            tag_ids: list[int] = []
            new_face_sets: dict[str, set[int]] = {}
            cursor = 0
            for face_def in aprilcube.FACE_DEFS:
                face_name = str(face_def[0])
                face_rows, face_cols, _down_cells, _right_cells = config.face_layout(face_def)
                count = int(face_rows * face_cols)
                face_ids = all_ids[cursor:cursor + count]
                tag_ids.extend(face_ids)
                new_face_sets[face_name] = set(face_ids)
                cursor += count
            config.tag_ids = tag_ids
            config.tag_pattern_mirrored = False
            config.compute()
            face_id_sets = new_face_sets
        return (aprilcube.build_tag_corner_map(config), face_id_sets)

    def detection_frame(self, image_bgr: np.ndarray) -> np.ndarray:
        color = strict_aprilcube_resize_bgr_if_needed(image_bgr, self.image_size)
        return realsense_undistort_frame(color, self.undistort_pack)

    def _visible_faces_for_ids(self, tag_ids: list[int]) -> set[str]:
        visible: set[str] = set()
        for tag_id in tag_ids:
            for face_name, face_ids in self.fallback_face_id_sets.items():
                if int(tag_id) in face_ids:
                    visible.add(str(face_name))
        return visible

    def _face_normals_ok(self, rvec: np.ndarray, visible_faces: set[str]) -> bool:
        rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        for face_name in visible_faces:
            for face_def in aprilcube.FACE_DEFS:
                if str(face_def[0]) != str(face_name):
                    continue
                normal = np.zeros(3, dtype=np.float64)
                normal[int(face_def[1])] = float(face_def[2])
                normal_cam = rot @ normal
                if float(normal_cam[2]) > 0.0:
                    return False
                break
        return True

    def _fallback_points_from_detections(self, detections: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        object_chunks: list[np.ndarray] = []
        image_chunks: list[np.ndarray] = []
        used_ids: list[int] = []
        for tag_id_raw, corners_raw in detections:
            tag_id = int(tag_id_raw)
            corners_3d = self.fallback_tag_corner_map.get(tag_id)
            if corners_3d is None:
                continue
            corners_2d = np.asarray(corners_raw, dtype=np.float64).reshape(4, 2)
            object_chunks.append(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3))
            image_chunks.append(corners_2d)
            used_ids.append(tag_id)
        if not object_chunks:
            return (np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), [])
        return (np.vstack(object_chunks).astype(np.float64), np.vstack(image_chunks).astype(np.float64), used_ids)

    def _solve_fallback_global_pnp(self, detections: list[tuple[int, np.ndarray]]) -> dict[str, Any] | None:
        object_points, image_points, used_ids = self._fallback_points_from_detections(detections)
        if object_points.shape[0] < 4:
            return None
        inliers = None
        if object_points.shape[0] >= 6:
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=object_points, imagePoints=image_points, cameraMatrix=self.detection_camera_matrix, distCoeffs=self.detector_dist_coeffs, iterationsCount=200, reprojectionError=float(self.fallback_ransac_reproj), confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=object_points, imagePoints=image_points, cameraMatrix=self.detection_camera_matrix, distCoeffs=self.detector_dist_coeffs, iterationsCount=200, reprojectionError=float(self.fallback_ransac_reproj), confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            try:
                success, rvec, tvec = cv2.solvePnP(objectPoints=object_points, imagePoints=image_points, cameraMatrix=self.detection_camera_matrix, distCoeffs=self.detector_dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                success, rvec, tvec = cv2.solvePnP(objectPoints=object_points, imagePoints=image_points, cameraMatrix=self.detection_camera_matrix, distCoeffs=self.detector_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            inliers = np.arange(object_points.shape[0], dtype=np.int32).reshape(-1, 1) if success else None
        if not success or rvec is None or tvec is None:
            return None
        if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
            return None
        if inliers is not None and len(inliers) >= 4:
            idx = np.asarray(inliers, dtype=np.int32).reshape(-1)
            try:
                rvec, tvec = cv2.solvePnPRefineLM(objectPoints=object_points[idx], imagePoints=image_points[idx], cameraMatrix=self.detection_camera_matrix, distCoeffs=self.detector_dist_coeffs, rvec=rvec, tvec=tvec)
            except cv2.error:
                pass
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, self.detection_camera_matrix, self.detector_dist_coeffs)
        projected = projected.reshape(-1, 2)
        corner_errors = np.linalg.norm(image_points - projected, axis=1)
        per_tag_reproj_error: dict[int, float] = {}
        for idx, tag_id in enumerate(used_ids):
            start = idx * 4
            end = start + 4
            per_tag_reproj_error[int(tag_id)] = float(np.mean(corner_errors[start:end]))
        visible_faces = self._visible_faces_for_ids(used_ids)
        if not self._face_normals_ok(np.asarray(rvec, dtype=np.float64).reshape(3, 1), visible_faces):
            return None
        return {'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'reproj_error': float(np.mean(corner_errors)), 'n_inliers': 0 if inliers is None else int(len(inliers)), 'used_ids': used_ids, 'visible_faces': visible_faces, 'per_tag_reproj_error': per_tag_reproj_error}

    def _fallback_pose(self, result: dict[str, Any]) -> dict[str, Any] | None:
        if not self.fallback_tag_corner_map:
            return None
        detections = [(int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2)) for tag_id, corners in result.get('detections', []) or [] if int(tag_id) in self.fallback_tag_corner_map]
        if not detections:
            return None
        solved = self._solve_fallback_global_pnp(detections)
        if solved is None:
            return None
        outlier_rejected_ids: list[int] = []
        if len(detections) >= 3:
            per_tag = solved['per_tag_reproj_error']
            per_tag_values = np.asarray([per_tag[int(tag_id)] for tag_id, _ in detections], dtype=np.float64)
            median_err = float(np.median(per_tag_values))
            tag_reproj_thresh = max(median_err * 3.0, 2.0)
            keep = [idx for idx, (tag_id, _corners) in enumerate(detections) if float(per_tag[int(tag_id)]) <= tag_reproj_thresh]
            if len(keep) < len(detections) and len(keep) >= 1:
                outlier_rejected_ids = [int(tag_id) for idx, (tag_id, _corners) in enumerate(detections) if idx not in keep]
                detections = [detections[idx] for idx in keep]
                solved = self._solve_fallback_global_pnp(detections)
                if solved is None:
                    return None
        reproj_error = float(solved['reproj_error'])
        if reproj_error > self.fallback_max_reproj:
            return None
        rvec = np.asarray(solved['rvec'], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(solved['tvec'], dtype=np.float64).reshape(3, 1)
        used_ids = [int(v) for v in solved['used_ids']]
        rot, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = tvec.reshape(3)
        fallback = dict(result)
        fallback.update({'success': True, 'failure_reason': '', 'pose_source': f'fallback_pnp_{self.fallback_layout}_aprilcube_style', 'rvec': rvec, 'tvec': tvec, 'T': transform, 'reproj_error': reproj_error, 'n_inliers': int(solved['n_inliers']), 'n_tags': len(used_ids), 'visible_faces': solved['visible_faces'], 'tag_ids': used_ids, 'detections': detections, 'per_tag_reproj_error': solved['per_tag_reproj_error'], 'fallback_outlier_rejected_ids': outlier_rejected_ids, 'fallback_original_failure_reason': str(result.get('failure_reason', '')), 'fallback_layout': self.fallback_layout})
        return fallback

    def process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        image_bgr = np.asarray(record['image_bgr'], dtype=np.uint8)
        detect_frame = self.detection_frame(image_bgr)
        timestamp = record.get('capture_timestamp', None)
        result = self.detector.process_frame(detect_frame, timestamp=None if timestamp is None else float(timestamp))
        detector_success = bool(result.get('success', False))
        detector_reproj = float(result.get('reproj_error', float('inf')))
        detector_n_tags = int(result.get('n_tags', 0) or 0)
        detector_usable = detector_success and detector_n_tags > 0 and np.isfinite(detector_reproj) and (detector_reproj <= self.fallback_max_reproj)
        if detector_usable:
            result = dict(result)
            result['pose_source'] = 'aprilcube_detector'
        else:
            rejected_reason = '' if not detector_success else 'detector_no_tags' if detector_n_tags <= 0 else f'detector_reproj_rejected:{detector_reproj:.2f}>{self.fallback_max_reproj:.2f}'
            fallback_seed = dict(result)
            if rejected_reason:
                fallback_seed['failure_reason'] = rejected_reason
            fallback = self._fallback_pose(fallback_seed)
            if fallback is not None:
                result = fallback
            else:
                result = fallback_seed
                result['success'] = False
        return {'success': bool(result.get('success', False)), 'n_tags': int(result.get('n_tags', 0)), 'result': result}

    def overlay_image(self, record: dict[str, Any], result: dict[str, Any]) -> np.ndarray:
        detect_frame = self.detection_frame(np.asarray(record['image_bgr'], dtype=np.uint8))
        vis = realsense_make_detector_input_vis(detect_frame)
        return self.detector.draw_result(vis, result)


def strict_aprilcube_precompute_pose_cache(pkl_path: Path, offsets: list[int], estimator: StrictAprilCubeEstimator) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    total = len(offsets)
    t0 = time.perf_counter()
    for idx, offset in enumerate(offsets):
        record = strict_aprilcube_load_frame_at(pkl_path, offset)
        pose = estimator.process_record(record)
        cache.append(pose)
        done = idx + 1
        if done == total or done % 10 == 0:
            elapsed = time.perf_counter() - t0
            fps = done / max(elapsed, 1e-09)
            print(f"\r[INFO] Offline pose detection {done}/{total} success={sum((int(v['success']) for v in cache))} fps={fps:.1f}", end='', flush=True)
    print()
    return cache

def strict_aprilcube_fill_missing_pose_cache(pose_cache: list[dict[str, Any]]) -> int:
    good_indices = [idx for idx, item in enumerate(pose_cache) if item['result'].get('success', False) and item['result'].get('rvec', None) is not None and (item['result'].get('tvec', None) is not None)]
    if not good_indices:
        return 0

    def make_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
        return transform

    def fill_one(idx: int, source: str, rvec: np.ndarray, tvec: np.ndarray) -> None:
        result = pose_cache[idx]['result']
        filled = dict(result)
        filled.update({'success': True, 'pose_source': source, 'pose_filled': True, 'fill_original_failure_reason': str(result.get('failure_reason', '')), 'failure_reason': '', 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': make_transform(rvec, tvec), 'reproj_error': float('inf'), 'n_inliers': 0})
        pose_cache[idx]['result'] = filled
        pose_cache[idx]['success'] = True
    filled_count = 0
    first_good = good_indices[0]
    first_result = pose_cache[first_good]['result']
    for idx in range(0, first_good):
        fill_one(idx, 'filled_next_pose', np.asarray(first_result['rvec'], dtype=np.float64).reshape(3, 1), np.asarray(first_result['tvec'], dtype=np.float64).reshape(3, 1))
        filled_count += 1
    for left_idx, right_idx in zip(good_indices[:-1], good_indices[1:]):
        if right_idx <= left_idx + 1:
            continue
        left = pose_cache[left_idx]['result']
        right = pose_cache[right_idx]['result']
        left_t = np.asarray(left['tvec'], dtype=np.float64).reshape(3)
        right_t = np.asarray(right['tvec'], dtype=np.float64).reshape(3)
        left_q = np.asarray(strict_aprilcube_rvec_to_wxyz(left['rvec']), dtype=np.float64)
        right_q = np.asarray(strict_aprilcube_rvec_to_wxyz(right['rvec']), dtype=np.float64)
        gap = right_idx - left_idx
        for idx in range(left_idx + 1, right_idx):
            alpha = float(idx - left_idx) / float(gap)
            tvec = ((1.0 - alpha) * left_t + alpha * right_t).reshape(3, 1)
            rvec = strict_aprilcube_wxyz_to_rvec(strict_aprilcube_slerp_wxyz(left_q, right_q, alpha))
            fill_one(idx, 'filled_interpolated_pose', rvec, tvec)
            filled_count += 1
    last_good = good_indices[-1]
    last_result = pose_cache[last_good]['result']
    for idx in range(last_good + 1, len(pose_cache)):
        fill_one(idx, 'filled_previous_pose', np.asarray(last_result['rvec'], dtype=np.float64).reshape(3, 1), np.asarray(last_result['tvec'], dtype=np.float64).reshape(3, 1))
        filled_count += 1
    return filled_count

def strict_aprilcube_default_output_pkl_path(source_pkl: Path) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    return source_pkl.with_name(f'014_offline_pose_vis_{source_pkl.stem}_{stamp}.pkl')

def strict_aprilcube_write_processed_pkl(*, source_pkl: Path, output_pkl: Path, header: dict[str, Any], footer: dict[str, Any] | None, offsets: list[int], estimator: StrictAprilCubeEstimator, pose_cache: list[dict[str, Any]], jpeg_quality: int, save_raw_jpeg: bool) -> None:
    output_pkl = output_pkl.expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    total = len(offsets)
    t0 = time.perf_counter()
    success_count = sum((int(item['success']) for item in pose_cache))
    with output_pkl.open('wb') as f:
        pickle.dump({'type': 'header', 'format': 'aprilcube_012_offline_pose_vis_stream_v1', 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_pkl': str(source_pkl), 'source_format': header.get('format', ''), 'source_metadata': header.get('metadata', {}), 'source_footer': footer, 'metadata': {'script': str(STRICT_APRILCUBE_THIS_FILE), 'intrinsics_yaml': str(estimator.intrinsics_yaml), 'cube_cfg': str(estimator.cube_cfg), 'image_size': tuple((int(v) for v in estimator.image_size)), 'detection_camera_matrix': estimator.detection_camera_matrix.tolist(), 'detector_dist_coeffs': estimator.detector_dist_coeffs.tolist(), 'undistort_for_detection': estimator.undistort_pack is not None, 'jpeg_quality': int(jpeg_quality), 'contains_raw_jpeg': bool(save_raw_jpeg), 'fallback_layout': estimator.fallback_layout, 'fallback_max_reproj': float(estimator.fallback_max_reproj), 'fallback_ransac_reproj': float(estimator.fallback_ransac_reproj), 'fill_missing_pose': any((bool(item['result'].get('pose_filled', False)) for item in pose_cache)), 'filled_pose_count': int(sum((bool(item['result'].get('pose_filled', False)) for item in pose_cache))), 'frame_count': int(total), 'success_count': int(success_count)}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx, offset in enumerate(offsets):
            record = strict_aprilcube_load_frame_at(source_pkl, offset)
            result = pose_cache[idx]['result']
            overlay_bgr = estimator.overlay_image(record, result)
            frame_record = {'type': 'frame', 'frame_index': int(idx), 'source_offset': int(offset), 'camera_name': str(record.get('camera_name', record.get('device_name', ''))), 'device_name': str(record.get('device_name', '')), 'loop_frame_idx': int(record.get('loop_frame_idx', idx)), 'capture_timestamp': record.get('capture_timestamp', None), 'source_shape': tuple((int(v) for v in np.asarray(record['image_bgr']).shape)), 'overlay_shape': tuple((int(v) for v in overlay_bgr.shape)), 'overlay_format': 'jpeg_bgr', 'overlay_jpeg': strict_aprilcube_encode_bgr_jpeg(overlay_bgr, jpeg_quality), 'pose': strict_aprilcube_sanitize_result(result)}
            if save_raw_jpeg:
                frame_record['raw_format'] = 'jpeg_bgr'
                frame_record['raw_jpeg'] = strict_aprilcube_encode_bgr_jpeg(record['image_bgr'], jpeg_quality)
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
            done = idx + 1
            if done == total or done % 10 == 0:
                elapsed = time.perf_counter() - t0
                fps = done / max(elapsed, 1e-09)
                print(f'\r[INFO] Writing processed pkl {done}/{total} success={success_count}/{total} fps={fps:.1f}', end='', flush=True)
        pickle.dump({'type': 'footer', 'frame_count': int(total), 'success_count': int(success_count), 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()
    print(f'[INFO] Saved processed pose visualization pkl: {output_pkl}')

def strict_aprilcube_add_optional_cube_mesh(server: viser.ViserServer, cube_cfg: Path) -> None:
    cube_dir = cube_cfg if cube_cfg.is_dir() else cube_cfg.parent
    obj_path = cube_dir / 'mujoco' / 'cube.obj'
    if not obj_path.exists():
        return
    try:
        mesh = trimesh.load(str(obj_path))
        server.scene.add_mesh_trimesh('/cube/mesh', mesh)
    except Exception as exc:
        print(f'[WARNING] Could not add cube mesh to viser: {type(exc).__name__}: {exc}')

def strict_aprilcube_update_cube_handle(cube_handle: Any, result: dict[str, Any]) -> None:
    if not result.get('success', False):
        cube_handle.visible = False
        return
    cube_handle.visible = True
    cube_handle.position = tuple((float(v) for v in np.asarray(result['tvec'], dtype=np.float64).reshape(3) / 1000.0))
    cube_handle.wxyz = strict_aprilcube_rvec_to_wxyz(result['rvec'])

def strict_aprilcube_main(args: StrictAprilCubeEstimationConfig) -> None:
    pkl_path = strict_aprilcube_resolve_pkl_path(args.pkl_path)
    header, offsets, footer = strict_aprilcube_build_stream_index(pkl_path)
    metadata = dict(header.get('metadata', {}))
    if header.get('format') == 'aprilcube_012_raw_with_pose_stream_v1':
        metadata = dict(header.get('raw_header', {}).get('metadata', metadata))
    if args.max_frames > 0:
        offsets = offsets[:int(args.max_frames)]
    estimator = StrictAprilCubeEstimator(None, metadata, args)
    pose_cache = strict_aprilcube_precompute_pose_cache(pkl_path, offsets, estimator)
    filled_count = 0
    if not args.no_fill_missing_pose:
        filled_count = strict_aprilcube_fill_missing_pose_cache(pose_cache)
    success_count = sum((int(item['success']) for item in pose_cache))
    print(f'[INFO] pkl={pkl_path}')
    print(f'[INFO] frames={len(offsets)} footer={footer}')
    print(f'[INFO] intrinsics_yaml={estimator.intrinsics_yaml}')
    print(f'[INFO] cube_cfg={estimator.cube_cfg}')
    print(f'[INFO] offline pose success={success_count}/{len(pose_cache)} filled={filled_count}')
    if args.output_pkl is not None:
        output_pkl = args.output_pkl
        if str(output_pkl) == 'auto':
            output_pkl = strict_aprilcube_default_output_pkl_path(pkl_path)
        strict_aprilcube_write_processed_pkl(source_pkl=pkl_path, output_pkl=Path(output_pkl), header=header, footer=footer, offsets=offsets, estimator=estimator, pose_cache=pose_cache, jpeg_quality=int(args.jpeg_quality), save_raw_jpeg=bool(args.save_raw_jpeg))
        if not args.show_viser:
            return
    if args.precompute_only:
        return
    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction('-y')
    server.scene.world_axes.visible = False
    server.gui.set_panel_label('AprilCube 012 PKL')
    server.scene.add_frame('/camera', wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), axes_length=0.05, axes_radius=0.002, origin_radius=0.0)
    cube_handle = server.scene.add_frame('/cube', axes_length=0.04, axes_radius=0.0015, origin_radius=0.002, visible=False)
    strict_aprilcube_add_optional_cube_mesh(server, estimator.cube_cfg)
    frame_idx = 0
    is_playing = len(offsets) > 1
    loop_playback = True
    last_step_time = time.monotonic()
    with server.gui.add_folder('Replay Controls'):
        play_checkbox = server.gui.add_checkbox('Play', initial_value=is_playing)
        loop_checkbox = server.gui.add_checkbox('Loop', initial_value=loop_playback)
        frame_slider = server.gui.add_slider('Frame', min=0, max=len(offsets) - 1, step=1, initial_value=0)
        status_text = server.gui.add_text('Status', initial_value='', disabled=True)
    with server.gui.add_folder('Images'):
        raw_image_handle = server.gui.add_image(np.zeros((120, 160, 3), dtype=np.uint8), label='Raw BGR frame', format='jpeg', jpeg_quality=80)
        overlay_image_handle = server.gui.add_image(np.zeros((120, 160, 3), dtype=np.uint8), label='Offline detection', format='jpeg', jpeg_quality=80)
    pose_markdown = server.gui.add_markdown('')
    server.gui.add_markdown('\n'.join([f'pkl: `{pkl_path}`', f'frames: `{len(offsets)}`', f'success: `{success_count}/{len(pose_cache)}`', f'intrinsics: `{estimator.intrinsics_yaml}`', f'cube_cfg: `{estimator.cube_cfg}`', f'undistort_for_detection: `{estimator.undistort_pack is not None}`']))

    def clamp_index(value: int) -> int:
        return max(0, min(int(value), len(offsets) - 1))

    def render_frame(idx: int) -> None:
        record = strict_aprilcube_load_frame_at(pkl_path, offsets[idx])
        result = pose_cache[idx]['result']
        raw_image_handle.image = strict_aprilcube_bgr_to_rgb(record['image_bgr'], args.max_width)
        overlay = estimator.overlay_image(record, result)
        overlay_image_handle.image = strict_aprilcube_bgr_to_rgb(overlay, args.max_width)
        strict_aprilcube_update_cube_handle(cube_handle, result)
        pose_markdown.content = strict_aprilcube_result_to_markdown(record, result, idx)
        status_text.value = f"{idx + 1}/{len(offsets)} success={bool(result.get('success', False))} tags={int(result.get('n_tags', 0))}"

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
        frame_idx = clamp_index(int(frame_slider.value))
        last_step_time = time.monotonic()
        render_frame(frame_idx)
    render_frame(frame_idx)
    print(f'[INFO] Viser server started: http://{args.host}:{args.port}')
    while True:
        if is_playing and len(offsets) > 1:
            now = time.monotonic()
            if now - last_step_time >= 1.0 / max(float(args.fps), 1e-06):
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
                render_frame(frame_idx)
                last_step_time = now
        time.sleep(0.005)


# ---- Pose result visualization ----
POSE_VIEWER_APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
POSE_VIEWER_DEFAULT_PKL = POSE_VIEWER_APRILCUBE_ROOT / 'recordings' / '012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'
POSE_VIEWER_SUPPORTED_FORMATS = {'aprilcube_012_offline_pose_vis_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1', 'aprilcube_raw_with_020_postprocessed_pose_stream_v1', 'aprilcube_deeptag_fused_stream_v1', 'deeptag_012_offline_stream_v1'}

def pose_viewer_build_stream_index(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get('format') not in POSE_VIEWER_SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported pkl format: {header.get('format', None)}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if obj.get('type') == 'frame':
                offsets.append(offset)
            elif obj.get('type') == 'footer':
                footer = obj
                break
    if not offsets:
        raise ValueError(f'No frame records found in {path}')
    return (header, offsets, footer)

def pose_viewer_load_frame(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get('type') != 'frame':
        raise ValueError(f'Offset {offset} is not a frame record')
    return obj

def pose_viewer_decode_jpeg_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Failed to decode JPEG image')
    return image

def pose_viewer_bgr_to_rgb(image_bgr: np.ndarray, max_width: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    if max_width <= 0:
        return image_rgb
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    scale = float(max_width) / float(w)
    return cv2.resize(image_rgb, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

def pose_viewer_rotation_matrix_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
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
    return tuple((float(v) for v in quat))

def pose_viewer_rvec_to_wxyz(rvec: Any) -> tuple[float, float, float, float]:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return pose_viewer_rotation_matrix_to_wxyz(rot)

def pose_viewer_pose_markdown(frame: dict[str, Any]) -> str:
    pose = frame.get('pose', {})
    lines = [f"frame_index: `{frame.get('frame_index', '?')}`", f"loop_frame_idx: `{frame.get('loop_frame_idx', '?')}`", f"camera: `{frame.get('camera_name', '')}`", f"timestamp: `{frame.get('capture_timestamp', None)}`", f"success: `{pose.get('success', False)}`", f"pose_source: `{pose.get('pose_source', '')}`", f"quality_level: `{pose.get('quality_level', '')}`", f"quality_reason: `{pose.get('quality_reason', '')}`", f"pose_filled: `{pose.get('pose_filled', False)}`", f"reproj_error: `{pose.get('reproj_error', None)}`", f"n_tags: `{pose.get('n_tags', 0)}`", f"visible_faces: `{pose.get('visible_faces', [])}`"]
    tvec = pose.get('tvec', None)
    if tvec is not None:
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        lines.append(f't_mm: `({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f})`')
    if pose.get('fill_original_failure_reason', ''):
        lines.append(f"fill_original_failure_reason: `{pose['fill_original_failure_reason']}`")
    return '\n'.join(lines)

def pose_viewer_update_cube(cube_handle: Any, frame: dict[str, Any]) -> None:
    pose = frame.get('pose', {})
    if not pose.get('success', False) or pose.get('rvec') is None or pose.get('tvec') is None:
        cube_handle.visible = False
        return
    cube_handle.visible = True
    cube_handle.position = tuple((float(v) for v in np.asarray(pose['tvec'], dtype=np.float64).reshape(3) / 1000.0))
    cube_handle.wxyz = pose_viewer_rvec_to_wxyz(pose['rvec'])

def pose_viewer_main(args: PoseVisualizationConfig) -> None:
    pkl_path = Path(args.pkl_path).expanduser().resolve()
    header, offsets, footer = pose_viewer_build_stream_index(pkl_path)
    server = viser.ViserServer(host=args.host, port=int(args.port))
    server.scene.set_up_direction('-y')
    server.scene.world_axes.visible = False
    server.gui.set_panel_label('AprilCube Pose PKL')
    server.scene.add_frame('/camera', wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0), axes_length=0.05, axes_radius=0.002, origin_radius=0.0)
    cube_handle = server.scene.add_frame('/cube', axes_length=0.04, axes_radius=0.0015, origin_radius=0.002, visible=False)
    frame_idx = 0
    is_playing = len(offsets) > 1
    loop_playback = True
    last_step_time = time.monotonic()
    with server.gui.add_folder('Replay'):
        play_checkbox = server.gui.add_checkbox('Play', initial_value=is_playing)
        loop_checkbox = server.gui.add_checkbox('Loop', initial_value=loop_playback)
        frame_slider = server.gui.add_slider('Frame', min=0, max=len(offsets) - 1, step=1, initial_value=0)
        status_text = server.gui.add_text('Status', initial_value='', disabled=True)
    with server.gui.add_folder('Images'):
        overlay_handle = server.gui.add_image(np.zeros((120, 160, 3), dtype=np.uint8), label='Overlay', format='jpeg', jpeg_quality=80)
    pose_text = server.gui.add_markdown('')
    server.gui.add_markdown('\n'.join([f'pkl: `{pkl_path}`', f'frames: `{len(offsets)}`', f"format: `{header.get('format', '')}`", f'footer: `{footer}`']))

    def clamp_idx(value: int) -> int:
        return max(0, min(int(value), len(offsets) - 1))

    def render(idx: int) -> None:
        frame = pose_viewer_load_frame(pkl_path, offsets[idx])
        overlay_bgr = pose_viewer_decode_jpeg_bgr(frame['overlay_jpeg'])
        overlay_handle.image = pose_viewer_bgr_to_rgb(overlay_bgr, int(args.max_width))
        pose_viewer_update_cube(cube_handle, frame)
        pose_text.content = pose_viewer_pose_markdown(frame)
        pose = frame.get('pose', {})
        status_text.value = f"{idx + 1}/{len(offsets)} source={pose.get('pose_source', '')} filled={pose.get('pose_filled', False)}"

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
    print(f'[INFO] Loaded {pkl_path} frames={len(offsets)}')
    print(f'[INFO] Viser server: http://localhost:{args.port}')
    while True:
        if is_playing and len(offsets) > 1:
            now = time.monotonic()
            if now - last_step_time >= 1.0 / max(float(args.fps), 1e-06):
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


# ---- DeepTag keypoint detection ----
DEEPTAG_DETECTION_THIS_FILE = Path(__file__).resolve()
DEEPTAG_DETECTION_APRILCUBE_ROOT = DEEPTAG_DETECTION_THIS_FILE.parent.parent
DEEPTAG_DETECTION_DEEPTAG_ROOT = DEEPTAG_DETECTION_APRILCUBE_ROOT / 'thirdparty' / 'deeptag-pytorch'
DEEPTAG_DETECTION_DEFAULT_INPUT_PKL = DEEPTAG_DETECTION_APRILCUBE_ROOT / 'recordings' / '012_rs_raw_frames_20260710_214336.pkl'
DEEPTAG_DETECTION_DEFAULT_MERGED_INPUT_PKL = DEEPTAG_DETECTION_APRILCUBE_ROOT / 'recordings' / '012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'
DEEPTAG_DETECTION_DEFAULT_OUTPUT_PKL = DEEPTAG_DETECTION_APRILCUBE_ROOT / 'recordings' / '016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl'

DEEPTAG_DETECTION_SUPPORTED_INPUT_FORMATS = {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'}

def deeptag_detection_build_stream_index(path: Path) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get('format') not in DEEPTAG_DETECTION_SUPPORTED_INPUT_FORMATS:
            raise ValueError(f"Unsupported pkl format: {header.get('format', None)}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if obj.get('type') == 'frame':
                offsets.append(offset)
            elif obj.get('type') == 'footer':
                footer = obj
                break
    if not offsets:
        raise ValueError(f'No frame records found in {path}')
    return (header, offsets, footer)

def deeptag_detection_load_frame_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get('type') != 'frame':
        raise ValueError(f'Offset {offset} is not a frame')
    return obj

def deeptag_detection_input_metadata(header: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_header = header.get('raw_header', {})
    pose_header = header.get('pose_header', {})
    if isinstance(raw_header, dict):
        metadata.update(raw_header.get('metadata', {}) or {})
    if isinstance(pose_header, dict):
        metadata.update(pose_header.get('metadata', {}) or {})
    metadata.update(header.get('metadata', {}) or {})
    return metadata

def deeptag_detection_encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode('.jpg', np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(quality, 100)))])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def deeptag_detection_load_deeptag_engine(*, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, tag_size_m: float, args: DeepTagDetectionConfig) -> Any:
    if not DEEPTAG_DETECTION_DEEPTAG_ROOT.exists():
        raise FileNotFoundError(f'DeepTag repo not found: {DEEPTAG_DETECTION_DEEPTAG_ROOT}')
    old_cwd = Path.cwd()
    os.chdir(DEEPTAG_DETECTION_DEEPTAG_ROOT)
    try:
        load_deeptag_models = importlib.import_module(
            "deeptag_model_setting"
        ).load_deeptag_models
        load_marker_codebook = importlib.import_module(
            "marker_dict_setting"
        ).load_marker_codebook
        detection_engine_class = importlib.import_module(
            "stag_decode.detection_engine"
        ).DetectionEngine
        device = 'cpu' if args.cpu else None
        model_detector, model_decoder, device, tag_type, grid_size_cand_list = load_deeptag_models('apriltag', device)
        codebook = load_marker_codebook(str(DEEPTAG_DETECTION_DEEPTAG_ROOT / 'codebook' / 'apriltag_codebook.txt'), tag_type)
        engine = detection_engine_class(model_detector, model_decoder, device, tag_type, grid_size_cand_list, stg2_iter_num=int(args.stg2_iter_num), min_center_score=float(args.min_center_score), min_corner_score=float(args.min_corner_score), batch_size_stg2=int(args.batch_size_stg2), hamming_dist=int(args.hamming_dist), cameraMatrix=np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3), distCoeffs=np.asarray(dist_coeffs, dtype=np.float32).reshape(-1), codebook=codebook, tag_real_size_in_meter_dict={-1: float(tag_size_m)})
        return engine
    finally:
        os.chdir(old_cwd)

def deeptag_detection_make_runtime(script012: Any, metadata: dict[str, Any], args: DeepTagDetectionConfig) -> dict[str, Any]:
    intrinsics_yaml = Path(args.intrinsics_yaml or metadata.get('intrinsics_yaml') or REALSENSE_DEFAULT_INTRINSICS_YAML).expanduser().resolve()
    cube_cfg = Path(args.cube_cfg or metadata.get('cube_cfg') or REALSENSE_DEFAULT_CUBE_CFG).expanduser().resolve()
    calib = realsense_load_intrinsics_yaml(intrinsics_yaml)
    image_size = tuple((int(v) for v in metadata.get('image_size', calib['image_size'])))
    raw_camera_matrix = np.asarray(metadata.get('raw_camera_matrix', calib['K']), dtype=np.float64).reshape(3, 3)
    raw_dist_coeffs = np.asarray(metadata.get('raw_dist_coeffs', calib['dist']), dtype=np.float64).reshape(-1)
    undistort_pack = None
    detection_camera_matrix = raw_camera_matrix.copy()
    detector_dist_coeffs = raw_dist_coeffs
    if bool(metadata.get('undistort_for_detection', True)) and (not args.no_undistort):
        undistort_pack = realsense_create_undistort_maps(calib, image_size)
        if undistort_pack is not None:
            detection_camera_matrix = undistort_pack[2]
            detector_dist_coeffs = np.zeros(5, dtype=np.float64)
        if metadata.get('detection_camera_matrix', None) is not None:
            detection_camera_matrix = np.asarray(metadata['detection_camera_matrix'], dtype=np.float64).reshape(3, 3)
        if metadata.get('detector_dist_coeffs', None) is not None:
            detector_dist_coeffs = np.asarray(metadata['detector_dist_coeffs'], dtype=np.float64).reshape(-1)
    cube_config, face_id_sets = aprilcube.load_cube_config(str(cube_cfg / 'config.json' if cube_cfg.is_dir() else cube_cfg))
    tag_corner_map = aprilcube.build_tag_corner_map(cube_config)
    april_post_detector = aprilcube.detector(cube_cfg, intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(detection_camera_matrix), dist_coeffs=detector_dist_coeffs, enable_filter=False, fast=True)
    april_post_detector.draw_result = lambda frame, result: frame
    april_draw_detector = aprilcube.detector(cube_cfg, intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(detection_camera_matrix), dist_coeffs=detector_dist_coeffs, enable_filter=False, fast=True)
    return {'intrinsics_yaml': intrinsics_yaml, 'cube_cfg': cube_cfg, 'image_size': image_size, 'undistort_pack': undistort_pack, 'detection_camera_matrix': detection_camera_matrix, 'detector_dist_coeffs': detector_dist_coeffs, 'cube_config': cube_config, 'face_id_sets': face_id_sets, 'tag_corner_map': tag_corner_map, 'april_post_detector': april_post_detector, 'april_draw_detector': april_draw_detector}

def deeptag_detection_detection_frame(script012: Any, runtime: dict[str, Any], image_bgr: np.ndarray) -> np.ndarray:
    target_w, target_h = runtime['image_size']
    h, w = image_bgr.shape[:2]
    if (w, h) != (target_w, target_h):
        image_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return realsense_undistort_frame(image_bgr, runtime['undistort_pack'])
DEEPTAG_DETECTION_CORNER_ORDER_TRANSFORMS = {'id': (0, 1, 2, 3), 'rot90': (1, 2, 3, 0), 'rev': (0, 3, 2, 1), 'rot180': (2, 3, 0, 1), 'rot270': (3, 0, 1, 2), 'rev_rot90': (1, 0, 3, 2), 'rev_rot180': (2, 1, 0, 3), 'rev_rot270': (3, 2, 1, 0)}

def deeptag_detection_quad_quality(corners: np.ndarray) -> float:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    area = float(abs(cv2.contourArea(corners.astype(np.float32))))
    if area <= 0.0:
        return 0.0
    edges = [float(np.linalg.norm(corners[(idx + 1) % 4] - corners[idx])) for idx in range(4)]
    min_edge = min(edges)
    max_edge = max(edges)
    if max_edge <= 1e-09:
        return 0.0
    return area * (min_edge / max_edge)

def deeptag_detection_deeptag_detections_to_raw_corners(engine: Any, decoded_tags: list[dict[str, Any]], *, valid_ids: set[int]) -> tuple[list[tuple[int, np.ndarray]], dict[str, int]]:
    best_by_id: dict[int, tuple[float, np.ndarray]] = {}
    rois = getattr(engine, 'rois_info', [])
    raw_valid = 0
    invalid_id = 0
    duplicate_id = 0
    for idx, decoded in enumerate(decoded_tags):
        if not decoded.get('is_valid', False):
            continue
        raw_valid += 1
        tag_id = int(decoded.get('tag_id', -1))
        if tag_id < 0 or idx >= len(rois):
            invalid_id += 1
            continue
        if tag_id not in valid_ids:
            invalid_id += 1
            continue
        roi_info = rois[idx]
        roi = roi_info.get('ordered_corners', roi_info) if isinstance(roi_info, dict) else roi_info
        main_idx = int(decoded.get('main_idx', 0))
        ordered = list(roi[main_idx:]) + list(roi[:main_idx])
        corners = np.asarray(ordered, dtype=np.float64).reshape(4, 2)
        quality = deeptag_detection_quad_quality(corners)
        if tag_id in best_by_id:
            duplicate_id += 1
            if quality <= best_by_id[tag_id][0]:
                continue
        best_by_id[tag_id] = (quality, corners)
    detections = [(tag_id, corners) for tag_id, (_quality, corners) in sorted(best_by_id.items())]
    stats = {'raw_valid_decoded': int(raw_valid), 'invalid_or_wrong_id': int(invalid_id), 'duplicate_id': int(duplicate_id), 'kept': int(len(detections))}
    return (detections, stats)

def deeptag_detection_apply_corner_order(detections: list[tuple[int, np.ndarray]], corner_order: str) -> list[tuple[int, np.ndarray]]:
    order = list(DEEPTAG_DETECTION_CORNER_ORDER_TRANSFORMS[corner_order])
    return [(int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2)[order]) for tag_id, corners in detections]

def deeptag_detection_to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [deeptag_detection_to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return sorted((deeptag_detection_to_json_compatible(item) for item in value))
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {'unit_tag', 'H_crop'}:
                continue
            out[str(key)] = deeptag_detection_to_json_compatible(item)
        return out
    return str(value)

def deeptag_detection_sanitize_decoded_tags(decoded_tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deeptag_detection_to_json_compatible(tag) for tag in decoded_tags]

def deeptag_detection_sanitize_pose_result(result: dict[str, Any]) -> dict[str, Any]:
    skip = {'debug_viz'}
    out: dict[str, Any] = {}
    for key, value in result.items():
        if key in skip:
            continue
        if key == 'detections':
            detections = []
            for item in value or []:
                if len(item) != 2:
                    continue
                tag_id, corners = item
                detections.append({'tag_id': int(tag_id), 'corners_xy': np.asarray(corners, dtype=np.float64).reshape(4, 2).tolist()})
            out[key] = detections
            continue
        out[key] = deeptag_detection_to_json_compatible(value)
    return out

def deeptag_detection_reset_aprilcube_single_frame_state(detector: Any) -> None:
    detector.prev_rvec = None
    detector.prev_tvec = None
    detector._prev_gray = None
    detector._prev_corners_2d = None
    detector._prev_corners_3d = None
    if getattr(detector, 'pose_filter', None) is not None:
        detector.pose_filter.reset()

def deeptag_detection_finite_pose_success(result: dict[str, Any]) -> bool:
    if not bool(result.get('success', False)):
        return False
    if result.get('rvec', None) is None or result.get('tvec', None) is None:
        return False
    values = [np.asarray(result['rvec'], dtype=np.float64).reshape(-1), np.asarray(result['tvec'], dtype=np.float64).reshape(-1), np.asarray([float(result.get('reproj_error', float('inf')))], dtype=np.float64)]
    return all((bool(np.all(np.isfinite(chunk))) for chunk in values))

def deeptag_detection_rvec_to_rot(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot

def deeptag_detection_rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    ra = deeptag_detection_rvec_to_rot(rvec_a)
    rb = deeptag_detection_rvec_to_rot(rvec_b)
    cos_angle = np.clip((np.trace(ra @ rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def deeptag_detection_translation_delta_mm(tvec_a: Any, tvec_b: Any) -> float:
    ta = np.asarray(tvec_a, dtype=np.float64).reshape(3)
    tb = np.asarray(tvec_b, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(ta - tb))

def deeptag_detection_visible_faces_for_ids(face_id_sets: dict[str, set[int]], tag_ids: list[int]) -> set[str]:
    visible: set[str] = set()
    for tag_id in tag_ids:
        for face_name, ids in face_id_sets.items():
            if int(tag_id) in ids:
                visible.add(str(face_name))
    return visible

def deeptag_detection_face_normals_ok(rvec: Any, visible_faces: set[str]) -> bool:
    rot = deeptag_detection_rvec_to_rot(rvec)
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

def deeptag_detection_per_tag_reprojection_errors(detections: list[tuple[int, np.ndarray]], tag_corner_map: dict[int, np.ndarray], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, rvec: Any, tvec: Any) -> dict[int, float]:
    per_tag: dict[int, float] = {}
    for tag_id, corners_2d in detections:
        corners_3d = tag_corner_map.get(int(tag_id))
        if corners_3d is None:
            continue
        projected, _ = cv2.projectPoints(np.asarray(corners_3d, dtype=np.float64).reshape(4, 3), np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1), camera_matrix, dist_coeffs)
        err = np.linalg.norm(np.asarray(corners_2d, dtype=np.float64).reshape(4, 2) - projected.reshape(4, 2), axis=1)
        per_tag[int(tag_id)] = float(np.mean(err))
    return per_tag

def deeptag_detection_solve_pose_from_detections(detections: list[tuple[int, np.ndarray]], runtime: dict[str, Any], *, seed_rvec: Any | None=None, seed_tvec: Any | None=None, max_reproj: float) -> dict[str, Any]:
    tag_corner_map = runtime['tag_corner_map']
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
        return {'success': False, 'failure_reason': 'no_cluster_detections'}
    object_points = np.vstack(object_chunks).astype(np.float64)
    image_points = np.vstack(image_chunks).astype(np.float64)
    if len(used) == 1:
        ok, rvec, tvec, reproj_err, inliers, meta = estimate_single_tag_cube_pose(used, tag_corner_map, runtime['face_id_sets'], runtime['detection_camera_matrix'], runtime['detector_dist_coeffs'], None if seed_rvec is None else np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1), None if seed_tvec is None else np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1), allow_corner_rotations=False)
    else:
        ok, rvec, tvec, reproj_err, inliers = estimate_pose(object_points, image_points, runtime['detection_camera_matrix'], runtime['detector_dist_coeffs'], None if seed_rvec is None else np.asarray(seed_rvec, dtype=np.float64).reshape(3, 1), None if seed_tvec is None else np.asarray(seed_tvec, dtype=np.float64).reshape(3, 1))
        meta = {}
    if not ok or rvec is None or tvec is None or (not np.all(np.isfinite(rvec))) or (not np.all(np.isfinite(tvec))) or (not np.isfinite(float(reproj_err))) or (float(np.asarray(tvec).reshape(3)[2]) <= 0.0):
        return {'success': False, 'failure_reason': 'cluster_pnp_failed', 'detections': used, 'n_tags': len(used), 'tag_ids': [tag_id for tag_id, _ in used], 'reproj_error': float('inf')}
    visible_faces = deeptag_detection_visible_faces_for_ids(runtime['face_id_sets'], [tag_id for tag_id, _ in used])
    if not deeptag_detection_face_normals_ok(rvec, visible_faces):
        return {'success': False, 'failure_reason': 'cluster_face_normal_away', 'detections': used, 'n_tags': len(used), 'tag_ids': [tag_id for tag_id, _ in used], 'reproj_error': float('inf')}
    for _iteration in range(2):
        per_tag = deeptag_detection_per_tag_reprojection_errors(used, tag_corner_map, runtime['detection_camera_matrix'], runtime['detector_dist_coeffs'], rvec, tvec)
        if len(per_tag) < 3:
            break
        vals = np.asarray(list(per_tag.values()), dtype=np.float64)
        median_err = float(np.median(vals))
        keep_thresh = min(max(median_err * 3.0, 5.0), float(max_reproj))
        keep_ids = {tag_id for tag_id, err in per_tag.items() if err <= keep_thresh}
        if len(keep_ids) == len(used) or len(keep_ids) < 1:
            break
        used = [(tag_id, corners) for tag_id, corners in used if tag_id in keep_ids]
        return deeptag_detection_solve_pose_from_detections(used, runtime, seed_rvec=rvec, seed_tvec=tvec, max_reproj=max_reproj)
    per_tag = deeptag_detection_per_tag_reprojection_errors(used, tag_corner_map, runtime['detection_camera_matrix'], runtime['detector_dist_coeffs'], rvec, tvec)
    if float(reproj_err) > float(max_reproj):
        return {'success': False, 'failure_reason': f'cluster_reproj_too_high:{float(reproj_err):.2f}>{float(max_reproj):.2f}', 'detections': used, 'n_tags': len(used), 'tag_ids': [tag_id for tag_id, _ in used], 'reproj_error': float('inf'), 'per_tag_reproj_error': per_tag}
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = deeptag_detection_rvec_to_rot(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    result = {'success': True, 'failure_reason': '', 'detections': used, 'n_tags': len(used), 'tag_ids': [int(tag_id) for tag_id, _ in used], 'visible_faces': visible_faces, 'predicted': False, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': transform, 'reproj_error': float(reproj_err), 'n_inliers': 0 if inliers is None else int(len(inliers)), 'per_tag_reproj_error': per_tag}
    result.update(meta)
    return result

def deeptag_detection_robust_cluster_pose(raw_detections: list[tuple[int, np.ndarray]], runtime: dict[str, Any], args: DeepTagDetectionConfig) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for tag_id, raw_corners in raw_detections:
        for order_name, order in DEEPTAG_DETECTION_CORNER_ORDER_TRANSFORMS.items():
            corners = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)[list(order)]
            ok, rvec, tvec, reproj, _inliers, meta = estimate_single_tag_cube_pose([(int(tag_id), corners)], runtime['tag_corner_map'], runtime['face_id_sets'], runtime['detection_camera_matrix'], runtime['detector_dist_coeffs'], allow_corner_rotations=False)
            if not ok or rvec is None or tvec is None or (not np.all(np.isfinite(rvec))) or (not np.all(np.isfinite(tvec))) or (not np.isfinite(float(reproj))) or (float(reproj) > float(args.robust_single_tag_max_reproj)):
                continue
            candidates.append({'tag_id': int(tag_id), 'corners': corners, 'corner_order': order_name, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'reproj_error': float(reproj), 'face': meta.get('single_tag_face', None)})
    if not candidates:
        return ({'success': False, 'failure_reason': 'robust_no_single_tag_candidates', 'detections': [], 'n_tags': 0, 'tag_ids': [], 'reproj_error': float('inf')}, [], {'candidate_count': 0, 'cluster_size': 0})
    best_cluster: list[dict[str, Any]] = []
    best_score: tuple[int, float, float] | None = None
    for seed in candidates:
        by_tag: dict[int, tuple[float, dict[str, Any]]] = {}
        for candidate in candidates:
            trans = deeptag_detection_translation_delta_mm(seed['tvec'], candidate['tvec'])
            rot = deeptag_detection_rotation_delta_deg(seed['rvec'], candidate['rvec'])
            if trans > float(args.robust_cluster_trans_mm) or rot > float(args.robust_cluster_rot_deg):
                continue
            score = trans / max(float(args.robust_cluster_trans_mm), 1e-09) + rot / max(float(args.robust_cluster_rot_deg), 1e-09) + float(candidate['reproj_error']) / max(float(args.robust_single_tag_max_reproj), 1e-09)
            tag_id = int(candidate['tag_id'])
            if tag_id not in by_tag or score < by_tag[tag_id][0]:
                by_tag[tag_id] = (score, candidate)
        cluster = [item[1] for item in by_tag.values()]
        if not cluster:
            continue
        mean_single_reproj = float(np.mean([item['reproj_error'] for item in cluster]))
        mean_seed_trans = float(np.mean([deeptag_detection_translation_delta_mm(seed['tvec'], item['tvec']) for item in cluster]))
        score_key = (len(cluster), -mean_single_reproj, -mean_seed_trans)
        if best_score is None or score_key > best_score:
            best_score = score_key
            best_cluster = cluster
    if len(best_cluster) < int(args.robust_min_tags):
        return ({'success': False, 'failure_reason': f'robust_cluster_too_small:{len(best_cluster)}<{int(args.robust_min_tags)}', 'detections': [(item['tag_id'], item['corners']) for item in best_cluster], 'n_tags': len(best_cluster), 'tag_ids': [int(item['tag_id']) for item in best_cluster], 'reproj_error': float('inf')}, [(item['tag_id'], item['corners']) for item in best_cluster], {'candidate_count': len(candidates), 'cluster_size': len(best_cluster)})
    seed = min(best_cluster, key=lambda item: item['reproj_error'])
    cluster_detections = [(int(item['tag_id']), np.asarray(item['corners'], dtype=np.float64).reshape(4, 2)) for item in sorted(best_cluster, key=lambda item: int(item['tag_id']))]
    pose = deeptag_detection_solve_pose_from_detections(cluster_detections, runtime, seed_rvec=seed['rvec'], seed_tvec=seed['tvec'], max_reproj=float(args.robust_max_reproj))
    pose['pose_source'] = 'deeptag_robust_pose_cluster'
    pose['pose_filled'] = False
    pose['robust_candidate_count'] = int(len(candidates))
    pose['robust_cluster_size'] = int(len(best_cluster))
    pose['robust_corner_orders'] = {int(item['tag_id']): str(item['corner_order']) for item in best_cluster}
    stats = {'candidate_count': int(len(candidates)), 'cluster_size': int(len(best_cluster)), 'cluster_tag_ids': [int(item['tag_id']) for item in best_cluster], 'cluster_corner_orders': {int(item['tag_id']): str(item['corner_order']) for item in best_cluster}}
    selected = pose.get('detections', cluster_detections) or cluster_detections
    return (pose, selected, stats)

def deeptag_detection_draw_overlay(image_bgr: np.ndarray, runtime: dict[str, Any], detections: list[tuple[int, np.ndarray]], pose: dict[str, Any]) -> np.ndarray:
    result = {'success': bool(pose.get('success', False)), 'detections': detections, 'rvec': np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'tvec': np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'reproj_error': float(pose.get('reproj_error', float('inf'))), 'n_tags': int(pose.get('n_tags', 0)), 'visible_faces': set(pose.get('visible_faces', []) or []), 'predicted': False}
    vis = runtime['april_draw_detector'].draw_result(image_bgr.copy(), result)
    y = 28
    lines = [f"DeepTag tags={pose.get('n_tags', 0)} reproj={float(pose.get('reproj_error', float('inf'))):.2f}px", f"ids={pose.get('tag_ids', [])}"]
    for line in lines:
        cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return vis

def deeptag_detection_main(args: DeepTagDetectionConfig) -> None:
    pkl_path = Path(args.pkl_path).expanduser().resolve()
    header, all_offsets, footer = deeptag_detection_build_stream_index(pkl_path)
    offsets = all_offsets[int(args.start_frame)::max(int(args.stride), 1)]
    if args.max_frames > 0:
        offsets = offsets[:int(args.max_frames)]
    script012 = None
    metadata = deeptag_detection_input_metadata(header)
    runtime = deeptag_detection_make_runtime(script012, metadata, args)
    tag_size_m = float(runtime['cube_config'].tag_size_mm) / 1000.0
    print(f'[INFO] Loading DeepTag models from {DEEPTAG_DETECTION_DEEPTAG_ROOT}')
    t0 = time.perf_counter()
    engine = deeptag_detection_load_deeptag_engine(camera_matrix=runtime['detection_camera_matrix'], dist_coeffs=runtime['detector_dist_coeffs'], tag_size_m=tag_size_m, args=args)
    print(f'[INFO] DeepTag loaded in {time.perf_counter() - t0:.2f}s')
    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    total_tags = 0
    with output_pkl.open('wb') as f:
        pickle.dump({'type': 'header', 'format': 'deeptag_012_offline_stream_v1', 'source_pkl': str(pkl_path), 'source_footer': footer, 'metadata': {'script': str(DEEPTAG_DETECTION_THIS_FILE), 'deeptag_root': str(DEEPTAG_DETECTION_DEEPTAG_ROOT), 'cube_cfg': str(runtime['cube_cfg']), 'intrinsics_yaml': str(runtime['intrinsics_yaml']), 'camera_matrix': runtime['detection_camera_matrix'].tolist(), 'dist_coeffs': runtime['detector_dist_coeffs'].tolist(), 'frame_count': len(offsets), 'tag_size_m': tag_size_m, 'corner_order': str(args.corner_order), 'postprocess': str(args.pose_mode), 'robust_min_tags': int(args.robust_min_tags), 'robust_cluster_trans_mm': float(args.robust_cluster_trans_mm), 'robust_cluster_rot_deg': float(args.robust_cluster_rot_deg), 'robust_max_reproj': float(args.robust_max_reproj), 'robust_single_tag_max_reproj': float(args.robust_single_tag_max_reproj), 'args': vars(args)}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for out_idx, offset in enumerate(offsets):
            record = deeptag_detection_load_frame_at(pkl_path, offset)
            frame = deeptag_detection_detection_frame(script012, runtime, np.asarray(record['image_bgr'], dtype=np.uint8))
            t_frame = time.perf_counter()
            stream = io.StringIO()
            ctx = contextlib.redirect_stdout(stream) if args.quiet_deeptag else contextlib.nullcontext()
            with ctx:
                decoded_tags = engine.process(frame, detect_scale=None if args.detect_scale < 0 else float(args.detect_scale))
            elapsed = time.perf_counter() - t_frame
            raw_detections, detection_stats = deeptag_detection_deeptag_detections_to_raw_corners(engine, decoded_tags, valid_ids=set((int(v) for v in runtime['cube_config'].tag_ids)))
            cluster_stats: dict[str, Any] = {}
            if str(args.pose_mode) == 'robust-cluster':
                pose_raw, detections, cluster_stats = deeptag_detection_robust_cluster_pose(raw_detections, runtime, args)
            else:
                detections = deeptag_detection_apply_corner_order(raw_detections, str(args.corner_order))
                post_detector = runtime['april_post_detector']
                deeptag_detection_reset_aprilcube_single_frame_state(post_detector)
                pose_raw = post_detector.process_detections(frame, detections, timestamp=float(record.get('capture_timestamp', out_idx)))
                deeptag_detection_reset_aprilcube_single_frame_state(post_detector)
                if not deeptag_detection_finite_pose_success(pose_raw):
                    pose_raw['success'] = False
                    pose_raw['rvec'] = None
                    pose_raw['tvec'] = None
                    pose_raw['T'] = None
                    pose_raw['reproj_error'] = float('inf')
                    if not pose_raw.get('failure_reason', ''):
                        pose_raw['failure_reason'] = 'non_finite_or_failed_pose'
                pose_raw['pose_source'] = 'deeptag_aprilcube_postprocess'
                pose_raw['pose_filled'] = False
            pose = deeptag_detection_sanitize_pose_result(pose_raw)
            overlay = deeptag_detection_draw_overlay(frame, runtime, detections, pose)
            success_count += int(bool(pose.get('success', False)))
            total_tags += int(pose.get('n_tags', 0))
            frame_record = {'type': 'frame', 'frame_index': int(out_idx), 'source_offset': int(offset), 'loop_frame_idx': int(record.get('loop_frame_idx', out_idx)), 'capture_timestamp': record.get('capture_timestamp', None), 'deeptag_elapsed_s': float(elapsed), 'detection_stats': detection_stats, 'cluster_stats': deeptag_detection_to_json_compatible(cluster_stats), 'decoded_tags': deeptag_detection_sanitize_decoded_tags(decoded_tags), 'raw_detections': [{'tag_id': int(tag_id), 'corners_xy': np.asarray(corners).tolist()} for tag_id, corners in raw_detections], 'detections': [{'tag_id': int(tag_id), 'corners_xy': np.asarray(corners).tolist()} for tag_id, corners in detections], 'pose': pose, 'overlay_jpeg': deeptag_detection_encode_bgr_jpeg(overlay, int(args.jpeg_quality)), 'overlay_format': 'jpeg_bgr'}
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[INFO] frame {out_idx + 1}/{len(offsets)} tags={pose.get('n_tags', 0)} success={pose.get('success', False)} reproj={float(pose.get('reproj_error', float('inf'))):.2f}px time={elapsed:.2f}s")
        pickle.dump({'type': 'footer', 'frame_count': len(offsets), 'success_count': int(success_count), 'avg_tags': total_tags / max(len(offsets), 1), 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[INFO] Saved DeepTag result pkl: {output_pkl}')
    print(f'[INFO] success={success_count}/{len(offsets)} avg_tags={total_tags / max(len(offsets), 1):.2f}')


# ---- Raw frame and strict pose merge ----
STRICT_POSE_MERGE_APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
STRICT_POSE_MERGE_DEFAULT_RAW_PKL = STRICT_POSE_MERGE_APRILCUBE_ROOT / 'recordings' / '012_rs_raw_frames_20260710_214336.pkl'
STRICT_POSE_MERGE_DEFAULT_POSE_PKL = STRICT_POSE_MERGE_APRILCUBE_ROOT / 'recordings' / '014_offline_pose_vis_012_rs_raw_frames_20260710_214336.pkl'
STRICT_POSE_MERGE_DEFAULT_OUTPUT_PKL = STRICT_POSE_MERGE_APRILCUBE_ROOT / 'recordings' / '012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'

def strict_pose_merge_build_stream_index(path: Path, expected_format: str) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as f:
        header = pickle.load(f)
        if not isinstance(header, dict) or header.get('format') != expected_format:
            raise ValueError(f"Unsupported pkl format in {path}: {header.get('format', None)}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if obj.get('type') == 'frame':
                offsets.append(offset)
            elif obj.get('type') == 'footer':
                footer = obj
                break
    if not offsets:
        raise ValueError(f'No frame records found in {path}')
    return (header, offsets, footer)

def strict_pose_merge_load_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get('type') != 'frame':
        raise ValueError(f'Offset {offset} in {path} is not a frame record')
    return obj

def strict_pose_merge_verify_merged(path: Path, expected_frames: int) -> tuple[dict[str, Any], dict[str, int]]:
    header, offsets, footer = strict_pose_merge_build_stream_index(path, 'aprilcube_012_raw_with_pose_stream_v1')
    if len(offsets) != expected_frames:
        raise ValueError(f'Merged frame count mismatch: {len(offsets)} != {expected_frames}')
    if footer is None or int(footer.get('frame_count', -1)) != expected_frames:
        raise ValueError(f'Merged footer frame_count mismatch in {path}')
    pose_sources: dict[str, int] = {}
    success_count = 0
    for offset in offsets:
        record = strict_pose_merge_load_at(path, offset)
        image = record.get('image_bgr', None)
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Merged frame at offset {offset} does not contain raw image_bgr ndarray')
        pose = record.get('pose', {})
        if pose.get('success', False):
            success_count += 1
        source = str(pose.get('pose_source', ''))
        pose_sources[source] = pose_sources.get(source, 0) + 1
    return (header, {'frame_count': len(offsets), 'success_count': success_count, **pose_sources})

def strict_pose_merge_main(args: RawPoseMergeConfig) -> None:
    raw_pkl = Path(args.raw_pkl).expanduser().resolve()
    pose_pkl = Path(args.pose_pkl).expanduser().resolve()
    output_pkl = Path(args.output_pkl).expanduser().resolve()
    raw_header, raw_offsets, raw_footer = strict_pose_merge_build_stream_index(raw_pkl, 'aprilcube_rs_raw_frame_stream_v1')
    pose_header, pose_offsets, pose_footer = strict_pose_merge_build_stream_index(pose_pkl, 'aprilcube_012_offline_pose_vis_stream_v1')
    if len(raw_offsets) != len(pose_offsets):
        raise ValueError(f'Frame count mismatch: raw={len(raw_offsets)} pose={len(pose_offsets)}')
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    total = len(raw_offsets)
    success_count = int(pose_header.get('metadata', {}).get('success_count', 0))
    filled_count = int(pose_header.get('metadata', {}).get('filled_pose_count', 0))
    t0 = time.perf_counter()
    with output_pkl.open('wb') as f:
        pickle.dump({'type': 'header', 'format': 'aprilcube_012_raw_with_pose_stream_v1', 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_raw_pkl': str(raw_pkl), 'source_pose_pkl': str(pose_pkl), 'raw_header': raw_header, 'raw_footer': raw_footer, 'pose_header': pose_header, 'pose_footer': pose_footer, 'metadata': {'script': str(Path(__file__).resolve()), 'method': 'OpenCV/AprilCube + reprojection filtering + pose interpolation', 'frame_count': int(total), 'success_count': int(success_count), 'filled_pose_count': int(filled_count), 'raw_image_field': 'image_bgr', 'raw_image_storage': 'original numpy ndarray from 012 pkl', 'overlay_field': 'overlay_jpeg', 'pose_field': 'pose'}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx, (raw_offset, pose_offset) in enumerate(zip(raw_offsets, pose_offsets, strict=True)):
            raw_record = strict_pose_merge_load_at(raw_pkl, raw_offset)
            pose_record = strict_pose_merge_load_at(pose_pkl, pose_offset)
            if int(pose_record.get('source_offset', -1)) != int(raw_offset):
                raise ValueError(f"source_offset mismatch at frame {idx}: pose={pose_record.get('source_offset')} raw={raw_offset}")
            if pose_record.get('capture_timestamp', None) != raw_record.get('capture_timestamp', None):
                raise ValueError(f'capture_timestamp mismatch at frame {idx}')
            image_bgr = raw_record['image_bgr']
            frame_record = {'type': 'frame', 'frame_index': int(idx), 'raw_source_offset': int(raw_offset), 'pose_source_offset': int(pose_offset), 'device_name': str(raw_record.get('device_name', '')), 'camera_name': str(raw_record.get('camera_name', '')), 'loop_frame_idx': int(raw_record.get('loop_frame_idx', idx)), 'capture_timestamp': raw_record.get('capture_timestamp', None), 'write_monotonic': raw_record.get('write_monotonic', None), 'shape': tuple((int(v) for v in np.asarray(image_bgr).shape)), 'dtype': str(np.asarray(image_bgr).dtype), 'image_bgr': image_bgr, 'overlay_shape': pose_record.get('overlay_shape', None), 'overlay_format': pose_record.get('overlay_format', 'jpeg_bgr'), 'overlay_jpeg': pose_record['overlay_jpeg'], 'pose': pose_record['pose']}
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
            done = idx + 1
            if done == total or done % 10 == 0:
                elapsed = time.perf_counter() - t0
                fps = done / max(elapsed, 1e-09)
                print(f'\r[INFO] Merging {done}/{total} fps={fps:.1f}', end='', flush=True)
        pickle.dump({'type': 'footer', 'frame_count': int(total), 'success_count': int(success_count), 'filled_pose_count': int(filled_count), 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()
    _, summary = strict_pose_merge_verify_merged(output_pkl, total)
    print(f'[INFO] Saved merged pkl: {output_pkl}')
    print(f'[INFO] Verified merged pkl: {summary}')
    if args.delete_inputs:
        for path in (raw_pkl, pose_pkl):
            if path == output_pkl:
                raise ValueError(f'Refusing to delete output pkl: {path}')
            path.unlink()
            print(f'[INFO] Deleted input pkl: {path}')


# ---- Dense DeepTag pose estimation ----
DENSE_DEEPTAG_THIS_FILE = Path(__file__).resolve()
DENSE_DEEPTAG_APRILCUBE_ROOT = DENSE_DEEPTAG_THIS_FILE.parent.parent
DENSE_DEEPTAG_DEEPTAG_ROOT = DENSE_DEEPTAG_APRILCUBE_ROOT / 'thirdparty' / 'deeptag-pytorch'
DENSE_DEEPTAG_DEFAULT_INPUT_PKL = DENSE_DEEPTAG_APRILCUBE_ROOT / 'recordings' / '016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl'
DENSE_DEEPTAG_DEFAULT_OUTPUT_PKL = DENSE_DEEPTAG_APRILCUBE_ROOT / 'recordings' / '020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336.pkl'
DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS = {'id': (0, 1, 2, 3), 'rot90': (1, 2, 3, 0), 'rev': (0, 3, 2, 1), 'rot180': (2, 3, 0, 1), 'rot270': (3, 0, 1, 2), 'rev_rot90': (1, 0, 3, 2), 'rev_rot180': (2, 1, 0, 3), 'rev_rot270': (3, 2, 1, 0)}

def dense_deeptag_build_stream_index(path: Path, expected_format: set[str] | None=None) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as f:
        header = pickle.load(f)
        if expected_format is not None and header.get('format') not in expected_format:
            raise ValueError(f"Unsupported pkl format in {path}: {header.get('format', None)}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if not isinstance(obj, dict):
                continue
            if obj.get('type') == 'frame':
                offsets.append(offset)
            elif obj.get('type') == 'footer':
                footer = obj
                break
    if not offsets:
        raise ValueError(f'No frame records found in {path}')
    return (header, offsets, footer)

def dense_deeptag_load_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get('type') != 'frame':
        raise ValueError(f'Offset {offset} in {path} is not a frame')
    return obj

def dense_deeptag_encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode('.jpg', np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def dense_deeptag_decode_jpeg_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Failed to decode JPEG')
    return image

def dense_deeptag_to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [dense_deeptag_to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return sorted((dense_deeptag_to_json_compatible(item) for item in value))
    if isinstance(value, dict):
        return {str(key): dense_deeptag_to_json_compatible(item) for key, item in value.items()}
    return str(value)

def dense_deeptag_rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot

def dense_deeptag_visible_faces_for_ids(face_id_sets: dict[str, set[int]], tag_ids: list[int]) -> set[str]:
    faces: set[str] = set()
    for tag_id in tag_ids:
        for face_name, ids in face_id_sets.items():
            if int(tag_id) in ids:
                faces.add(str(face_name))
    return faces

def dense_deeptag_face_normals_ok(rvec: Any, visible_faces: set[str]) -> bool:
    rot = dense_deeptag_rotation_from_rvec(rvec)
    for face_name in visible_faces:
        for face_def in aprilcube.FACE_DEFS:
            if str(face_def[0]) != str(face_name):
                continue
            normal = np.zeros(3, dtype=np.float64)
            normal[int(face_def[1])] = float(face_def[2])
            if float((rot @ normal)[2]) > 0.0:
                return False
            break
    return True

def dense_deeptag_dense_local_annotations(num_points: int) -> np.ndarray:
    n = int(round(np.sqrt(int(num_points))))
    if n * n != int(num_points) or n < 3:
        raise ValueError(f'Unsupported dense keypoint count: {num_points}')
    grid_size = n - 2
    unit_tag = UnitArucoTag(grid_size, [0] * (grid_size * grid_size))
    anno = np.asarray(get_fine_grid_points_anno(unit_tag, step_elem_num=1), dtype=np.float64)
    return anno.reshape(-1, anno.shape[-1])[:, :2]

def dense_deeptag_local_to_cube_affine(tag_corners_3d: np.ndarray, corner_order: str) -> np.ndarray:
    stage1_corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float64)
    dense_corners = stage1_corners.copy()
    dense_corners[:, 0] *= -1.0
    order = np.asarray(DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS[str(corner_order)], dtype=np.int64)
    local = np.c_[dense_corners[order], np.ones(4, dtype=np.float64)]
    target = np.asarray(tag_corners_3d, dtype=np.float64).reshape(4, 3)
    affine_t, *_ = np.linalg.lstsq(local, target, rcond=None)
    return affine_t

def dense_deeptag_dense_points_for_frame(
    frame: dict[str, Any],
    *,
    tag_corner_map: dict[int, np.ndarray],
    face_id_sets: dict[str, set[int]] | None = None,
    min_tags: int,
    require_validated_corner_order: bool = True,
    unclustered_corner_order: str = 'rot180',
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int], dict[str, Any]]:
    if str(unclustered_corner_order) not in DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS:
        raise ValueError(
            f'Unsupported unclustered DeepTag corner order: {unclustered_corner_order}'
        )
    unclustered_corner_order = str(unclustered_corner_order)
    cluster_orders = frame.get('cluster_stats', {}).get('cluster_corner_orders', {}) or {}
    cluster_orders = {
        int(k): str(v)
        for k, v in cluster_orders.items()
        if str(v) in DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS
    }
    order_votes: dict[str, int] = {}
    for order in cluster_orders.values():
        if order in DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS:
            order_votes[order] = order_votes.get(order, 0) + 1
    dominant_order = (
        max(order_votes.items(), key=lambda item: item[1])[0]
        if order_votes
        else unclustered_corner_order
    )
    decoded_by_id: dict[int, dict[str, Any]] = {}
    for decoded in frame.get('decoded_tags', []) or []:
        if not decoded.get('is_valid', False):
            continue
        tag_id = int(decoded.get('tag_id', -1))
        if tag_id in tag_corner_map:
            decoded_by_id[tag_id] = decoded
    decoded_tag_ids = sorted(decoded_by_id)
    single_face_joint_ids: set[int] = set()
    if require_validated_corner_order and not cluster_orders and face_id_sets:
        face_candidates: list[tuple[int, str, set[int]]] = []
        decoded_id_set = set(decoded_tag_ids)
        for face_name, face_ids in face_id_sets.items():
            matching = decoded_id_set & {int(v) for v in face_ids}
            face_candidates.append((len(matching), str(face_name), matching))
        face_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if face_candidates and face_candidates[0][0] >= int(min_tags):
            single_face_joint_ids = set(face_candidates[0][2])
    individually_or_jointly_validated_ids = set(cluster_orders) | single_face_joint_ids
    unvalidated_order_tag_ids = [
        int(tag_id)
        for tag_id in decoded_tag_ids
        if int(tag_id) not in individually_or_jointly_validated_ids
    ]
    obj_chunks: list[np.ndarray] = []
    img_chunks: list[np.ndarray] = []
    tag_ids: list[int] = []
    point_counts: dict[int, int] = {}
    for tag_id in sorted(decoded_by_id):
        if (
            require_validated_corner_order
            and int(tag_id) not in individually_or_jointly_validated_ids
        ):
            continue
        decoded = decoded_by_id[tag_id]
        image_points = np.asarray(decoded.get('keypoints_in_images', []), dtype=np.float64).reshape(-1, 2)
        if image_points.shape[0] < 4:
            continue
        local_xy = dense_deeptag_dense_local_annotations(image_points.shape[0])
        corner_order = cluster_orders.get(
            int(tag_id),
            unclustered_corner_order
            if int(tag_id) in single_face_joint_ids
            else dominant_order,
        )
        affine_t = dense_deeptag_local_to_cube_affine(tag_corner_map[tag_id], corner_order)
        object_points = np.c_[local_xy, np.ones(local_xy.shape[0], dtype=np.float64)] @ affine_t
        obj_chunks.append(object_points.astype(np.float64))
        img_chunks.append(image_points.astype(np.float64))
        tag_ids.append(int(tag_id))
        point_counts[int(tag_id)] = int(image_points.shape[0])
    if len(tag_ids) < int(min_tags):
        validation_name = 'validated_order_tags' if require_validated_corner_order else 'tags'
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            tag_ids,
            point_counts,
            {
                'reason': f'dense_{validation_name}_too_small:{len(tag_ids)}<{int(min_tags)}',
                'point_order_validation': ('per_tag_cluster_or_single_face_joint_pnp' if require_validated_corner_order else 'dominant_fallback'),
                'unclustered_corner_order': unclustered_corner_order,
                'decoded_tag_ids': decoded_tag_ids,
                'validated_order_tag_ids': sorted(individually_or_jointly_validated_ids),
                'single_face_joint_order_tag_ids': sorted(single_face_joint_ids),
                'rejected_unvalidated_order_tag_ids': unvalidated_order_tag_ids if require_validated_corner_order else [],
            },
        )
    return (
        np.vstack(obj_chunks),
        np.vstack(img_chunks),
        tag_ids,
        point_counts,
        {
            'cluster_corner_order_count': int(len(cluster_orders)),
            'point_order_validation': ('per_tag_cluster_or_single_face_joint_pnp' if require_validated_corner_order else 'dominant_fallback'),
            'unclustered_corner_order': unclustered_corner_order,
            'corner_order_fallback': None if require_validated_corner_order else dominant_order,
            'used_fallback_order_tag_ids': [] if require_validated_corner_order else [int(tag_id) for tag_id in tag_ids if int(tag_id) not in cluster_orders],
            'decoded_tag_ids': decoded_tag_ids,
            'validated_order_tag_ids': sorted(individually_or_jointly_validated_ids),
            'single_face_joint_order_tag_ids': sorted(single_face_joint_ids),
            'rejected_unvalidated_order_tag_ids': unvalidated_order_tag_ids if require_validated_corner_order else [],
        },
    )

def dense_deeptag_project_errors(object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)

def dense_deeptag_face_def_by_name(face_name: str) -> tuple:
    for face_def in aprilcube.FACE_DEFS:
        if str(face_def[0]) == str(face_name):
            return face_def
    raise KeyError(f'Unknown face name: {face_name}')

def dense_deeptag_face_local_basis(cube_config: Any, face_name: str) -> tuple[np.ndarray, np.ndarray]:
    face_def = dense_deeptag_face_def_by_name(face_name)
    _name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
    rot_cube_face = np.zeros((3, 3), dtype=np.float64)
    rot_cube_face[int(right_ax), 0] = float(right_sign)
    rot_cube_face[int(down_ax), 1] = float(down_sign)
    rot_cube_face[int(normal_ax), 2] = float(normal_sign)
    t_cube_face = np.zeros(3, dtype=np.float64)
    t_cube_face[int(normal_ax)] = float(normal_sign) * float(cube_config.box_dims[int(normal_ax)]) / 2.0
    return (rot_cube_face, t_cube_face)

def dense_deeptag_cube_points_to_face_points(cube_config: Any, face_name: str, cube_points: np.ndarray) -> np.ndarray:
    rot_cube_face, t_cube_face = dense_deeptag_face_local_basis(cube_config, face_name)
    points = np.asarray(cube_points, dtype=np.float64).reshape(-1, 3)
    face_points = (rot_cube_face.T @ (points - t_cube_face).T).T
    face_points[:, 2] = 0.0
    return face_points

def dense_deeptag_face_pose_to_cube_pose(cube_config: Any, face_name: str, face_rvec: np.ndarray, face_tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_cube_face, t_cube_face = dense_deeptag_face_local_basis(cube_config, face_name)
    rot_cam_face = dense_deeptag_rotation_from_rvec(face_rvec)
    rot_cam_cube = rot_cam_face @ rot_cube_face.T
    t_cam_cube = np.asarray(face_tvec, dtype=np.float64).reshape(3) - rot_cam_cube @ t_cube_face
    cube_rvec, _ = cv2.Rodrigues(rot_cam_cube)
    return (cube_rvec.reshape(3, 1), t_cam_cube.reshape(3, 1))

def dense_deeptag_transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = dense_deeptag_rotation_from_rvec(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform

def dense_deeptag_inlier_tag_coverage_failure(raw_tag_ids: list[int], used_tag_ids: list[int], *, min_tags: int, min_inlier_tag_fraction: float, coverage_check_min_raw_tags: int, max_required_inlier_tags: int) -> str:
    raw_count = len(set((int(tag_id) for tag_id in raw_tag_ids)))
    used_count = len(set((int(tag_id) for tag_id in used_tag_ids)))
    if raw_count < int(coverage_check_min_raw_tags):
        return ''
    required = int(np.ceil(raw_count * max(0.0, float(min_inlier_tag_fraction))))
    required = max(int(min_tags), required)
    required = min(max(required, 1), int(max_required_inlier_tags))
    if used_count < required:
        return f'dense_inlier_tags_low:{used_count}<{required}(raw={raw_count})'
    return ''

def dense_deeptag_best_single_face_ippe_pose(face_points: np.ndarray, cube_points: np.ndarray, image_points: np.ndarray, *, cube_config: Any, face_name: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> tuple[bool, np.ndarray | None, np.ndarray | None, float, int]:
    try:
        retval, rvecs, tvecs, _errs = cv2.solvePnPGeneric(face_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        retval, rvecs, tvecs = (0, (), ())
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    if retval:
        for face_rvec, face_tvec in zip(rvecs, tvecs):
            face_rvec = np.asarray(face_rvec, dtype=np.float64).reshape(3, 1)
            face_tvec = np.asarray(face_tvec, dtype=np.float64).reshape(3, 1)
            rot_cam_face = dense_deeptag_rotation_from_rvec(face_rvec)
            if float((rot_cam_face @ np.array([0.0, 0.0, 1.0], dtype=np.float64))[2]) > 0.0:
                continue
            cube_rvec, cube_tvec = dense_deeptag_face_pose_to_cube_pose(cube_config, face_name, face_rvec, face_tvec)
            if float(cube_tvec.reshape(3)[2]) <= 0.0:
                continue
            errors = dense_deeptag_project_errors(cube_points, image_points, cube_rvec, cube_tvec, camera_matrix, dist_coeffs)
            candidates.append((float(np.mean(errors)), cube_rvec, cube_tvec))
    if not candidates:
        try:
            ok, face_rvec, face_tvec = cv2.solvePnP(face_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            ok, face_rvec, face_tvec = (False, None, None)
        if ok and face_rvec is not None and (face_tvec is not None):
            cube_rvec, cube_tvec = dense_deeptag_face_pose_to_cube_pose(cube_config, face_name, face_rvec, face_tvec)
            if float(cube_tvec.reshape(3)[2]) > 0.0:
                errors = dense_deeptag_project_errors(cube_points, image_points, cube_rvec, cube_tvec, camera_matrix, dist_coeffs)
                candidates.append((float(np.mean(errors)), cube_rvec, cube_tvec))
    if not candidates:
        return (False, None, None, float('inf'), int(retval or 0))
    candidates.sort(key=lambda item: item[0])
    reproj, rvec, tvec = candidates[0]
    return (True, rvec, tvec, reproj, len(candidates))

def dense_deeptag_required_inliers_for_tag(
    point_count: int,
    *,
    min_fraction: float,
    min_points: int,
) -> int:
    return min(
        int(point_count),
        max(4, int(min_points), int(np.ceil(float(point_count) * float(min_fraction)))),
    )


def dense_deeptag_solve_single_face_dense_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    tag_ids: list[int],
    point_counts: dict[int, int],
    *,
    cube_config: Any,
    face_name: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    max_reproj: float,
    point_reject_px: float,
    tag_reject_px: float,
    min_tags: int,
    min_inlier_tag_fraction: float,
    coverage_check_min_raw_tags: int,
    max_required_inlier_tags: int,
    min_point_inlier_fraction: float = 0.25,
    min_per_tag_inlier_fraction: float = 0.25,
    min_per_tag_inlier_points: int = 12,
) -> dict[str, Any]:
    if object_points.shape[0] < 4:
        return {'success': False, 'failure_reason': 'dense_single_face_no_points', 'reproj_error': float('inf')}
    face_points = dense_deeptag_cube_points_to_face_points(cube_config, face_name, object_points)
    active = np.ones(object_points.shape[0], dtype=bool)
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    candidate_count = 0
    rejected_points = 0
    rejected_tags: list[int] = []
    for _iteration in range(3):
        if int(active.sum()) < 4:
            break
        ok, next_rvec, next_tvec, _reproj, candidate_count = dense_deeptag_best_single_face_ippe_pose(face_points[active], object_points[active], image_points[active], cube_config=cube_config, face_name=face_name, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        if not ok or next_rvec is None or next_tvec is None:
            break
        rvec, tvec = (next_rvec, next_tvec)
        errors = dense_deeptag_project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
        active_errors = errors[active]
        if active_errors.size == 0:
            break
        point_thresh = min(max(float(np.median(active_errors)) * 3.0, 2.0), float(point_reject_px))
        point_keep = errors <= point_thresh
        tag_keep_ids: set[int] = set()
        start = 0
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            tag_active = active[start:end] & point_keep[start:end]
            required_tag_points = dense_deeptag_required_inliers_for_tag(
                count,
                min_fraction=min_per_tag_inlier_fraction,
                min_points=min_per_tag_inlier_points,
            )
            if int(tag_active.sum()) >= required_tag_points:
                mean_err = float(np.mean(errors[start:end][tag_active]))
                if mean_err <= float(tag_reject_px):
                    tag_keep_ids.add(int(tag_id))
            start = end
        next_active = active & point_keep
        start = 0
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            if int(tag_id) not in tag_keep_ids:
                next_active[start:end] = False
            start = end
        if np.array_equal(next_active, active):
            break
        rejected_points += int(active.sum() - next_active.sum())
        rejected_tags = [int(tag_id) for tag_id in tag_ids if int(tag_id) not in tag_keep_ids]
        active = next_active
    if rvec is None or tvec is None or int(active.sum()) < 4:
        return {'success': False, 'failure_reason': 'dense_single_face_ippe_failed', 'reproj_error': float('inf'), 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    ok, final_rvec, final_tvec, _final_reproj, candidate_count = dense_deeptag_best_single_face_ippe_pose(face_points[active], object_points[active], image_points[active], cube_config=cube_config, face_name=face_name, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    if ok and final_rvec is not None and (final_tvec is not None):
        rvec, tvec = (final_rvec, final_tvec)
    errors = dense_deeptag_project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    active_errors = errors[active]
    reproj = float(np.mean(active_errors))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return {'success': False, 'failure_reason': f'dense_single_face_reproj_too_high:{reproj:.2f}>{float(max_reproj):.2f}', 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    used_ids: list[int] = []
    per_tag_reproj: dict[int, float] = {}
    per_tag_inliers: dict[int, int] = {}
    start = 0
    for tag_id in tag_ids:
        count = int(point_counts[int(tag_id)])
        end = start + count
        tag_active = active[start:end]
        if int(tag_active.sum()) > 0:
            used_ids.append(int(tag_id))
            per_tag_reproj[int(tag_id)] = float(np.mean(errors[start:end][tag_active]))
            per_tag_inliers[int(tag_id)] = int(tag_active.sum())
        start = end
    if len(used_ids) < int(min_tags):
        return {'success': False, 'failure_reason': f'dense_single_face_final_tags_too_small:{len(used_ids)}<{int(min_tags)}', 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    coverage_failure = dense_deeptag_inlier_tag_coverage_failure(tag_ids, used_ids, min_tags=min_tags, min_inlier_tag_fraction=min_inlier_tag_fraction, coverage_check_min_raw_tags=coverage_check_min_raw_tags, max_required_inlier_tags=max_required_inlier_tags)
    if coverage_failure:
        return {'success': False, 'failure_reason': coverage_failure, 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    point_inlier_fraction = float(active.sum()) / max(float(object_points.shape[0]), 1.0)
    if point_inlier_fraction < float(min_point_inlier_fraction):
        return {'success': False, 'failure_reason': f'dense_single_face_point_retention_low:{point_inlier_fraction:.3f}<{float(min_point_inlier_fraction):.3f}', 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'n_points': int(active.sum()), 'n_points_raw': int(object_points.shape[0]), 'point_inlier_fraction': point_inlier_fraction, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    if not dense_deeptag_face_normals_ok(rvec, {face_name}):
        return {'success': False, 'failure_reason': 'dense_single_face_normal_away', 'reproj_error': float('inf'), 'raw_reproj_error': reproj}
    if len(used_ids) >= 2:
        quality_level = 'B'
        quality_reason = f'dense_singleface_face_frame:{len(used_ids)}tags;point_retention:{point_inlier_fraction:.2f}'
    else:
        quality_level = 'C'
        quality_reason = 'dense_singletag_face_frame'
    return {'success': True, 'failure_reason': '', 'pose_source': 'deeptag_dense_keypoints_single_face_ippe_cfg_transform', 'quality_level': quality_level, 'quality_reason': quality_reason, 'pose_filled': False, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': dense_deeptag_transform_from_rvec_tvec(rvec, tvec), 'reproj_error': reproj, 'n_points': int(active.sum()), 'n_points_raw': int(object_points.shape[0]), 'point_inlier_fraction': point_inlier_fraction, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'visible_faces': {face_name}, 'single_face_name': face_name, 'single_face_ippe_candidates': int(candidate_count), 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}

def dense_deeptag_solve_dense_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    tag_ids: list[int],
    point_counts: dict[int, int],
    *,
    cube_config: Any,
    face_id_sets: dict[str, set[int]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_reproj: float,
    max_reproj: float,
    point_reject_px: float,
    tag_reject_px: float,
    min_tags: int,
    min_inlier_tag_fraction: float,
    coverage_check_min_raw_tags: int,
    max_required_inlier_tags: int,
    min_point_inlier_fraction: float = 0.25,
    min_per_tag_inlier_fraction: float = 0.25,
    min_per_tag_inlier_points: int = 12,
) -> dict[str, Any]:
    if object_points.shape[0] < 4:
        return {'success': False, 'failure_reason': 'dense_no_points', 'reproj_error': float('inf')}
    raw_visible_faces = dense_deeptag_visible_faces_for_ids(face_id_sets, tag_ids)
    if len(raw_visible_faces) == 1:
        return dense_deeptag_solve_single_face_dense_pose(object_points, image_points, tag_ids, point_counts, cube_config=cube_config, face_name=next(iter(raw_visible_faces)), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, max_reproj=max_reproj, point_reject_px=point_reject_px, tag_reject_px=tag_reject_px, min_tags=min_tags, min_inlier_tag_fraction=min_inlier_tag_fraction, coverage_check_min_raw_tags=coverage_check_min_raw_tags, max_required_inlier_tags=max_required_inlier_tags, min_point_inlier_fraction=min_point_inlier_fraction, min_per_tag_inlier_fraction=min_per_tag_inlier_fraction, min_per_tag_inlier_points=min_per_tag_inlier_points)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, iterationsCount=300, reprojectionError=float(ransac_reproj), confidence=0.995, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        ok, rvec, tvec, inliers = (False, None, None, None)
    if not ok or rvec is None or tvec is None or (float(np.asarray(tvec).reshape(3)[2]) <= 0.0):
        return {'success': False, 'failure_reason': 'dense_pnp_failed', 'reproj_error': float('inf')}
    active = np.ones(object_points.shape[0], dtype=bool)
    if inliers is not None and len(inliers) >= 4:
        active[:] = False
        active[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    rejected_points = 0
    rejected_tags: list[int] = []
    for _iteration in range(2):
        if int(active.sum()) < 4:
            break
        try:
            rvec, tvec = cv2.solvePnPRefineLM(object_points[active], image_points[active], camera_matrix, dist_coeffs, rvec, tvec)
        except cv2.error:
            pass
        errors = dense_deeptag_project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
        active_errors = errors[active]
        if active_errors.size == 0:
            break
        point_thresh = min(max(float(np.median(active_errors)) * 3.0, 2.0), float(point_reject_px))
        point_keep = errors <= point_thresh
        tag_keep_ids: set[int] = set()
        start = 0
        per_tag_mean: dict[int, float] = {}
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            tag_active = active[start:end] & point_keep[start:end]
            required_tag_points = dense_deeptag_required_inliers_for_tag(
                count,
                min_fraction=min_per_tag_inlier_fraction,
                min_points=min_per_tag_inlier_points,
            )
            if int(tag_active.sum()) >= required_tag_points:
                mean_err = float(np.mean(errors[start:end][tag_active]))
                per_tag_mean[int(tag_id)] = mean_err
                if mean_err <= float(tag_reject_px):
                    tag_keep_ids.add(int(tag_id))
            start = end
        next_active = active & point_keep
        start = 0
        for tag_id in tag_ids:
            count = int(point_counts[int(tag_id)])
            end = start + count
            if int(tag_id) not in tag_keep_ids:
                next_active[start:end] = False
            start = end
        if np.array_equal(next_active, active):
            break
        rejected_points += int(active.sum() - next_active.sum())
        rejected_tags = [int(tag_id) for tag_id in tag_ids if int(tag_id) not in tag_keep_ids]
        active = next_active
    if int(active.sum()) < 4:
        return {'success': False, 'failure_reason': 'dense_too_few_inlier_points', 'reproj_error': float('inf'), 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    try:
        rvec, tvec = cv2.solvePnPRefineLM(object_points[active], image_points[active], camera_matrix, dist_coeffs, rvec, tvec)
    except cv2.error:
        pass
    errors = dense_deeptag_project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    active_errors = errors[active]
    reproj = float(np.mean(active_errors))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return {'success': False, 'failure_reason': f'dense_reproj_too_high:{reproj:.2f}>{float(max_reproj):.2f}', 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    used_ids: list[int] = []
    per_tag_reproj: dict[int, float] = {}
    per_tag_inliers: dict[int, int] = {}
    start = 0
    for tag_id in tag_ids:
        count = int(point_counts[int(tag_id)])
        end = start + count
        tag_active = active[start:end]
        if int(tag_active.sum()) > 0:
            used_ids.append(int(tag_id))
            per_tag_reproj[int(tag_id)] = float(np.mean(errors[start:end][tag_active]))
            per_tag_inliers[int(tag_id)] = int(tag_active.sum())
        start = end
    if len(used_ids) < int(min_tags):
        return {'success': False, 'failure_reason': f'dense_final_tags_too_small:{len(used_ids)}<{int(min_tags)}', 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    coverage_failure = dense_deeptag_inlier_tag_coverage_failure(tag_ids, used_ids, min_tags=min_tags, min_inlier_tag_fraction=min_inlier_tag_fraction, coverage_check_min_raw_tags=coverage_check_min_raw_tags, max_required_inlier_tags=max_required_inlier_tags)
    if coverage_failure:
        return {'success': False, 'failure_reason': coverage_failure, 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    point_inlier_fraction = float(active.sum()) / max(float(object_points.shape[0]), 1.0)
    if point_inlier_fraction < float(min_point_inlier_fraction):
        return {'success': False, 'failure_reason': f'dense_point_retention_low:{point_inlier_fraction:.3f}<{float(min_point_inlier_fraction):.3f}', 'reproj_error': float('inf'), 'raw_reproj_error': reproj, 'n_points': int(active.sum()), 'n_points_raw': int(object_points.shape[0]), 'point_inlier_fraction': point_inlier_fraction, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}
    visible_faces = dense_deeptag_visible_faces_for_ids(face_id_sets, used_ids)
    if not dense_deeptag_face_normals_ok(rvec, visible_faces):
        return {'success': False, 'failure_reason': 'dense_face_normal_away', 'reproj_error': float('inf'), 'raw_reproj_error': reproj}
    if len(visible_faces) >= 2:
        quality_level = 'A'
        quality_reason = f'dense_multiface:{len(visible_faces)}faces/{len(used_ids)}tags;point_retention:{point_inlier_fraction:.2f}'
    elif len(used_ids) >= 2:
        quality_level = 'B'
        quality_reason = f'dense_multitag_singleface:{len(used_ids)}tags;point_retention:{point_inlier_fraction:.2f}'
    else:
        quality_level = 'C'
        quality_reason = 'dense_single_tag_planar'
    return {'success': True, 'failure_reason': '', 'pose_source': 'deeptag_dense_keypoints_all_point_pnp', 'quality_level': quality_level, 'quality_reason': quality_reason, 'pose_filled': False, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': dense_deeptag_transform_from_rvec_tvec(rvec, tvec), 'reproj_error': reproj, 'n_points': int(active.sum()), 'n_points_raw': int(object_points.shape[0]), 'point_inlier_fraction': point_inlier_fraction, 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'visible_faces': visible_faces, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}

def dense_deeptag_sanitize_pose(pose: dict[str, Any]) -> dict[str, Any]:
    return dense_deeptag_to_json_compatible(pose)

def dense_deeptag_make_runtime(header: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(header.get('metadata', {}) or {})
    cube_cfg = Path(metadata['cube_cfg']).expanduser().resolve()
    cfg_path = cube_cfg / 'config.json' if cube_cfg.is_dir() else cube_cfg
    cube_config, face_id_sets = aprilcube.load_cube_config(str(cfg_path))
    camera_matrix = np.asarray(metadata['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get('dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    draw_detector = aprilcube.detector(cube_cfg, intrinsic_cfg={'fx': float(camera_matrix[0, 0]), 'fy': float(camera_matrix[1, 1]), 'cx': float(camera_matrix[0, 2]), 'cy': float(camera_matrix[1, 2])}, dist_coeffs=dist_coeffs, enable_filter=False, fast=True)
    return {'metadata': metadata, 'cube_cfg': cube_cfg, 'cube_config': cube_config, 'face_id_sets': face_id_sets, 'tag_corner_map': aprilcube.build_tag_corner_map(cube_config), 'camera_matrix': camera_matrix, 'dist_coeffs': dist_coeffs, 'draw_detector': draw_detector}

def dense_deeptag_make_source_frame_loader(header: dict[str, Any]) -> tuple[Path | None, dict[int, int], Any | None, tuple | None]:
    source = header.get('source_pkl', '')
    if not source:
        return (None, {}, None, None)
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        return (None, {}, None, None)
    source_header, source_offsets, _source_footer = dense_deeptag_build_stream_index(source_path, {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'})
    offset_set = {int(offset): int(offset) for offset in source_offsets}
    script012 = None
    metadata: dict[str, Any] = {}
    if source_header.get('format') == 'aprilcube_012_raw_with_pose_stream_v1':
        metadata.update(source_header.get('raw_header', {}).get('metadata', {}) or {})
    metadata.update(source_header.get('metadata', {}) or {})
    try:
        intrinsics_yaml = Path(metadata.get('intrinsics_yaml')).expanduser().resolve()
        calib = realsense_load_intrinsics_yaml(intrinsics_yaml)
        image_size = tuple((int(v) for v in metadata.get('image_size', calib['image_size'])))
        undistort_pack = None
        if bool(metadata.get('undistort_for_detection', True)):
            undistort_pack = realsense_create_undistort_maps(calib, image_size)
        return (source_path, offset_set, script012, undistort_pack)
    except Exception:
        return (source_path, offset_set, None, None)

def dense_deeptag_source_detection_frame(source_path: Path | None, source_offsets: dict[int, int], script012: Any | None, undistort_pack: tuple | None, source_offset: int) -> np.ndarray | None:
    if source_path is None or int(source_offset) not in source_offsets:
        return None
    try:
        record = dense_deeptag_load_at(source_path, source_offsets[int(source_offset)])
        image = np.asarray(record['image_bgr'], dtype=np.uint8)
        if script012 is not None:
            return realsense_undistort_frame(image, undistort_pack)
        return image
    except Exception:
        return None

def dense_deeptag_draw_overlay(base_bgr: np.ndarray, runtime: dict[str, Any], pose: dict[str, Any]) -> np.ndarray:
    result = {'success': bool(pose.get('success', False)), 'detections': [], 'rvec': np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'tvec': np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'reproj_error': float(pose.get('reproj_error', float('inf'))), 'n_tags': int(pose.get('n_tags', 0)), 'visible_faces': set(pose.get('visible_faces', []) or []), 'predicted': False}
    vis = runtime['draw_detector'].draw_result(base_bgr.copy(), result)
    text = f"DenseDeepTag success={pose.get('success', False)} tags={pose.get('n_tags', 0)} pts={pose.get('n_points', 0)} reproj={float(pose.get('reproj_error', float('inf'))):.2f}px"
    cv2.rectangle(vis, (8, 8), (900, 42), (0, 0, 0), -1)
    cv2.putText(vis, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return vis

def dense_deeptag_main(args: DenseDeepTagPoseConfig) -> None:
    input_pkl = Path(args.deeptag_pkl).expanduser().resolve()
    header, all_offsets, footer = dense_deeptag_build_stream_index(input_pkl, {'deeptag_012_offline_stream_v1'})
    offsets = all_offsets[int(args.start_frame)::max(int(args.stride), 1)]
    if args.max_frames > 0:
        offsets = offsets[:int(args.max_frames)]
    runtime = dense_deeptag_make_runtime(header)
    unclustered_corner_order = str(
        runtime['metadata'].get('corner_order', 'rot180')
    )
    if unclustered_corner_order not in DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS:
        raise ValueError(
            'DeepTag input metadata has unsupported corner_order: '
            f'{unclustered_corner_order}'
        )
    source_path, source_offsets, script012, undistort_pack = dense_deeptag_make_source_frame_loader(header)
    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    total_points = 0
    t0 = time.perf_counter()
    with output_pkl.open('wb') as f:
        pickle.dump({'type': 'header', 'format': 'deeptag_012_offline_stream_v1', 'source_pkl': str(input_pkl), 'source_footer': footer, 'metadata': {'script': str(DENSE_DEEPTAG_THIS_FILE), 'method': 'DeepTag dense keypoints with per-tag validated point order and point-retention gates; single-face frames use cfg face-frame IPPE then fixed face-to-cube transform; multiface frames use cube-frame all-point PnP; no temporal filter', 'cube_cfg': str(runtime['cube_cfg']), 'camera_matrix': runtime['camera_matrix'].tolist(), 'dist_coeffs': runtime['dist_coeffs'].tolist(), 'frame_count': int(len(offsets)), 'min_tags': int(args.min_tags), 'ransac_reproj': float(args.ransac_reproj), 'max_reproj': float(args.max_reproj), 'point_reject_px': float(args.point_reject_px), 'tag_reject_px': float(args.tag_reject_px), 'min_inlier_tag_fraction': float(args.min_inlier_tag_fraction), 'coverage_check_min_raw_tags': int(args.coverage_check_min_raw_tags), 'max_required_inlier_tags': int(args.max_required_inlier_tags), 'require_validated_corner_order': bool(args.require_validated_corner_order), 'unclustered_corner_order': unclustered_corner_order, 'min_point_inlier_fraction': float(args.min_point_inlier_fraction), 'min_per_tag_inlier_fraction': float(args.min_per_tag_inlier_fraction), 'min_per_tag_inlier_points': int(args.min_per_tag_inlier_points), 'input_header': dense_deeptag_to_json_compatible(header)}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for out_idx, offset in enumerate(offsets):
            frame = dense_deeptag_load_at(input_pkl, offset)
            object_points, image_points, tag_ids, point_counts, dense_stats = dense_deeptag_dense_points_for_frame(frame, tag_corner_map=runtime['tag_corner_map'], face_id_sets=runtime['face_id_sets'], min_tags=int(args.min_tags), require_validated_corner_order=bool(args.require_validated_corner_order), unclustered_corner_order=unclustered_corner_order)
            if object_points.shape[0] >= 4:
                pose = dense_deeptag_solve_dense_pose(object_points, image_points, tag_ids, point_counts, cube_config=runtime['cube_config'], face_id_sets=runtime['face_id_sets'], camera_matrix=runtime['camera_matrix'], dist_coeffs=runtime['dist_coeffs'], ransac_reproj=float(args.ransac_reproj), max_reproj=float(args.max_reproj), point_reject_px=float(args.point_reject_px), tag_reject_px=float(args.tag_reject_px), min_tags=int(args.min_tags), min_inlier_tag_fraction=float(args.min_inlier_tag_fraction), coverage_check_min_raw_tags=int(args.coverage_check_min_raw_tags), max_required_inlier_tags=int(args.max_required_inlier_tags), min_point_inlier_fraction=float(args.min_point_inlier_fraction), min_per_tag_inlier_fraction=float(args.min_per_tag_inlier_fraction), min_per_tag_inlier_points=int(args.min_per_tag_inlier_points))
            else:
                pose = {'success': False, 'failure_reason': str(dense_stats.get('reason', 'dense_no_points')), 'reproj_error': float('inf'), 'n_tags': len(tag_ids), 'tag_ids': tag_ids, 'pose_source': 'deeptag_dense_keypoints_all_point_pnp', 'pose_filled': False}
            pose['dense_stats'] = {**dense_stats, 'raw_tag_ids': tag_ids, 'raw_point_counts': point_counts}
            pose_sanitized = dense_deeptag_sanitize_pose(pose)
            success_count += int(bool(pose.get('success', False)))
            total_points += int(pose.get('n_points', 0) or 0)
            base = None
            if not args.no_source_overlay:
                base = dense_deeptag_source_detection_frame(source_path, source_offsets, script012, undistort_pack, int(frame.get('source_offset', -1)))
            if base is None:
                base = dense_deeptag_decode_jpeg_bgr(frame['overlay_jpeg'])
            overlay = dense_deeptag_draw_overlay(base, runtime, pose)
            frame_record = {'type': 'frame', 'frame_index': int(frame.get('frame_index', out_idx)), 'source_offset': int(frame.get('source_offset', -1)), 'loop_frame_idx': int(frame.get('loop_frame_idx', out_idx)), 'capture_timestamp': frame.get('capture_timestamp', None), 'pose': pose_sanitized, 'dense_point_count': int(object_points.shape[0]), 'overlay_jpeg': dense_deeptag_encode_bgr_jpeg(overlay, int(args.jpeg_quality)), 'overlay_format': 'jpeg_bgr', 'cluster_stats': frame.get('cluster_stats', {}), 'detection_stats': frame.get('detection_stats', {})}
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
            done = out_idx + 1
            if done == len(offsets) or done % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f'\r[INFO] dense pose {done}/{len(offsets)} success={success_count} fps={done / max(elapsed, 1e-09):.1f}', end='', flush=True)
        pickle.dump({'type': 'footer', 'frame_count': int(len(offsets)), 'success_count': int(success_count), 'avg_inlier_points': float(total_points / max(success_count, 1)), 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()
    print(f'[INFO] Saved dense DeepTag pose pkl: {output_pkl}')
    print(f'[INFO] success={success_count}/{len(offsets)}')


# ---- Single-frame recovery primitives and benchmark ----
@dataclass
class RecoveryPoseCandidate:
    success: bool
    method: str
    frame_index: int
    pose: dict[str, Any]
    tag_ids: list[int]
    reproj_error: float
    edge_score: float | None = None
    failure_reason: str = ''

def pose_recovery_build_stream_index(path: Path, formats: set[str] | None=None) -> tuple[dict[str, Any], list[int], dict[str, Any] | None]:
    offsets: list[int] = []
    footer = None
    with path.open('rb') as f:
        header = pickle.load(f)
        if formats is not None and header.get('format') not in formats:
            raise ValueError(f"Unsupported format {header.get('format')} for {path}")
        while True:
            offset = f.tell()
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, dict) and obj.get('type') == 'frame':
                offsets.append(offset)
            elif isinstance(obj, dict) and obj.get('type') == 'footer':
                footer = obj
                break
    return (header, offsets, footer)

def pose_recovery_load_at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as f:
        f.seek(int(offset))
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get('type') != 'frame':
        raise ValueError(f'Offset {offset} in {path} is not a frame')
    return obj

def pose_recovery_load_pose_records(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any] | None]:
    header, offsets, footer = pose_recovery_build_stream_index(path, None)
    frames: dict[int, dict[str, Any]] = {}
    for offset in offsets:
        frame = pose_recovery_load_at(path, offset)
        frames[int(frame['frame_index'])] = frame
    return (header, frames, footer)

def pose_recovery_rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot

def pose_recovery_pose_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = pose_recovery_rotation_from_rvec(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform

def pose_recovery_visible_faces_for_ids(face_id_sets: dict[str, set[int]], tag_ids: list[int]) -> set[str]:
    visible: set[str] = set()
    for tag_id in tag_ids:
        for face, ids in face_id_sets.items():
            if int(tag_id) in ids:
                visible.add(str(face))
    return visible

def pose_recovery_face_normals_ok(rvec: np.ndarray, visible_faces: set[str]) -> bool:
    rot = pose_recovery_rotation_from_rvec(rvec)
    for face_name in visible_faces:
        for face_def in aprilcube.FACE_DEFS:
            if str(face_def[0]) != str(face_name):
                continue
            normal = np.zeros(3, dtype=np.float64)
            normal[int(face_def[1])] = float(face_def[2])
            if float((rot @ normal)[2]) > 0.0:
                return False
    return True

def pose_recovery_project_errors(object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return np.linalg.norm(np.asarray(image_points, dtype=np.float64).reshape(-1, 2) - projected.reshape(-1, 2), axis=1)

def pose_recovery_detections_to_points(detections: list[tuple[int, np.ndarray]], tag_corner_map: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int]]:
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
        return (np.empty((0, 3)), np.empty((0, 2)), [])
    return (np.vstack(obj), np.vstack(img), ids)

def pose_recovery_solve_pose_from_detections(detections: list[tuple[int, np.ndarray]], *, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, method: str, frame_index: int, min_tags: int, max_reproj: float) -> RecoveryPoseCandidate:
    object_points, image_points, tag_ids = pose_recovery_detections_to_points(detections, tag_corner_map)
    if len(tag_ids) < int(min_tags) or object_points.shape[0] < 8:
        return RecoveryPoseCandidate(False, method, frame_index, {}, tag_ids, float('inf'), failure_reason=f'tags_too_small:{len(tag_ids)}')
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, iterationsCount=300, reprojectionError=3.0, confidence=0.995, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        ok, rvec, tvec, inliers = (False, None, None, None)
    if not ok or rvec is None or tvec is None or (float(np.asarray(tvec).reshape(3)[2]) <= 0.0):
        return RecoveryPoseCandidate(False, method, frame_index, {}, tag_ids, float('inf'), failure_reason='pnp_failed')
    active = np.ones(object_points.shape[0], dtype=bool)
    if inliers is not None and len(inliers) >= 8:
        active[:] = False
        active[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    used_tags: list[int] = []
    for idx, tag_id in enumerate(tag_ids):
        if int(active[idx * 4:idx * 4 + 4].sum()) >= 3:
            used_tags.append(int(tag_id))
    if len(used_tags) < int(min_tags):
        return RecoveryPoseCandidate(False, method, frame_index, {}, used_tags, float('inf'), failure_reason=f'inlier_tags_too_small:{len(used_tags)}')
    try:
        rvec, tvec = cv2.solvePnPRefineLM(object_points[active], image_points[active], camera_matrix, dist_coeffs, rvec, tvec)
    except cv2.error:
        pass
    errors = pose_recovery_project_errors(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = float(np.mean(errors[active]))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return RecoveryPoseCandidate(False, method, frame_index, {}, used_tags, reproj, failure_reason=f'reproj_too_high:{reproj:.2f}')
    faces = pose_recovery_visible_faces_for_ids(face_id_sets, used_tags)
    if not pose_recovery_face_normals_ok(rvec, faces):
        return RecoveryPoseCandidate(False, method, frame_index, {}, used_tags, reproj, failure_reason='face_normal_away')
    pose = {'success': True, 'pose_source': method, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': pose_recovery_pose_transform(rvec, tvec), 'reproj_error': reproj, 'n_tags': len(used_tags), 'tag_ids': used_tags, 'visible_faces': sorted(faces), 'pose_filled': False}
    return RecoveryPoseCandidate(True, method, frame_index, pose, used_tags, reproj)

def pose_recovery_make_variants(gray: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    variants: list[tuple[str, np.ndarray, float]] = [('gray', gray, 1.0), ('preprocess', _preprocess(gray), 1.0), ('clahe', _preprocess_clahe(gray, clip_limit=2.5, tile_grid_size=(8, 8)), 1.0), ('sharpen', _sharpen(gray), 1.0), ('gamma07', _gamma_correct(gray, 0.7), 1.0), ('gamma13', _gamma_correct(gray, 1.3), 1.0), ('contrast', _linear_contrast(gray, 1.35, -18.0), 1.0)]
    big = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    variants.append(('scale15_preprocess', _preprocess(big), 1.5))
    return variants

def pose_recovery_detect_sweep(gray: np.ndarray, *, config: Any, valid_ids: set[int]) -> list[tuple[int, np.ndarray]]:
    detectors = [create_detector(config.dict_id, fast=False), create_fallback_detector(config.dict_id)]
    best: dict[int, tuple[float, np.ndarray]] = {}
    for _name, image, scale in pose_recovery_make_variants(gray):
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

def pose_recovery_face_board_pose(detections: list[tuple[int, np.ndarray]], *, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, min_tags: int, max_reproj: float) -> RecoveryPoseCandidate:
    best: RecoveryPoseCandidate | None = None
    for face_name, ids in face_id_sets.items():
        face_dets = [(tag_id, corners) for tag_id, corners in detections if int(tag_id) in ids]
        candidate = pose_recovery_solve_pose_from_detections(face_dets, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, method=f'face_board_{face_name}', frame_index=frame_index, min_tags=min_tags, max_reproj=max_reproj)
        if candidate.success and (best is None or candidate.reproj_error < best.reproj_error):
            best = candidate
    if best is None:
        return RecoveryPoseCandidate(False, 'face_board', frame_index, {}, [], float('inf'), failure_reason='no_face_board_pose')
    best.method = 'face_board'
    best.pose['pose_source'] = 'face_board'
    return best

def pose_recovery_deeptag_cross_validated_pose(deeptag_frame: dict[str, Any], april_detections: list[tuple[int, np.ndarray]], *, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, min_tags: int, max_reproj: float) -> RecoveryPoseCandidate:
    april_by_id = {int(tag_id): np.asarray(corners, dtype=np.float64).reshape(4, 2) for tag_id, corners in april_detections}
    detections: list[tuple[int, np.ndarray]] = []
    for decoded in deeptag_frame.get('decoded_tags', []) or []:
        if not decoded.get('is_valid', False):
            continue
        tag_id = int(decoded.get('tag_id', -1))
        if tag_id not in april_by_id:
            continue
        pts = np.asarray(decoded.get('keypoints_in_images', []), dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        center_dist = float(np.linalg.norm(pts.mean(axis=0) - april_by_id[tag_id].mean(axis=0)))
        if center_dist > 18.0:
            continue
        detections.append((tag_id, april_by_id[tag_id]))
    return pose_recovery_solve_pose_from_detections(detections, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, method='deeptag_apriltag_cross_validated', frame_index=frame_index, min_tags=min_tags, max_reproj=max_reproj)

def pose_recovery_cube_corners(config: Any) -> np.ndarray:
    x, y, z = [float(v) / 2.0 for v in config.box_dims]
    return np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=np.float64)
POSE_RECOVERY_CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

def pose_recovery_prepare_edge_distance_map(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

def pose_recovery_edge_alignment_score_from_distance_map(dist: np.ndarray, pose: dict[str, Any], *, config: Any, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> float:
    if not pose.get('success', False):
        return 0.0
    dist = np.asarray(dist, dtype=np.float32)
    rvec = np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1)
    corners_2d, _ = cv2.projectPoints(pose_recovery_cube_corners(config), rvec, tvec, camera_matrix, dist_coeffs)
    corners_2d = corners_2d.reshape(-1, 2)
    h, w = dist.shape[:2]
    hits = 0
    total = 0
    for a, b in POSE_RECOVERY_CUBE_EDGES:
        p0, p1 = (corners_2d[a], corners_2d[b])
        length = float(np.linalg.norm(p1 - p0))
        samples = max(4, min(40, int(length / 4.0)))
        for t in np.linspace(0.05, 0.95, samples):
            p = p0 * (1.0 - t) + p1 * t
            x, y = (int(round(p[0])), int(round(p[1])))
            if 0 <= x < w and 0 <= y < h:
                total += 1
                if float(dist[y, x]) <= 2.5:
                    hits += 1
    return float(hits / max(total, 1))

def pose_recovery_edge_alignment_score(gray: np.ndarray, pose: dict[str, Any], *, config: Any, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> float:
    if not pose.get('success', False):
        return 0.0
    return pose_recovery_edge_alignment_score_from_distance_map(pose_recovery_prepare_edge_distance_map(gray), pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

def pose_recovery_pkl_pose_candidate(frame: dict[str, Any], method: str, frame_index: int, min_tags: int, max_reproj: float) -> RecoveryPoseCandidate:
    pose = frame.get('pose', {})
    n_tags = int(pose.get('n_tags', 0) or 0)
    try:
        reproj = float(pose.get('reproj_error', float('inf')))
    except (TypeError, ValueError):
        reproj = float('inf')
    if not pose.get('success', False) or n_tags < int(min_tags) or (not np.isfinite(reproj)) or (reproj > float(max_reproj)) or (pose.get('rvec') is None) or (pose.get('tvec') is None):
        return RecoveryPoseCandidate(False, method, frame_index, {}, [], reproj, failure_reason='candidate_not_usable')
    return RecoveryPoseCandidate(True, method, frame_index, dict(pose), [int(v) for v in pose.get('tag_ids', []) or []], reproj)

# ---- Single-frame candidate fusion ----
SINGLE_FRAME_FUSION_THIS_FILE = Path(__file__).resolve()
SINGLE_FRAME_FUSION_APRILCUBE_ROOT = SINGLE_FRAME_FUSION_THIS_FILE.parent.parent
SINGLE_FRAME_FUSION_DEFAULT_RAW_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'
SINGLE_FRAME_FUSION_DEFAULT_DEEPTAG_RAW_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/016_deeptag_robust_cluster_012_rs_raw_frames_20260710_214336.pkl'
SINGLE_FRAME_FUSION_DEFAULT_DEEPTAG_POSE_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336_faceframe_alltags_coverage_mintag2.pkl'
SINGLE_FRAME_FUSION_DEFAULT_APRIL_STRICT_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/014_offline_pose_vis_012_rs_raw_frames_20260710_214336_aprilcube_style_nofill_notagfix.pkl'
SINGLE_FRAME_FUSION_DEFAULT_LOOSE_DEEPTAG_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/020_deeptag_dense_keypoints_pose_012_rs_raw_frames_20260710_214336_faceframe_alltags.pkl'
SINGLE_FRAME_FUSION_DEFAULT_OLD_APRIL_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'
SINGLE_FRAME_FUSION_DEFAULT_OUTPUT_PKL = SINGLE_FRAME_FUSION_APRILCUBE_ROOT / 'recordings/023_fused_all_single_frame_recovery.pkl'

def single_frame_fusion_encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode('.jpg', np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def single_frame_fusion_finite_pose(pose: dict[str, Any], *, min_tags: int, max_reproj: float) -> bool:
    if not bool(pose.get('success', False)):
        return False
    if bool(pose.get('pose_filled', False)) or bool(pose.get('predicted', False)):
        return False
    if int(pose.get('n_tags', 0) or 0) < int(min_tags):
        return False
    if int(pose.get('n_points_raw', 0) or 0) > 0:
        point_fraction = float(pose.get('point_inlier_fraction', 0.0) or 0.0)
        if point_fraction < 0.25:
            return False
    if pose.get('rvec') is None or pose.get('tvec') is None:
        return False
    try:
        chunks = [np.asarray(pose['rvec'], dtype=np.float64).reshape(-1), np.asarray(pose['tvec'], dtype=np.float64).reshape(-1), np.asarray([float(pose.get('reproj_error', float('inf')))], dtype=np.float64)]
    except (TypeError, ValueError):
        return False
    if not all((bool(np.all(np.isfinite(chunk))) for chunk in chunks)):
        return False
    return float(pose.get('reproj_error', float('inf'))) <= float(max_reproj)

def single_frame_fusion_pkl_pose_candidate_no_temporal(bm: Any, frame: dict[str, Any], method: str, frame_index: int, min_tags: int, max_reproj: float) -> Any:
    pose = frame.get('pose', {}) if isinstance(frame, dict) else {}
    if bool(pose.get('pose_filled', False)) or bool(pose.get('predicted', False)):
        return RecoveryPoseCandidate(False, method, frame_index, {}, [], float('inf'), failure_reason='temporal_or_filled_pose')
    return pose_recovery_pkl_pose_candidate(frame, method, frame_index, min_tags, max_reproj)

def single_frame_fusion_copy_pose_with_stage(pose: dict[str, Any], *, source: str, quality_level: str, quality_reason: str, edge_score: float | None=None) -> dict[str, Any]:
    out = copy.deepcopy(pose)
    out['success'] = True
    out['pose_source_original'] = str(out.get('pose_source', ''))
    out['pose_source'] = source
    out['quality_level'] = quality_level
    out['quality_reason'] = quality_reason
    out['pose_filled'] = False
    out['fused_pose'] = True
    out['single_frame_only'] = True
    if edge_score is not None:
        out['edge_score'] = float(edge_score)
    return out

def single_frame_fusion_failure_pose(reason: str) -> dict[str, Any]:
    return {'success': False, 'pose_source': 'fused_failed', 'quality_level': 'Z', 'quality_reason': reason, 'reproj_error': float('inf'), 'pose_filled': False, 'single_frame_only': True}

def single_frame_fusion_minimal_pose(pose: dict[str, Any]) -> dict[str, Any]:
    keys = {'success', 'failure_reason', 'n_tags', 'n_points', 'n_points_raw', 'point_inlier_fraction', 'n_inliers', 'reproj_error', 'tag_ids', 'visible_faces', 'pose_source', 'pose_filled', 'quality_level', 'quality_reason', 'edge_score', 'rvec', 'tvec', 'T', 'per_tag_inlier_points', 'dense_stats'}
    return {key: copy.deepcopy(value) for key, value in pose.items() if key in keys}

def single_frame_fusion_draw_overlay(bm: Any, script012: Any, draw_detector: Any, detect_frame: np.ndarray, pose: dict[str, Any], label: str, reason: str, quality: int) -> bytes:
    base = realsense_make_detector_input_vis(detect_frame)
    result = {'success': bool(pose.get('success', False)), 'detections': [], 'rvec': np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'tvec': np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'reproj_error': float(pose.get('reproj_error', float('inf'))), 'n_tags': int(pose.get('n_tags', 0) or 0), 'visible_faces': set(pose.get('visible_faces', []) or []), 'predicted': False}
    vis = draw_detector.draw_result(base.copy(), result)
    cv2.rectangle(vis, (8, 8), (1100, 66), (0, 0, 0), -1)
    cv2.putText(vis, f"Fused {pose.get('quality_level', 'Z')}: {label}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, reason[:110], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return single_frame_fusion_encode_bgr_jpeg(vis, quality)

def single_frame_fusion_accept_recovery(bm: Any, candidate: Any, gray: np.ndarray, *, config: Any, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, edge_threshold: float) -> bool:
    if not candidate.success:
        return False
    if candidate.edge_score is None:
        candidate.edge_score = pose_recovery_edge_alignment_score(gray, candidate.pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    return float(candidate.edge_score) >= float(edge_threshold)

def single_frame_fusion_tag_center_multiface_pose(bm: Any, detections: list[tuple[int, np.ndarray]], *, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, max_reproj: float) -> Any:
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
    visible_faces = pose_recovery_visible_faces_for_ids(face_id_sets, tag_ids)
    if len(tag_ids) < 4 or len(visible_faces) < 2:
        return RecoveryPoseCandidate(False, 'tag_center_multiface_pnp', frame_index, {}, tag_ids, float('inf'), failure_reason=f'center_tags_or_faces_too_small:{len(tag_ids)}tags/{len(visible_faces)}faces')
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, camera_matrix, dist_coeffs, iterationsCount=500, reprojectionError=5.0, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        ok, rvec, tvec, inliers = (False, None, None, None)
    if not ok or rvec is None or tvec is None or (float(np.asarray(tvec).reshape(3)[2]) <= 0.0):
        return RecoveryPoseCandidate(False, 'tag_center_multiface_pnp', frame_index, {}, tag_ids, float('inf'), failure_reason='center_pnp_failed')
    active = np.ones(len(tag_ids), dtype=bool)
    if inliers is not None and len(inliers) >= 4:
        active[:] = False
        active[np.asarray(inliers, dtype=np.int64).reshape(-1)] = True
    used_ids = [int(tag_ids[i]) for i in range(len(tag_ids)) if bool(active[i])]
    used_faces = pose_recovery_visible_faces_for_ids(face_id_sets, used_ids)
    if len(used_ids) < 4 or len(used_faces) < 2:
        return RecoveryPoseCandidate(False, 'tag_center_multiface_pnp', frame_index, {}, used_ids, float('inf'), failure_reason=f'center_inliers_too_small:{len(used_ids)}tags/{len(used_faces)}faces')
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj[active], img[active], camera_matrix, dist_coeffs, rvec, tvec)
    except cv2.error:
        pass
    errors = pose_recovery_project_errors(obj, img, rvec, tvec, camera_matrix, dist_coeffs)
    reproj = float(np.mean(errors[active]))
    if not np.isfinite(reproj) or reproj > float(max_reproj):
        return RecoveryPoseCandidate(False, 'tag_center_multiface_pnp', frame_index, {}, used_ids, reproj, failure_reason=f'center_reproj_too_high:{reproj:.2f}')
    if not pose_recovery_face_normals_ok(np.asarray(rvec, dtype=np.float64).reshape(3, 1), used_faces):
        return RecoveryPoseCandidate(False, 'tag_center_multiface_pnp', frame_index, {}, used_ids, reproj, failure_reason='center_face_normal_away')
    pose = {'success': True, 'pose_source': 'tag_center_multiface_pnp', 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': pose_recovery_pose_transform(rvec, tvec), 'reproj_error': reproj, 'n_tags': len(used_ids), 'tag_ids': used_ids, 'visible_faces': sorted(used_faces), 'pose_filled': False, 'reproj_metric': 'tag_center_mean_px'}
    return RecoveryPoseCandidate(True, 'tag_center_multiface_pnp', frame_index, pose, used_ids, reproj)

def single_frame_fusion_apriltag_single_tag_pose(bm: Any, detections: list[tuple[int, np.ndarray]], gray: np.ndarray, *, config: Any, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, max_reproj: float, preferred_tag_id: int | None = None) -> Any:
    best: Any | None = None
    ordered_detections = sorted(
        detections,
        key=lambda item: (
            0
            if preferred_tag_id is not None and int(item[0]) == int(preferred_tag_id)
            else 1
        ),
    )
    for tag_id, corners in ordered_detections:
        try:
            ok, rvec, tvec, reproj, _inliers, meta = estimate_single_tag_cube_pose([(int(tag_id), np.asarray(corners, dtype=np.float64).reshape(4, 2))], tag_corner_map, face_id_sets, camera_matrix, dist_coeffs, allow_corner_rotations=not bool(config.tag_pattern_mirrored))
        except cv2.error:
            ok, rvec, tvec, reproj, meta = (False, None, None, float('inf'), {})
        if not ok or rvec is None or tvec is None or (not np.isfinite(reproj)) or (reproj > float(max_reproj)):
            continue
        if float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
            continue
        face_name = meta.get('single_tag_face', None)
        tag_id = int(meta.get('single_tag_id', tag_id))
        pose = {'success': True, 'pose_source': 'apriltag_single_tag_cfg_pose', 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': pose_recovery_pose_transform(rvec, tvec), 'reproj_error': float(reproj), 'n_tags': 1, 'tag_ids': [tag_id], 'visible_faces': [str(face_name)] if face_name else [], 'pose_filled': False, 'single_tag_cfg_pose': True, 'single_tag_meta': meta}
        edge_score = pose_recovery_edge_alignment_score(gray, pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        candidate = RecoveryPoseCandidate(True, 'apriltag_single_tag_cfg_pose', frame_index, pose, [tag_id], float(reproj), edge_score=edge_score)
        candidate_priority = int(
            preferred_tag_id is not None and int(tag_id) == int(preferred_tag_id)
        )
        best_priority = (
            -1
            if best is None
            else int(
                preferred_tag_id is not None
                and int(best.tag_ids[0]) == int(preferred_tag_id)
            )
        )
        if best is None or (candidate_priority, candidate.edge_score, -candidate.reproj_error) > (best_priority, best.edge_score or 0.0, -best.reproj_error):
            best = candidate
    if best is None:
        return RecoveryPoseCandidate(False, 'apriltag_single_tag_cfg_pose', frame_index, {}, [], float('inf'), failure_reason='no_single_tag_cfg_candidate')
    return best

def single_frame_fusion_deeptag_single_tag_dense_pose(bm: Any, dense020: Any, deeptag_frame: dict[str, Any], gray: np.ndarray, *, config: Any, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, max_reproj: float, unclustered_corner_order: str = 'rot180', preferred_tag_id: int | None = None) -> Any:
    object_points, image_points, tag_ids, point_counts, dense_stats = dense_deeptag_dense_points_for_frame(deeptag_frame, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, min_tags=1, unclustered_corner_order=unclustered_corner_order)
    if object_points.shape[0] < 4:
        return RecoveryPoseCandidate(False, 'deeptag_single_tag_dense_pose', frame_index, {}, tag_ids, float('inf'), failure_reason=str(dense_stats.get('reason', 'dense_single_tag_no_points')))
    pose = dense_deeptag_solve_dense_pose(object_points, image_points, tag_ids, point_counts, cube_config=config, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, ransac_reproj=4.0, max_reproj=float(max_reproj), point_reject_px=8.0, tag_reject_px=8.0, min_tags=1, min_inlier_tag_fraction=0.0, coverage_check_min_raw_tags=999, max_required_inlier_tags=4)
    if not bool(pose.get('success', False)):
        return RecoveryPoseCandidate(False, 'deeptag_single_tag_dense_pose', frame_index, {}, tag_ids, float(pose.get('raw_reproj_error', pose.get('reproj_error', float('inf')))), failure_reason=str(pose.get('failure_reason', 'dense_single_tag_failed')))
    used_ids = [int(v) for v in pose.get('tag_ids', []) or []]
    if len(used_ids) != 1:
        return RecoveryPoseCandidate(False, 'deeptag_single_tag_dense_pose', frame_index, {}, used_ids, float(pose.get('reproj_error', float('inf'))), failure_reason=f'dense_single_tag_used_count:{len(used_ids)}')
    pose = copy.deepcopy(pose)
    pose['pose_source'] = 'deeptag_single_tag_dense_pose'
    pose['preferred_single_tag_id'] = preferred_tag_id
    pose['preferred_single_tag_used'] = bool(
        preferred_tag_id is not None and used_ids == [int(preferred_tag_id)]
    )
    pose['dense_stats'] = {**dense_stats, 'raw_tag_ids': tag_ids, 'raw_point_counts': point_counts}
    edge_score = pose_recovery_edge_alignment_score(gray, pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    return RecoveryPoseCandidate(True, 'deeptag_single_tag_dense_pose', frame_index, pose, used_ids, float(pose.get('reproj_error', float('inf'))), edge_score=edge_score)

def single_frame_fusion_main(args: SingleFrameFusionConfig) -> None:
    bm = None
    script012 = None
    dense020 = None
    raw_header, raw_offsets, raw_footer = pose_recovery_build_stream_index(args.raw_pkl, {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'})
    dt_header, dt_frames, dt_footer = pose_recovery_load_pose_records(args.deeptag_pose_pkl)
    ap_header, ap_frames, ap_footer = pose_recovery_load_pose_records(args.april_strict_pkl)
    deeptag_raw_header, deeptag_raw_offsets, deeptag_raw_footer = pose_recovery_build_stream_index(args.deeptag_raw_pkl, {'deeptag_012_offline_stream_v1'})
    unclustered_corner_order = str(
        (deeptag_raw_header.get('metadata', {}) or {}).get(
            'corner_order', 'rot180'
        )
    )
    if unclustered_corner_order not in DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS:
        raise ValueError(
            'DeepTag input metadata has unsupported corner_order: '
            f'{unclustered_corner_order}'
        )
    loose_header, loose_frames, loose_footer = pose_recovery_load_pose_records(args.loose_deeptag_pkl)
    old_header, old_frames, old_footer = pose_recovery_load_pose_records(args.old_april_pkl)
    metadata: dict[str, Any] = {}
    if raw_header.get('format') == 'aprilcube_012_raw_with_pose_stream_v1':
        metadata.update(raw_header.get('raw_header', {}).get('metadata', {}) or {})
    metadata.update(raw_header.get('metadata', {}) or {})
    cube_cfg = Path(metadata['cube_cfg']).expanduser().resolve()
    config, face_id_sets = aprilcube.load_cube_config(str(cube_cfg / 'config.json' if cube_cfg.is_dir() else cube_cfg))
    tag_corner_map = aprilcube.build_tag_corner_map(config)
    valid_ids = set((int(v) for v in tag_corner_map))
    calib = realsense_load_intrinsics_yaml(metadata.get('intrinsics_yaml'))
    image_size = tuple((int(v) for v in metadata.get('image_size', calib['image_size'])))
    undistort_pack = realsense_create_undistort_maps(calib, image_size) if bool(metadata.get('undistort_for_detection', True)) else None
    camera_matrix = np.asarray(metadata.get('detection_camera_matrix', calib['K']), dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    if undistort_pack is not None:
        camera_matrix = np.asarray(metadata.get('detection_camera_matrix', undistort_pack[2]), dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    draw_detector = aprilcube.detector(cube_cfg, intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(camera_matrix), dist_coeffs=dist_coeffs, enable_filter=False, fast=True)
    raw_offset_by_frame = {int(pose_recovery_load_at(args.raw_pkl, offset).get('frame_index', idx)): int(offset) for idx, offset in enumerate(raw_offsets)}
    deeptag_offset_by_frame = {int(pose_recovery_load_at(args.deeptag_raw_pkl, offset).get('frame_index', idx)): int(offset) for idx, offset in enumerate(deeptag_raw_offsets)}
    frame_indices = sorted(dt_frames)
    if set(frame_indices) != set(ap_frames):
        raise ValueError('DeepTag pose pkl and April strict pkl have different frame indices')
    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with args.output_pkl.open('wb') as f:
        pickle.dump({'type': 'header', 'format': 'aprilcube_deeptag_fused_stream_v1', 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_raw_pkl': str(args.raw_pkl.resolve()), 'source_deeptag_pose_pkl': str(args.deeptag_pose_pkl.resolve()), 'source_april_strict_pkl': str(args.april_strict_pkl.resolve()), 'source_deeptag_raw_pkl': str(args.deeptag_raw_pkl.resolve()), 'source_loose_deeptag_pkl': str(args.loose_deeptag_pkl.resolve()), 'source_old_april_pkl': str(args.old_april_pkl.resolve()), 'raw_footer': raw_footer, 'deeptag_footer': dt_footer, 'april_footer': ap_footer, 'deeptag_raw_footer': deeptag_raw_footer, 'loose_footer': loose_footer, 'old_footer': old_footer, 'metadata': {'script': str(SINGLE_FRAME_FUSION_THIS_FILE), 'method': 'single-frame cascade with optional preferred single-tag DeepTag priority; no temporal filter or fill', 'frame_count': len(frame_indices), 'min_tags': int(args.min_tags), 'max_reproj': float(args.max_reproj), 'edge_threshold': float(args.edge_threshold), 'single_tag_edge_threshold': float(args.single_tag_edge_threshold), 'single_tag_max_reproj': float(args.single_tag_max_reproj), 'preferred_single_tag_id': args.preferred_single_tag_id, 'prefer_deeptag_single_tag': bool(args.prefer_deeptag_single_tag), 'unclustered_corner_order': unclustered_corner_order}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for out_idx, frame_index in enumerate(frame_indices):
            dt_frame = dt_frames[frame_index]
            ap_frame = ap_frames[frame_index]
            raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(frame_index)])
            image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
            detect_frame = realsense_undistort_frame(image, undistort_pack)
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            selected = 'failed'
            selected_candidate = None
            pose_candidates: dict[str, Any] = {'deeptag_dense': single_frame_fusion_minimal_pose(dt_frame.get('pose', {})), 'aprilcube_strict': single_frame_fusion_minimal_pose(ap_frame.get('pose', {}))}
            dt_pose = dt_frame.get('pose', {})
            ap_pose = ap_frame.get('pose', {})
            if single_frame_fusion_finite_pose(dt_pose, min_tags=int(args.min_tags), max_reproj=float(args.max_reproj)):
                fused_pose = single_frame_fusion_copy_pose_with_stage(dt_pose, source='stage1_deeptag_dense_coverage_mintag2', quality_level=str(dt_pose.get('quality_level', 'B')), quality_reason=f"deeptag_dense_reproj:{float(dt_pose.get('reproj_error', float('inf'))):.2f};point_retention:{float(dt_pose.get('point_inlier_fraction', 0.0)):.2f}")
                selected = 'stage1_deeptag'
            elif single_frame_fusion_finite_pose(ap_pose, min_tags=int(args.min_tags), max_reproj=float(args.max_reproj)):
                fused_pose = single_frame_fusion_copy_pose_with_stage(ap_pose, source='stage2_aprilcube_strict_mintag2', quality_level='B', quality_reason=f"aprilcube_strict_reproj:{float(ap_pose.get('reproj_error', float('inf'))):.2f}")
                selected = 'stage2_aprilcube_strict'
            else:
                detections = pose_recovery_detect_sweep(gray, config=config, valid_ids=valid_ids)
                deeptag_raw = pose_recovery_load_at(args.deeptag_raw_pkl, deeptag_offset_by_frame[int(frame_index)])
                stage_candidates = [('stage3_tag_center_multiface_pnp', 'C', single_frame_fusion_tag_center_multiface_pose(bm, detections, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), max_reproj=2.0)), ('stage4_single_face_board', 'D', pose_recovery_face_board_pose(detections, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), min_tags=int(args.min_tags), max_reproj=float(args.max_reproj))), ('stage5_apriltag_preproc_sweep', 'E', pose_recovery_solve_pose_from_detections(detections, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, method='apriltag_preproc_sweep', frame_index=int(frame_index), min_tags=int(args.min_tags), max_reproj=float(args.max_reproj))), ('stage6_deeptag_apriltag_cross_validated', 'F', pose_recovery_deeptag_cross_validated_pose(deeptag_raw, detections, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), min_tags=int(args.min_tags), max_reproj=float(args.max_reproj)))]
                loose_sources = [single_frame_fusion_pkl_pose_candidate_no_temporal(bm, loose_frames.get(frame_index, {}), 'loose_deeptag_edge_checked', int(frame_index), int(args.min_tags), float(args.max_reproj)), single_frame_fusion_pkl_pose_candidate_no_temporal(bm, old_frames.get(frame_index, {}), 'old_april_edge_checked', int(frame_index), int(args.min_tags), float(args.max_reproj))]
                best_edge = None
                for cand in loose_sources:
                    if not cand.success:
                        continue
                    cand.edge_score = pose_recovery_edge_alignment_score(gray, cand.pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
                    if cand.edge_score >= float(args.edge_threshold) and (best_edge is None or (cand.edge_score, -cand.reproj_error) > (best_edge.edge_score or 0.0, -best_edge.reproj_error)):
                        best_edge = cand
                stage8 = ('stage8_apriltag_single_tag_cfg_edge', 'H', single_frame_fusion_apriltag_single_tag_pose(bm, detections, gray, config=config, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), max_reproj=float(args.single_tag_max_reproj), preferred_tag_id=args.preferred_single_tag_id))
                stage9 = ('stage9_deeptag_single_tag_dense_edge', 'I', single_frame_fusion_deeptag_single_tag_dense_pose(bm, dense020, deeptag_raw, gray, config=config, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), max_reproj=float(args.single_tag_max_reproj), unclustered_corner_order=unclustered_corner_order, preferred_tag_id=args.preferred_single_tag_id))
                stage_candidates.append(('stage7_edge_checked_loose_candidate', 'G', best_edge or RecoveryPoseCandidate(False, 'edge_checked_loose_candidates', int(frame_index), {}, [], float('inf'), failure_reason='no_edge_accepted_candidate')))
                stage_candidates.extend([stage9, stage8] if args.prefer_deeptag_single_tag else [stage8, stage9])
                pose_candidates['recovery_detected_tag_ids'] = [int(v[0]) for v in detections]
                fused_pose = single_frame_fusion_failure_pose('no_single_frame_method_accepted')
                for source, quality, candidate in stage_candidates:
                    pose_candidates[source] = {'success': bool(candidate.success), 'failure_reason': candidate.failure_reason, 'n_tags': len(candidate.tag_ids), 'tag_ids': candidate.tag_ids, 'reproj_error': candidate.reproj_error, 'edge_score': candidate.edge_score, 'pose_source': candidate.pose.get('pose_source', candidate.method) if candidate.pose else candidate.method}
                    if single_frame_fusion_accept_recovery(bm, candidate, gray, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, edge_threshold=float(args.single_tag_edge_threshold) if source in {'stage8_apriltag_single_tag_cfg_edge', 'stage9_deeptag_single_tag_dense_edge'} else float(args.edge_threshold)):
                        fused_pose = single_frame_fusion_copy_pose_with_stage(candidate.pose, source=source, quality_level=quality, quality_reason=f'{source}_reproj:{candidate.reproj_error:.2f};edge:{float(candidate.edge_score):.2f}', edge_score=candidate.edge_score)
                        selected = source
                        selected_candidate = candidate
                        break
            quality = str(fused_pose.get('quality_level', 'Z'))
            source = str(fused_pose.get('pose_source', ''))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(fused_pose.get('success', False)))
            overlay_jpeg = single_frame_fusion_draw_overlay(bm, script012, draw_detector, detect_frame, fused_pose, source, str(fused_pose.get('quality_reason', '')), int(args.jpeg_quality))
            frame_record = {'type': 'frame', 'frame_index': int(frame_index), 'source_offset': int(raw_offset_by_frame[int(frame_index)]), 'loop_frame_idx': int(raw_record.get('loop_frame_idx', frame_index)), 'capture_timestamp': raw_record.get('capture_timestamp', None), 'overlay_shape': tuple((int(v) for v in detect_frame.shape)), 'overlay_format': 'jpeg_bgr', 'overlay_jpeg': overlay_jpeg, 'pose': fused_pose, 'pose_candidates': pose_candidates, 'selected_stage': selected, 'selected_candidate_reproj': None if selected_candidate is None else float(selected_candidate.reproj_error), 'selected_candidate_edge_score': None if selected_candidate is None else float(selected_candidate.edge_score or 0.0)}
            pickle.dump(frame_record, f, protocol=pickle.HIGHEST_PROTOCOL)
            done = out_idx + 1
            if done == len(frame_indices) or done % 25 == 0:
                elapsed = time.perf_counter() - t0
                print(f'\r[INFO] fused all {done}/{len(frame_indices)} success={success_count} fps={done / max(elapsed, 1e-09):.1f}', end='', flush=True)
        footer = {'type': 'footer', 'frame_count': len(frame_indices), 'success_count': int(success_count), 'quality_counts': quality_counts, 'source_counts': source_counts, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}
        pickle.dump(footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print()
    print(f'[INFO] saved {args.output_pkl}')
    print(f'[INFO] success={success_count}/{len(frame_indices)} quality_counts={quality_counts}')
    print(f'[INFO] source_counts={source_counts}')


# ---- Bidirectional temporal outlier rejection ----
TEMPORAL_OUTLIER_THIS_FILE = Path(__file__).resolve()


def temporal_outlier_rotation_matrix(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot


def temporal_outlier_rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    rot_a = temporal_outlier_rotation_matrix(rvec_a)
    rot_b = temporal_outlier_rotation_matrix(rvec_b)
    cosine = np.clip((np.trace(rot_a.T @ rot_b) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def temporal_outlier_translation_delta_mm(tvec_a: Any, tvec_b: Any) -> float:
    a = np.asarray(tvec_a, dtype=np.float64).reshape(3)
    b = np.asarray(tvec_b, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(a - b))


def temporal_outlier_interpolated_pose(
    before_pose: dict[str, Any],
    after_pose: dict[str, Any],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    before_t = np.asarray(before_pose['tvec'], dtype=np.float64).reshape(3)
    after_t = np.asarray(after_pose['tvec'], dtype=np.float64).reshape(3)
    tvec = ((1.0 - alpha) * before_t + alpha * after_t).reshape(3, 1)
    rotations = Rotation.from_matrix(np.stack([
        temporal_outlier_rotation_matrix(before_pose['rvec']),
        temporal_outlier_rotation_matrix(after_pose['rvec']),
    ]))
    rvec = Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0].reshape(3, 1)
    return rvec, tvec


def temporal_outlier_rejection_main(args: TemporalOutlierRejectionConfig) -> None:
    header, frames, footer = pose_recovery_load_pose_records(args.input_pkl)
    indices = sorted(frames)
    valid_indices = [
        idx
        for idx in indices
        if bool(frames[idx].get('pose', {}).get('success', False))
        and frames[idx].get('pose', {}).get('rvec') is not None
        and frames[idx].get('pose', {}).get('tvec') is not None
    ]
    rejected: dict[int, dict[str, Any]] = {}
    for position, idx in enumerate(valid_indices):
        if position == 0 or position + 1 >= len(valid_indices):
            continue
        before_idx = int(valid_indices[position - 1])
        after_idx = int(valid_indices[position + 1])
        before_gap = int(idx - before_idx)
        after_gap = int(after_idx - idx)
        if before_gap <= 0 or after_gap <= 0:
            continue
        if before_gap > int(args.max_neighbor_gap) or after_gap > int(args.max_neighbor_gap):
            continue
        before_pose = frames[before_idx]['pose']
        pose = frames[idx]['pose']
        after_pose = frames[after_idx]['pose']
        bracket_gap = int(after_idx - before_idx)
        alpha = float(idx - before_idx) / float(bracket_gap)
        predicted_rvec, predicted_tvec = temporal_outlier_interpolated_pose(
            before_pose,
            after_pose,
            alpha,
        )
        rotation_residual = temporal_outlier_rotation_delta_deg(pose['rvec'], predicted_rvec)
        translation_residual = temporal_outlier_translation_delta_mm(pose['tvec'], predicted_tvec)
        rotation_before = temporal_outlier_rotation_delta_deg(before_pose['rvec'], pose['rvec'])
        rotation_after = temporal_outlier_rotation_delta_deg(pose['rvec'], after_pose['rvec'])
        translation_before = temporal_outlier_translation_delta_mm(before_pose['tvec'], pose['tvec'])
        translation_after = temporal_outlier_translation_delta_mm(pose['tvec'], after_pose['tvec'])
        bracket_rotation = temporal_outlier_rotation_delta_deg(before_pose['rvec'], after_pose['rvec'])
        bracket_translation = temporal_outlier_translation_delta_mm(before_pose['tvec'], after_pose['tvec'])
        rotation_rate = bracket_rotation / max(float(bracket_gap), 1.0)
        translation_rate = bracket_translation / max(float(bracket_gap), 1.0)
        rotation_threshold = max(float(args.rotation_residual_deg), rotation_rate * 3.0 + 8.0)
        translation_threshold = max(float(args.translation_residual_mm), translation_rate * 3.0 + 6.0)
        rotation_outlier = (
            rotation_residual > rotation_threshold
            and min(rotation_before, rotation_after) > float(args.min_two_sided_rotation_jump_deg)
        )
        translation_outlier = (
            translation_residual > translation_threshold
            and min(translation_before, translation_after) > float(args.min_two_sided_translation_jump_mm)
        )
        if not rotation_outlier and not translation_outlier:
            continue
        rejected[int(idx)] = {
            'rotation_outlier': bool(rotation_outlier),
            'translation_outlier': bool(translation_outlier),
            'rotation_residual_deg': float(rotation_residual),
            'translation_residual_mm': float(translation_residual),
            'rotation_before_deg': float(rotation_before),
            'rotation_after_deg': float(rotation_after),
            'translation_before_mm': float(translation_before),
            'translation_after_mm': float(translation_after),
            'rotation_threshold_deg': float(rotation_threshold),
            'translation_threshold_mm': float(translation_threshold),
            'before_frame': int(before_idx),
            'after_frame': int(after_idx),
            'bracket_rotation_deg': float(bracket_rotation),
            'bracket_translation_mm': float(bracket_translation),
        }
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    with args.output_pkl.open('wb') as f:
        out_header = copy.deepcopy(header)
        out_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        out_header['source_input_pkl'] = str(args.input_pkl.resolve())
        out_header['metadata'] = {
            **(out_header.get('metadata', {}) or {}),
            'script': str(TEMPORAL_OUTLIER_THIS_FILE),
            'method': 'reject two-sided SE(3) temporal spikes before outline recovery or interpolation',
            'temporal_outlier_max_neighbor_gap': int(args.max_neighbor_gap),
            'temporal_outlier_rotation_residual_deg': float(args.rotation_residual_deg),
            'temporal_outlier_translation_residual_mm': float(args.translation_residual_mm),
        }
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            if idx in rejected:
                original_pose = copy.deepcopy(frame.get('pose', {}))
                metrics = copy.deepcopy(rejected[idx])
                frame['pose_temporal_outlier_original'] = original_pose
                frame.setdefault('pose_candidates', {})['stage_temporal_outlier_rejection'] = {
                    'success': False,
                    'failure_reason': 'bidirectional_temporal_outlier',
                    **metrics,
                }
                frame['pose'] = {
                    'success': False,
                    'failure_reason': 'bidirectional_temporal_outlier',
                    'pose_source': 'fused_failed',
                    'quality_level': 'Z',
                    'quality_reason': (
                        f"temporal_outlier;rot_residual:{metrics['rotation_residual_deg']:.1f};"
                        f"trans_residual:{metrics['translation_residual_mm']:.1f};"
                        f"bracket:{metrics['before_frame']}-{metrics['after_frame']}"
                    ),
                    'pose_filled': False,
                    'single_frame_only': False,
                    'temporal_outlier_rejected': True,
                    'temporal_outlier_metrics': metrics,
                    'reproj_error': float('inf'),
                    'n_tags': 0,
                    'tag_ids': [],
                    'visible_faces': [],
                }
                frame['selected_stage'] = 'stage_temporal_outlier_rejection'
            pose = frame.get('pose', {})
            quality = str(pose.get('quality_level', 'Z'))
            source = str(pose.get('pose_source', ''))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(pose.get('success', False)))
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        out_footer = {
            'type': 'footer',
            'frame_count': int(len(indices)),
            'success_count': int(success_count),
            'temporal_outlier_rejected_count': int(len(rejected)),
            'temporal_outlier_rejected_frames': sorted((int(v) for v in rejected)),
            'temporal_outlier_metrics': rejected,
            'quality_counts': quality_counts,
            'source_counts': source_counts,
            'input_footer': footer,
            'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        pickle.dump(out_footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[INFO] saved {args.output_pkl}')
    print(f'[INFO] temporal outliers rejected={len(rejected)} frames={sorted(rejected)}')
    print(f'[INFO] success={success_count}/{len(indices)}')


# ---- Temporal outline recovery ----
OUTLINE_RECOVERY_THIS_FILE = Path(__file__).resolve()
OUTLINE_RECOVERY_APRILCUBE_ROOT = OUTLINE_RECOVERY_THIS_FILE.parent.parent
OUTLINE_RECOVERY_DEFAULT_INPUT_PKL = OUTLINE_RECOVERY_APRILCUBE_ROOT / 'recordings/023_fused_all_single_frame_recovery_edge045_centerpnp_singletag.pkl'
OUTLINE_RECOVERY_DEFAULT_RAW_PKL = OUTLINE_RECOVERY_APRILCUBE_ROOT / 'recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'
OUTLINE_RECOVERY_DEFAULT_OUTPUT_PKL = OUTLINE_RECOVERY_APRILCUBE_ROOT / 'recordings/024_temporal_outline_refine_recovery.pkl'

def outline_recovery_encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode('.jpg', np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def outline_recovery_rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot

def outline_recovery_rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    ra = Rotation.from_matrix(outline_recovery_rotation_from_rvec(rvec_a))
    rb = Rotation.from_matrix(outline_recovery_rotation_from_rvec(rvec_b))
    return float(np.degrees((rb * ra.inv()).magnitude()))

def outline_recovery_interpolate_pose(prev_pose: dict[str, Any], next_pose: dict[str, Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    t0 = np.asarray(prev_pose['tvec'], dtype=np.float64).reshape(3, 1)
    t1 = np.asarray(next_pose['tvec'], dtype=np.float64).reshape(3, 1)
    tvec = (1.0 - alpha) * t0 + alpha * t1
    r0 = Rotation.from_matrix(outline_recovery_rotation_from_rvec(prev_pose['rvec']))
    r1 = Rotation.from_matrix(outline_recovery_rotation_from_rvec(next_pose['rvec']))
    r = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))([alpha])[0]
    rvec = r.as_rotvec().reshape(3, 1)
    return (rvec.astype(np.float64), tvec.astype(np.float64))

def outline_recovery_cube_corners(config: Any) -> np.ndarray:
    x, y, z = [float(v) / 2.0 for v in config.box_dims]
    return np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z], [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=np.float64)
OUTLINE_RECOVERY_CUBE_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

def outline_recovery_edge_distance_cost(dist: np.ndarray, corners_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> float:
    if float(np.asarray(tvec, dtype=np.float64).reshape(3)[2]) <= 0.0:
        return 10000.0
    projected, _ = cv2.projectPoints(corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = projected.reshape(-1, 2)
    h, w = dist.shape[:2]
    values: list[float] = []
    outside = 0
    for a, b in OUTLINE_RECOVERY_CUBE_EDGES:
        p0, p1 = (pts[a], pts[b])
        length = float(np.linalg.norm(p1 - p0))
        samples = max(8, min(70, int(length / 3.0)))
        for t in np.linspace(0.05, 0.95, samples):
            p = p0 * (1.0 - t) + p1 * t
            x, y = (int(round(p[0])), int(round(p[1])))
            if 0 <= x < w and 0 <= y < h:
                values.append(min(float(dist[y, x]), 12.0))
            else:
                outside += 1
                values.append(12.0)
    if not values:
        return 10000.0
    return float(np.mean(values) + 0.05 * outside)

def outline_recovery_detected_tag_points(bm: Any, gray: np.ndarray, *, config: Any, tag_corner_map: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    detections = pose_recovery_detect_sweep(gray, config=config, valid_ids=set((int(v) for v in tag_corner_map)))
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
        return (np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), [])
    return (np.vstack(obj_chunks), np.vstack(img_chunks), tag_ids)

def outline_recovery_tag_reprojection_error(object_points: np.ndarray, image_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> tuple[float, float]:
    if object_points.shape[0] == 0:
        return (float('inf'), float('inf'))
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)
    return (float(np.mean(errors)), float(np.max(errors)))

def outline_recovery_refine_pose_from_outline(gray: np.ndarray, init_rvec: np.ndarray, init_tvec: np.ndarray, *, config: Any, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, tag_object_points: np.ndarray | None=None, tag_image_points: np.ndarray | None=None, tag_anchor_weight: float=0.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    corners_3d = outline_recovery_cube_corners(config)
    init_cost = outline_recovery_edge_distance_cost(dist, corners_3d, init_rvec, init_tvec, camera_matrix, dist_coeffs)
    init_t = np.asarray(init_tvec, dtype=np.float64).reshape(3)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_rot = Rotation.from_rotvec(np.asarray(x[:3], dtype=np.float64))
        init_rot = Rotation.from_matrix(outline_recovery_rotation_from_rvec(init_rvec))
        rot = delta_rot * init_rot
        rvec = rot.as_rotvec().reshape(3, 1)
        tvec = (init_t + np.asarray(x[3:6], dtype=np.float64)).reshape(3, 1)
        return (rvec, tvec)

    def objective(x: np.ndarray) -> float:
        rvec, tvec = unpack(x)
        data_cost = outline_recovery_edge_distance_cost(dist, corners_3d, rvec, tvec, camera_matrix, dist_coeffs)
        rot_reg = float(np.linalg.norm(x[:3]) / 0.22) ** 2
        t_reg = float(np.linalg.norm(x[3:6] / np.array([22.0, 22.0, 35.0], dtype=np.float64))) ** 2
        tag_cost = 0.0
        if tag_anchor_weight > 0.0 and tag_object_points is not None and (tag_image_points is not None) and (tag_object_points.shape[0] > 0):
            tag_mean, _tag_max = outline_recovery_tag_reprojection_error(tag_object_points, tag_image_points, rvec, tvec, camera_matrix, dist_coeffs)
            tag_cost = min(float(tag_mean), 50.0) / 5.0
        return data_cost + float(tag_anchor_weight) * tag_cost + 0.25 * rot_reg + 0.2 * t_reg
    best = np.zeros(6, dtype=np.float64)
    seeds = [np.zeros(6, dtype=np.float64), np.array([0.0, 0.0, 0.0, -8.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, -8.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 8.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0, -12.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 12.0])]
    bounds = [(-0.32, 0.32), (-0.32, 0.32), (-0.32, 0.32), (-35.0, 35.0), (-35.0, 35.0), (-45.0, 45.0)]
    best_value = objective(best)
    for seed in seeds:
        result = minimize(objective, seed, method='Powell', bounds=bounds, options={'maxiter': 90, 'xtol': 0.001, 'ftol': 0.001, 'disp': False})
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best = np.asarray(result.x, dtype=np.float64)
    rvec, tvec = unpack(best)
    return (rvec, tvec, init_cost, outline_recovery_edge_distance_cost(dist, corners_3d, rvec, tvec, camera_matrix, dist_coeffs))

def outline_recovery_draw_overlay(script012: Any, draw_detector: Any, detect_frame: np.ndarray, pose: dict[str, Any], quality: int) -> bytes:
    base = realsense_make_detector_input_vis(detect_frame)
    result = {'success': bool(pose.get('success', False)), 'detections': [], 'rvec': np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'tvec': np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'reproj_error': float(pose.get('reproj_error', float('inf'))), 'n_tags': int(pose.get('n_tags', 0) or 0), 'visible_faces': set(pose.get('visible_faces', []) or []), 'predicted': False}
    vis = draw_detector.draw_result(base.copy(), result)
    cv2.rectangle(vis, (8, 8), (1180, 66), (0, 0, 0), -1)
    cv2.putText(vis, f"Temporal outline: {pose.get('pose_source', '')}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, str(pose.get('quality_reason', ''))[:120], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return outline_recovery_encode_bgr_jpeg(vis, quality)

def outline_recovery_input_pose_usable(frame: dict[str, Any], *, reject_loose_input: bool) -> bool:
    pose = frame.get('pose', {})
    if not bool(pose.get('success', False)):
        return False
    if bool(reject_loose_input) and str(pose.get('pose_source', '')) == 'stage7_edge_checked_loose_candidate':
        return False
    return True

def outline_recovery_main(args: OutlinePoseRecoveryConfig) -> None:
    bm = None
    script012 = None
    header, frames, footer = pose_recovery_load_pose_records(args.input_pkl)
    raw_header, raw_offsets, raw_footer = pose_recovery_build_stream_index(args.raw_pkl, {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'})
    metadata: dict[str, Any] = {}
    if raw_header.get('format') == 'aprilcube_012_raw_with_pose_stream_v1':
        metadata.update(raw_header.get('raw_header', {}).get('metadata', {}) or {})
    metadata.update(raw_header.get('metadata', {}) or {})
    cube_cfg = Path(metadata['cube_cfg']).expanduser().resolve()
    config, _face_id_sets = aprilcube.load_cube_config(str(cube_cfg / 'config.json' if cube_cfg.is_dir() else cube_cfg))
    tag_corner_map = aprilcube.build_tag_corner_map(config)
    calib = realsense_load_intrinsics_yaml(metadata.get('intrinsics_yaml'))
    image_size = tuple((int(v) for v in metadata.get('image_size', calib['image_size'])))
    undistort_pack = realsense_create_undistort_maps(calib, image_size) if bool(metadata.get('undistort_for_detection', True)) else None
    camera_matrix = np.asarray(metadata.get('detection_camera_matrix', calib['K']), dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    if undistort_pack is not None:
        camera_matrix = np.asarray(metadata.get('detection_camera_matrix', undistort_pack[2]), dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    raw_offset_by_frame = {int(pose_recovery_load_at(args.raw_pkl, offset).get('frame_index', idx)): int(offset) for idx, offset in enumerate(raw_offsets)}
    draw_detector = aprilcube.detector(cube_cfg, intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(camera_matrix), dist_coeffs=dist_coeffs, enable_filter=False, fast=True)
    indices = sorted(frames)
    success_indices = [idx for idx in indices if outline_recovery_input_pose_usable(frames[idx], reject_loose_input=bool(args.reject_loose_input))]
    recovered: dict[int, dict[str, Any]] = {}
    rejected: dict[int, str] = {}
    for idx in indices:
        if outline_recovery_input_pose_usable(frames[idx], reject_loose_input=bool(args.reject_loose_input)):
            continue
        prevs = [v for v in success_indices if v < idx]
        nexts = [v for v in success_indices if v > idx]
        if not prevs or not nexts:
            rejected[idx] = 'no_bracketing_success_pose'
            continue
        prev_idx, next_idx = (prevs[-1], nexts[0])
        gap = int(next_idx - prev_idx)
        if gap > int(args.max_gap):
            rejected[idx] = f'bracket_gap_too_large:{gap}>{int(args.max_gap)}'
            continue
        alpha = float(idx - prev_idx) / float(max(gap, 1))
        init_rvec, init_tvec = outline_recovery_interpolate_pose(frames[prev_idx]['pose'], frames[next_idx]['pose'], alpha)
        raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
        image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
        detect_frame = realsense_undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        init_pose = {'success': True, 'rvec': init_rvec, 'tvec': init_tvec, 'pose_filled': True}
        init_edge = pose_recovery_edge_alignment_score(gray, init_pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        tag_object_points, tag_image_points, detected_tag_ids = outline_recovery_detected_tag_points(bm, gray, config=config, tag_corner_map=tag_corner_map)
        use_tag_anchor = len(detected_tag_ids) == 1
        if not use_tag_anchor and init_edge >= float(args.use_interp_if_edge):
            opt_rvec, opt_tvec = (init_rvec, init_tvec)
            init_cost = outline_recovery_edge_distance_cost(cv2.distanceTransform(255 - cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140), cv2.DIST_L2, 3), outline_recovery_cube_corners(config), init_rvec, init_tvec, camera_matrix, dist_coeffs)
            opt_cost = init_cost
            used_interp_direct = True
        else:
            used_interp_direct = False
            opt_rvec, opt_tvec, init_cost, opt_cost = outline_recovery_refine_pose_from_outline(gray, init_rvec, init_tvec, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, tag_object_points=tag_object_points if use_tag_anchor else None, tag_image_points=tag_image_points if use_tag_anchor else None, tag_anchor_weight=float(args.tag_anchor_weight) if use_tag_anchor else 0.0)
        opt_pose = {'success': True, 'rvec': opt_rvec, 'tvec': opt_tvec, 'pose_filled': True}
        opt_edge = pose_recovery_edge_alignment_score(gray, opt_pose, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        tag_mean_reproj, tag_max_reproj = outline_recovery_tag_reprojection_error(tag_object_points, tag_image_points, opt_rvec, opt_tvec, camera_matrix, dist_coeffs) if len(detected_tag_ids) > 0 else (float('inf'), float('inf'))
        trans_delta = float(np.linalg.norm(np.asarray(opt_tvec).reshape(3) - np.asarray(init_tvec).reshape(3)))
        rot_delta = outline_recovery_rotation_delta_deg(init_rvec, opt_rvec)
        accept_edge = float(args.tag_anchor_accept_edge) if use_tag_anchor else float(args.accept_edge)
        if opt_edge < accept_edge:
            rejected[idx] = f'edge_too_low:{opt_edge:.3f}<{accept_edge:.3f}'
            continue
        if use_tag_anchor and tag_mean_reproj > float(args.tag_anchor_max_reproj):
            rejected[idx] = f'tag_anchor_reproj_too_high:{tag_mean_reproj:.2f}>{float(args.tag_anchor_max_reproj):.2f}'
            continue
        if not use_tag_anchor and opt_edge < init_edge + float(args.min_improvement) and (init_edge < float(args.accept_edge)):
            rejected[idx] = f'edge_improvement_too_small:{init_edge:.3f}->{opt_edge:.3f}'
            continue
        if trans_delta > float(args.max_translation_delta_mm):
            rejected[idx] = f'translation_delta_too_large:{trans_delta:.1f}'
            continue
        if rot_delta > float(args.max_rotation_delta_deg):
            rejected[idx] = f'rotation_delta_too_large:{rot_delta:.1f}'
            continue
        pose = {'success': True, 'pose_source': 'stage10_temporal_tag_outline_refine' if use_tag_anchor else 'stage10_temporal_interp' if used_interp_direct else 'stage10_temporal_outline_refine', 'quality_level': 'T', 'quality_reason': f'bracket:{prev_idx}-{next_idx};edge:{init_edge:.2f}->{opt_edge:.2f};cost:{init_cost:.2f}->{opt_cost:.2f};dt:{trans_delta:.1f}mm;dr:{rot_delta:.1f}deg;tag_anchor:{(detected_tag_ids if use_tag_anchor else [])};tag_reproj:{tag_mean_reproj:.2f};interp_direct:{used_interp_direct}', 'pose_filled': True, 'temporal_recovery': True, 'single_frame_only': False, 'rvec': opt_rvec, 'tvec': opt_tvec, 'T': pose_recovery_pose_transform(opt_rvec, opt_tvec), 'reproj_error': float('nan'), 'n_tags': 0, 'tag_ids': [], 'visible_faces': [], 'edge_score': float(opt_edge), 'init_edge_score': float(init_edge), 'outline_cost': float(opt_cost), 'init_outline_cost': float(init_cost), 'prev_success_frame': int(prev_idx), 'next_success_frame': int(next_idx), 'interpolation_alpha': float(alpha), 'temporal_init_rvec': init_rvec, 'temporal_init_tvec': init_tvec, 'temporal_delta_t_mm': float(trans_delta), 'temporal_delta_r_deg': float(rot_delta), 'detected_tag_ids': detected_tag_ids, 'tag_anchor_used': bool(use_tag_anchor), 'tag_anchor_reproj_error': float(tag_mean_reproj), 'tag_anchor_max_reproj_error': float(tag_max_reproj), 'interp_direct': bool(used_interp_direct)}
        recovered[idx] = pose
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    with args.output_pkl.open('wb') as f:
        out_header = copy.deepcopy(header)
        out_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        out_header['source_input_pkl'] = str(args.input_pkl.resolve())
        out_header['source_raw_pkl'] = str(args.raw_pkl.resolve())
        out_header['raw_footer'] = raw_footer
        out_header.setdefault('metadata', {})
        out_header['metadata'] = {**(out_header.get('metadata', {}) or {}), 'script': str(OUTLINE_RECOVERY_THIS_FILE), 'method': 'temporal interpolation followed by current-frame RGB cube-outline edge refinement', 'max_gap': int(args.max_gap), 'accept_edge': float(args.accept_edge), 'min_improvement': float(args.min_improvement)}
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            if idx in recovered:
                raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
                image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
                detect_frame = realsense_undistort_frame(image, undistort_pack)
                frame['pose_original'] = copy.deepcopy(frame.get('pose', {}))
                frame['pose'] = recovered[idx]
                frame['selected_stage'] = 'stage10_temporal_outline_refine'
                frame['overlay_jpeg'] = outline_recovery_draw_overlay(script012, draw_detector, detect_frame, recovered[idx], int(args.jpeg_quality))
                frame['overlay_format'] = 'jpeg_bgr'
                frame['overlay_shape'] = tuple((int(v) for v in detect_frame.shape))
            elif not outline_recovery_input_pose_usable(frame, reject_loose_input=bool(args.reject_loose_input)):
                original_pose = copy.deepcopy(frame.get('pose', {}))
                frame['pose_original'] = original_pose
                frame['pose'] = {'success': False, 'pose_source': 'fused_failed', 'quality_level': 'Z', 'quality_reason': rejected.get(idx, 'input_pose_rejected'), 'failure_reason': rejected.get(idx, 'input_pose_rejected'), 'reproj_error': float('inf'), 'pose_filled': False, 'single_frame_only': False}
                frame.setdefault('pose_candidates', {})
                frame['pose_candidates']['stage10_temporal_outline_refine'] = {'success': False, 'failure_reason': rejected.get(idx, 'not_attempted')}
            pose = frame.get('pose', {})
            quality = str(pose.get('quality_level', 'Z'))
            source = str(pose.get('pose_source', ''))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(pose.get('success', False)))
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        out_footer = {'type': 'footer', 'frame_count': len(indices), 'success_count': int(success_count), 'recovered_count': len(recovered), 'recovered_frames': sorted((int(v) for v in recovered)), 'remaining_failed_frames': [int(idx) for idx in indices if not bool((recovered.get(idx) or frames[idx].get('pose', {})).get('success', False))], 'rejected_temporal_reasons': rejected, 'quality_counts': quality_counts, 'source_counts': source_counts, 'input_footer': footer, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}
        pickle.dump(out_footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[INFO] saved {args.output_pkl}')
    print(f'[INFO] recovered={len(recovered)} frames={sorted(recovered)}')
    print(f'[INFO] success={success_count}/{len(indices)}')
    print(f"[INFO] remaining_failed={out_footer['remaining_failed_frames']}")


# ---- Short local bracket interpolation ----
TEMPORAL_COMPLETION_THIS_FILE = Path(__file__).resolve()
TEMPORAL_COMPLETION_APRILCUBE_ROOT = TEMPORAL_COMPLETION_THIS_FILE.parent.parent
TEMPORAL_COMPLETION_DEFAULT_INPUT_PKL = TEMPORAL_COMPLETION_APRILCUBE_ROOT / 'recordings/024_temporal_outline_refine_recovery_conservative_fixed.pkl'
TEMPORAL_COMPLETION_DEFAULT_RAW_PKL = TEMPORAL_COMPLETION_APRILCUBE_ROOT / 'recordings/012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl'
TEMPORAL_COMPLETION_DEFAULT_OUTPUT_PKL = TEMPORAL_COMPLETION_APRILCUBE_ROOT / 'recordings/025_global_temporal_filter_fill_final.pkl'

def temporal_completion_encode_bgr_jpeg(image_bgr: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode('.jpg', np.asarray(image_bgr, dtype=np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(int(quality), 100)))])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def temporal_completion_rotation_from_rvec(rvec: Any) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return rot

def temporal_completion_make_pose_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = temporal_completion_rotation_from_rvec(rvec)
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform

def temporal_completion_draw_overlay(script012: Any, draw_detector: Any, detect_frame: np.ndarray, pose: dict[str, Any], quality: int) -> bytes:
    base = realsense_make_detector_input_vis(detect_frame)
    result = {'success': bool(pose.get('success', False)), 'detections': [], 'rvec': np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'tvec': np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1) if pose.get('success', False) else None, 'reproj_error': float(pose.get('reproj_error', float('inf'))), 'n_tags': int(pose.get('n_tags', 0) or 0), 'visible_faces': set(pose.get('visible_faces', []) or []), 'predicted': False}
    vis = draw_detector.draw_result(base.copy(), result)
    cv2.rectangle(vis, (8, 8), (1180, 66), (0, 0, 0), -1)
    cv2.putText(vis, f"Local bracket interpolation: {pose.get('pose_source', '')}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, str(pose.get('quality_reason', ''))[:120], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return temporal_completion_encode_bgr_jpeg(vis, quality)

def temporal_completion_main(args: TemporalPoseCompletionConfig) -> None:
    bm = None
    script012 = None
    header, frames, footer = pose_recovery_load_pose_records(args.input_pkl)
    raw_header, raw_offsets, raw_footer = pose_recovery_build_stream_index(args.raw_pkl, {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'})
    metadata: dict[str, Any] = {}
    if raw_header.get('format') == 'aprilcube_012_raw_with_pose_stream_v1':
        metadata.update(raw_header.get('raw_header', {}).get('metadata', {}) or {})
    metadata.update(raw_header.get('metadata', {}) or {})
    cube_cfg = Path(metadata['cube_cfg']).expanduser().resolve()
    config, _face_id_sets = aprilcube.load_cube_config(str(cube_cfg / 'config.json' if cube_cfg.is_dir() else cube_cfg))
    calib = realsense_load_intrinsics_yaml(metadata.get('intrinsics_yaml'))
    image_size = tuple((int(v) for v in metadata.get('image_size', calib['image_size'])))
    undistort_pack = realsense_create_undistort_maps(calib, image_size) if bool(metadata.get('undistort_for_detection', True)) else None
    camera_matrix = np.asarray(metadata.get('detection_camera_matrix', calib['K']), dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    if undistort_pack is not None:
        camera_matrix = np.asarray(metadata.get('detection_camera_matrix', undistort_pack[2]), dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64).reshape(-1)
    raw_offset_by_frame = {int(pose_recovery_load_at(args.raw_pkl, offset).get('frame_index', idx)): int(offset) for idx, offset in enumerate(raw_offsets)}
    draw_detector = aprilcube.detector(cube_cfg, intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(camera_matrix), dist_coeffs=dist_coeffs, enable_filter=False, fast=True)
    indices = sorted(frames)
    valid_indices = [idx for idx in indices if bool(frames[idx].get('pose', {}).get('success', False))]
    failed_indices = [idx for idx in indices if idx not in valid_indices]
    if len(valid_indices) < 2:
        args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input_pkl, args.output_pkl)
        print(
            '[WARN] Skipping local bracket interpolation: '
            f'need at least 2 valid poses, got {len(valid_indices)}; '
            f'copied input stream to {args.output_pkl}'
        )
        return
    filled: dict[int, dict[str, Any]] = {}
    rejected: dict[int, str] = {}
    for idx in failed_indices:
        prevs = [v for v in valid_indices if v < idx]
        nexts = [v for v in valid_indices if v > idx]
        if not prevs or not nexts:
            rejected[idx] = 'no_bracketing_valid_pose'
            continue
        prev_idx, next_idx = (prevs[-1], nexts[0])
        gap = int(next_idx - prev_idx)
        if gap > int(args.max_bracket_gap):
            rejected[idx] = f'bracket_gap_too_large:{gap}>{int(args.max_bracket_gap)}'
            continue
        prev_timestamp = frames[prev_idx].get('capture_timestamp', None)
        current_timestamp = frames[idx].get('capture_timestamp', None)
        next_timestamp = frames[next_idx].get('capture_timestamp', None)
        use_capture_time = all(
            value is not None and np.isfinite(float(value))
            for value in (prev_timestamp, current_timestamp, next_timestamp)
        ) and float(next_timestamp) > float(prev_timestamp)
        if use_capture_time:
            alpha = (
                (float(current_timestamp) - float(prev_timestamp))
                / (float(next_timestamp) - float(prev_timestamp))
            )
            interpolation_clock = 'capture_timestamp'
        else:
            alpha = float(idx - prev_idx) / float(next_idx - prev_idx)
            interpolation_clock = 'frame_index'
        if not (0.0 < alpha < 1.0):
            rejected[idx] = f'invalid_interpolation_alpha:{alpha:.6f}'
            continue
        prev_pose = frames[prev_idx]['pose']
        next_pose = frames[next_idx]['pose']
        prev_tvec = np.asarray(prev_pose['tvec'], dtype=np.float64).reshape(3)
        next_tvec = np.asarray(next_pose['tvec'], dtype=np.float64).reshape(3)
        tvec = ((1.0 - alpha) * prev_tvec + alpha * next_tvec).reshape(3, 1)
        endpoint_rotations = Rotation.from_matrix(np.stack([
            temporal_completion_rotation_from_rvec(prev_pose['rvec']),
            temporal_completion_rotation_from_rvec(next_pose['rvec']),
        ], axis=0))
        rvec = Slerp([0.0, 1.0], endpoint_rotations)([alpha]).as_rotvec()[0].reshape(3, 1).astype(np.float64)
        raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
        image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
        detect_frame = realsense_undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        edge_score = pose_recovery_edge_alignment_score(gray, {'success': True, 'rvec': rvec, 'tvec': tvec}, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        filled[idx] = {'success': True, 'pose_source': 'stage11_local_bracket_se3_interpolation', 'quality_level': 'F', 'quality_reason': f'local_bracket_se3_interpolation;bracket:{prev_idx}-{next_idx};gap:{gap};alpha:{alpha:.4f};clock:{interpolation_clock};edge:{edge_score:.2f}', 'pose_filled': True, 'temporal_filter_fill': True, 'local_temporal_interpolation': True, 'single_frame_only': False, 'rvec': rvec, 'tvec': tvec, 'T': temporal_completion_make_pose_transform(rvec, tvec), 'reproj_error': float('nan'), 'n_tags': 0, 'tag_ids': [], 'visible_faces': [], 'edge_score': float(edge_score), 'prev_success_frame': int(prev_idx), 'next_success_frame': int(next_idx), 'bracket_gap': int(gap), 'interpolation_alpha': float(alpha), 'interpolation_clock': interpolation_clock}
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    quality_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    success_count = 0
    with args.output_pkl.open('wb') as f:
        out_header = copy.deepcopy(header)
        out_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        out_header['source_input_pkl'] = str(args.input_pkl.resolve())
        out_header['source_raw_pkl'] = str(args.raw_pkl.resolve())
        out_header['raw_footer'] = raw_footer
        out_header['metadata'] = {**(out_header.get('metadata', {}) or {}), 'script': str(TEMPORAL_COMPLETION_THIS_FILE), 'method': 'fill only short missing runs by local bracketed translation interpolation and rotation SLERP', 'max_bracket_gap': int(args.max_bracket_gap)}
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            if idx in filled:
                raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
                image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
                detect_frame = realsense_undistort_frame(image, undistort_pack)
                frame['pose_original'] = copy.deepcopy(frame.get('pose', {}))
                frame['pose'] = filled[idx]
                frame['selected_stage'] = 'stage11_local_bracket_se3_interpolation'
                frame['overlay_jpeg'] = temporal_completion_draw_overlay(script012, draw_detector, detect_frame, filled[idx], int(args.jpeg_quality))
                frame['overlay_format'] = 'jpeg_bgr'
                frame['overlay_shape'] = tuple((int(v) for v in detect_frame.shape))
            elif idx in rejected:
                frame['pose_original'] = copy.deepcopy(frame.get('pose', {}))
                frame['pose'] = {'success': False, 'pose_source': 'fused_failed', 'quality_level': 'Z', 'quality_reason': rejected[idx], 'failure_reason': rejected[idx], 'reproj_error': float('inf'), 'pose_filled': False, 'single_frame_only': False}
            pose = frame.get('pose', {})
            quality = str(pose.get('quality_level', 'Z'))
            source = str(pose.get('pose_source', ''))
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            success_count += int(bool(pose.get('success', False)))
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        out_footer = {'type': 'footer', 'frame_count': len(indices), 'success_count': int(success_count), 'filled_count': len(filled), 'filled_frames': sorted((int(v) for v in filled)), 'remaining_failed_frames': [int(idx) for idx in indices if idx not in valid_indices and idx not in filled], 'rejected_temporal_fill_reasons': rejected, 'quality_counts': quality_counts, 'source_counts': source_counts, 'input_footer': footer, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}
        pickle.dump(out_footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[INFO] saved {args.output_pkl}')
    print(f'[INFO] filled={len(filled)} frames={sorted(filled)}')
    print(f'[INFO] success={success_count}/{len(indices)}')
    print(f"[INFO] remaining_failed={out_footer['remaining_failed_frames']}")


# ---- Constrained full-trajectory temporal smoothing ----
TEMPORAL_SMOOTHING_THIS_FILE = Path(__file__).resolve()


def temporal_smoothing_quality_weight(pose: dict[str, Any]) -> float:
    quality_weight = {
        'A': 4.0,
        'B': 3.0,
        'C': 2.0,
        'D': 1.5,
        'E': 1.3,
        'F': 0.8,
        'G': 1.2,
        # A planar AprilTag/IPPE pose is useful as a fallback but is less
        # stable than the dense DeepTag estimate on the near-frontal middle
        # finger view.
        'H': 0.7,
        'I': 2.0,
        'T': 0.8,
    }.get(str(pose.get('quality_level', '')), 1.0)
    if bool(pose.get('pose_filled', False)):
        quality_weight *= 0.65
    return float(quality_weight)


def temporal_smoothing_weighted_pose(
    samples: list[tuple[int, float | None, dict[str, Any]]],
    *,
    target_idx: int,
    target_timestamp: float | None,
    sigma_frames: float,
    sigma_seconds: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    frame_sigma = max(float(sigma_frames), 1e-6)
    time_sigma = max(float(sigma_seconds), 1e-6)
    weights: list[float] = []
    translations: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    target_pose = next(pose for idx, _timestamp, pose in samples if int(idx) == int(target_idx))
    reference_quat = Rotation.from_rotvec(
        np.asarray(target_pose['rvec'], dtype=np.float64).reshape(3)
    ).as_quat()
    for idx, timestamp, pose in samples:
        if (
            target_timestamp is not None
            and timestamp is not None
            and np.isfinite(float(target_timestamp))
            and np.isfinite(float(timestamp))
        ):
            distance = abs(float(timestamp) - float(target_timestamp))
            sigma = time_sigma
        else:
            distance = abs(int(idx) - int(target_idx))
            sigma = frame_sigma
        weight = float(np.exp(-0.5 * (float(distance) / sigma) ** 2))
        weight *= temporal_smoothing_quality_weight(pose)
        if int(idx) == int(target_idx):
            weight *= 2.0
        quat = Rotation.from_rotvec(
            np.asarray(pose['rvec'], dtype=np.float64).reshape(3)
        ).as_quat()
        if float(np.dot(quat, reference_quat)) < 0.0:
            quat = -quat
        weights.append(weight)
        translations.append(np.asarray(pose['tvec'], dtype=np.float64).reshape(3))
        quaternions.append(quat)
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights /= max(float(np.sum(normalized_weights)), 1e-12)
    tvec = np.sum(
        np.stack(translations, axis=0) * normalized_weights[:, None], axis=0
    ).reshape(3, 1)
    quat = np.sum(
        np.stack(quaternions, axis=0) * normalized_weights[:, None], axis=0
    )
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    rvec = Rotation.from_quat(quat).as_rotvec().reshape(3, 1)
    return rvec, tvec, len(samples)


def temporal_smoothing_limit_pose_delta(
    source_pose: dict[str, Any],
    candidate_rvec: np.ndarray,
    candidate_tvec: np.ndarray,
    *,
    max_translation_delta_mm: float,
    max_rotation_delta_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    source_tvec = np.asarray(source_pose['tvec'], dtype=np.float64).reshape(3)
    candidate_t = np.asarray(candidate_tvec, dtype=np.float64).reshape(3)
    translation_delta = candidate_t - source_tvec
    translation_norm = float(np.linalg.norm(translation_delta))
    if translation_norm > float(max_translation_delta_mm) > 0.0:
        candidate_t = source_tvec + translation_delta * (
            float(max_translation_delta_mm) / translation_norm
        )
    source_rotation = Rotation.from_rotvec(
        np.asarray(source_pose['rvec'], dtype=np.float64).reshape(3)
    )
    candidate_rotation = Rotation.from_rotvec(
        np.asarray(candidate_rvec, dtype=np.float64).reshape(3)
    )
    relative_rotation = source_rotation.inv() * candidate_rotation
    rotation_delta_deg = float(np.degrees(relative_rotation.magnitude()))
    if rotation_delta_deg > float(max_rotation_delta_deg) > 0.0:
        scale = float(max_rotation_delta_deg) / rotation_delta_deg
        candidate_rotation = source_rotation * Rotation.from_rotvec(
            relative_rotation.as_rotvec() * scale
        )
    limited_rvec = candidate_rotation.as_rotvec().reshape(3, 1)
    limited_tvec = candidate_t.reshape(3, 1)
    applied_translation = temporal_outlier_translation_delta_mm(
        source_pose['tvec'], limited_tvec
    )
    applied_rotation = temporal_outlier_rotation_delta_deg(
        source_pose['rvec'], limited_rvec
    )
    return limited_rvec, limited_tvec, applied_translation, applied_rotation


def temporal_smoothing_blend_from_source(
    source_pose: dict[str, Any],
    target_rvec: np.ndarray,
    target_tvec: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    source_t = np.asarray(source_pose['tvec'], dtype=np.float64).reshape(3)
    target_t = np.asarray(target_tvec, dtype=np.float64).reshape(3)
    tvec = ((1.0 - alpha) * source_t + alpha * target_t).reshape(3, 1)
    source_rotation = Rotation.from_rotvec(
        np.asarray(source_pose['rvec'], dtype=np.float64).reshape(3)
    )
    target_rotation = Rotation.from_rotvec(
        np.asarray(target_rvec, dtype=np.float64).reshape(3)
    )
    relative = source_rotation.inv() * target_rotation
    rotation = source_rotation * Rotation.from_rotvec(relative.as_rotvec() * alpha)
    return rotation.as_rotvec().reshape(3, 1), tvec


def temporal_smoothing_draw_overlay(
    draw_detector: Any,
    detect_frame: np.ndarray,
    pose: dict[str, Any],
    quality: int,
) -> bytes:
    base = realsense_make_detector_input_vis(detect_frame)
    result = {
        'success': bool(pose.get('success', False)),
        'detections': [],
        'rvec': np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1),
        'tvec': np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1),
        'reproj_error': float(pose.get('reproj_error', float('inf'))),
        'n_tags': int(pose.get('n_tags', 0) or 0),
        'visible_faces': set(pose.get('visible_faces', []) or []),
        'predicted': False,
    }
    vis = draw_detector.draw_result(base.copy(), result)
    cv2.rectangle(vis, (8, 8), (1180, 66), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"Constrained temporal smoothing: {pose.get('pose_source', '')}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        (
            f"dt={float(pose.get('temporal_smoothing_translation_delta_mm', 0.0)):.2f}mm "
            f"dr={float(pose.get('temporal_smoothing_rotation_delta_deg', 0.0)):.2f}deg "
            f"edge={float(pose.get('temporal_smoothing_edge_after', 0.0)):.2f}"
        ),
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return temporal_completion_encode_bgr_jpeg(vis, quality)


def temporal_pose_smoothing_main(args: TemporalPoseSmoothingConfig) -> None:
    header, frames, footer = pose_recovery_load_pose_records(args.input_pkl)
    raw_header, raw_offsets, raw_footer = pose_recovery_build_stream_index(
        args.raw_pkl,
        {'aprilcube_rs_raw_frame_stream_v1', 'aprilcube_012_raw_with_pose_stream_v1'},
    )
    metadata: dict[str, Any] = {}
    if raw_header.get('format') == 'aprilcube_012_raw_with_pose_stream_v1':
        metadata.update(raw_header.get('raw_header', {}).get('metadata', {}) or {})
    metadata.update(raw_header.get('metadata', {}) or {})
    cube_cfg = Path(metadata['cube_cfg']).expanduser().resolve()
    config, _face_id_sets = aprilcube.load_cube_config(
        str(cube_cfg / 'config.json' if cube_cfg.is_dir() else cube_cfg)
    )
    calib = realsense_load_intrinsics_yaml(metadata.get('intrinsics_yaml'))
    image_size = tuple((int(v) for v in metadata.get('image_size', calib['image_size'])))
    undistort_pack = (
        realsense_create_undistort_maps(calib, image_size)
        if bool(metadata.get('undistort_for_detection', True))
        else None
    )
    camera_matrix = np.asarray(
        metadata.get('detection_camera_matrix', calib['K']), dtype=np.float64
    ).reshape(3, 3)
    dist_coeffs = np.asarray(
        metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64
    ).reshape(-1)
    if undistort_pack is not None:
        camera_matrix = np.asarray(
            metadata.get('detection_camera_matrix', undistort_pack[2]),
            dtype=np.float64,
        ).reshape(3, 3)
        dist_coeffs = np.asarray(
            metadata.get('detector_dist_coeffs', np.zeros(5)), dtype=np.float64
        ).reshape(-1)
    raw_offset_by_frame = {
        int(pose_recovery_load_at(args.raw_pkl, offset).get('frame_index', idx)): int(offset)
        for idx, offset in enumerate(raw_offsets)
    }
    draw_detector = aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=realsense_camera_matrix_to_intrinsic_dict(camera_matrix),
        dist_coeffs=dist_coeffs,
        enable_filter=False,
        fast=True,
    )
    indices = sorted(frames)
    source_poses = {idx: copy.deepcopy(frames[idx].get('pose', {})) for idx in indices}
    capture_timestamps = {
        idx: (
            float(frames[idx]['capture_timestamp'])
            if frames[idx].get('capture_timestamp') is not None
            and np.isfinite(float(frames[idx]['capture_timestamp']))
            else None
        )
        for idx in indices
    }
    if any((not bool(source_poses[idx].get('success', False))) for idx in indices):
        failed = [idx for idx in indices if not bool(source_poses[idx].get('success', False))]
        args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input_pkl, args.output_pkl)
        print(
            '[WARN] Skipping constrained temporal smoothing because the pose stream '
            f'is incomplete; failed frames={failed}; copied input to {args.output_pkl}'
        )
        return
    smoothed_poses: dict[int, dict[str, Any]] = {}
    smoothed_count = 0
    edge_rejected_count = 0
    for idx in indices:
        source_pose = source_poses[idx]
        target_timestamp = capture_timestamps[idx]
        samples = [
            (neighbor_idx, capture_timestamps[neighbor_idx], source_poses[neighbor_idx])
            for neighbor_idx in indices
            if abs(int(neighbor_idx) - int(idx)) <= int(args.window_radius)
            and (
                target_timestamp is None
                or capture_timestamps[neighbor_idx] is None
                or abs(float(capture_timestamps[neighbor_idx]) - float(target_timestamp))
                <= float(args.window_seconds)
            )
        ]
        candidate_rvec, candidate_tvec, source_count = temporal_smoothing_weighted_pose(
            samples,
            target_idx=int(idx),
            target_timestamp=target_timestamp,
            sigma_frames=float(args.sigma_frames),
            sigma_seconds=float(args.sigma_seconds),
        )
        is_filled = bool(source_pose.get('pose_filled', False)) or str(
            source_pose.get('quality_level', '')
        ) in {'F', 'T'}
        max_translation = (
            float(args.max_filled_translation_delta_mm)
            if is_filled
            else float(args.max_measured_translation_delta_mm)
        )
        max_rotation = (
            float(args.max_filled_rotation_delta_deg)
            if is_filled
            else float(args.max_measured_rotation_delta_deg)
        )
        candidate_rvec, candidate_tvec, _dt, _dr = temporal_smoothing_limit_pose_delta(
            source_pose,
            candidate_rvec,
            candidate_tvec,
            max_translation_delta_mm=max_translation,
            max_rotation_delta_deg=max_rotation,
        )
        raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
        image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
        detect_frame = realsense_undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        source_edge = pose_recovery_edge_alignment_score(
            gray,
            source_pose,
            config=config,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        blend = 1.0
        candidate_pose = {'success': True, 'rvec': candidate_rvec, 'tvec': candidate_tvec}
        candidate_edge = pose_recovery_edge_alignment_score(
            gray,
            candidate_pose,
            config=config,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        while (
            candidate_edge < source_edge - float(args.max_edge_score_drop)
            and blend > 0.125
        ):
            blend *= 0.5
            candidate_rvec, candidate_tvec = temporal_smoothing_blend_from_source(
                source_pose,
                candidate_rvec,
                candidate_tvec,
                0.5,
            )
            candidate_pose = {'success': True, 'rvec': candidate_rvec, 'tvec': candidate_tvec}
            candidate_edge = pose_recovery_edge_alignment_score(
                gray,
                candidate_pose,
                config=config,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
        edge_rejected = candidate_edge < source_edge - float(args.max_edge_score_drop)
        if edge_rejected:
            candidate_rvec = np.asarray(source_pose['rvec'], dtype=np.float64).reshape(3, 1)
            candidate_tvec = np.asarray(source_pose['tvec'], dtype=np.float64).reshape(3, 1)
            candidate_edge = float(source_edge)
            edge_rejected_count += 1
        translation_delta = temporal_outlier_translation_delta_mm(
            source_pose['tvec'], candidate_tvec
        )
        rotation_delta = temporal_outlier_rotation_delta_deg(
            source_pose['rvec'], candidate_rvec
        )
        smoothed = copy.deepcopy(source_pose)
        smoothed['rvec'] = np.asarray(candidate_rvec, dtype=np.float64).reshape(3, 1)
        smoothed['tvec'] = np.asarray(candidate_tvec, dtype=np.float64).reshape(3, 1)
        smoothed['T'] = temporal_completion_make_pose_transform(
            smoothed['rvec'], smoothed['tvec']
        )
        smoothed['temporal_smoothed'] = bool(
            translation_delta > 1e-9 or rotation_delta > 1e-9
        )
        smoothed['temporal_smoothing_source_count'] = int(source_count)
        smoothed['temporal_smoothing_window_radius'] = int(args.window_radius)
        smoothed['temporal_smoothing_translation_delta_mm'] = float(translation_delta)
        smoothed['temporal_smoothing_rotation_delta_deg'] = float(rotation_delta)
        smoothed['temporal_smoothing_edge_before'] = float(source_edge)
        smoothed['temporal_smoothing_edge_after'] = float(candidate_edge)
        smoothed['temporal_smoothing_edge_rejected'] = bool(edge_rejected)
        smoothed_poses[int(idx)] = smoothed
        smoothed_count += int(bool(smoothed['temporal_smoothed']))
    args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_pkl.open('wb') as f:
        out_header = copy.deepcopy(header)
        out_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        out_header['source_input_pkl'] = str(args.input_pkl.resolve())
        out_header['source_raw_pkl'] = str(args.raw_pkl.resolve())
        out_header['raw_footer'] = raw_footer
        out_header['metadata'] = {
            **(out_header.get('metadata', {}) or {}),
            'script': str(TEMPORAL_SMOOTHING_THIS_FILE),
            'method': 'quality-weighted symmetric SE(3) smoothing with per-frame motion caps and RGB edge-alignment guard',
            'temporal_smoothing_window_radius': int(args.window_radius),
            'temporal_smoothing_sigma_frames': float(args.sigma_frames),
            'temporal_smoothing_window_seconds': float(args.window_seconds),
            'temporal_smoothing_sigma_seconds': float(args.sigma_seconds),
            'temporal_smoothing_max_measured_translation_delta_mm': float(args.max_measured_translation_delta_mm),
            'temporal_smoothing_max_measured_rotation_delta_deg': float(args.max_measured_rotation_delta_deg),
            'temporal_smoothing_max_filled_translation_delta_mm': float(args.max_filled_translation_delta_mm),
            'temporal_smoothing_max_filled_rotation_delta_deg': float(args.max_filled_rotation_delta_deg),
            'temporal_smoothing_max_edge_score_drop': float(args.max_edge_score_drop),
        }
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            frame['pose_before_temporal_smoothing'] = copy.deepcopy(frame.get('pose', {}))
            frame['pose'] = smoothed_poses[idx]
            frame['pose_temporally_smoothed'] = copy.deepcopy(smoothed_poses[idx])
            frame['selected_stage_before_temporal_smoothing'] = frame.get('selected_stage', '')
            frame['selected_stage'] = 'stage12_constrained_temporal_smoothing'
            raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
            image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
            detect_frame = realsense_undistort_frame(image, undistort_pack)
            frame['overlay_jpeg'] = temporal_smoothing_draw_overlay(
                draw_detector,
                detect_frame,
                smoothed_poses[idx],
                int(args.jpeg_quality),
            )
            frame['overlay_format'] = 'jpeg_bgr'
            frame['overlay_shape'] = tuple((int(v) for v in detect_frame.shape))
            pickle.dump(frame, f, protocol=pickle.HIGHEST_PROTOCOL)
        out_footer = {
            'type': 'footer',
            'frame_count': int(len(indices)),
            'success_count': int(len(indices)),
            'smoothed_count': int(smoothed_count),
            'edge_rejected_count': int(edge_rejected_count),
            'input_footer': footer,
            'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        pickle.dump(out_footer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[INFO] saved {args.output_pkl}')
    print(
        f'[INFO] smoothed={smoothed_count}/{len(indices)} '
        f'edge_rejected={edge_rejected_count}'
    )


def adjacent_rgb_flow_pose_valid(pose: dict[str, Any]) -> bool:
    if not bool(pose.get("success", False)):
        return False
    try:
        values = np.r_[
            np.asarray(pose["rvec"], dtype=np.float64).reshape(3),
            np.asarray(pose["tvec"], dtype=np.float64).reshape(3),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def adjacent_rgb_flow_rotation_delta_deg(first: Any, second: Any) -> float:
    first_matrix = pose_recovery_rotation_from_rvec(first)
    second_matrix = pose_recovery_rotation_from_rvec(second)
    relative = second_matrix @ first_matrix.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def adjacent_rgb_flow_best_quad_agreement(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4, 2)
    second = np.asarray(second, dtype=np.float64).reshape(4, 2)
    errors: list[float] = []
    for candidate in (second, second[::-1]):
        for shift in range(4):
            shifted = np.roll(candidate, shift, axis=0)
            errors.append(float(np.mean(np.linalg.norm(first - shifted, axis=1))))
    return min(errors)


def adjacent_rgb_flow_runtime(raw_header: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if raw_header.get("format") == "aprilcube_012_raw_with_pose_stream_v1":
        metadata.update(raw_header.get("raw_header", {}).get("metadata", {}) or {})
    metadata.update(raw_header.get("metadata", {}) or {})
    cube_cfg = Path(metadata["cube_cfg"]).expanduser().resolve()
    config_path = cube_cfg / "config.json" if cube_cfg.is_dir() else cube_cfg
    config, face_id_sets = aprilcube.load_cube_config(str(config_path))
    tag_corner_map = aprilcube.build_tag_corner_map(config)
    calib = realsense_load_intrinsics_yaml(metadata.get("intrinsics_yaml"))
    image_size = tuple(int(value) for value in metadata.get("image_size", calib["image_size"]))
    undistort_pack = (
        realsense_create_undistort_maps(calib, image_size)
        if bool(metadata.get("undistort_for_detection", True))
        else None
    )
    default_matrix = undistort_pack[2] if undistort_pack is not None else calib["K"]
    camera_matrix = np.asarray(
        metadata.get("detection_camera_matrix", default_matrix), dtype=np.float64
    ).reshape(3, 3)
    dist_coeffs = np.asarray(
        metadata.get("detector_dist_coeffs", np.zeros(5)), dtype=np.float64
    ).reshape(-1)
    return {
        "metadata": metadata,
        "image_size": image_size,
        "undistort_pack": undistort_pack,
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "config": config,
        "face_id_sets": face_id_sets,
        "tag_corner_map": tag_corner_map,
    }


def adjacent_rgb_flow_detection_frame(image: np.ndarray, runtime: dict[str, Any]) -> np.ndarray:
    width, height = runtime["image_size"]
    if image.shape[:2] != (height, width):
        raise ValueError(
            f"RGB-flow raw image is {image.shape[1]}x{image.shape[0]}, "
            f"expected {width}x{height}"
        )
    return realsense_undistort_frame(image, runtime["undistort_pack"])


def adjacent_rgb_flow_select_anchor_tag(
    pose: dict[str, Any], runtime: dict[str, Any]
) -> tuple[int, np.ndarray, np.ndarray]:
    tag_corner_map = runtime["tag_corner_map"]
    tag_ids = [
        int(value)
        for value in (pose.get("tag_ids", []) or pose.get("detected_tag_ids", []) or [])
        if int(value) in tag_corner_map
    ]
    if not tag_ids:
        raise RuntimeError("RGB-flow anchor has no cube tag identity")
    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(3, 1)
    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for tag_id in sorted(set(tag_ids)):
        object_points = np.asarray(tag_corner_map[tag_id], dtype=np.float64).reshape(4, 3)
        quad, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            runtime["camera_matrix"],
            runtime["dist_coeffs"],
        )
        quad = quad.reshape(4, 2).astype(np.float32)
        area = abs(float(cv2.contourArea(quad)))
        candidates.append((area, tag_id, object_points, quad))
    _area, tag_id, object_points, quad = max(candidates, key=lambda item: item[0])
    return tag_id, object_points, quad


def adjacent_rgb_flow_recover_pair(
    *,
    anchor_image: np.ndarray,
    target_image: np.ndarray,
    anchor_pose: dict[str, Any],
    anchor_frame: int,
    target_frame: int,
    runtime: dict[str, Any],
    config: AdjacentRgbFlowRecoveryConfig,
) -> dict[str, Any]:
    tag_id, object_points, anchor_quad = adjacent_rgb_flow_select_anchor_tag(
        anchor_pose, runtime
    )
    anchor_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(anchor_gray)
    center = anchor_quad.mean(axis=0, keepdims=True)
    feature_quad = center + float(config.feature_mask_scale) * (anchor_quad - center)
    cv2.fillConvexPoly(mask, np.rint(feature_quad).astype(np.int32), 255)
    anchor_points = cv2.goodFeaturesToTrack(
        anchor_gray,
        maxCorners=int(config.max_features),
        qualityLevel=float(config.feature_quality),
        minDistance=float(config.feature_min_distance),
        mask=mask,
        blockSize=5,
    )
    if anchor_points is None or len(anchor_points) < int(config.min_features):
        count = 0 if anchor_points is None else len(anchor_points)
        raise RuntimeError(f"not_enough_anchor_features:{count}<{config.min_features}")
    lk = {
        "winSize": (int(config.lk_window), int(config.lk_window)),
        "maxLevel": int(config.lk_levels),
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            0.001,
        ),
    }
    target_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        anchor_gray, target_gray, anchor_points, None, **lk
    )
    if target_points is None or forward_status is None:
        raise RuntimeError("forward_lk_failed")
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        target_gray, anchor_gray, target_points, None, **lk
    )
    if backward_points is None or backward_status is None:
        raise RuntimeError("backward_lk_failed")
    anchor_xy = anchor_points.reshape(-1, 2)
    target_xy = target_points.reshape(-1, 2)
    backward_xy = backward_points.reshape(-1, 2)
    fb_error = np.linalg.norm(anchor_xy - backward_xy, axis=1)
    good = (
        (forward_status.reshape(-1) > 0)
        & (backward_status.reshape(-1) > 0)
        & np.isfinite(target_xy).all(axis=1)
        & np.isfinite(fb_error)
        & (fb_error <= float(config.max_fb_error))
    )
    good_count = int(good.sum())
    if good_count < int(config.min_good_tracks):
        raise RuntimeError(f"not_enough_consistent_tracks:{good_count}<{config.min_good_tracks}")
    homography, inlier_values = cv2.findHomography(
        anchor_xy[good], target_xy[good], cv2.RANSAC, float(config.homography_ransac_px)
    )
    if homography is None or inlier_values is None:
        raise RuntimeError("homography_failed")
    inliers = inlier_values.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    inlier_ratio = float(inlier_count / max(good_count, 1))
    if inlier_count < int(config.min_homography_inliers):
        raise RuntimeError(
            f"not_enough_homography_inliers:{inlier_count}<{config.min_homography_inliers}"
        )
    if inlier_ratio < float(config.min_homography_inlier_ratio):
        raise RuntimeError(
            f"homography_inlier_ratio:{inlier_ratio:.3f}<{config.min_homography_inlier_ratio:.3f}"
        )
    predicted_tracks = cv2.perspectiveTransform(
        anchor_xy[good].reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    residuals = np.linalg.norm(predicted_tracks - target_xy[good], axis=1)
    homography_median = float(np.median(residuals[inliers]))
    fb_median = float(np.median(fb_error[good]))
    if homography_median > float(config.max_homography_median_px):
        raise RuntimeError("homography_residual_too_high")
    if fb_median > float(config.max_fb_median_px):
        raise RuntimeError("forward_backward_residual_too_high")
    target_quad = cv2.perspectiveTransform(
        anchor_quad.reshape(-1, 1, 2), homography
    ).reshape(4, 2)
    height, width = target_gray.shape[:2]
    corners_inside = int(
        np.sum(
            (target_quad[:, 0] >= 0.0)
            & (target_quad[:, 0] < width)
            & (target_quad[:, 1] >= 0.0)
            & (target_quad[:, 1] < height)
        )
    )
    if corners_inside < int(config.min_tag_corners_inside):
        raise RuntimeError("propagated_tag_outside_image")
    detections = pose_recovery_detect_sweep(
        target_gray,
        config=runtime["config"],
        valid_ids=set(int(value) for value in runtime["tag_corner_map"]),
    )
    detected_by_id = {
        int(detected_id): np.asarray(corners, dtype=np.float64).reshape(4, 2)
        for detected_id, corners in detections
    }
    current_detected = tag_id in detected_by_id
    if not current_detected and not bool(config.allow_missing_current_tag):
        raise RuntimeError(f"current_frame_missing_tag:{tag_id}")
    agreement = (
        adjacent_rgb_flow_best_quad_agreement(target_quad, detected_by_id[tag_id])
        if current_detected
        else float("nan")
    )
    if current_detected and agreement > float(config.max_current_tag_agreement_px):
        raise RuntimeError("current_tag_flow_disagreement")
    anchor_rvec = np.asarray(anchor_pose["rvec"], dtype=np.float64).reshape(3, 1)
    anchor_tvec = np.asarray(anchor_pose["tvec"], dtype=np.float64).reshape(3, 1)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        target_quad.astype(np.float64),
        runtime["camera_matrix"],
        runtime["dist_coeffs"],
        anchor_rvec.copy(),
        anchor_tvec.copy(),
        True,
        cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
        raise RuntimeError("tracked_corner_pnp_failed")
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            target_quad.astype(np.float64),
            runtime["camera_matrix"],
            runtime["dist_coeffs"],
            rvec,
            tvec,
        )
    except cv2.error:
        pass
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        runtime["camera_matrix"],
        runtime["dist_coeffs"],
    )
    flow_reprojection = float(
        np.mean(np.linalg.norm(projected.reshape(4, 2) - target_quad, axis=1))
    )
    translation_delta = float(
        np.linalg.norm(np.asarray(tvec).reshape(3) - anchor_tvec.reshape(3))
    )
    rotation_delta = adjacent_rgb_flow_rotation_delta_deg(anchor_rvec, rvec)
    provisional = {"success": True, "rvec": rvec, "tvec": tvec}
    edge_score = float(
        pose_recovery_edge_alignment_score(
            target_gray,
            provisional,
            config=runtime["config"],
            camera_matrix=runtime["camera_matrix"],
            dist_coeffs=runtime["dist_coeffs"],
        )
    )
    if flow_reprojection > float(config.max_flow_corner_reproj_px):
        raise RuntimeError("flow_corner_reprojection_too_high")
    if translation_delta > float(config.max_translation_delta_mm):
        raise RuntimeError("adjacent_translation_too_large")
    if rotation_delta > float(config.max_rotation_delta_deg):
        raise RuntimeError("adjacent_rotation_too_large")
    if edge_score < float(config.min_edge_score):
        raise RuntimeError("rgb_cube_edge_score_too_low")
    visible_faces = sorted(
        pose_recovery_visible_faces_for_ids(runtime["face_id_sets"], [tag_id])
    )
    return {
        "success": True,
        "failure_reason": "",
        "pose_source": "stage10_adjacent_bidirectional_rgb_flow_tag_pnp",
        "quality_level": "T",
        "quality_reason": (
            f"anchor:{anchor_frame};tag:{tag_id};tracks:{good_count};"
            f"inliers:{inlier_count};fb:{fb_median:.3f}px;H:{homography_median:.3f}px;"
            f"edge:{edge_score:.3f};dt:{translation_delta:.3f}mm;dr:{rotation_delta:.3f}deg"
        ),
        "pose_filled": True,
        "predicted": True,
        "temporal_recovery": True,
        "single_frame_only": False,
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "T": pose_recovery_pose_transform(rvec, tvec),
        "reproj_error": flow_reprojection,
        "reproj_metric": "optical_flow_propagated_tag_corner_mean_px",
        "n_tags": 1,
        "tag_ids": [tag_id],
        "visible_faces": visible_faces,
        "edge_score": edge_score,
        "flow_anchor_frame": int(anchor_frame),
        "flow_target_frame": int(target_frame),
        "flow_anchor_pose_source": str(anchor_pose.get("pose_source", "")),
        "flow_good_track_count": good_count,
        "flow_homography_inlier_count": inlier_count,
        "flow_homography_inlier_ratio": inlier_ratio,
        "flow_fb_median_px": fb_median,
        "flow_homography_median_px": homography_median,
        "flow_current_tag_corner_agreement_px": agreement,
        "flow_translation_delta_mm": translation_delta,
        "flow_rotation_delta_deg": rotation_delta,
        "current_frame_tag_anchor_used": current_detected,
    }


def adjacent_rgb_flow_failed_runs(
    frames: dict[int, dict[str, Any]]
) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index in sorted(frames):
        if adjacent_rgb_flow_pose_valid(frames[index].get("pose", {}) or {}):
            if current:
                runs.append(current)
                current = []
        else:
            current.append(index)
    if current:
        runs.append(current)
    return runs


def adjacent_rgb_flow_recovery_main(config: AdjacentRgbFlowRecoveryConfig) -> None:
    header, frames, footer = pose_recovery_load_pose_records(config.input_pkl)
    raw_header, raw_offsets, raw_footer = pose_recovery_build_stream_index(
        config.raw_pkl,
        {"aprilcube_rs_raw_frame_stream_v1", "aprilcube_012_raw_with_pose_stream_v1"},
    )
    runtime = adjacent_rgb_flow_runtime(raw_header)
    raw_offsets_by_frame = {
        int(pose_recovery_load_at(config.raw_pkl, offset).get("frame_index", position)): int(
            offset
        )
        for position, offset in enumerate(raw_offsets)
    }
    image_cache: dict[int, np.ndarray] = {}

    def image_for(index: int) -> np.ndarray:
        if index not in image_cache:
            raw = pose_recovery_load_at(config.raw_pkl, raw_offsets_by_frame[index])
            image_cache[index] = adjacent_rgb_flow_detection_frame(
                np.asarray(raw["image_bgr"], dtype=np.uint8), runtime
            )
        return image_cache[index]

    recovered: dict[int, dict[str, Any]] = {}
    rejected: dict[int, list[str]] = {}
    for run in adjacent_rgb_flow_failed_runs(frames):
        if len(run) > int(config.max_gap_frames):
            for index in run:
                rejected.setdefault(index, []).append(
                    f"gap_too_long:{len(run)}>{config.max_gap_frames}"
                )
            continue
        remaining = set(run)
        while remaining:
            proposals: dict[int, list[dict[str, Any]]] = {}
            left_target = min(remaining)
            left_anchor = left_target - 1
            if left_anchor in frames and adjacent_rgb_flow_pose_valid(
                frames[left_anchor].get("pose", {}) or {}
            ):
                try:
                    proposals.setdefault(left_target, []).append(
                        adjacent_rgb_flow_recover_pair(
                            anchor_image=image_for(left_anchor),
                            target_image=image_for(left_target),
                            anchor_pose=frames[left_anchor]["pose"],
                            anchor_frame=left_anchor,
                            target_frame=left_target,
                            runtime=runtime,
                            config=config,
                        )
                    )
                except Exception as exc:
                    rejected.setdefault(left_target, []).append(f"left:{exc}")
            right_target = max(remaining)
            right_anchor = right_target + 1
            if right_anchor in frames and adjacent_rgb_flow_pose_valid(
                frames[right_anchor].get("pose", {}) or {}
            ):
                try:
                    proposals.setdefault(right_target, []).append(
                        adjacent_rgb_flow_recover_pair(
                            anchor_image=image_for(right_anchor),
                            target_image=image_for(right_target),
                            anchor_pose=frames[right_anchor]["pose"],
                            anchor_frame=right_anchor,
                            target_frame=right_target,
                            runtime=runtime,
                            config=config,
                        )
                    )
                except Exception as exc:
                    rejected.setdefault(right_target, []).append(f"right:{exc}")
            if not proposals:
                break
            progressed = False
            for target_index, candidates in proposals.items():
                best = max(
                    candidates,
                    key=lambda pose: (
                        float(pose.get("edge_score", 0.0)),
                        int(pose.get("flow_good_track_count", 0)),
                    ),
                )
                original = copy.deepcopy(frames[target_index].get("pose", {}) or {})
                frames[target_index]["pose_before_adjacent_rgb_flow"] = original
                frames[target_index].setdefault("pose_candidates", {})[
                    "stage10_adjacent_bidirectional_rgb_flow"
                ] = copy.deepcopy(best)
                frames[target_index]["pose"] = best
                recovered[target_index] = best
                remaining.remove(target_index)
                progressed = True
            if not progressed:
                break
    output = config.output_pkl.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    success_count = 0
    source_counts: dict[str, int] = {}
    try:
        with temporary.open("wb") as stream:
            out_header = copy.deepcopy(header)
            out_header["created_wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            out_header["source_input_pkl"] = str(config.input_pkl.resolve())
            out_header["metadata"] = {
                **(out_header.get("metadata", {}) or {}),
                "script": str(Path(__file__).resolve()),
                "method": (
                    "embedded automatic short-gap bidirectional adjacent RGB LK flow, "
                    "homography, tag-corner PnP, and RGB-edge gating"
                ),
                "target_name": config.target_name,
                "max_gap_frames": int(config.max_gap_frames),
                "recovered_frames": sorted(recovered),
                "rejected_recovery_reasons": rejected,
            }
            pickle.dump(out_header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for index in sorted(frames):
                frame = frames[index]
                pose = frame.get("pose", {}) or {}
                if adjacent_rgb_flow_pose_valid(pose):
                    success_count += 1
                source = str(pose.get("pose_source", "failed"))
                source_counts[source] = source_counts.get(source, 0) + 1
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            out_footer = copy.deepcopy(footer or {})
            out_footer.update(
                {
                    "type": "footer",
                    "frame_count": len(frames),
                    "success_count": success_count,
                    "pose_source_counts": source_counts,
                    "adjacent_rgb_flow_recovered_count": len(recovered),
                    "adjacent_rgb_flow_recovered_frames": sorted(recovered),
                    "source_footer": footer,
                    "raw_footer": raw_footer,
                    "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            pickle.dump(out_footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"[INFO] adjacent RGB-flow recovered={len(recovered)} "
        f"success={success_count}/{len(frames)} output={output}"
    )


# -----------------------------------------------------------------------------
# Embedded synchronized multi-camera stream reader (formerly scripts/utils).
# -----------------------------------------------------------------------------
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload
mcstream_MULTI_CAMERA_STREAM_FORMAT = 'consens_multi_camera_sync_stream'
mcstream_MULTI_CAMERA_STREAM_VERSION = 1

@dataclass(frozen=True)
class mcstream_MultiCameraStreamIndex:
    path: Path
    header: dict[str, Any]
    sample_offsets: tuple[int, ...]
    footer: dict[str, Any] | None

    @property
    def complete(self) -> bool:
        return self.footer is not None

def mcstream_is_stream_header(record: Any) -> bool:
    return isinstance(record, dict) and record.get('type') == 'header' and (record.get('format') == mcstream_MULTI_CAMERA_STREAM_FORMAT)

def mcstream_iter_stream_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield header/sample/footer records without retaining prior samples."""
    resolved = Path(path).expanduser().resolve()
    with resolved.open('rb') as stream:
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                return
            if not isinstance(record, dict):
                raise ValueError(f'Expected dict record at byte {stream.tell()} in {resolved}, got {type(record).__name__}')
            yield record

def mcstream_iter_stream_samples(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one synchronized sample at a time with bounded memory use."""
    saw_header = False
    for record in mcstream_iter_stream_records(path):
        record_type = record.get('type')
        if not saw_header:
            if not mcstream_is_stream_header(record):
                raise ValueError(f'Not a {mcstream_MULTI_CAMERA_STREAM_FORMAT} recording: {path}')
            saw_header = True
            continue
        if record_type == 'sample':
            sample = record.get('sample')
            if not isinstance(sample, dict):
                raise ValueError('Streaming sample record does not contain a dict sample')
            yield sample
        elif record_type == 'footer':
            return

def mcstream_scan_stream_index(path: str | Path) -> mcstream_MultiCameraStreamIndex:
    """Build byte offsets while holding at most one decoded sample in memory."""
    resolved = Path(path).expanduser().resolve()
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    offsets: list[int] = []
    with resolved.open('rb') as stream:
        while True:
            offset = stream.tell()
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                raise ValueError(f'Expected dict record at byte {offset} in {resolved}, got {type(record).__name__}')
            record_type = record.get('type')
            if record_type == 'header':
                if not mcstream_is_stream_header(record):
                    raise ValueError(f'Unsupported streaming header in {resolved}')
                header = record
            elif record_type == 'sample':
                offsets.append(offset)
            elif record_type == 'footer':
                footer = record
            del record
    if header is None:
        raise ValueError(f'No streaming multi-camera header in {resolved}')
    return mcstream_MultiCameraStreamIndex(path=resolved, header=header, sample_offsets=tuple(offsets), footer=footer)

def mcstream_load_sample_at_offset(path: str | Path, offset: int) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open('rb') as stream:
        stream.seek(int(offset))
        record = pickle.load(stream)
    if not isinstance(record, dict) or record.get('type') != 'sample':
        raise ValueError(f'Byte offset {offset} is not a sample record in {resolved}')
    sample = record.get('sample')
    if not isinstance(sample, dict):
        raise ValueError(f'Sample record at byte {offset} has no dict sample')
    return sample

class mcstream_StreamingSampleSequence(Sequence[dict[str, Any]]):
    """Random-access sample view that loads only the requested sample."""

    def __init__(self, index: mcstream_MultiCameraStreamIndex) -> None:
        self.index = index

    def __len__(self) -> int:
        return len(self.index.sample_offsets)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]:
        ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        return mcstream_load_sample_at_offset(self.index.path, self.index.sample_offsets[normalized])

# -----------------------------------------------------------------------------
# Embedded multi-camera raw -> four-target 020 adapter.
# -----------------------------------------------------------------------------
import argparse
import copy
import gc
import hashlib
import importlib.util
import pickle
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import yaml
mc_FILE_PATH = Path(__file__).resolve()
mc_PROJECT_ROOT = APRILCUBE_ROOT.parent.parent
if str(mc_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(mc_PROJECT_ROOT))
mc_FINALIZE_020_PATH = mc_PROJECT_ROOT / 'thirdparty' / 'aprilcube' / 'src' / '020_finalize_pose_postprocess.py'
mc_EXPECTED_D435_INTRINSICS = Path('/home/ps/RobotCamCalib1/outputs/intrinsics_d435_color_charuco_1920x1080_0716_130910_offline_filtered.yaml')
mc_EXPECTED_MIDDLE_INTRINSICS = Path('/home/ps/RobotCamCalib1/outputs/intrinsics_charuco_scale0p25_2592x1944_0712_225925.yaml')
mc_VERIFIED_D435_SDK_SERIAL = '244222070135'
mc_POSE_SIDECAR_FORMAT = 'consensv2_multi_cam_020_pose_sidecar_v1'
mc_EXPECTED_CUBE_CFG_BY_TARGET = {'wrist_Q': mc_PROJECT_ROOT / 'thirdparty/aprilcube/cubes/cube_april_36h11_100_123_2x2x2_outer62p5mm', 'index_Q': mc_PROJECT_ROOT / 'thirdparty/aprilcube/cubes/cube_april_36h11_6_11_1x1x1_15mm', 'thumb_Q': mc_PROJECT_ROOT / 'thirdparty/aprilcube/cubes/cube_april_36h11_12_17_1x1x1_15mm', 'middle_Q': mc_PROJECT_ROOT / 'thirdparty/aprilcube/cubes/cube_april_36h11_0_5_1x1x1_15mm'}
mc_SINGLE_TAG_EDGE_THRESHOLD_BY_TARGET = {'index_Q': 0.4, 'thumb_Q': 0.3, 'middle_Q': 0.08}
mc_SINGLE_TAG_MAX_REPROJ_BY_TARGET = {'middle_Q': 1.5}
mc_DEFAULT_SINGLE_TAG_EDGE_THRESHOLD = 0.6
mc_DEFAULT_SINGLE_TAG_MAX_REPROJ = 1.0

@dataclass(frozen=True)
class mc_PoseTarget:
    name: str
    worker_name: str
    camera_name: str
    cube_metadata_key: str
    intrinsics_kind: str
mc_TARGETS: tuple[mc_PoseTarget, ...] = (mc_PoseTarget('wrist_Q', 'rs', 'd435', 'wrist_cube_cfg', 'rs'), mc_PoseTarget('index_Q', 'thumb', 'thumb_web_cam', 'index_cube_cfg', 'cv2'), mc_PoseTarget('thumb_Q', 'thumb', 'thumb_web_cam', 'thumb_cube_cfg', 'cv2'), mc_PoseTarget('middle_Q', 'middle', 'middle_finger_cam', 'middle_cube_cfg', 'cv2'))

@dataclass(frozen=True)
class mc_TargetRuntime:
    spec: mc_PoseTarget
    intrinsics: dict[str, Any]
    recorded_intrinsics_path: Path
    cube_cfg: Path
    recorded_cube_cfg: Path
    single_tag_edge_threshold: float
    single_tag_max_reproj: float

def mc_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while (chunk := stream.read(8 * 1024 * 1024)):
            digest.update(chunk)
    return digest.hexdigest()

def mc_load_finalize_020() -> Any:
    """Return this already-loaded monolithic module.

    The former adapter imported 020 through a second module instance.  Keeping
    one instance avoids duplicated JAX/DeepTag state and is required for the
    standalone-file contract.
    """
    return sys.modules[__name__]

def mc_load_final_sidecar_smoother() -> Any:
    return type('_EmbeddedStage13', (), {'run': staticmethod(stage13_run)})

def mc_require_all_target_poses(summary: dict[str, Any], target_names: list[str]) -> None:
    frame_count = int(summary.get('frame_count', -1))
    success_counts = summary.get('success_counts', {}) or {}
    incomplete = {name: {'success': int(success_counts.get(name, 0)), 'required': frame_count, 'missing': frame_count - int(success_counts.get(name, 0))} for name in target_names if int(success_counts.get(name, 0)) != frame_count}
    if incomplete:
        raise RuntimeError(f'020 produced an intermediate sidecar, but final stage13 smoothing is forbidden until every cube has a pose on every frame. Run the target-specific direct/RGB-flow/local-interpolation recovery first: {incomplete}')

def mc_load_intrinsics(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f'Intrinsics YAML is not a mapping: {path}')
    dist = data.get('dist', data.get('D'))
    if dist is None:
        raise ValueError(f'Intrinsics YAML has no dist/D field: {path}')
    return {'path': path, 'image_size': tuple((int(v) for v in data['image_size'])), 'K': np.asarray(data['K'], dtype=np.float64).reshape(3, 3), 'dist': np.asarray(dist, dtype=np.float64).reshape(-1), 'camera_model': str(data.get('camera_model', 'pinhole')), 'distortion_model': str(data.get('distortion_model', '')), 'rms': float(data.get('rms', float('nan'))), 'mean_reproj_error': float(data.get('mean_reproj_error', float('nan'))), 'num_samples': int(data.get('num_samples', 0))}

def mc_target_intrinsics_path(metadata: dict[str, Any], target: mc_PoseTarget) -> Path:
    if target.intrinsics_kind == 'rs':
        value = metadata.get('rs_intrinsics_yaml')
    else:
        mapping = metadata.get('cv2_camera_to_intrinsics_yaml', {})
        value = mapping.get(target.camera_name) if isinstance(mapping, dict) else None
    if not value:
        raise ValueError(f'No recorded intrinsics path for {target.name}/{target.camera_name}')
    return Path(str(value)).expanduser().resolve()

def mc_build_target_runtimes(metadata: dict[str, Any], target_names: set[str] | None=None) -> list[mc_TargetRuntime]:
    runtimes: list[mc_TargetRuntime] = []
    for target in mc_TARGETS:
        if target_names is not None and target.name not in target_names:
            continue
        recorded_intrinsics_path = mc_target_intrinsics_path(metadata, target)
        intrinsics_path = recorded_intrinsics_path
        if target.name == 'wrist_Q' and recorded_intrinsics_path != mc_EXPECTED_D435_INTRINSICS.resolve():
            raise ValueError(f'D435 intrinsics mismatch: recorded={recorded_intrinsics_path}, expected={mc_EXPECTED_D435_INTRINSICS.resolve()}')
        if target.name == 'middle_Q':
            intrinsics_path = mc_EXPECTED_MIDDLE_INTRINSICS.resolve()
            if recorded_intrinsics_path != intrinsics_path:
                print(f'[WARN] Overriding recorded middle_finger_cam intrinsics: recorded={recorded_intrinsics_path}, effective={intrinsics_path}')
        intrinsics = mc_load_intrinsics(intrinsics_path)
        cube_value = metadata.get(target.cube_metadata_key)
        if not cube_value:
            raise ValueError(f'metadata.{target.cube_metadata_key} is missing')
        recorded_cube_cfg = Path(str(cube_value)).expanduser().resolve()
        cube_cfg = mc_EXPECTED_CUBE_CFG_BY_TARGET[target.name].resolve()
        if not (cube_cfg / 'config.json').is_file():
            raise FileNotFoundError(f'Cube config is missing for {target.name}: {cube_cfg}')
        runtimes.append(mc_TargetRuntime(target, intrinsics, recorded_intrinsics_path, cube_cfg, recorded_cube_cfg, mc_SINGLE_TAG_EDGE_THRESHOLD_BY_TARGET.get(target.name, mc_DEFAULT_SINGLE_TAG_EDGE_THRESHOLD), mc_SINGLE_TAG_MAX_REPROJ_BY_TARGET.get(target.name, mc_DEFAULT_SINGLE_TAG_MAX_REPROJ)))
    return runtimes

def mc_load_source_recording(source_path: Path) -> dict[str, Any]:
    with source_path.open('rb') as stream:
        first_record = pickle.load(stream)
    if mcstream_is_stream_header(first_record):
        index = mcstream_scan_stream_index(source_path)
        if not index.complete:
            raise ValueError(f'Streaming recording has no complete footer: {source_path}')
        footer = index.footer or {}
        return {'metadata': index.header.get('metadata', {}), 'num_samples': len(index.sample_offsets), 'record_hz': index.header.get('record_hz'), 'measured_record_hz': footer.get('measured_record_hz', index.header.get('measured_record_hz')), 'samples': mcstream_StreamingSampleSequence(index), '_source_storage': 'streaming'}
    if not isinstance(first_record, dict):
        raise ValueError(f'Expected recording dict or streaming header, got {type(first_record).__name__}')
    first_record['_source_storage'] = 'monolithic'
    return first_record

def mc_source_frame(sample: dict[str, Any], target: mc_PoseTarget) -> tuple[np.ndarray, float]:
    try:
        image = sample['worker_raw_frames'][target.worker_name][target.camera_name]
        timestamp = float(sample['worker_timestamps'][target.worker_name][target.camera_name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Missing frame/timestamp for {target.name}/{target.camera_name}') from exc
    return (image, timestamp)

def mc_validate_recording(data: Any, *, source_path: Path, max_frames: int | None, target_names: set[str] | None=None) -> tuple[dict[str, Any], Sequence[dict[str, Any]], list[mc_TargetRuntime]]:
    if not isinstance(data, dict) or not isinstance(data.get('samples'), Sequence):
        raise ValueError(f'Expected multi-camera recording with a sample sequence: {source_path}')
    metadata = data.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Recording has no metadata mapping')
    samples = data['samples']
    declared_count = int(data.get('num_samples', len(samples)))
    if declared_count != len(samples):
        raise ValueError(f'num_samples mismatch: declared={declared_count}, actual={len(samples)}')
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError('--max-frames must be positive')
        samples = samples[:max_frames]
    if not samples:
        raise ValueError('Recording contains no samples')
    if str(metadata.get('rs_color_format', '')).upper() != 'BGR8':
        raise ValueError(f"Expected rs_color_format=BGR8, got {metadata.get('rs_color_format')}")
    if not bool(metadata.get('rs_strict_profile', False)):
        raise ValueError('Recording did not declare rs_strict_profile=True')
    runtimes = mc_build_target_runtimes(metadata, target_names)
    checked_streams: set[tuple[str, str]] = set()
    for runtime in runtimes:
        target = runtime.spec
        stream_key = (target.worker_name, target.camera_name)
        if stream_key in checked_streams:
            continue
        checked_streams.add(stream_key)
        (target_w, target_h) = runtime.intrinsics['image_size']
        timestamps: list[float] = []
        for (sample_index, sample) in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f'Sample {sample_index} is not a mapping')
            (image, timestamp) = mc_source_frame(sample, target)
            if not isinstance(image, np.ndarray):
                raise ValueError(f'Sample {sample_index} {target.camera_name} is not ndarray')
            if image.shape != (target_h, target_w, 3) or image.dtype != np.uint8:
                raise ValueError(f'Sample {sample_index} {target.camera_name} mismatch: shape={image.shape}, dtype={image.dtype}, expected={(target_h, target_w, 3)}/uint8')
            timestamps.append(timestamp)
        if any((right <= left for (left, right) in zip(timestamps, timestamps[1:]))):
            raise ValueError(f'{target.camera_name} timestamps are not strictly increasing')
    return (metadata, samples, runtimes)

def mc_build_sample_manifest(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{'sample_index': sample_index, 'step_idx': int(sample.get('step_idx', sample_index)), 'time_monotonic': sample.get('time_monotonic'), 'time_wall': sample.get('time_wall'), 'physical_camera_timestamps': copy.deepcopy(sample.get('physical_camera_timestamps', {}))} for (sample_index, sample) in enumerate(samples)]

def mc_write_012_stream(*, source_path: Path, source_metadata: dict[str, Any], samples: Sequence[dict[str, Any]], runtime: mc_TargetRuntime, output_path: Path) -> None:
    target = runtime.spec
    intrinsics = runtime.intrinsics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.unlink(missing_ok=True)
    source_stat = source_path.stat()
    header = {'type': 'header', 'format': 'aprilcube_rs_raw_frame_stream_v1', 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_multi_cam_pkl': str(source_path), 'source_multi_cam_identity': {'size': int(source_stat.st_size), 'mtime_ns': int(source_stat.st_mtime_ns)}, 'metadata': {'script': str(mc_FILE_PATH), 'method': 'camera/target extraction from monolithic synchronized samples', 'source_format': 'consensv2_multi_cam_monolithic_dict_v1', 'pose_target': target.name, 'source_camera_name': target.camera_name, 'source_worker_name': target.worker_name, 'verified_d435_sdk_serial': mc_VERIFIED_D435_SDK_SERIAL if target.camera_name == 'd435' else None, 'intrinsics_yaml': str(intrinsics['path']), 'recorded_intrinsics_yaml': str(runtime.recorded_intrinsics_path), 'cube_cfg': str(runtime.cube_cfg), 'recorded_cube_cfg': str(runtime.recorded_cube_cfg), 'single_tag_edge_threshold': runtime.single_tag_edge_threshold, 'single_tag_max_reproj': runtime.single_tag_max_reproj, 'image_size': tuple((int(v) for v in intrinsics['image_size'])), 'fps': float(source_metadata.get('rs_fps' if target.intrinsics_kind == 'rs' else 'record_hz', 0.0)), 'undistort_for_detection': True, 'raw_camera_matrix': intrinsics['K'].tolist(), 'raw_dist_coeffs': intrinsics['dist'].tolist(), 'camera_model': intrinsics['camera_model'], 'distortion_model': intrinsics['distortion_model'], 'raw_image_field': 'image_bgr', 'raw_image_storage': 'original ndarray from source multi-camera sample', 'frame_count': len(samples)}}
    try:
        with temporary_path.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for (sample_index, sample) in enumerate(samples):
                (image, timestamp) = mc_source_frame(sample, target)
                frame = {'type': 'frame', 'frame_index': int(sample_index), 'source_sample_index': int(sample_index), 'source_step_idx': int(sample.get('step_idx', sample_index)), 'device_name': target.camera_name, 'camera_name': target.camera_name, 'loop_frame_idx': int(sample.get('step_idx', sample_index)), 'capture_timestamp': timestamp, 'source_sample_time_monotonic': sample.get('time_monotonic'), 'source_sample_time_wall': sample.get('time_wall'), 'shape': tuple((int(v) for v in image.shape)), 'dtype': str(image.dtype), 'image_bgr': image}
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
                done = sample_index + 1
                if done == len(samples) or done % 20 == 0:
                    print(f'\r[INFO] Extract {target.name}/{target.camera_name}: {done}/{len(samples)}', end='', flush=True)
            pickle.dump({'type': 'footer', 'frame_count': len(samples), 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print()
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

def mc_copy_final_pose_stream(final_pose_path: Path, result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(result_path.suffix + '.tmp')
    temporary_path.unlink(missing_ok=True)
    try:
        shutil.copy2(final_pose_path, temporary_path)
        temporary_path.replace(result_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

def mc_compact_pose_result(pose_frame: dict[str, Any], runtime: mc_TargetRuntime) -> dict[str, Any]:
    result = {'camera_name': runtime.spec.camera_name, 'capture_timestamp': pose_frame.get('capture_timestamp'), 'pose': copy.deepcopy(pose_frame.get('pose', {})), 'selected_stage': pose_frame.get('selected_stage', ''), 'overlay_shape': pose_frame.get('overlay_shape'), 'overlay_format': pose_frame.get('overlay_format'), 'overlay_jpeg': pose_frame.get('overlay_jpeg')}
    for key in ('pose_candidates', 'pose_before_temporal_smoothing', 'pose_temporally_smoothed', 'selected_stage_before_temporal_smoothing'):
        if key in pose_frame:
            result[key] = copy.deepcopy(pose_frame[key])
    return result

def mc_sidecar_target_metadata(runtime: mc_TargetRuntime) -> dict[str, Any]:
    return {'camera_name': runtime.spec.camera_name, 'worker_name': runtime.spec.worker_name, 'intrinsics_yaml': str(runtime.intrinsics['path']), 'recorded_intrinsics_yaml': str(runtime.recorded_intrinsics_path), 'intrinsics_yaml_sha256': mc_sha256_file(runtime.intrinsics['path']), 'camera_model': runtime.intrinsics['camera_model'], 'image_size': runtime.intrinsics['image_size'], 'cube_cfg': str(runtime.cube_cfg), 'recorded_cube_cfg': str(runtime.recorded_cube_cfg), 'single_tag_edge_threshold': runtime.single_tag_edge_threshold, 'single_tag_max_reproj': runtime.single_tag_max_reproj}

def mc_write_combined_sidecar(*, finalize020: Any, source_path: Path, source_metadata: dict[str, Any], sample_manifest: list[dict[str, Any]], runtimes: list[mc_TargetRuntime], result_paths: dict[str, Path], output_path: Path) -> dict[str, Any]:
    indexed: dict[str, tuple[dict[str, Any], list[int], dict[str, Any] | None]] = {}
    for runtime in runtimes:
        indexed[runtime.spec.name] = finalize020.build_stream_index(result_paths[runtime.spec.name])
        frame_count = len(indexed[runtime.spec.name][1])
        if frame_count != len(sample_manifest):
            raise ValueError(f'{runtime.spec.name} frame count mismatch: {frame_count} != {len(sample_manifest)}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.unlink(missing_ok=True)
    source_stat = source_path.stat()
    success_counts = {runtime.spec.name: 0 for runtime in runtimes}
    pose_source_counts: dict[str, dict[str, int]] = {runtime.spec.name: {} for runtime in runtimes}
    try:
        with temporary_path.open('wb') as stream:
            pickle.dump({'type': 'header', 'format': mc_POSE_SIDECAR_FORMAT, 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_multi_cam_pkl': str(source_path), 'source_multi_cam_identity': {'size': int(source_stat.st_size), 'mtime_ns': int(source_stat.st_mtime_ns)}, 'source_multi_cam_metadata': copy.deepcopy(source_metadata), 'metadata': {'script': str(mc_FILE_PATH), 'pipeline_script': str(mc_FINALIZE_020_PATH), 'pipeline_script_sha256': mc_sha256_file(mc_FINALIZE_020_PATH), 'mapping_key': 'sample_index', 'frame_count': len(sample_manifest), 'contains_raw_images': False, 'contains_overlay_jpeg': True, 'verified_d435_sdk_serial': mc_VERIFIED_D435_SDK_SERIAL, 'targets': {runtime.spec.name: mc_sidecar_target_metadata(runtime) for runtime in runtimes}}}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for (sample_index, manifest) in enumerate(sample_manifest):
                pose_results: dict[str, Any] = {}
                for runtime in runtimes:
                    target_name = runtime.spec.name
                    (_header, offsets, _footer) = indexed[target_name]
                    pose_frame = finalize020.load_at(result_paths[target_name], offsets[sample_index])
                    if int(pose_frame.get('frame_index', -1)) != sample_index:
                        raise ValueError(f'{target_name} frame index mismatch at {sample_index}')
                    compact = mc_compact_pose_result(pose_frame, runtime)
                    pose_results[target_name] = compact
                    pose = compact['pose'] or {}
                    success_counts[target_name] += int(bool(pose.get('success', False)))
                    source = str(pose.get('pose_source', ''))
                    counts = pose_source_counts[target_name]
                    counts[source] = counts.get(source, 0) + 1
                pickle.dump({'type': 'frame', **copy.deepcopy(manifest), 'pose_results': pose_results, 'poses': {name: copy.deepcopy(result['pose']) for (name, result) in pose_results.items()}}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            summary = {'frame_count': len(sample_manifest), 'success_counts': success_counts, 'pose_source_counts': pose_source_counts}
            pickle.dump({'type': 'footer', **summary, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(output_path)
        return summary
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

def mc_write_merged_sidecar(*, finalize020: Any, source_path: Path, sample_manifest: list[dict[str, Any]], runtimes: list[mc_TargetRuntime], result_paths: dict[str, Path], existing_path: Path, output_path: Path) -> dict[str, Any]:
    """Replace selected target results while preserving all other sidecar targets."""
    if not existing_path.is_file():
        raise FileNotFoundError(f'Existing sidecar does not exist: {existing_path}')
    indexed: dict[str, tuple[dict[str, Any], list[int], dict[str, Any] | None]] = {}
    for runtime in runtimes:
        target_name = runtime.spec.name
        indexed[target_name] = finalize020.build_stream_index(result_paths[target_name])
        frame_count = len(indexed[target_name][1])
        if frame_count != len(sample_manifest):
            raise ValueError(f'{target_name} frame count mismatch: {frame_count} != {len(sample_manifest)}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.unlink(missing_ok=True)
    selected_names = [runtime.spec.name for runtime in runtimes]
    source_stat = source_path.stat()
    try:
        with existing_path.open('rb') as source_stream, temporary_path.open('wb') as output_stream:
            header = pickle.load(source_stream)
            if not isinstance(header, dict) or header.get('format') != mc_POSE_SIDECAR_FORMAT:
                raise ValueError(f"Unsupported existing sidecar format: {header.get('format')}")
            if Path(str(header.get('source_multi_cam_pkl', ''))).resolve() != source_path:
                raise ValueError('Existing sidecar source PKL does not match requested source')
            existing_identity = header.get('source_multi_cam_identity', {})
            expected_identity = {'size': int(source_stat.st_size), 'mtime_ns': int(source_stat.st_mtime_ns)}
            if existing_identity != expected_identity:
                raise ValueError(f'Existing sidecar source identity mismatch: existing={existing_identity}, current={expected_identity}')
            updated_header = copy.deepcopy(header)
            metadata = updated_header.setdefault('metadata', {})
            targets_metadata = metadata.setdefault('targets', {})
            for runtime in runtimes:
                targets_metadata[runtime.spec.name] = mc_sidecar_target_metadata(runtime)
            history = metadata.setdefault('update_history', [])
            history.append({'updated_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'reprocessed_targets': selected_names, 'pipeline_script': str(mc_FINALIZE_020_PATH), 'pipeline_script_sha256': mc_sha256_file(mc_FINALIZE_020_PATH)})
            pickle.dump(updated_header, output_stream, protocol=pickle.HIGHEST_PROTOCOL)
            all_target_names = list(targets_metadata)
            success_counts = {name: 0 for name in all_target_names}
            pose_source_counts: dict[str, dict[str, int]] = {name: {} for name in all_target_names}
            for (sample_index, manifest) in enumerate(sample_manifest):
                record = pickle.load(source_stream)
                if not isinstance(record, dict) or record.get('type') != 'frame':
                    raise ValueError(f'Expected existing sidecar frame at index {sample_index}')
                if int(record.get('sample_index', -1)) != sample_index:
                    raise ValueError(f'Existing sidecar sample index mismatch at {sample_index}')
                if int(record.get('step_idx', -1)) != int(manifest['step_idx']):
                    raise ValueError(f'Existing sidecar step_idx mismatch at {sample_index}')
                pose_results = record.setdefault('pose_results', {})
                poses = record.setdefault('poses', {})
                for runtime in runtimes:
                    target_name = runtime.spec.name
                    (_result_header, offsets, _result_footer) = indexed[target_name]
                    pose_frame = finalize020.load_at(result_paths[target_name], offsets[sample_index])
                    if int(pose_frame.get('frame_index', -1)) != sample_index:
                        raise ValueError(f'{target_name} frame index mismatch at {sample_index}')
                    compact = mc_compact_pose_result(pose_frame, runtime)
                    pose_results[target_name] = compact
                    poses[target_name] = copy.deepcopy(compact['pose'])
                for target_name in all_target_names:
                    if target_name not in pose_results:
                        raise ValueError(f'Existing frame {sample_index} lacks {target_name}')
                    pose = pose_results[target_name].get('pose', {}) or {}
                    success_counts[target_name] += int(bool(pose.get('success', False)))
                    source = str(pose.get('pose_source', ''))
                    counts = pose_source_counts[target_name]
                    counts[source] = counts.get(source, 0) + 1
                pickle.dump(record, output_stream, protocol=pickle.HIGHEST_PROTOCOL)
            old_footer = pickle.load(source_stream)
            if not isinstance(old_footer, dict) or old_footer.get('type') != 'footer':
                raise ValueError('Existing sidecar footer is missing')
            if int(old_footer.get('frame_count', -1)) != len(sample_manifest):
                raise ValueError('Existing sidecar footer frame count mismatch')
            summary = {'frame_count': len(sample_manifest), 'success_counts': success_counts, 'pose_source_counts': pose_source_counts, 'reprocessed_targets': selected_names}
            pickle.dump({'type': 'footer', **summary, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}, output_stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(output_path)
        return summary
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

def mc_run(args: argparse.Namespace) -> None:
    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    temp_root = args.temp_root.expanduser().resolve()
    requested_target_names = set(args.targets)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    print(f'[INFO] Loading recording: {source_path}')
    load_started = time.perf_counter()
    data = mc_load_source_recording(source_path)
    print(f"[INFO] Loaded {data.get('_source_storage', 'unknown')} recording in {time.perf_counter() - load_started:.2f}s")
    (source_metadata, samples, runtimes) = mc_validate_recording(data, source_path=source_path, max_frames=args.max_frames, target_names=requested_target_names)
    sample_manifest = mc_build_sample_manifest(samples)
    print(f'[INFO] Validated frames={len(samples)} targets={[r.spec.name for r in runtimes]}')
    for runtime in runtimes:
        print(f"[INFO] {runtime.spec.name}: camera={runtime.spec.camera_name} size={runtime.intrinsics['image_size']} model={runtime.intrinsics['camera_model']} rms={runtime.intrinsics['rms']:.3f}px cube={runtime.cube_cfg.name} recorded_intrinsics={runtime.recorded_intrinsics_path.name} recorded_cube={runtime.recorded_cube_cfg.name} single_tag_edge={runtime.single_tag_edge_threshold:.2f} single_tag_reproj={runtime.single_tag_max_reproj:.2f}px")
    if args.validate_only:
        return
    if temp_root.exists() and any(temp_root.iterdir()):
        raise FileExistsError(f'Temporary root is not empty: {temp_root}. Remove it or choose --temp-root.')
    temp_root.mkdir(parents=True, exist_ok=True)
    raw_paths: dict[str, Path] = {}
    result_paths: dict[str, Path] = {}
    try:
        for runtime in runtimes:
            target_root = temp_root / runtime.spec.name
            raw_path = target_root / f'{source_path.stem}_{runtime.spec.name}_012_raw.pkl'
            mc_write_012_stream(source_path=source_path, source_metadata=source_metadata, samples=samples, runtime=runtime, output_path=raw_path)
            raw_paths[runtime.spec.name] = raw_path
        del samples
        del data
        gc.collect()
        finalize020 = mc_load_finalize_020()
        results_root = temp_root / 'pose_results'
        strict_index_stream: Path | None = None
        for runtime in runtimes:
            target_name = runtime.spec.name
            target_root = temp_root / target_name
            work_dir = target_root / '020_work'
            unused_merged_output = target_root / 'unused_020_with_raw.pkl'
            print(f'[INFO] Starting 020 target={target_name} camera={runtime.spec.camera_name}')
            final_pose_path = finalize020._process_extracted_target_stream(raw_pkl=raw_paths[target_name], output_pkl=unused_merged_output, work_dir=work_dir, merge_final_raw=False, single_tag_edge_threshold=runtime.single_tag_edge_threshold, single_tag_max_reproj=runtime.single_tag_max_reproj, preferred_single_tag_id=2 if target_name == 'middle_Q' else None, prefer_deeptag_single_tag=target_name == 'middle_Q', enable_temporal_outline_recovery=target_name == 'wrist_Q', enable_adjacent_rgb_flow_recovery=False, target_name=target_name)
            result_path = results_root / f'{target_name}_final_pose.pkl'
            mc_copy_final_pose_stream(final_pose_path, result_path)
            result_paths[target_name] = result_path
            if target_name == 'index_Q':
                strict_source = work_dir / f"strict_aprilcube_pose_{raw_paths[target_name].stem}.pkl"
                strict_index_stream = results_root / 'index_Q_strict_aprilcube.pkl'
                mc_copy_final_pose_stream(strict_source, strict_index_stream)
            shutil.rmtree(target_root)
            print(f'[INFO] Completed 020 target={target_name}')
        if args.merge_existing is not None:
            summary = mc_write_merged_sidecar(finalize020=finalize020, source_path=source_path, sample_manifest=sample_manifest, runtimes=runtimes, result_paths=result_paths, existing_path=args.merge_existing.expanduser().resolve(), output_path=output_path)
        else:
            all_target_names = {target.name for target in mc_TARGETS}
            if requested_target_names != all_target_names:
                raise ValueError('A partial --targets run requires --merge-existing so the other target results are preserved.')
            summary = mc_write_combined_sidecar(finalize020=finalize020, source_path=source_path, source_metadata=source_metadata, sample_manifest=sample_manifest, runtimes=runtimes, result_paths=result_paths, output_path=output_path)
        print(f'[INFO] Saved combined pose sidecar: {output_path}')
        print(f'[INFO] Sidecar summary: {summary}')
        all_target_names = {target.name for target in mc_TARGETS}
        if requested_target_names == all_target_names:
            if strict_index_stream is None:
                raise RuntimeError('The complete run did not preserve index_Q strict poses')
            summary = run_embedded_pose_recovery_patches(
                source_path=source_path,
                initial_sidecar=output_path,
                strict_index_stream=strict_index_stream,
                output_sidecar=output_path,
                work_dir=temp_root / 'complete_pose_recovery',
            )
            print(f'[INFO] Complete recovered sidecar summary: {summary}')
        if not args.skip_final_global_smoothing:
            all_target_names = [target.name for target in mc_TARGETS]
            mc_require_all_target_poses(summary, all_target_names)
            final_output_path = args.final_output.expanduser().resolve() if args.final_output is not None else output_path.with_name(f'{source_path.stem}_post_progress{output_path.suffix}')
            final_qa_path = args.final_qa.expanduser().resolve() if args.final_qa is not None else final_output_path.with_suffix('.smoothing_qa.json')
            print('[INFO] All target poses complete; starting mandatory stage13 final sidecar smoothing')
            final_smoother = mc_load_final_sidecar_smoother()
            final_smoother.run(SimpleNamespace(source=source_path, sidecar=output_path, output=final_output_path, qa=final_qa_path, targets=all_target_names, window_radius=4, window_seconds=0.18, sigma_seconds=0.075, max_measured_translation_mm=4.0, max_measured_rotation_deg=4.0, max_filled_translation_mm=10.0, max_filled_rotation_deg=8.0, max_edge_score_drop=0.04, overwrite=True))
            print(f'[INFO] Final complete+smoothed sidecar: {final_output_path}')
    finally:
        if args.keep_temp:
            print(f'[INFO] Keeping temporary files: {temp_root}')
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f'[INFO] Removed temporary files: {temp_root}')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Embedded Wuji-left four-point retargeting.
#
# This implementation is namespace-prefixed from the former two retargeting
# scripts. It is executable code in this file, preserving the original
# numerical path without loading either script at runtime.
# -----------------------------------------------------------------------------
import argparse
import json
import pickle
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
rtbase_FILE_PATH = Path(__file__).resolve()
rtbase_REPO_ROOT = APRILCUBE_ROOT.parent.parent
rtbase_PYROKI_SRC = rtbase_REPO_ROOT / 'thirdparty/pyroki/src'
if str(rtbase_REPO_ROOT) not in sys.path:
    sys.path.append(str(rtbase_REPO_ROOT))
if str(rtbase_PYROKI_SRC) not in sys.path:
    sys.path.append(str(rtbase_PYROKI_SRC))
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
import yaml
from scipy.spatial.transform import Rotation
rtbase_DEFAULT_PKL_PATH = rtbase_REPO_ROOT / 'thirdparty/aprilcube/recordings/021_hand_back_sync_raw_frames_20260712_233831.pkl'
rtbase_DEFAULT_URDF_PATH = rtbase_REPO_ROOT / 'thirdparty/wuji-description/hand/body-with-soft/urdf/left_simplified_w_fingereye.urdf'
rtbase_DEFAULT_FINGERTIP_GEOMETRY_URDF_PATH = rtbase_REPO_ROOT / 'thirdparty/xarm7_wuji_left_description/xarm7_wuji_left_w_fingereye_v2.urdf'
rtbase_DEFAULT_CONTACT_CONFIG = rtbase_REPO_ROOT / 'configs/retarget/left_wuji_fingertip_contact_keypoints_v2.yaml'
rtbase_DEFAULT_OUTPUT_DIR = rtbase_REPO_ROOT / 'outputs/retargeting/index_middle_cube_calibration_021'
rtbase_DEFAULT_INDEX_ONLY_RESULT = rtbase_REPO_ROOT / 'outputs/retargeting/index_cube_calibration_021/optimized_unconstrained_four_point_se3_from_bounded.npz'
rtbase_EXPECTED_PKL_FORMAT = 'aprilcube_hand_back_software_synced_raw_v1'
rtbase_SOURCE_POSE_FIELD = 'hand_back_cube_obj_poses'
rtbase_ROOT_LINK_NAME = 'left_palm_link'
rtbase_POSITION_WEIGHT = 40.0
rtbase_NUMERICAL_ACTIVE_WEIGHT = 0.001
rtbase_INACTIVE_JOINT_WEIGHT = 2.0
rtbase_LAMBDA_INITIAL = 1.0
rtbase_DEFAULT_SOLVER_ITERATIONS = 100
rtbase_DEFAULT_ALTERNATING_ITERATIONS = 30
rtbase_T_PALM_CUBE_INITIAL = np.asarray([[0.0, 0.0, -1.0, -0.07175], [0.0, -1.0, 0.0, 0.005444], [-1.0, 0.0, 0.0, 0.011613], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

@dataclass(frozen=True)
class rtbase_FingerSpec:
    name: str
    robot_link_name: str
    robot_joint_prefix: str
    obj_mesh_name: str
rtbase_FINGER_SPECS = (rtbase_FingerSpec('index', 'left_finger2_link4', 'left_finger2_', 'index_v2.obj'), rtbase_FingerSpec('middle', 'left_finger3_link4', 'left_finger3_', 'middle_v2.obj'))
rtbase_ACTIVE_JOINT_NAMES = tuple((f'left_finger{finger_index}_joint{joint_index}' for finger_index in (2, 3) for joint_index in range(1, 5)))

@dataclass
class rtbase_SequenceSolution:
    qpos: np.ndarray
    target_obj_poses: np.ndarray
    predicted_obj_poses: np.ndarray
    target_keypoints: np.ndarray
    predicted_keypoints: np.ndarray
    mean_error_m: np.ndarray
    max_error_m: np.ndarray
    elapsed_seconds: float

def rtbase_transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]

def rtbase_make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = translation
    return output

def rtbase_rpy_transform(xyz: list[float], rpy: list[float]) -> np.ndarray:
    return rtbase_make_transform(Rotation.from_euler('xyz', rpy).as_matrix(), np.asarray(xyz, dtype=np.float64))

def rtbase_validate_transform(transform: np.ndarray, label: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f'{label} must be a finite 4x4 matrix')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-08):
        raise ValueError(f'{label} has an invalid homogeneous last row')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-05):
        raise ValueError(f'{label} rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-05):
        raise ValueError(f'{label} rotation determinant is not +1')
    return transform

def rtbase_load_urdf(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(str(path), filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent))

def rtbase_load_contact_keypoints(config_path: Path) -> np.ndarray:
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if config.get('coordinate_frame') != 'per_finger_obj':
        raise ValueError('Contact points must be expressed in per_finger_obj')
    if config.get('coordinate_unit') != 'm':
        raise ValueError('Contact points must use metres')
    fingers = config.get('fingers', {})
    output: list[np.ndarray] = []
    for spec in rtbase_FINGER_SPECS:
        payload = fingers.get(spec.name)
        if not isinstance(payload, dict):
            raise KeyError(f'Contact config has no {spec.name} entry')
        if Path(payload.get('obj_path', '')).name != spec.obj_mesh_name:
            raise ValueError(f'{spec.name} points do not reference {spec.obj_mesh_name}')
        points = np.asarray(payload.get('keypoints_obj_m'), dtype=np.float64)
        if points.shape != (4, 3) or not np.all(np.isfinite(points)):
            raise ValueError(f'{spec.name} keypoints must have finite shape (4, 3)')
        output.append(points)
    return np.stack(output, axis=0)

def rtbase_load_link_from_obj_transforms(urdf_path: Path) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    output: list[np.ndarray] = []
    for spec in rtbase_FINGER_SPECS:
        matches: list[np.ndarray] = []
        for link in root.findall('link'):
            if link.attrib.get('name') != spec.robot_link_name:
                continue
            for visual in link.findall('visual'):
                mesh = visual.find('./geometry/mesh')
                if mesh is None:
                    continue
                if Path(mesh.attrib.get('filename', '')).name != spec.obj_mesh_name:
                    continue
                scale = np.asarray([float(value) for value in mesh.attrib.get('scale', '1 1 1').split()])
                if scale.shape != (3,) or not np.allclose(scale, 0.001, atol=1e-12):
                    raise ValueError(f'{spec.obj_mesh_name} scale must be 0.001, got {scale.tolist()}')
                origin = visual.find('origin')
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if origin is not None:
                    xyz = [float(v) for v in origin.attrib.get('xyz', '0 0 0').split()]
                    rpy = [float(v) for v in origin.attrib.get('rpy', '0 0 0').split()]
                matches.append(rtbase_rpy_transform(xyz, rpy))
        if len(matches) != 1:
            raise ValueError(f'Expected one {spec.obj_mesh_name} visual on {spec.robot_link_name}; found {len(matches)}')
        output.append(matches[0])
    return np.stack(output, axis=0)

def rtbase_load_source_poses(pkl_path: Path, max_frames: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    poses: list[np.ndarray] = []
    timestamps: list[float] = []
    source_record_indices: list[int] = []
    total_bytes = pkl_path.stat().st_size
    record_index = -1
    with pkl_path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('type') != 'header':
            raise ValueError('Source PKL does not begin with a header')
        if header.get('format') != rtbase_EXPECTED_PKL_FORMAT:
            raise ValueError(f"Unexpected PKL format: {header.get('format')!r}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            record_index += 1
            if not isinstance(record, dict) or record.get('type') != 'frame_pair':
                continue
            payload = record.get(rtbase_SOURCE_POSE_FIELD)
            if not isinstance(payload, dict):
                raise KeyError(f'Record {record_index} is missing {rtbase_SOURCE_POSE_FIELD}')
            if payload.get('reference_frame') != 'hand_back_cube':
                raise ValueError(f'Record {record_index} has the wrong reference frame')
            objects = payload.get('objects', {})
            current: list[np.ndarray] = []
            valid = True
            for spec in rtbase_FINGER_SPECS:
                entry = objects.get(spec.name)
                if not isinstance(entry, dict) or not bool(entry.get('success', False)):
                    valid = False
                    break
                current.append(rtbase_validate_transform(entry.get('T_hand_back_cube_obj'), f'record {record_index} {spec.name} pose'))
            if not valid:
                print(f'\n[WARN] Skipping record {record_index}: invalid finger pose')
                continue
            poses.append(np.stack(current, axis=0))
            timestamps.append(float(record.get('pair_timestamp', len(poses) - 1)))
            source_record_indices.append(record_index)
            if len(poses) == 1 or len(poses) % 25 == 0:
                ratio = stream.tell() / max(total_bytes, 1)
                print(f'\r[INFO] Reading source: {len(poses)} valid frames ({ratio * 100.0:.1f}%)', end='', flush=True)
            if max_frames is not None and len(poses) >= max_frames:
                break
    print()
    if not poses:
        raise ValueError('No valid index+middle frames found')
    return (np.asarray(poses, dtype=np.float64), np.asarray(timestamps, dtype=np.float64), np.asarray(source_record_indices, dtype=np.int32))

class rtbase_IndexMiddleIKSolver:

    def __init__(self, urdf: yourdfpy.URDF, robot: pk.Robot, link_from_obj: np.ndarray, keypoints_obj: np.ndarray, natural_qpos: np.ndarray, active_mask: np.ndarray, max_iterations: int) -> None:
        self.urdf = urdf
        self.robot = robot
        self.link_from_obj = np.asarray(link_from_obj, dtype=np.float64)
        self.keypoints_obj = np.asarray(keypoints_obj, dtype=np.float64)
        self.keypoints_link = np.stack([rtbase_transform_points(transform, points) for (transform, points) in zip(link_from_obj, keypoints_obj)], axis=0).astype(np.float32)
        self.natural_qpos = np.asarray(natural_qpos, dtype=np.float32)
        self.active_mask = np.asarray(active_mask, dtype=np.float32)
        self.lower = np.asarray(robot.joints.lower_limits, dtype=np.float32)
        self.upper = np.asarray(robot.joints.upper_limits, dtype=np.float32)
        self.link_indices = np.asarray([robot.links.names.index(spec.robot_link_name) for spec in rtbase_FINGER_SPECS], dtype=np.int32)
        self.max_iterations = int(max_iterations)
        self._build_solver()

    def _build_solver(self) -> None:
        num_joints = len(self.natural_qpos)
        num_fingers = len(rtbase_FINGER_SPECS)

        class TargetVar(jaxls.Var[jax.Array], default_factory=lambda : jnp.zeros((num_fingers, 4, 3), dtype=jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...

        class PreviousVar(jaxls.Var[jax.Array], default_factory=lambda : jnp.zeros((num_joints,), dtype=jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...
        joint_var = self.robot.joint_var_cls(0)
        target_var = TargetVar(0)
        previous_var = PreviousVar(0)
        robot = self.robot
        link_indices = jnp.asarray(self.link_indices, dtype=jnp.int32)
        points_link = jnp.asarray(self.keypoints_link, dtype=jnp.float32)
        natural = jnp.asarray(self.natural_qpos, dtype=jnp.float32)
        weights = jnp.asarray(self.active_mask * rtbase_NUMERICAL_ACTIVE_WEIGHT + (1.0 - self.active_mask) * rtbase_INACTIVE_JOINT_WEIGHT, dtype=jnp.float32)

        @jaxls.Cost.factory
        def alignment_cost(values: jaxls.VarValues, var_q: jaxls.Var[jnp.ndarray], var_target: jaxls.Var[jnp.ndarray]) -> jax.Array:
            root_from_links = jaxlie.SE3(robot.forward_kinematics(cfg=values[var_q])[link_indices])
            predicted = jnp.einsum('fij,fkj->fki', root_from_links.rotation().as_matrix(), points_link) + root_from_links.translation()[:, None, :]
            return ((predicted - values[var_target]) * rtbase_POSITION_WEIGHT).reshape(-1)

        @jaxls.Cost.factory
        def seed_cost(values: jaxls.VarValues, var_q: jaxls.Var[jnp.ndarray], var_previous: jaxls.Var[jnp.ndarray]) -> jax.Array:
            return ((values[var_q] - values[var_previous]) * weights).reshape(-1)
        self.joint_var = joint_var
        self.target_var = target_var
        self.previous_var = previous_var
        self.problem = jaxls.LeastSquaresProblem(costs=[alignment_cost(joint_var, target_var), pk.costs.rest_cost(joint_var, natural, weights), seed_cost(joint_var, previous_var), pk.costs.limit_constraint(robot, joint_var)], variables=[joint_var, target_var, previous_var]).analyze(use_onp=True)

    def solve_frame(self, target_keypoints: np.ndarray, initial_qpos: np.ndarray) -> np.ndarray:
        values = jaxls.VarValues.make([self.joint_var.with_value(jnp.asarray(initial_qpos, dtype=jnp.float32)), self.target_var.with_value(jnp.asarray(target_keypoints, dtype=jnp.float32)), self.previous_var.with_value(jnp.asarray(initial_qpos, dtype=jnp.float32))])
        solution = self.problem.solve(initial_vals=values, verbose=False, linear_solver='conjugate_gradient', trust_region=jaxls.TrustRegionConfig(lambda_initial=rtbase_LAMBDA_INITIAL), termination=jaxls.TerminationConfig(max_iterations=self.max_iterations))
        qpos = np.asarray(solution[self.joint_var], dtype=np.float32)
        qpos = np.clip(qpos, self.lower, self.upper)
        qpos[self.active_mask < 0.5] = self.natural_qpos[self.active_mask < 0.5]
        return qpos

    def solve_sequence(self, palm_from_cube: np.ndarray, cube_from_obj: np.ndarray) -> rtbase_SequenceSolution:
        started = time.monotonic()
        q_values: list[np.ndarray] = []
        target_obj_values: list[np.ndarray] = []
        predicted_obj_values: list[np.ndarray] = []
        target_point_values: list[np.ndarray] = []
        predicted_point_values: list[np.ndarray] = []
        mean_errors: list[float] = []
        max_errors: list[float] = []
        previous = self.natural_qpos.copy()
        frame_count = len(cube_from_obj)
        for (frame_index, frame_cube_objs) in enumerate(cube_from_obj):
            target_objs = np.stack([palm_from_cube @ cube_obj for cube_obj in frame_cube_objs], axis=0)
            target_points = np.stack([rtbase_transform_points(transform, points) for (transform, points) in zip(target_objs, self.keypoints_obj)], axis=0)
            qpos = self.solve_frame(target_points, previous)
            previous = qpos
            self.urdf.update_cfg(qpos)
            predicted_objs: list[np.ndarray] = []
            predicted_points: list[np.ndarray] = []
            for (spec, link_obj, points_obj) in zip(rtbase_FINGER_SPECS, self.link_from_obj, self.keypoints_obj):
                palm_link = np.asarray(self.urdf.get_transform(spec.robot_link_name, rtbase_ROOT_LINK_NAME, collision_geometry=False), dtype=np.float64)
                palm_obj = palm_link @ link_obj
                predicted_objs.append(palm_obj)
                predicted_points.append(rtbase_transform_points(palm_obj, points_obj))
            predicted_objs_array = np.stack(predicted_objs, axis=0)
            predicted_points_array = np.stack(predicted_points, axis=0)
            errors = np.linalg.norm(predicted_points_array - target_points, axis=-1)
            q_values.append(qpos.copy())
            target_obj_values.append(target_objs)
            predicted_obj_values.append(predicted_objs_array)
            target_point_values.append(target_points)
            predicted_point_values.append(predicted_points_array)
            mean_errors.append(float(np.mean(errors)))
            max_errors.append(float(np.max(errors)))
            if frame_index == 0 or (frame_index + 1) % 50 == 0:
                print(f'\r[INFO] IK [{frame_index + 1:4d}/{frame_count}] mean={np.mean(mean_errors) * 1000.0:.3f} mm', end='', flush=True)
        print()
        return rtbase_SequenceSolution(qpos=np.asarray(q_values, dtype=np.float32), target_obj_poses=np.asarray(target_obj_values, dtype=np.float64), predicted_obj_poses=np.asarray(predicted_obj_values, dtype=np.float64), target_keypoints=np.asarray(target_point_values, dtype=np.float64), predicted_keypoints=np.asarray(predicted_point_values, dtype=np.float64), mean_error_m=np.asarray(mean_errors, dtype=np.float64), max_error_m=np.asarray(max_errors, dtype=np.float64), elapsed_seconds=time.monotonic() - started)

def rtbase_fit_global_transform(cube_from_obj: np.ndarray, keypoints_obj: np.ndarray, predicted_keypoints: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cube_points = np.asarray([[rtbase_transform_points(transform, points) for (transform, points) in zip(frame_transforms, keypoints_obj)] for frame_transforms in cube_from_obj], dtype=np.float64).reshape(-1, 3)
    robot_points = np.asarray(predicted_keypoints, dtype=np.float64).reshape(-1, 3)
    source_centroid = np.mean(cube_points, axis=0)
    target_centroid = np.mean(robot_points, axis=0)
    source_centered = cube_points - source_centroid
    target_centered = robot_points - target_centroid
    covariance = source_centered.T @ target_centered
    (u, singular_values, vt) = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = rtbase_make_transform(rotation, translation)
    fitted = (rotation @ cube_points.T).T + translation
    residual = np.linalg.norm(fitted - robot_points, axis=-1)
    return (transform, {'point_count': int(len(cube_points)), 'singular_values': singular_values.tolist(), 'fixed_q_mean_error_mm': float(np.mean(residual) * 1000.0)})

def rtbase_run_alternating(solver: rtbase_IndexMiddleIKSolver, cube_from_obj: np.ndarray, keypoints_obj: np.ndarray, initial_transform: np.ndarray, max_iterations: int) -> tuple[np.ndarray, rtbase_SequenceSolution, list[dict[str, Any]], bool]:
    transform = rtbase_validate_transform(initial_transform, 'initial transform').copy()
    best_transform = transform.copy()
    best_solution: rtbase_SequenceSolution | None = None
    best_error = float('inf')
    previous_error = float('inf')
    converged = False
    history: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        print(f'[INFO] alternating iteration {iteration + 1}')
        solution = solver.solve_sequence(transform, cube_from_obj)
        mean_error = float(np.mean(solution.mean_error_m))
        if mean_error < best_error:
            best_error = mean_error
            best_transform = transform.copy()
            best_solution = solution
        (next_transform, fit_info) = rtbase_fit_global_transform(cube_from_obj, keypoints_obj, solution.predicted_keypoints)
        rotation_change = float(np.linalg.norm(Rotation.from_matrix(next_transform[:3, :3] @ transform[:3, :3].T).as_rotvec()))
        translation_change = float(np.linalg.norm(next_transform[:3, 3] - transform[:3, 3]))
        history.append({'iteration': iteration + 1, 'mean_error_mm': mean_error * 1000.0, 'T_before': transform.tolist(), 'T_after': next_transform.tolist(), 'translation_change_m': translation_change, 'rotation_change_rad': rotation_change, 'registration': fit_info})
        error_change = abs(previous_error - mean_error)
        transform = next_transform
        if translation_change < 1e-07 and rotation_change < 1e-06 and (error_change < 1e-07):
            converged = True
            break
        previous_error = mean_error
    print('[INFO] final IK')
    final_solution = solver.solve_sequence(transform, cube_from_obj)
    final_error = float(np.mean(final_solution.mean_error_m))
    if final_error < best_error or best_solution is None:
        return (transform, final_solution, history, converged)
    return (best_transform, best_solution, history, converged)

def rtbase_stats_mm(values_m: np.ndarray) -> dict[str, float]:
    values = np.asarray(values_m, dtype=np.float64) * 1000.0
    return {'mean_mm': float(np.mean(values)), 'median_mm': float(np.median(values)), 'p95_mm': float(np.percentile(values, 95.0)), 'max_mm': float(np.max(values))}

def rtbase_summarize(name: str, transform: np.ndarray, solution: rtbase_SequenceSolution, history: list[dict[str, Any]], converged: bool, active_indices: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    errors = np.linalg.norm(solution.predicted_keypoints - solution.target_keypoints, axis=-1)
    q_active = solution.qpos[:, active_indices]
    margins = np.minimum(q_active - lower[active_indices][None, :], upper[active_indices][None, :] - q_active)
    delta_rotation = transform[:3, :3] @ rtbase_T_PALM_CUBE_INITIAL[:3, :3].T
    return {'name': name, 'fingers': [spec.name for spec in rtbase_FINGER_SPECS], 'frame_count': int(len(solution.qpos)), 'T_left_palm_link_hand_back_cube': transform.tolist(), 'translation_m': transform[:3, 3].tolist(), 'translation_delta_m': (transform[:3, 3] - rtbase_T_PALM_CUBE_INITIAL[:3, 3]).tolist(), 'rotation_delta_rotvec_deg': (Rotation.from_matrix(delta_rotation).as_rotvec() * 180.0 / np.pi).tolist(), 'all_keypoint_error': rtbase_stats_mm(errors.reshape(-1)), 'per_frame_mean_keypoint_error': rtbase_stats_mm(np.mean(errors, axis=(1, 2))), 'per_finger_keypoint_error': {spec.name: rtbase_stats_mm(errors[:, finger_index, :].reshape(-1)) for (finger_index, spec) in enumerate(rtbase_FINGER_SPECS)}, 'joint_limits': {'minimum_margin_rad': float(np.min(margins)), 'frames_at_limit': int(np.sum(np.any(margins < 1e-05, axis=1)))}, 'active_natural_pose_weight': rtbase_NUMERICAL_ACTIVE_WEIGHT, 'active_seed_weight': rtbase_NUMERICAL_ACTIVE_WEIGHT, 'ik_elapsed_seconds': float(solution.elapsed_seconds), 'alternating_iterations': len(history), 'converged': bool(converged), 'history': history}

def rtbase_save_npz(path: Path, name: str, transform: np.ndarray, solution: rtbase_SequenceSolution, cube_from_obj: np.ndarray, timestamps: np.ndarray, source_indices: np.ndarray, keypoints_obj: np.ndarray, link_from_obj: np.ndarray, joint_names: list[str], active_indices: np.ndarray) -> None:
    np.savez_compressed(path, schema=np.asarray('consens.left_wuji_index_middle_cube_calibration.v1'), result_name=np.asarray(name), finger_names=np.asarray([spec.name for spec in rtbase_FINGER_SPECS]), qpos=solution.qpos, joint_names=np.asarray(joint_names), active_joint_indices=active_indices, active_joint_names=np.asarray(rtbase_ACTIVE_JOINT_NAMES), timestamps=timestamps, source_record_indices=source_indices, T_left_palm_link_hand_back_cube=transform, T_hand_back_cube_obj=cube_from_obj, target_T_left_palm_link_obj=solution.target_obj_poses, predicted_T_left_palm_link_obj=solution.predicted_obj_poses, target_keypoints=solution.target_keypoints, predicted_keypoints=solution.predicted_keypoints, mean_keypoint_error_m=solution.mean_error_m, max_keypoint_error_m=solution.max_error_m, keypoints_obj_m=keypoints_obj, T_robot_link_obj=link_from_obj)

def rtbase_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optimize index+middle contact surfaces and one global cube SE(3).')
    parser.add_argument('pkl_path', nargs='?', type=Path, default=rtbase_DEFAULT_PKL_PATH)
    parser.add_argument('--urdf', type=Path, default=rtbase_DEFAULT_URDF_PATH)
    parser.add_argument('--fingertip-geometry-urdf', type=Path, default=rtbase_DEFAULT_FINGERTIP_GEOMETRY_URDF_PATH, help='URDF whose visual origins define the v2 fingertip OBJ frames.')
    parser.add_argument('--contact-keypoints', type=Path, default=rtbase_DEFAULT_CONTACT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=rtbase_DEFAULT_OUTPUT_DIR)
    parser.add_argument('--index-only-result', type=Path, default=rtbase_DEFAULT_INDEX_ONLY_RESULT)
    parser.add_argument('--initial-transform-npz', type=Path, help='Run one refinement path initialized from the T_left_palm_link_hand_back_cube stored in this NPZ.')
    parser.add_argument('--start-name', default='refined_best', help='Result-name suffix used with --initial-transform-npz.')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--solver-iterations', type=int, default=rtbase_DEFAULT_SOLVER_ITERATIONS)
    parser.add_argument('--alternating-iterations', type=int, default=rtbase_DEFAULT_ALTERNATING_ITERATIONS)
    return parser.parse_args()

def rtbase_main() -> None:
    args = rtbase_parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    geometry_urdf_path = args.fingertip_geometry_urdf.expanduser().resolve()
    contact_path = args.contact_keypoints.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    index_only_path = args.index_only_result.expanduser().resolve()
    for path in (pkl_path, urdf_path, geometry_urdf_path, contact_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError('--max-frames must be positive')
    if args.solver_iterations <= 0 or args.alternating_iterations <= 0:
        raise ValueError('Iteration counts must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints_obj = rtbase_load_contact_keypoints(contact_path)
    link_from_obj = rtbase_load_link_from_obj_transforms(geometry_urdf_path)
    cache_path = output_dir / 'source_index_middle_poses.npz'
    if cache_path.is_file() and args.max_frames is None:
        with np.load(cache_path, allow_pickle=False) as archive:
            cube_from_obj = np.asarray(archive['T_hand_back_cube_obj'], dtype=np.float64)
            timestamps = np.asarray(archive['timestamps'], dtype=np.float64)
            source_indices = np.asarray(archive['source_record_indices'], dtype=np.int32)
        print(f'[INFO] Loaded source cache: {len(cube_from_obj)} frames')
    else:
        (cube_from_obj, timestamps, source_indices) = rtbase_load_source_poses(pkl_path, args.max_frames)
        if args.max_frames is None:
            np.savez_compressed(cache_path, T_hand_back_cube_obj=cube_from_obj, timestamps=timestamps, source_record_indices=source_indices)
    urdf = rtbase_load_urdf(urdf_path)
    joint_names = list(urdf.actuated_joint_names)
    lower = np.asarray([float(urdf.joint_map[name].limit.lower) for name in joint_names], dtype=np.float32)
    upper = np.asarray([float(urdf.joint_map[name].limit.upper) for name in joint_names], dtype=np.float32)
    natural = np.clip(np.zeros(len(joint_names), dtype=np.float32), lower, upper)
    active_indices = np.asarray([joint_names.index(name) for name in rtbase_ACTIVE_JOINT_NAMES], dtype=np.int32)
    active_mask = np.zeros(len(joint_names), dtype=np.float32)
    active_mask[active_indices] = 1.0
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=jnp.asarray(natural))
    solver = rtbase_IndexMiddleIKSolver(urdf=urdf, robot=robot, link_from_obj=link_from_obj, keypoints_obj=keypoints_obj, natural_qpos=natural, active_mask=active_mask, max_iterations=int(args.solver_iterations))
    if args.initial_transform_npz is not None:
        initial_transform_path = args.initial_transform_npz.expanduser().resolve()
        if not initial_transform_path.is_file():
            raise FileNotFoundError(initial_transform_path)
        with np.load(initial_transform_path, allow_pickle=False) as archive:
            resumed_transform = rtbase_validate_transform(np.asarray(archive['T_left_palm_link_hand_back_cube'], dtype=np.float64), 'resumed transform')
        starts: list[tuple[str, np.ndarray]] = [(str(args.start_name), resumed_transform)]
        print(f'[INFO] Refining transform from: {initial_transform_path}')
    else:
        starts = [('theoretical', rtbase_T_PALM_CUBE_INITIAL)]
        if index_only_path.is_file():
            with np.load(index_only_path, allow_pickle=False) as archive:
                index_only_transform = np.asarray(archive['T_left_palm_link_hand_back_cube'], dtype=np.float64)
            starts.append(('index_only', index_only_transform))
    results: dict[str, dict[str, Any]] = {}
    result_paths: dict[str, str] = {}
    for (start_name, initial_transform) in starts:
        print(f'[INFO] Starting path: {start_name}')
        (transform, solution, history, converged) = rtbase_run_alternating(solver, cube_from_obj, keypoints_obj, initial_transform, int(args.alternating_iterations))
        result_name = f'unconstrained_from_{start_name}'
        summary = rtbase_summarize(result_name, transform, solution, history, converged, active_indices, lower, upper)
        result_path = output_dir / f'{result_name}.npz'
        rtbase_save_npz(result_path, result_name, transform, solution, cube_from_obj, timestamps, source_indices, keypoints_obj, link_from_obj, joint_names, active_indices)
        results[result_name] = summary
        result_paths[result_name] = str(result_path)
    best_name = min(results, key=lambda name: results[name]['all_keypoint_error']['mean_mm'])
    report = {'schema': 'consens.left_wuji_index_middle_cube_experiments.v1', 'source_pkl': str(pkl_path), 'urdf': str(urdf_path), 'fingertip_geometry_urdf': str(geometry_urdf_path), 'contact_keypoints': str(contact_path), 'scope': 'index+middle; 8 points; 8 active joints; one constant unconstrained cube SE(3)', 'best_result': best_name, 'result_npz': result_paths, 'experiments': results}
    report_path = output_dir / 'experiment_summary.json'
    with report_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print('\n[RESULT] index+middle all-keypoint errors')
    for (name, summary) in results.items():
        print(f"  {name:32s} {summary['all_keypoint_error']['mean_mm']:.3f} mm")
    print(f'[RESULT] best: {best_name}')
    print('[RESULT] T_left_palm_link_hand_back_cube:\n' + np.array2string(np.asarray(results[best_name]['T_left_palm_link_hand_back_cube']), precision=9, suppress_small=True))
    print(f'[RESULT] summary: {report_path}')

base = SimpleNamespace(
    ACTIVE_JOINT_NAMES=rtbase_ACTIVE_JOINT_NAMES,
    FINGER_SPECS=rtbase_FINGER_SPECS,
    FingerSpec=rtbase_FingerSpec,
    INACTIVE_JOINT_WEIGHT=rtbase_INACTIVE_JOINT_WEIGHT,
    IndexMiddleIKSolver=rtbase_IndexMiddleIKSolver,
    LAMBDA_INITIAL=rtbase_LAMBDA_INITIAL,
    NUMERICAL_ACTIVE_WEIGHT=rtbase_NUMERICAL_ACTIVE_WEIGHT,
    POSITION_WEIGHT=rtbase_POSITION_WEIGHT,
    SequenceSolution=rtbase_SequenceSolution,
    T_PALM_CUBE_INITIAL=rtbase_T_PALM_CUBE_INITIAL,
    jax=jax,
    jaxlie=jaxlie,
    jaxls=jaxls,
    jnp=jnp,
    load_contact_keypoints=rtbase_load_contact_keypoints,
    load_link_from_obj_transforms=rtbase_load_link_from_obj_transforms,
    load_source_poses=rtbase_load_source_poses,
    load_urdf=rtbase_load_urdf,
    make_transform=rtbase_make_transform,
    pk=pk,
    stats_mm=rtbase_stats_mm,
    summarize=rtbase_summarize,
    transform_points=rtbase_transform_points,
    validate_transform=rtbase_validate_transform,
)

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Literal
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
rt_FILE_PATH = Path(__file__).resolve()
rt_REPO_ROOT = APRILCUBE_ROOT.parent.parent
rt_DEFAULT_PKL_PATH = rt_REPO_ROOT / 'thirdparty/aprilcube/recordings/021_hand_back_sync_raw_frames_20260712_233831.pkl'
rt_DEFAULT_URDF_PATH = rt_REPO_ROOT / 'thirdparty/wuji-description/hand/body-with-soft/urdf/left_simplified_w_fingereye.urdf'
rt_DEFAULT_FINGERTIP_GEOMETRY_URDF_PATH = rt_REPO_ROOT / 'thirdparty/xarm7_wuji_left_description/xarm7_wuji_left_w_fingereye_v2.urdf'
rt_DEFAULT_CONTACT_CONFIG = rt_REPO_ROOT / 'configs/retarget/left_wuji_fingertip_contact_keypoints_v2.yaml'
rt_DEFAULT_OUTPUT_DIR = rt_REPO_ROOT / 'outputs/retargeting/021_new_intrinsics_recovered/three_finger'
rt_EXPECTED_MIDDLE_INTRINSICS = '/home/ps/RobotCamCalib1/outputs/intrinsics_charuco_scale0p25_2592x1944_0712_225925.yaml'
rt_DEFAULT_MULTI_CAM_WRIST_EXTRINSICS = Path('/home/ps/RobotCamCalib1/outputs/extrinsics_wrist_Q_thumb_web_cam_middle_finger_cam_apriltag_grid_offline_2samples_0712_030212_0712_031300.yaml')
rt_MULTI_CAM_RAW_FORMAT = 'consens_multi_camera_sync_stream'
rt_MULTI_CAM_SIDECAR_FORMAT = 'consensv2_multi_cam_020_pose_sidecar_v1'
rt_MULTI_CAM_TARGET_MAPPING = {'thumb': {'sidecar_target': 'thumb_Q', 'camera': 'thumb_web_cam', 'cube_dir': 'cube_april_36h11_12_17_1x1x1_15mm'}, 'index': {'sidecar_target': 'index_Q', 'camera': 'thumb_web_cam', 'cube_dir': 'cube_april_36h11_6_11_1x1x1_15mm'}, 'middle': {'sidecar_target': 'middle_Q', 'camera': 'middle_finger_cam', 'cube_dir': 'cube_april_36h11_0_5_1x1x1_15mm'}}
rt_SCHEMA = 'consens.left_wuji_three_finger_cube_calibration.v1'
rt_COMPACT_PKL_FORMAT = 'consens.left_wuji_three_finger_retarget.compact.v2'
rt_OPTIMIZED_CUBE_OFFSET_KEY = 'T_left_palm_link_hand_back_cube_6d_optimized'
rt_DEFAULT_SOLVER_ITERATIONS = 100
rt_DEFAULT_ALTERNATING_ITERATIONS = 30
rt_DEFAULT_TEMPORAL_BRANCH_WEIGHT = 5e-05
rt_DEFAULT_TEMPORAL_SEED_WEIGHT = 0.25
rt_DEFAULT_JOINT_VELOCITY_WEIGHT = 9.3
rt_DEFAULT_JOINT_ACCELERATION_WEIGHT = 0.62
rt_DEFAULT_JOINT_JERK_WEIGHT = 0.026
rt_DEFAULT_JOINT_OPTIMIZATION_ITERATIONS = 800
rt_DEFAULT_JOINT_MAX_STEP_DEG = 2.1
rt_DEFAULT_JOINT_MAX_STEP_WEIGHT = 165.0
rt_Mode = Literal['fixed', 'translation', 'rotation', 'se3']
rt_FINGER_SPECS = (base.FingerSpec('thumb', 'left_finger1_link4', 'left_finger1_', 'thumb.obj'), base.FingerSpec('index', 'left_finger2_link4', 'left_finger2_', 'index_v2.obj'), base.FingerSpec('middle', 'left_finger3_link4', 'left_finger3_', 'middle_v2.obj'))
rt_ACTIVE_JOINT_NAMES = tuple((f'left_finger{finger_index}_joint{joint_index}' for finger_index in (1, 2, 3) for joint_index in range(1, 5)))
base.FINGER_SPECS = rt_FINGER_SPECS
base.ACTIVE_JOINT_NAMES = rt_ACTIVE_JOINT_NAMES
base.NUMERICAL_ACTIVE_WEIGHT = 0.0
# In the original two-module implementation the assignments above mutated the
# imported base module globals.  The embedded namespace is a facade, so keep
# the namespace-prefixed base globals in lockstep explicitly.
rtbase_FINGER_SPECS = rt_FINGER_SPECS
rtbase_ACTIVE_JOINT_NAMES = rt_ACTIVE_JOINT_NAMES
rtbase_NUMERICAL_ACTIVE_WEIGHT = 0.0

class rt_BatchedThreeFingerIKSolver(base.IndexMiddleIKSolver):
    """Solve all independent frame IK problems in parallel on the accelerator."""

    def _build_solver(self) -> None:
        num_joints = len(self.natural_qpos)
        num_fingers = len(rt_FINGER_SPECS)

        class TargetVar(base.jaxls.Var[base.jax.Array], default_factory=lambda : base.jnp.zeros((num_fingers, 4, 3), dtype=base.jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...

        class PointWeightVar(base.jaxls.Var[base.jax.Array], default_factory=lambda : base.jnp.ones((num_fingers, 4), dtype=base.jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...

        class PreviousVar(base.jaxls.Var[base.jax.Array], default_factory=lambda : base.jnp.zeros((num_joints,), dtype=base.jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...

        class SeedWeightVar(base.jaxls.Var[base.jax.Array], default_factory=lambda : base.jnp.zeros((num_joints,), dtype=base.jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...
        joint_var = self.robot.joint_var_cls(0)
        target_var = TargetVar(0)
        point_weight_var = PointWeightVar(0)
        previous_var = PreviousVar(0)
        seed_weight_var = SeedWeightVar(0)
        robot = self.robot
        link_indices = base.jnp.asarray(self.link_indices, dtype=base.jnp.int32)
        points_link = base.jnp.asarray(self.keypoints_link, dtype=base.jnp.float32)
        natural = base.jnp.asarray(self.natural_qpos, dtype=base.jnp.float32)
        inactive_weights = base.jnp.asarray((1.0 - self.active_mask) * base.INACTIVE_JOINT_WEIGHT, dtype=base.jnp.float32)

        @base.jaxls.Cost.factory
        def alignment_cost(values: base.jaxls.VarValues, var_q: base.jaxls.Var[base.jnp.ndarray], var_target: base.jaxls.Var[base.jnp.ndarray], var_weight: base.jaxls.Var[base.jnp.ndarray]) -> base.jax.Array:
            root_from_links = base.jaxlie.SE3(robot.forward_kinematics(cfg=values[var_q])[link_indices])
            predicted = base.jnp.einsum('fij,fkj->fki', root_from_links.rotation().as_matrix(), points_link) + root_from_links.translation()[:, None, :]
            point_scale = base.jnp.sqrt(base.jnp.maximum(values[var_weight], 0.0))[..., None]
            return ((predicted - values[var_target]) * point_scale * base.POSITION_WEIGHT).reshape(-1)

        @base.jaxls.Cost.factory
        def inactive_seed_cost(values: base.jaxls.VarValues, var_q: base.jaxls.Var[base.jnp.ndarray], var_previous: base.jaxls.Var[base.jnp.ndarray], var_seed_weight: base.jaxls.Var[base.jnp.ndarray]) -> base.jax.Array:
            return ((values[var_q] - values[var_previous]) * values[var_seed_weight]).reshape(-1)
        self.joint_var = joint_var
        self.target_var = target_var
        self.point_weight_var = point_weight_var
        self.previous_var = previous_var
        self.seed_weight_var = seed_weight_var
        self.problem = base.jaxls.LeastSquaresProblem(costs=[alignment_cost(joint_var, target_var, point_weight_var), base.pk.costs.rest_cost(joint_var, natural, inactive_weights), inactive_seed_cost(joint_var, previous_var, seed_weight_var), base.pk.costs.limit_constraint(robot, joint_var)], variables=[joint_var, target_var, point_weight_var, previous_var, seed_weight_var]).analyze(use_onp=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.point_weights = np.asarray(kwargs.pop('point_weights'), dtype=np.float32)
        self.ignored_joint_indices = np.asarray(kwargs.pop('ignored_joint_indices'), dtype=np.int32)
        self.temporal_branch_weight = float(kwargs.pop('temporal_branch_weight', rt_DEFAULT_TEMPORAL_BRANCH_WEIGHT))
        if self.temporal_branch_weight < 0.0:
            raise ValueError('temporal_branch_weight must be nonnegative')
        super().__init__(*args, **kwargs)
        solver = self

        def solve_one(target: Any, point_weight: Any, initial: Any, seed_weight: Any) -> Any:
            values = base.jaxls.VarValues.make([solver.joint_var.with_value(initial), solver.target_var.with_value(target), solver.point_weight_var.with_value(point_weight), solver.previous_var.with_value(initial), solver.seed_weight_var.with_value(seed_weight)])
            solution = solver.problem.solve(initial_vals=values, verbose=False, linear_solver='conjugate_gradient', trust_region=base.jaxls.TrustRegionConfig(lambda_initial=base.LAMBDA_INITIAL), termination=base.jaxls.TerminationConfig(max_iterations=solver.max_iterations))
            return solution[solver.joint_var]
        self._solve_batch_jax = base.jax.jit(base.jax.vmap(solve_one))
        link_indices = base.jnp.asarray(self.link_indices, dtype=base.jnp.int32)
        robot = self.robot

        def fk_one(qpos: Any) -> Any:
            return base.jaxlie.SE3(robot.forward_kinematics(cfg=qpos)[link_indices]).as_matrix()
        self._fk_batch_jax = base.jax.jit(base.jax.vmap(fk_one))

    def solve_q_batch(self, targets: np.ndarray, point_weights: np.ndarray, initial_qpos: np.ndarray, active_seed_weight: float=0.0) -> np.ndarray:
        seed_weight = (1.0 - self.active_mask) * base.INACTIVE_JOINT_WEIGHT + self.active_mask * float(active_seed_weight)
        seed_weights = np.tile(seed_weight[None, :], (len(np.asarray(targets)), 1)).astype(np.float32)
        qpos = self._solve_batch_jax(base.jnp.asarray(targets, dtype=base.jnp.float32), base.jnp.asarray(point_weights, dtype=base.jnp.float32), base.jnp.asarray(initial_qpos, dtype=base.jnp.float32), base.jnp.asarray(seed_weights, dtype=base.jnp.float32))
        base.jax.block_until_ready(qpos)
        output = np.asarray(qpos, dtype=np.float32)
        output = np.clip(output, self.lower[None, :], self.upper[None, :])
        output[:, self.active_mask < 0.5] = self.natural_qpos[self.active_mask < 0.5]
        return output

    def refine_sequence_temporally(self, palm_from_cube: np.ndarray, cube_from_obj: np.ndarray, initial_qpos: np.ndarray, active_seed_weight: float) -> base.SequenceSolution:
        if active_seed_weight <= 0.0:
            raise ValueError('active_seed_weight must be positive')
        target_obj_poses = np.einsum('ij,tfjk->tfik', np.asarray(palm_from_cube, dtype=np.float64), np.asarray(cube_from_obj, dtype=np.float64))
        target_keypoints = np.einsum('tfij,fkj->tfki', target_obj_poses[:, :, :3, :3], self.keypoints_obj) + target_obj_poses[:, :, None, :3, 3]
        qpos = np.asarray(initial_qpos, dtype=np.float32).copy()
        for frame_index in range(1, len(qpos)):
            qpos[frame_index] = self.solve_q_batch(target_keypoints[frame_index:frame_index + 1], self.point_weights[frame_index:frame_index + 1], qpos[frame_index - 1:frame_index], active_seed_weight=active_seed_weight)[0]
        result = self.evaluate_qpos(qpos, target_obj_poses, target_keypoints)
        active_delta = np.diff(qpos[:, self.active_mask > 0.5], axis=0)
        maximum_active_step_deg = float(np.rad2deg(np.max(np.abs(active_delta))))
        print(f'[INFO] Sequential temporal IK frames={len(qpos)} mean={np.mean(result.mean_error_m) * 1000.0:.3f} mm max_active_step={maximum_active_step_deg:.3f} deg seed_weight={active_seed_weight:.6f}')
        return result

    def optimize_sequence_jointly(self, palm_from_cube: np.ndarray, cube_from_obj: np.ndarray, timestamps: np.ndarray, initial_qpos: np.ndarray, velocity_weight: float, acceleration_weight: float, jerk_weight: float, max_step_deg: float, max_step_weight: float, max_iterations: int) -> tuple[base.SequenceSolution, dict[str, Any]]:
        """Jointly optimize all frames for contact fit and C3 time smoothness."""
        if min(velocity_weight, acceleration_weight, jerk_weight, max_step_deg, max_step_weight) < 0.0:
            raise ValueError('Joint temporal weights must be nonnegative')
        if max_iterations <= 0:
            raise ValueError('Joint temporal max_iterations must be positive')
        target_obj_poses = np.einsum('ij,tfjk->tfik', np.asarray(palm_from_cube, dtype=np.float64), np.asarray(cube_from_obj, dtype=np.float64))
        target_keypoints = np.einsum('tfij,fkj->tfki', target_obj_poses[:, :, :3, :3], self.keypoints_obj) + target_obj_poses[:, :, None, :3, 3]
        timestamps = np.asarray(timestamps, dtype=np.float64)
        dt = np.diff(timestamps)
        if len(dt) < 3 or np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
            raise ValueError('Joint temporal optimization needs >=4 monotonic timestamps')
        initial_qpos = np.asarray(initial_qpos, dtype=np.float32)
        active_indices = np.flatnonzero(self.active_mask > 0.5).astype(np.int32)
        inactive_qpos = base.jnp.asarray(initial_qpos, dtype=base.jnp.float32)
        target_jax = base.jnp.asarray(target_keypoints, dtype=base.jnp.float32)
        weights_jax = base.jnp.asarray(self.point_weights, dtype=base.jnp.float32)
        link_from_obj_jax = base.jnp.asarray(self.link_from_obj, dtype=base.jnp.float32)
        points_obj_jax = base.jnp.asarray(self.keypoints_obj, dtype=base.jnp.float32)
        dt_jax = base.jnp.asarray(dt, dtype=base.jnp.float32)
        active_indices_jax = base.jnp.asarray(active_indices, dtype=base.jnp.int32)
        data_weight_sum = base.jnp.maximum(base.jnp.sum(weights_jax), 1.0)

        def objective(active_flat: Any) -> Any:
            active_qpos = active_flat.reshape((len(initial_qpos), len(active_indices)))
            qpos = inactive_qpos.at[:, active_indices_jax].set(active_qpos)
            root_from_links = self._fk_batch_jax(qpos)
            root_from_obj = base.jnp.einsum('tfij,fjk->tfik', root_from_links, link_from_obj_jax)
            predicted_keypoints = base.jnp.einsum('tfij,fkj->tfki', root_from_obj[:, :, :3, :3], points_obj_jax) + root_from_obj[:, :, None, :3, 3]
            residual = predicted_keypoints - target_jax
            data_mm2 = base.jnp.sum(residual * residual * weights_jax[..., None]) / data_weight_sum * 1000000.0
            velocity = base.jnp.diff(active_qpos, axis=0) / dt_jax[:, None]
            step_deg = base.jnp.abs(base.jnp.diff(active_qpos, axis=0)) * (180.0 / np.pi)
            step_excess_deg = base.jnp.maximum(step_deg - float(max_step_deg), 0.0)
            acceleration_dt = 0.5 * (dt_jax[1:] + dt_jax[:-1])
            acceleration = base.jnp.diff(velocity, axis=0) / acceleration_dt[:, None]
            jerk_dt = (dt_jax[2:] + dt_jax[1:-1] + dt_jax[:-2]) / 3.0
            jerk = base.jnp.diff(acceleration, axis=0) / jerk_dt[:, None]
            return data_mm2 + float(velocity_weight) * base.jnp.mean(velocity * velocity) + float(acceleration_weight) * base.jnp.mean(acceleration * acceleration) + float(jerk_weight) * base.jnp.mean(jerk * jerk) + float(max_step_weight) * base.jnp.mean(base.jnp.sum(step_excess_deg ** 2, axis=1))
        value_and_grad = base.jax.jit(base.jax.value_and_grad(objective))

        def scipy_value_and_grad(flat: np.ndarray) -> tuple[float, np.ndarray]:
            (value, gradient) = value_and_grad(base.jnp.asarray(flat, dtype=base.jnp.float32))
            base.jax.block_until_ready(value)
            return (float(value), np.asarray(gradient, dtype=np.float64))
        x0 = initial_qpos[:, active_indices].astype(np.float64).reshape(-1)
        active_lower = self.lower[active_indices].astype(np.float64)
        active_upper = self.upper[active_indices].astype(np.float64)
        bounds = list(zip(np.tile(active_lower, len(initial_qpos)), np.tile(active_upper, len(initial_qpos))))
        (initial_loss, _) = scipy_value_and_grad(x0)
        optimized = minimize(scipy_value_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': int(max_iterations), 'ftol': 1e-10, 'gtol': 1e-07, 'maxls': 40})
        qpos = initial_qpos.copy()
        qpos[:, active_indices] = np.asarray(optimized.x.reshape((len(initial_qpos), len(active_indices))), dtype=np.float32)
        result = self.evaluate_qpos(qpos, target_obj_poses, target_keypoints)
        (final_loss, _) = scipy_value_and_grad(optimized.x)
        metadata = {'method': 'confidence_weighted_four_point_plus_velocity_acceleration_jerk', 'optimizer': 'scipy_L-BFGS-B_with_JAX_gradient', 'success': bool(optimized.success), 'status': int(optimized.status), 'message': str(optimized.message), 'iterations': int(optimized.nit), 'function_evaluations': int(optimized.nfev), 'initial_total_loss': float(initial_loss), 'final_total_loss': float(final_loss), 'velocity_weight': float(velocity_weight), 'acceleration_weight': float(acceleration_weight), 'jerk_weight': float(jerk_weight), 'maximum_step_target_deg': float(max_step_deg), 'maximum_step_weight': float(max_step_weight), 'max_iterations': int(max_iterations)}
        print(f'[INFO] Joint temporal optimization success={optimized.success} iterations={optimized.nit} loss={initial_loss:.6f}->{final_loss:.6f} mean={np.mean(result.mean_error_m) * 1000.0:.3f} mm')
        return (result, metadata)

    def evaluate_qpos(self, qpos: np.ndarray, target_obj_poses: np.ndarray, target_keypoints: np.ndarray) -> base.SequenceSolution:
        started = time.monotonic()
        link_poses = self._fk_batch_jax(base.jnp.asarray(qpos, dtype=base.jnp.float32))
        base.jax.block_until_ready(link_poses)
        link_poses_np = np.asarray(link_poses, dtype=np.float64)
        predicted_obj_poses = np.einsum('tfij,fjk->tfik', link_poses_np, self.link_from_obj)
        predicted_keypoints = np.einsum('tfij,fkj->tfki', predicted_obj_poses[:, :, :3, :3], self.keypoints_obj) + predicted_obj_poses[:, :, None, :3, 3]
        errors = np.linalg.norm(predicted_keypoints - target_keypoints, axis=-1)
        weights = np.asarray(self.point_weights, dtype=np.float64)
        weight_sum = np.sum(weights, axis=(1, 2))
        if np.any(weight_sum <= 0.0):
            raise ValueError('Every frame must retain at least one positive-weight point')
        mean_error_m = np.sum(errors * weights, axis=(1, 2)) / weight_sum
        max_error_m = np.max(np.where(weights > 0.0, errors, -np.inf), axis=(1, 2))
        return base.SequenceSolution(qpos=np.asarray(qpos, dtype=np.float32), target_obj_poses=np.asarray(target_obj_poses, dtype=np.float64), predicted_obj_poses=predicted_obj_poses, target_keypoints=np.asarray(target_keypoints, dtype=np.float64), predicted_keypoints=predicted_keypoints, mean_error_m=mean_error_m, max_error_m=max_error_m, elapsed_seconds=time.monotonic() - started)

    def solve_sequence(self, palm_from_cube: np.ndarray, cube_from_obj: np.ndarray, initial_qpos: np.ndarray | None=None) -> base.SequenceSolution:
        started = time.monotonic()
        target_obj_poses = np.einsum('ij,tfjk->tfik', np.asarray(palm_from_cube, dtype=np.float64), np.asarray(cube_from_obj, dtype=np.float64))
        target_keypoints = np.einsum('tfij,fkj->tfki', target_obj_poses[:, :, :3, :3], self.keypoints_obj) + target_obj_poses[:, :, None, :3, 3]
        frame_count = len(target_obj_poses)
        if self.point_weights.shape != (frame_count, len(rt_FINGER_SPECS), 4):
            raise ValueError(f'Point weights must have shape ({frame_count}, {len(rt_FINGER_SPECS)}, 4); got {self.point_weights.shape}')
        seeds = np.tile(self.natural_qpos[None, :], (frame_count, 1)) if initial_qpos is None else np.asarray(initial_qpos, dtype=np.float32)
        primary = self.solve_q_batch(target_keypoints, self.point_weights, seeds)
        forward_seed = primary.copy()
        backward_seed = primary.copy()
        if frame_count > 1:
            forward_seed[1:] = primary[:-1]
            backward_seed[:-1] = primary[1:]
        forward = self.solve_q_batch(target_keypoints, self.point_weights, forward_seed)
        backward = self.solve_q_batch(target_keypoints, self.point_weights, backward_seed)
        candidates = np.stack([primary, forward, backward], axis=0)
        candidate_errors = []
        for candidate in candidates:
            evaluated = self.evaluate_qpos(candidate, target_obj_poses, target_keypoints)
            residual = evaluated.predicted_keypoints - target_keypoints
            candidate_errors.append(np.sum(residual * residual * self.point_weights[..., None], axis=(1, 2, 3)))
        emissions = np.stack(candidate_errors, axis=0)
        if self.temporal_branch_weight > 0.0 and frame_count > 1:
            state_count = len(candidates)
            active_mask = np.asarray(self.active_mask, dtype=np.float64)
            accumulated = np.empty_like(emissions, dtype=np.float64)
            backpointers = np.zeros((frame_count, state_count), dtype=np.int32)
            accumulated[:, 0] = emissions[:, 0]
            for frame_index in range(1, frame_count):
                difference = candidates[:, frame_index - 1, None, :] - candidates[None, :, frame_index, :]
                transition = self.temporal_branch_weight * np.sum(difference * difference * active_mask[None, None, :], axis=-1)
                total = accumulated[:, frame_index - 1, None] + transition
                backpointers[frame_index] = np.argmin(total, axis=0)
                accumulated[:, frame_index] = emissions[:, frame_index] + np.min(total, axis=0)
            choice = np.zeros(frame_count, dtype=np.int32)
            choice[-1] = int(np.argmin(accumulated[:, -1]))
            for frame_index in range(frame_count - 1, 0, -1):
                choice[frame_index - 1] = backpointers[frame_index, choice[frame_index]]
        else:
            choice = np.argmin(emissions, axis=0)
        selected = candidates[choice, np.arange(frame_count)]
        ignored_frames = np.flatnonzero(np.all(self.point_weights[:, 0, :] <= 0.0, axis=1))
        for frame_index in ignored_frames:
            if 0 < frame_index < frame_count - 1:
                selected[frame_index, self.ignored_joint_indices] = 0.5 * (selected[frame_index - 1, self.ignored_joint_indices] + selected[frame_index + 1, self.ignored_joint_indices])
            elif frame_index > 0:
                selected[frame_index, self.ignored_joint_indices] = selected[frame_index - 1, self.ignored_joint_indices]
            elif frame_count > 1:
                selected[frame_index, self.ignored_joint_indices] = selected[frame_index + 1, self.ignored_joint_indices]
        result = self.evaluate_qpos(selected, target_obj_poses, target_keypoints)
        result.elapsed_seconds = time.monotonic() - started
        active_delta = np.diff(selected[:, self.active_mask > 0.5], axis=0)
        maximum_active_step_deg = float(np.rad2deg(np.max(np.abs(active_delta)))) if active_delta.size else 0.0
        print(f'[INFO] Batched IK frames={frame_count} mean={np.mean(result.mean_error_m) * 1000.0:.3f} mm max_active_step={maximum_active_step_deg:.3f} deg elapsed={result.elapsed_seconds:.3f}s')
        return result

def rt_flatten_cube_points(cube_from_obj: np.ndarray, keypoints_obj: np.ndarray) -> np.ndarray:
    return np.asarray([[base.transform_points(transform, points) for (transform, points) in zip(frame_transforms, keypoints_obj)] for frame_transforms in cube_from_obj], dtype=np.float64).reshape(-1, 3)

def rt_kabsch_rotation(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(source, dtype=np.float64).T @ np.asarray(target, dtype=np.float64)
    (u, singular_values, vt) = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    return (rotation, singular_values)

def rt_fit_transform_for_mode(mode: rt_Mode, cube_from_obj: np.ndarray, keypoints_obj: np.ndarray, predicted_keypoints: np.ndarray, point_weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = rt_flatten_cube_points(cube_from_obj, keypoints_obj)
    target = np.asarray(predicted_keypoints, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(point_weights, dtype=np.float64).reshape(-1)
    valid = weights > 0.0
    source = source[valid]
    target = target[valid]
    weights = weights[valid]
    if not len(source):
        raise ValueError('Global registration has no positive-weight points')
    sqrt_weights = np.sqrt(weights)[:, None]
    initial = base.T_PALM_CUBE_INITIAL
    singular_values = np.zeros(3, dtype=np.float64)
    if mode == 'translation':
        rotation = initial[:3, :3].copy()
        translation = np.average(target - (rotation @ source.T).T, axis=0, weights=weights)
    elif mode == 'rotation':
        translation = initial[:3, 3].copy()
        (rotation, singular_values) = rt_kabsch_rotation(source * sqrt_weights, (target - translation[None, :]) * sqrt_weights)
    elif mode == 'se3':
        source_centroid = np.average(source, axis=0, weights=weights)
        target_centroid = np.average(target, axis=0, weights=weights)
        (rotation, singular_values) = rt_kabsch_rotation((source - source_centroid) * sqrt_weights, (target - target_centroid) * sqrt_weights)
        translation = target_centroid - rotation @ source_centroid
    else:
        raise ValueError(f'Mode {mode!r} does not optimize a transform')
    transform = base.make_transform(rotation, translation)
    fitted = (rotation @ source.T).T + translation
    errors = np.linalg.norm(fitted - target, axis=-1)
    return (transform, {'mode': mode, 'point_count': int(len(source)), 'weight_sum': float(np.sum(weights)), 'singular_values': singular_values.tolist(), 'fixed_q_mean_error_mm': float(np.mean(errors) * 1000.0), 'fixed_q_rmse_mm': float(np.sqrt(np.mean(errors ** 2)) * 1000.0)})

def rt_transform_change(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    rotation_rad = float(np.linalg.norm(Rotation.from_matrix(current[:3, :3] @ previous[:3, :3].T).as_rotvec()))
    return (translation_m, rotation_rad)

def rt_objective_rmse_m(solution: base.SequenceSolution, point_weights: np.ndarray) -> float:
    residual = np.asarray(solution.predicted_keypoints - solution.target_keypoints, dtype=np.float64)
    squared_distance = np.sum(residual * residual, axis=-1)
    weights = np.asarray(point_weights, dtype=np.float64)
    return float(np.sqrt(np.sum(squared_distance * weights) / np.sum(weights)))

def rt_trajectory_smoothness_stats(qpos: np.ndarray, timestamps: np.ndarray, active_indices: np.ndarray) -> dict[str, Any]:
    active = np.asarray(qpos, dtype=np.float64)[:, active_indices]
    dt = np.diff(np.asarray(timestamps, dtype=np.float64))
    velocity = np.diff(active, axis=0) / dt[:, None]
    acceleration_dt = 0.5 * (dt[1:] + dt[:-1])
    acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
    jerk_dt = (dt[2:] + dt[1:-1] + dt[:-2]) / 3.0
    jerk = np.diff(acceleration, axis=0) / jerk_dt[:, None]
    step_deg = np.rad2deg(np.diff(active, axis=0))

    def stats(values: np.ndarray) -> dict[str, float]:
        absolute = np.abs(np.asarray(values, dtype=np.float64))
        return {'rms': float(np.sqrt(np.mean(absolute * absolute))), 'p95_abs': float(np.percentile(absolute, 95.0)), 'max_abs': float(np.max(absolute))}
    flat_index = int(np.argmax(np.abs(step_deg)))
    (transition, joint) = np.unravel_index(flat_index, step_deg.shape)
    per_finger: dict[str, Any] = {}
    for (finger_index, finger_name) in enumerate(('thumb', 'index', 'middle')):
        selection = slice(4 * finger_index, 4 * (finger_index + 1))
        finger_steps = step_deg[:, selection]
        local_flat = int(np.argmax(np.abs(finger_steps)))
        (local_transition, local_joint) = np.unravel_index(local_flat, finger_steps.shape)
        per_finger[finger_name] = {**stats(finger_steps), 'maximum_transition': [int(local_transition), int(local_transition + 1)], 'maximum_joint_name': rt_ACTIVE_JOINT_NAMES[4 * finger_index + int(local_joint)]}
    return {'step_deg': {**stats(step_deg), 'maximum_transition': [int(transition), int(transition + 1)], 'maximum_joint_name': rt_ACTIVE_JOINT_NAMES[int(joint)]}, 'velocity_rad_s': stats(velocity), 'acceleration_rad_s2': stats(acceleration), 'jerk_rad_s3': stats(jerk), 'per_finger_step_deg': per_finger}

def rt_run_path(mode: rt_Mode, start_name: str, initial_transform: np.ndarray, solver: rt_BatchedThreeFingerIKSolver, cube_from_obj: np.ndarray, keypoints_obj: np.ndarray, point_weights: np.ndarray, alternating_iterations: int) -> tuple[np.ndarray, base.SequenceSolution, dict[str, Any]]:
    transform = base.validate_transform(initial_transform, f'{mode}/{start_name} start').copy()
    if mode == 'translation':
        transform[:3, :3] = base.T_PALM_CUBE_INITIAL[:3, :3]
    elif mode == 'rotation':
        transform[:3, 3] = base.T_PALM_CUBE_INITIAL[:3, 3]
    history: list[dict[str, Any]] = []
    best_transform = transform.copy()
    best_solution: base.SequenceSolution | None = None
    best_rmse = float('inf')
    converged = mode == 'fixed'
    max_iterations = 1 if mode == 'fixed' else int(alternating_iterations)
    q_seed: np.ndarray | None = None
    for iteration in range(max_iterations):
        print(f'[INFO] {mode}/{start_name}: IK-registration iteration {iteration + 1}/{max_iterations}')
        solution = solver.solve_sequence(transform, cube_from_obj, q_seed)
        q_seed = solution.qpos
        rmse = rt_objective_rmse_m(solution, point_weights)
        if rmse < best_rmse:
            best_rmse = rmse
            best_transform = transform.copy()
            best_solution = solution
        if mode == 'fixed':
            break
        (next_transform, registration) = rt_fit_transform_for_mode(mode, cube_from_obj, keypoints_obj, solution.predicted_keypoints, point_weights)
        (translation_change_m, rotation_change_rad) = rt_transform_change(transform, next_transform)
        history.append({'iteration': iteration + 1, 'objective_rmse_mm': rmse * 1000.0, 'T_before': transform.tolist(), 'T_after': next_transform.tolist(), 'translation_change_m': translation_change_m, 'rotation_change_rad': rotation_change_rad, 'registration': registration})
        transform = next_transform
        if translation_change_m < 1e-07 and rotation_change_rad < 1e-06:
            converged = True
            break
    if mode != 'fixed':
        print(f'[INFO] {mode}/{start_name}: final IK')
        final_solution = solver.solve_sequence(transform, cube_from_obj, q_seed)
        final_rmse = rt_objective_rmse_m(final_solution, point_weights)
        if final_rmse < best_rmse or best_solution is None:
            best_rmse = final_rmse
            best_transform = transform.copy()
            best_solution = final_solution
    assert best_solution is not None
    path_summary = {'start_name': start_name, 'mode': mode, 'objective_rmse_mm': best_rmse * 1000.0, 'converged': bool(converged), 'iterations': len(history), 'T_left_palm_link_hand_back_cube': best_transform.tolist(), 'history': history}
    return (best_transform, best_solution, path_summary)

def rt_deduplicate_starts(starts: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    output: list[tuple[str, np.ndarray]] = []
    for (name, transform) in starts:
        if any((np.allclose(transform, existing, atol=1e-12) for (_, existing) in output)):
            continue
        output.append((name, transform))
    return output

def rt_summarize_best(mode: rt_Mode, best_start: str, transform: np.ndarray, solution: base.SequenceSolution, point_weights: np.ndarray, path_summaries: list[dict[str, Any]], ignored_observations: list[dict[str, Any]], active_indices: np.ndarray, lower: np.ndarray, upper: np.ndarray, temporal_branch_weight: float) -> dict[str, Any]:
    summary = base.summarize(mode, transform, solution, [], any((item['converged'] for item in path_summaries)), active_indices, lower, upper)
    residual = np.asarray(solution.predicted_keypoints - solution.target_keypoints, dtype=np.float64)
    errors = np.linalg.norm(residual, axis=-1)
    weights = np.asarray(point_weights, dtype=np.float64)
    valid = weights > 0.0
    per_frame_mean = np.sum(errors * weights, axis=(1, 2)) / np.sum(weights, axis=(1, 2))
    summary['all_keypoint_error'] = base.stats_mm(errors[valid])
    summary['per_frame_mean_keypoint_error'] = base.stats_mm(per_frame_mean)
    summary['per_finger_keypoint_error'] = {spec.name: base.stats_mm(errors[:, finger_index, :][weights[:, finger_index, :] > 0.0]) for (finger_index, spec) in enumerate(rt_FINGER_SPECS)}
    squared_distance = np.sum(residual * residual, axis=-1)
    summary['unweighted_objective_rmse_mm'] = float(np.sqrt(np.mean(squared_distance)) * 1000.0)
    reliable = weights >= 0.75
    summary['reliable_direct_observation_rmse_mm'] = float(np.sqrt(np.mean(squared_distance[reliable])) * 1000.0)
    summary['per_finger_confidence_weighted_rmse_mm'] = {spec.name: float(np.sqrt(np.sum(squared_distance[:, finger_index, :] * weights[:, finger_index, :]) / np.sum(weights[:, finger_index, :])) * 1000.0) for (finger_index, spec) in enumerate(rt_FINGER_SPECS)}
    summary['observation_confidence'] = {'minimum': float(np.min(weights)), 'mean': float(np.mean(weights)), 'reliable_point_fraction': float(np.mean(reliable))}
    summary.update({'mode': mode, 'best_start': best_start, 'objective': 'equal_weight_four_contact_point_squared_distance', 'objective_rmse_mm': float(np.sqrt(np.sum(np.sum(residual * residual, axis=-1) * weights) / np.sum(weights)) * 1000.0), 'active_natural_pose_weight': 0.0, 'active_temporal_smoothness_weight': float(temporal_branch_weight), 'ignored_observations': ignored_observations, 'path_summaries': path_summaries})
    return summary

def rt_save_npz(path: Path, mode: rt_Mode, best_start: str, transform: np.ndarray, solution: base.SequenceSolution, cube_from_obj: np.ndarray, timestamps: np.ndarray, source_indices: np.ndarray, keypoints_obj: np.ndarray, link_from_obj: np.ndarray, joint_names: list[str], active_indices: np.ndarray, point_weights: np.ndarray) -> None:
    np.savez_compressed(path, schema=np.asarray(rt_SCHEMA), result_name=np.asarray(f'three_finger_{mode}'), mode=np.asarray(mode), best_start=np.asarray(best_start), finger_names=np.asarray([spec.name for spec in rt_FINGER_SPECS]), qpos=solution.qpos, joint_names=np.asarray(joint_names), active_joint_indices=active_indices, active_joint_names=np.asarray(rt_ACTIVE_JOINT_NAMES), timestamps=timestamps, source_record_indices=source_indices, T_left_palm_link_hand_back_cube=transform, T_hand_back_cube_obj=cube_from_obj, target_T_left_palm_link_obj=solution.target_obj_poses, predicted_T_left_palm_link_obj=solution.predicted_obj_poses, target_keypoints=solution.target_keypoints, predicted_keypoints=solution.predicted_keypoints, mean_keypoint_error_m=solution.mean_error_m, max_keypoint_error_m=solution.max_error_m, keypoints_obj_m=keypoints_obj, T_robot_link_obj=link_from_obj, keypoint_weights=point_weights)

def rt_write_compact_pkl(path: Path, source_pkl: Path, mode: rt_Mode, best_start: str, summary: dict[str, Any], transform: np.ndarray, solution: base.SequenceSolution, timestamps: np.ndarray, source_indices: np.ndarray, joint_names: list[str], active_indices: np.ndarray, point_weights: np.ndarray, source_validation: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    header = {'type': 'header', 'format': rt_COMPACT_PKL_FORMAT, 'source_pkl': str(source_pkl), 'source_validation': source_validation, 'mode': mode, 'best_start': best_start, 'finger_names': [spec.name for spec in rt_FINGER_SPECS], 'joint_names': joint_names, 'active_joint_indices': active_indices, 'wujihand_qpos_joint_order': [joint_names[int(index)] for index in active_indices], 'wujihand_qpos_dimension': int(len(active_indices)), 'wujihand_qpos_unit': 'radian', rt_OPTIMIZED_CUBE_OFFSET_KEY: transform, 'cube_offset_transform_convention': 'p_left_palm_link = T_left_palm_link_hand_back_cube_6d_optimized @ p_hand_back_cube', 'summary': summary}
    try:
        with temporary.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            errors = np.linalg.norm(solution.predicted_keypoints - solution.target_keypoints, axis=-1)
            for frame_index in range(len(solution.qpos)):
                pickle.dump({'type': 'frame', 'frame_index': frame_index, 'source_record_index': int(source_indices[frame_index]), 'timestamp': float(timestamps[frame_index]), 'wujihand_qpos': solution.qpos[frame_index, active_indices], 'target_T_left_palm_link_obj': solution.target_obj_poses[frame_index], 'predicted_T_left_palm_link_obj': solution.predicted_obj_poses[frame_index], 'target_keypoints': solution.target_keypoints[frame_index], 'predicted_keypoints': solution.predicted_keypoints[frame_index], 'keypoint_error_m': errors[frame_index], 'keypoint_weight': point_weights[frame_index]}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def rt__is_direct_pose(pose: dict[str, Any]) -> bool:
    source = str(pose.get('pose_source', ''))
    return not bool(pose.get('pose_filled', False)) and (not any((token in source for token in ('rgb_flow', 'interpolation'))))

def rt__interpolate_transform(previous: np.ndarray, following: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = (1.0 - alpha) * previous[:3, 3] + alpha * following[:3, 3]
    relative = previous[:3, :3].T @ following[:3, :3]
    output[:3, :3] = previous[:3, :3] @ Rotation.from_rotvec(alpha * Rotation.from_matrix(relative).as_rotvec()).as_matrix()
    return output

def rt_calibrate_observation_confidences(pose_rows: list[list[dict[str, Any]]], transforms: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate continuous weights from image quality and held-out tag motion."""
    frame_count = len(pose_rows)
    confidences = np.zeros((frame_count, len(rt_FINGER_SPECS)), dtype=np.float32)
    report: dict[str, Any] = {'method': 'continuous image-quality likelihood multiplied by direct-tag leave-one-out SE3 temporal consistency', 'ground_truth_status': 'No external mocap ground truth; direct tag detections are held out one frame at a time as a measurable proxy validation set.', 'per_finger': {}}

    def finite_values(values: list[float], fallback: float) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        return array if len(array) else np.asarray([fallback], dtype=np.float64)

    def lower_is_better(value: float, center: float, scale: float) -> float:
        if not np.isfinite(value):
            return 0.25
        excess = max(float(value) - float(center), 0.0)
        return float(np.exp(-excess / max(float(scale), 1e-06)))

    def stats(values: list[float]) -> dict[str, float]:
        array = finite_values(values, 0.0)
        return {'mean': float(np.mean(array)), 'median': float(np.median(array)), 'p95': float(np.percentile(array, 95.0)), 'max': float(np.max(array))}
    for (finger_index, spec) in enumerate(rt_FINGER_SPECS):
        records = [row[finger_index] for row in pose_rows]
        direct = np.asarray([rt__is_direct_pose(record) for record in records])
        direct_indices = np.flatnonzero(direct)
        direct_reprojection = finite_values([float(records[index].get('reproj_error', np.nan)) for index in direct_indices], 1.0)
        reprojection_center = float(np.median(direct_reprojection))
        reprojection_scale = max(float(np.percentile(direct_reprojection, 90.0) - reprojection_center), 0.25)
        direct_edges = finite_values([float(records[index].get('edge_score', np.nan)) for index in direct_indices], 0.5)
        edge_low = float(np.percentile(direct_edges, 10.0))
        edge_high = max(float(np.percentile(direct_edges, 90.0)), edge_low + 0.05)
        previous_direct = np.full(frame_count, -1, dtype=np.int32)
        following_direct = np.full(frame_count, -1, dtype=np.int32)
        last = -1
        for frame_index in range(frame_count):
            previous_direct[frame_index] = last
            if direct[frame_index]:
                last = frame_index
        last = -1
        for frame_index in range(frame_count - 1, -1, -1):
            following_direct[frame_index] = last
            if direct[frame_index]:
                last = frame_index
        innovation_translation_mm = np.full(frame_count, np.nan, dtype=np.float64)
        innovation_rotation_deg = np.full(frame_count, np.nan, dtype=np.float64)
        bracket_seconds = np.full(frame_count, np.nan, dtype=np.float64)
        for frame_index in range(frame_count):
            before = int(previous_direct[frame_index])
            after = int(following_direct[frame_index])
            if before < 0 or after < 0:
                continue
            duration = float(timestamps[after] - timestamps[before])
            if duration <= 0.0:
                continue
            alpha = float((timestamps[frame_index] - timestamps[before]) / duration)
            predicted = rt__interpolate_transform(transforms[before, finger_index], transforms[after, finger_index], alpha)
            observed = transforms[frame_index, finger_index]
            innovation_translation_mm[frame_index] = np.linalg.norm(observed[:3, 3] - predicted[:3, 3]) * 1000.0
            innovation_rotation_deg[frame_index] = np.rad2deg(np.linalg.norm(Rotation.from_matrix(predicted[:3, :3].T @ observed[:3, :3]).as_rotvec()))
            bracket_seconds[frame_index] = duration
        held_out_translation = innovation_translation_mm[direct]
        held_out_rotation = innovation_rotation_deg[direct]
        translation_scale = max(float(np.nanpercentile(held_out_translation, 95.0)), 0.5)
        rotation_scale = max(float(np.nanpercentile(held_out_rotation, 95.0)), 0.5)
        flow_indices = [index for (index, record) in enumerate(records) if 'rgb_flow' in str(record.get('pose_source', ''))]
        flow_reprojection = finite_values([float(records[index].get('reproj_error', np.nan)) for index in flow_indices], reprojection_center + reprojection_scale)
        flow_fb = finite_values([float(records[index].get('flow_fb_median_px', np.nan)) for index in flow_indices], 1.0)
        flow_agreement = finite_values([float(records[index].get('flow_current_tag_corner_agreement_px', np.nan)) for index in flow_indices], 5.0)
        flow_reprojection_reference = max(float(np.median(flow_reprojection)), 0.5)
        flow_fb_reference = max(float(np.median(flow_fb)), 0.25)
        flow_agreement_reference = max(float(np.median(flow_agreement)), 1.0)
        median_dt = float(np.median(np.diff(timestamps)))
        for (frame_index, record) in enumerate(records):
            source = str(record.get('pose_source', ''))
            reprojection = float(record.get('reproj_error', np.nan))
            reprojection_score = lower_is_better(reprojection, reprojection_center, reprojection_scale)
            point_score = float(np.clip(record.get('point_inlier_fraction', 1.0), 0.05, 1.0))
            edge = float(record.get('edge_score', np.nan))
            edge_score = 1.0 if not np.isfinite(edge) else float(np.clip((edge - edge_low) / (edge_high - edge_low), 0.1, 1.0))
            if direct[frame_index]:
                image_score = float((reprojection_score * point_score * edge_score) ** (1.0 / 3.0))
                multi_tag_bonus = min(max(int(record.get('n_tags', 1)) - 1, 0), 2)
                confidences[frame_index, finger_index] = np.clip(0.6 + 0.3 * image_score + 0.05 * multi_tag_bonus, 0.6, 1.0)
                continue
            translation_innovation = innovation_translation_mm[frame_index]
            rotation_innovation = innovation_rotation_deg[frame_index]
            temporal_score = 0.35
            if np.isfinite(translation_innovation) and np.isfinite(rotation_innovation):
                temporal_score = float(np.exp(-0.5 * ((translation_innovation / translation_scale) ** 2 + (rotation_innovation / rotation_scale) ** 2)))
            if 'rgb_flow' in source:
                fb = float(record.get('flow_fb_median_px', np.nan))
                agreement = float(record.get('flow_current_tag_corner_agreement_px', np.nan))
                if not np.isfinite(reprojection):
                    reprojection = 2.0 * flow_reprojection_reference
                if not np.isfinite(fb):
                    fb = 2.0 * flow_fb_reference
                if not np.isfinite(agreement):
                    agreement = 2.0 * flow_agreement_reference
                inlier_ratio = float(np.clip(record.get('flow_homography_inlier_ratio', 0.0), 0.01, 1.0))
                flow_score = (1.0 / (1.0 + (max(reprojection, 0.0) / flow_reprojection_reference) ** 2) * 1.0 / (1.0 + (max(fb, 0.0) / flow_fb_reference) ** 2) * 1.0 / (1.0 + (max(agreement, 0.0) / flow_agreement_reference) ** 2) * inlier_ratio * temporal_score) ** 0.2
                if bool(record.get('flow_anchor_is_filled', False)):
                    flow_score *= 0.65
                confidences[frame_index, finger_index] = np.clip(0.04 + 0.46 * flow_score, 0.04, 0.5)
            else:
                span = bracket_seconds[frame_index]
                span_score = 0.25 if not np.isfinite(span) else float(np.exp(-max(span - 2.0 * median_dt, 0.0) / (4.0 * median_dt)))
                confidences[frame_index, finger_index] = np.clip(0.04 + 0.36 * np.sqrt(span_score * temporal_score), 0.04, 0.4)
        finger_confidence = confidences[:, finger_index]
        report['per_finger'][spec.name] = {'direct_frame_count': int(np.sum(direct)), 'filled_frame_count': int(np.sum(~direct)), 'held_out_direct_frame_count': int(np.sum(np.isfinite(held_out_translation) & np.isfinite(held_out_rotation))), 'held_out_direct_translation_error_mm': stats(held_out_translation.tolist()), 'held_out_direct_rotation_error_deg': stats(held_out_rotation.tolist()), 'direct_reprojection_center_px': reprojection_center, 'direct_reprojection_scale_px': reprojection_scale, 'temporal_translation_scale_mm': translation_scale, 'temporal_rotation_scale_deg': rotation_scale, 'confidence': stats(finger_confidence.tolist()), 'minimum_confidence_frames': np.argsort(finger_confidence)[:8].tolist()}
    return (confidences, report)

def rt_load_multi_cam_sidecar_poses(raw_path: Path, sidecar_path: Path, max_frames: int | None, wrist_extrinsics_override: Path | None=None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 0717 multi-camera poses, correcting thumb/index by physical Cube ID."""
    with raw_path.open('rb') as stream:
        raw_header = pickle.load(stream)
    if not isinstance(raw_header, dict) or raw_header.get('format') != rt_MULTI_CAM_RAW_FORMAT:
        raise ValueError(f"Unsupported multi-camera raw format: {raw_header.get('format')!r}")
    raw_frame_count = int(raw_header.get('num_samples', -1))
    if raw_frame_count <= 0:
        raise ValueError('Multi-camera raw header has no positive num_samples')
    with sidecar_path.open('rb') as stream:
        sidecar_header_preview = pickle.load(stream)
    if not isinstance(sidecar_header_preview, dict) or sidecar_header_preview.get('format') != rt_MULTI_CAM_SIDECAR_FORMAT:
        raise ValueError(f"Unsupported pose sidecar format: {sidecar_header_preview.get('format')!r}")
    wrist_target_metadata = sidecar_header_preview.get('metadata', {}).get('targets', {}).get('wrist_Q')
    if not isinstance(wrist_target_metadata, dict):
        raise KeyError('Sidecar metadata has no wrist_Q target')
    final_smoothing = sidecar_header_preview.get('metadata', {}).get('final_global_smoothing', {})
    required_targets = {'wrist_Q', 'index_Q', 'thumb_Q', 'middle_Q'}
    applied_counts = final_smoothing.get('applied_counts', {}) or {}
    if not bool(final_smoothing.get('complete', False)) or not bool(final_smoothing.get('completion_barrier_passed', False)) or int(final_smoothing.get('frame_count', -1)) != raw_frame_count or (set(final_smoothing.get('targets', [])) != required_targets) or any((int(applied_counts.get(name, -1)) != raw_frame_count for name in required_targets)):
        raise ValueError('Retargeting requires a complete stage13 sidecar: all wrist/index/thumb/middle poses must exist before final global smoothing. Run this 020 pipeline from the multi-camera raw PKL first.')
    expected_wrist_cube_dir = Path(str(wrist_target_metadata.get('cube_cfg', ''))).name
    raw_extrinsics_value = raw_header.get('metadata', {}).get('wrist_extrinsics_yaml')
    candidates: list[Path] = []
    if wrist_extrinsics_override is not None:
        candidates.append(wrist_extrinsics_override.expanduser().resolve())
    else:
        if raw_extrinsics_value:
            candidates.append(Path(str(raw_extrinsics_value)).expanduser().resolve())
        candidates.append(rt_DEFAULT_MULTI_CAM_WRIST_EXTRINSICS.expanduser().resolve())
    wrist_extrinsics_path: Path | None = None
    wrist_payload: dict[str, Any] | None = None
    rejected_extrinsics: list[dict[str, str]] = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            rejected_extrinsics.append({'path': str(candidate), 'reason': 'file_not_found'})
            continue
        with candidate.open('r', encoding='utf-8') as stream:
            payload = yaml.safe_load(stream)
        actual_wrist_cube_dir = Path(str(payload.get('inputs', {}).get('aprilcube_cfg_dir', ''))).name
        if actual_wrist_cube_dir != expected_wrist_cube_dir:
            rejected_extrinsics.append({'path': str(candidate), 'reason': f'cube_cfg_mismatch:{actual_wrist_cube_dir!r}!={expected_wrist_cube_dir!r}'})
            continue
        wrist_extrinsics_path = candidate
        wrist_payload = payload
        break
    if wrist_extrinsics_path is None or wrist_payload is None:
        raise ValueError(f'No wrist extrinsics use the same Q cube model as wrist_Q: expected={expected_wrist_cube_dir!r}, rejected={rejected_extrinsics}')
    world_from_camera = {camera: base.validate_transform(np.asarray(wrist_payload[f'Q_T_{camera}'], dtype=np.float64), f'{wrist_extrinsics_path}:Q_T_{camera}') for camera in ('thumb_web_cam', 'middle_finger_cam')}
    poses: list[np.ndarray] = []
    pose_quality_rows: list[list[dict[str, Any]]] = []
    pose_source_counts: dict[str, dict[str, int]] = {spec.name: {} for spec in rt_FINGER_SPECS}
    timestamps: list[float] = []
    source_indices: list[int] = []
    footer: dict[str, Any] | None = None
    with sidecar_path.open('rb') as stream:
        sidecar_header = pickle.load(stream)
        if not isinstance(sidecar_header, dict) or sidecar_header.get('format') != rt_MULTI_CAM_SIDECAR_FORMAT:
            raise ValueError(f"Unsupported pose sidecar format: {sidecar_header.get('format')!r}")
        if Path(str(sidecar_header.get('source_multi_cam_pkl', ''))).resolve() != raw_path:
            raise ValueError('Pose sidecar source_multi_cam_pkl does not match raw PKL')
        expected_identity = {'size': int(raw_path.stat().st_size), 'mtime_ns': int(raw_path.stat().st_mtime_ns)}
        if sidecar_header.get('source_multi_cam_identity') != expected_identity:
            raise ValueError('Pose sidecar source identity does not match raw PKL')
        target_metadata = sidecar_header.get('metadata', {}).get('targets', {})
        for (finger_name, mapping) in rt_MULTI_CAM_TARGET_MAPPING.items():
            target_name = str(mapping['sidecar_target'])
            metadata = target_metadata.get(target_name)
            if not isinstance(metadata, dict):
                raise KeyError(f'Sidecar metadata has no target {target_name!r}')
            actual_cube_dir = Path(str(metadata.get('cube_cfg', ''))).name
            if actual_cube_dir != mapping['cube_dir']:
                raise ValueError(f"{finger_name} must use physical Cube {mapping['cube_dir']}, but sidecar target {target_name} uses {actual_cube_dir!r}")
            if metadata.get('camera_name') != mapping['camera']:
                raise ValueError(f"{target_name} camera mismatch: {metadata.get('camera_name')!r}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get('type') == 'footer':
                footer = record
                break
            if record.get('type') != 'frame':
                continue
            sample_index = int(record.get('sample_index', -1))
            if sample_index != len(source_indices):
                raise ValueError(f'Sidecar sample_index is not contiguous: got {sample_index}, expected {len(source_indices)}')
            frame_poses = record.get('poses', {})
            current: list[np.ndarray] = []
            current_quality: list[dict[str, Any]] = []
            for spec in rt_FINGER_SPECS:
                mapping = rt_MULTI_CAM_TARGET_MAPPING[spec.name]
                target_name = str(mapping['sidecar_target'])
                pose = frame_poses.get(target_name)
                if not isinstance(pose, dict) or not bool(pose.get('success', False)) or pose.get('T') is None:
                    raise ValueError(f'Sidecar frame {sample_index} has no valid {target_name} pose')
                if not bool(pose.get('final_global_smoothing_applied', False)):
                    raise ValueError(f'Sidecar frame {sample_index} {target_name} bypassed the mandatory final global smoothing stage')
                camera_from_obj = np.asarray(pose['T'], dtype=np.float64).copy()
                camera_from_obj[:3, 3] *= 0.001
                camera_from_obj = base.validate_transform(camera_from_obj, f'sidecar frame {sample_index} {target_name}')
                current.append(world_from_camera[str(mapping['camera'])] @ camera_from_obj)
                current_quality.append(pose)
                pose_source = str(pose.get('pose_source', 'unknown'))
                source_counts = pose_source_counts[spec.name]
                source_counts[pose_source] = source_counts.get(pose_source, 0) + 1
            poses.append(np.stack(current, axis=0))
            pose_quality_rows.append(current_quality)
            timestamps.append(float(record.get('time_monotonic', sample_index)))
            source_indices.append(sample_index)
            if max_frames is not None and len(poses) >= max_frames:
                break
    if not poses:
        raise ValueError('Pose sidecar contains no usable three-finger frames')
    if max_frames is None:
        if len(poses) != raw_frame_count:
            raise ValueError(f'Raw/sidecar frame count mismatch: {raw_frame_count} != {len(poses)}')
        if footer is None or int(footer.get('frame_count', -1)) != len(poses):
            raise ValueError('Pose sidecar footer frame_count mismatch')
        required_targets = {'wrist_Q', 'index_Q', 'thumb_Q', 'middle_Q'}
        success_counts = footer.get('success_counts', {}) or {}
        applied_counts = footer.get('final_global_smoothing_applied_counts', {}) or {}
        if not bool(footer.get('final_global_smoothing_complete', False)) or any((int(success_counts.get(name, -1)) != len(poses) for name in required_targets)) or any((int(applied_counts.get(name, -1)) != len(poses) for name in required_targets)):
            raise ValueError('Pose sidecar footer does not certify complete stage13 smoothing for every target/frame')
    mapping_summary = {finger_name: {'sidecar_target': str(mapping['sidecar_target']), 'camera': str(mapping['camera']), 'physical_cube': str(mapping['cube_dir'])} for (finger_name, mapping) in rt_MULTI_CAM_TARGET_MAPPING.items()}
    poses_array = np.asarray(poses, dtype=np.float64)
    timestamps_array = np.asarray(timestamps, dtype=np.float64)
    (observation_confidences, confidence_calibration) = rt_calibrate_observation_confidences(pose_quality_rows, poses_array, timestamps_array)
    validation = {'source_kind': 'multi_camera_020_pose_sidecar', 'raw_format': rt_MULTI_CAM_RAW_FORMAT, 'pose_sidecar': str(sidecar_path), 'pose_sidecar_format': rt_MULTI_CAM_SIDECAR_FORMAT, 'wrist_extrinsics': str(wrist_extrinsics_path), 'wrist_cube_cfg': expected_wrist_cube_dir, 'rejected_wrist_extrinsics': rejected_extrinsics, 'frame_count': int(len(poses)), 'finger_mapping_by_physical_cube_id': mapping_summary, 'observation_confidence_calibration': confidence_calibration, 'pose_source_counts': pose_source_counts, 'final_global_smoothing': final_smoothing}
    return (validation, poses_array, timestamps_array, np.asarray(source_indices, dtype=np.int32), observation_confidences)

def rt_validate_source_pkl(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        header = pickle.load(stream)
    metadata = header.get('metadata', {})
    middle_intrinsics = metadata.get('camera_intrinsics_yaml', {}).get('middle_finger_cam')
    if middle_intrinsics != rt_EXPECTED_MIDDLE_INTRINSICS:
        raise ValueError(f'Source PKL does not contain the corrected middle intrinsics: {middle_intrinsics!r}')
    recovery = metadata.get('rgb_pose_recovery', {})
    recovered_count = recovery.get('quality_summary', {}).get('recovered_pose_count')
    if recovered_count != 6:
        raise ValueError('Source PKL does not contain the expected six recovered middle poses')
    return {'middle_intrinsics': middle_intrinsics, 'recovery_algorithm': recovery.get('algorithm'), 'recovered_pose_count': int(recovered_count)}

def rt_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Three-finger four-point global cube-pose experiments.')
    parser.add_argument('pkl_path', nargs='?', type=Path, default=rt_DEFAULT_PKL_PATH)
    parser.add_argument('--pose-sidecar', type=Path, help='Aligned consensv2_multi_cam_020_pose_sidecar_v1 input. When set, pkl_path is the original multi-camera raw stream.')
    parser.add_argument('--wrist-extrinsics', type=Path, help='Optional Q_T_camera YAML override. Its AprilCube model must match the sidecar wrist_Q cube model.')
    parser.add_argument('--urdf', type=Path, default=rt_DEFAULT_URDF_PATH)
    parser.add_argument('--fingertip-geometry-urdf', type=Path, default=rt_DEFAULT_FINGERTIP_GEOMETRY_URDF_PATH, help='URDF whose visual origins define the v2 fingertip OBJ frames.')
    parser.add_argument('--contact-keypoints', type=Path, default=rt_DEFAULT_CONTACT_CONFIG)
    parser.add_argument('--output-dir', type=Path, default=rt_DEFAULT_OUTPUT_DIR)
    parser.add_argument('--modes', nargs='+', choices=('fixed', 'translation', 'rotation', 'se3'), default=('fixed', 'translation', 'rotation', 'se3'))
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--solver-iterations', type=int, default=rt_DEFAULT_SOLVER_ITERATIONS)
    parser.add_argument('--alternating-iterations', type=int, default=rt_DEFAULT_ALTERNATING_ITERATIONS)
    parser.add_argument('--temporal-branch-weight', type=float, default=rt_DEFAULT_TEMPORAL_BRANCH_WEIGHT, help='Dynamic-programming penalty in m^2/rad^2 for switching between per-frame IK branches.')
    parser.add_argument('--temporal-seed-weight', type=float, default=rt_DEFAULT_TEMPORAL_SEED_WEIGHT, help='Active-joint continuity weight used by the final chronological sequential IK refinement.')
    parser.add_argument('--temporal-refine-only', action='store_true', help='Keep the existing three_finger_se3 offset and run only the final chronological sequential IK refinement.')
    parser.add_argument('--joint-temporal-optimize', action='store_true', help='Jointly optimize all frame qpos using confidence-weighted four-point contact error plus velocity, acceleration, and jerk penalties.')
    parser.add_argument('--joint-velocity-weight', type=float, default=rt_DEFAULT_JOINT_VELOCITY_WEIGHT)
    parser.add_argument('--joint-acceleration-weight', type=float, default=rt_DEFAULT_JOINT_ACCELERATION_WEIGHT)
    parser.add_argument('--joint-jerk-weight', type=float, default=rt_DEFAULT_JOINT_JERK_WEIGHT)
    parser.add_argument('--joint-optimization-iterations', type=int, default=rt_DEFAULT_JOINT_OPTIMIZATION_ITERATIONS)
    parser.add_argument('--joint-max-step-deg', type=float, default=rt_DEFAULT_JOINT_MAX_STEP_DEG)
    parser.add_argument('--joint-max-step-weight', type=float, default=rt_DEFAULT_JOINT_MAX_STEP_WEIGHT)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def rt_main() -> None:
    args = rt_parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    geometry_urdf_path = args.fingertip_geometry_urdf.expanduser().resolve()
    contact_path = args.contact_keypoints.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pose_sidecar_path = args.pose_sidecar.expanduser().resolve() if args.pose_sidecar is not None else None
    required_paths = [pkl_path, urdf_path, geometry_urdf_path, contact_path]
    if pose_sidecar_path is not None:
        required_paths.append(pose_sidecar_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError('--max-frames must be positive')
    if args.solver_iterations <= 0 or args.alternating_iterations <= 0:
        raise ValueError('Iteration counts must be positive')
    if args.temporal_branch_weight < 0.0:
        raise ValueError('--temporal-branch-weight must be nonnegative')
    if args.temporal_seed_weight < 0.0:
        raise ValueError('--temporal-seed-weight must be nonnegative')
    if min(args.joint_velocity_weight, args.joint_acceleration_weight, args.joint_jerk_weight, args.joint_max_step_deg, args.joint_max_step_weight) < 0.0:
        raise ValueError('Joint temporal weights must be nonnegative')
    if args.joint_optimization_iterations <= 0:
        raise ValueError('--joint-optimization-iterations must be positive')
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints_obj = base.load_contact_keypoints(contact_path)
    link_from_obj = base.load_link_from_obj_transforms(geometry_urdf_path)
    if pose_sidecar_path is None:
        source_validation = rt_validate_source_pkl(pkl_path)
        (cube_from_obj, timestamps, source_indices) = base.load_source_poses(pkl_path, args.max_frames)
        observation_confidence = np.ones((len(cube_from_obj), len(rt_FINGER_SPECS)), dtype=np.float32)
    else:
        (source_validation, cube_from_obj, timestamps, source_indices, observation_confidence) = rt_load_multi_cam_sidecar_poses(pkl_path, pose_sidecar_path, args.max_frames, args.wrist_extrinsics)
    if pose_sidecar_path is None and args.max_frames is None and (len(cube_from_obj) != 363):
        raise ValueError(f'Expected 363 valid frames, got {len(cube_from_obj)}')
    point_weights = np.repeat(observation_confidence[:, :, None], 4, axis=2)
    ignored_observations: list[dict[str, Any]] = []
    if pose_sidecar_path is None:
        ignored_rows = np.flatnonzero(source_indices == 275)
        if args.max_frames is None and ignored_rows.tolist() != [275]:
            raise ValueError(f'Expected source frame 275 at row 275; got rows {ignored_rows.tolist()}')
        point_weights[ignored_rows, 0, :] = 0.0
        ignored_observations = [{'source_frame_index': 275, 'finger': 'thumb', 'keypoint_weights': [0.0, 0.0, 0.0, 0.0], 'other_fingers_unchanged': True}]
    urdf = base.load_urdf(urdf_path)
    joint_names = list(urdf.actuated_joint_names)
    lower = np.asarray([float(urdf.joint_map[name].limit.lower) for name in joint_names], dtype=np.float32)
    upper = np.asarray([float(urdf.joint_map[name].limit.upper) for name in joint_names], dtype=np.float32)
    natural = np.clip(np.zeros(len(joint_names), dtype=np.float32), lower, upper)
    active_indices = np.asarray([joint_names.index(name) for name in rt_ACTIVE_JOINT_NAMES], dtype=np.int32)
    thumb_joint_indices = np.asarray([joint_names.index(f'left_finger1_joint{i}') for i in range(1, 5)], dtype=np.int32)
    active_mask = np.zeros(len(joint_names), dtype=np.float32)
    active_mask[active_indices] = 1.0
    robot = base.pk.Robot.from_urdf(urdf, default_joint_cfg=base.jnp.asarray(natural))
    solver = rt_BatchedThreeFingerIKSolver(point_weights=point_weights, ignored_joint_indices=thumb_joint_indices, urdf=urdf, robot=robot, link_from_obj=link_from_obj, keypoints_obj=keypoints_obj, natural_qpos=natural, active_mask=active_mask, max_iterations=int(args.solver_iterations), temporal_branch_weight=float(args.temporal_branch_weight))
    requested_modes = list(dict.fromkeys(args.modes))
    results: dict[str, dict[str, Any]] = {}
    selected: dict[str, tuple[np.ndarray, base.SequenceSolution]] = {}
    run_config = {'schema': 'consens.left_wuji_three_finger_experiment_config.v1', 'source_pkl': str(pkl_path), 'source_validation': source_validation, 'urdf': str(urdf_path), 'fingertip_geometry_urdf': str(geometry_urdf_path), 'contact_keypoints': str(contact_path), 'fingers': [spec.name for spec in rt_FINGER_SPECS], 'frame_count': int(len(cube_from_obj)), 'point_count_per_frame': 12, 'active_joint_names': list(rt_ACTIVE_JOINT_NAMES), 'T_left_palm_link_hand_back_cube_initial': base.T_PALM_CUBE_INITIAL.tolist(), 'requested_modes': requested_modes, 'solver_iterations': int(args.solver_iterations), 'alternating_iterations': int(args.alternating_iterations), 'objective': 'equal-weight squared 3D distance over 3 fingers x 4 points', 'active_joint_regularization': 0.0, 'active_temporal_regularization': float(args.temporal_branch_weight), 'active_temporal_seed_weight': float(args.temporal_seed_weight), 'joint_temporal_optimize': bool(args.joint_temporal_optimize), 'joint_velocity_weight': float(args.joint_velocity_weight), 'joint_acceleration_weight': float(args.joint_acceleration_weight), 'joint_jerk_weight': float(args.joint_jerk_weight), 'joint_optimization_iterations': int(args.joint_optimization_iterations), 'joint_max_step_deg': float(args.joint_max_step_deg), 'joint_max_step_weight': float(args.joint_max_step_weight), 'joint_limits_from_urdf': True, 'ignored_observations': ignored_observations}
    (output_dir / 'run_config.json').write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.temporal_refine_only:
        if not args.overwrite:
            raise ValueError('--temporal-refine-only requires --overwrite')
        if args.temporal_seed_weight <= 0.0:
            raise ValueError('--temporal-refine-only requires positive --temporal-seed-weight')
        npz_path = output_dir / 'three_finger_se3.npz'
        json_path = output_dir / 'three_finger_se3.json'
        pkl_path_out = output_dir / 'three_finger_se3.pkl'
        report_path = output_dir / 'experiment_summary.json'
        for path in (npz_path, json_path, pkl_path_out, report_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(npz_path, allow_pickle=False) as archive:
            transform = np.asarray(archive['T_left_palm_link_hand_back_cube'], dtype=np.float64)
            initial_qpos = np.asarray(archive['qpos'], dtype=np.float32)
            best_start = str(np.asarray(archive['best_start']).item())
            stored_timestamps = np.asarray(archive['timestamps'], dtype=np.float64)
        if initial_qpos.shape != (len(cube_from_obj), len(joint_names)):
            raise ValueError(f'Existing qpos shape does not match current source/URDF: {initial_qpos.shape}')
        if not np.allclose(stored_timestamps, timestamps, atol=1e-09, rtol=0.0):
            raise ValueError('Existing retarget timestamps do not match current source')
        previous_summary = json.loads(json_path.read_text(encoding='utf-8'))
        refined = solver.refine_sequence_temporally(transform, cube_from_obj, initial_qpos, float(args.temporal_seed_weight))
        smoothness_before_joint = rt_trajectory_smoothness_stats(refined.qpos, timestamps, active_indices)
        joint_metadata: dict[str, Any] | None = None
        if args.joint_temporal_optimize:
            (refined, joint_metadata) = solver.optimize_sequence_jointly(transform, cube_from_obj, timestamps, refined.qpos, float(args.joint_velocity_weight), float(args.joint_acceleration_weight), float(args.joint_jerk_weight), float(args.joint_max_step_deg), float(args.joint_max_step_weight), int(args.joint_optimization_iterations))
            joint_metadata['smoothness_before'] = smoothness_before_joint
            joint_metadata['smoothness_after'] = rt_trajectory_smoothness_stats(refined.qpos, timestamps, active_indices)
        summary = rt_summarize_best('se3', best_start, transform, refined, point_weights, list(previous_summary.get('path_summaries', [])), ignored_observations, active_indices, lower, upper, float(args.temporal_branch_weight))
        summary['active_temporal_seed_weight'] = float(args.temporal_seed_weight)
        summary['temporal_refine_only'] = True
        summary['trajectory_smoothness'] = rt_trajectory_smoothness_stats(refined.qpos, timestamps, active_indices)
        if joint_metadata is not None:
            summary['joint_temporal_optimization'] = joint_metadata
        rt_save_npz(npz_path, 'se3', best_start, transform, refined, cube_from_obj, timestamps, source_indices, keypoints_obj, link_from_obj, joint_names, active_indices, point_weights)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        rt_write_compact_pkl(pkl_path_out, pkl_path, 'se3', best_start, summary, transform, refined, timestamps, source_indices, joint_names, active_indices, point_weights, source_validation, True)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report.update(run_config)
        report['schema'] = 'consens.left_wuji_three_finger_cube_experiments.v1'
        report.setdefault('results', {})['se3'] = summary
        report['best_optimized_mode'] = 'se3'
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f"[RESULT] temporally refined three_finger_se3 objective_rmse={summary['objective_rmse_mm']:.6f} mm")
        return
    (fixed_transform, fixed_solution, fixed_path) = rt_run_path('fixed', 'theoretical', base.T_PALM_CUBE_INITIAL, solver, cube_from_obj, keypoints_obj, point_weights, int(args.alternating_iterations))
    selected['fixed'] = (fixed_transform, fixed_solution)
    fixed_summary = rt_summarize_best('fixed', 'theoretical', fixed_transform, fixed_solution, point_weights, [fixed_path], ignored_observations, active_indices, lower, upper, float(args.temporal_branch_weight))
    results['fixed'] = fixed_summary

    def run_optimized_mode(mode: rt_Mode, starts: list[tuple[str, np.ndarray]]) -> None:
        path_results: list[tuple[np.ndarray, base.SequenceSolution, dict[str, Any]]] = []
        for (start_name, start_transform) in rt_deduplicate_starts(starts):
            path_results.append(rt_run_path(mode, start_name, start_transform, solver, cube_from_obj, keypoints_obj, point_weights, int(args.alternating_iterations)))
        best_index = min(range(len(path_results)), key=lambda index: path_results[index][2]['objective_rmse_mm'])
        (transform, solution, best_path) = path_results[best_index]
        if mode in requested_modes and args.temporal_seed_weight > 0.0:
            solution = solver.refine_sequence_temporally(transform, cube_from_obj, solution.qpos, float(args.temporal_seed_weight))
        joint_metadata: dict[str, Any] | None = None
        if mode in requested_modes and args.joint_temporal_optimize:
            smoothness_before_joint = rt_trajectory_smoothness_stats(solution.qpos, timestamps, active_indices)
            (solution, joint_metadata) = solver.optimize_sequence_jointly(transform, cube_from_obj, timestamps, solution.qpos, float(args.joint_velocity_weight), float(args.joint_acceleration_weight), float(args.joint_jerk_weight), float(args.joint_max_step_deg), float(args.joint_max_step_weight), int(args.joint_optimization_iterations))
            joint_metadata['smoothness_before'] = smoothness_before_joint
            joint_metadata['smoothness_after'] = rt_trajectory_smoothness_stats(solution.qpos, timestamps, active_indices)
        path_summaries = [value[2] for value in path_results]
        selected[mode] = (transform, solution)
        results[mode] = rt_summarize_best(mode, str(best_path['start_name']), transform, solution, point_weights, path_summaries, ignored_observations, active_indices, lower, upper, float(args.temporal_branch_weight))
        results[mode]['active_temporal_seed_weight'] = float(args.temporal_seed_weight)
        results[mode]['trajectory_smoothness'] = rt_trajectory_smoothness_stats(solution.qpos, timestamps, active_indices)
        if joint_metadata is not None:
            results[mode]['joint_temporal_optimization'] = joint_metadata
    (fixed_translation_start, _) = rt_fit_transform_for_mode('translation', cube_from_obj, keypoints_obj, fixed_solution.predicted_keypoints, point_weights)
    (fixed_rotation_start, _) = rt_fit_transform_for_mode('rotation', cube_from_obj, keypoints_obj, fixed_solution.predicted_keypoints, point_weights)
    (fixed_se3_start, _) = rt_fit_transform_for_mode('se3', cube_from_obj, keypoints_obj, fixed_solution.predicted_keypoints, point_weights)
    if 'translation' in requested_modes or 'se3' in requested_modes:
        run_optimized_mode('translation', [('theoretical', base.T_PALM_CUBE_INITIAL), ('fixed_registration', fixed_translation_start)])
    if 'rotation' in requested_modes or 'se3' in requested_modes:
        run_optimized_mode('rotation', [('theoretical', base.T_PALM_CUBE_INITIAL), ('fixed_registration', fixed_rotation_start)])
    if 'se3' in requested_modes:
        run_optimized_mode('se3', [('theoretical', base.T_PALM_CUBE_INITIAL), ('fixed_registration', fixed_se3_start), ('translation_best', selected['translation'][0]), ('rotation_best', selected['rotation'][0])])
    modes_to_write = [mode for mode in requested_modes if mode in selected]
    for mode in modes_to_write:
        (transform, solution) = selected[mode]
        summary = results[mode]
        stem = f'three_finger_{mode}'
        npz_path = output_dir / f'{stem}.npz'
        json_path = output_dir / f'{stem}.json'
        pkl_path_out = output_dir / f'{stem}.pkl'
        if not args.overwrite:
            for path in (npz_path, json_path, pkl_path_out):
                if path.exists():
                    raise FileExistsError(path)
        rt_save_npz(npz_path, mode, str(summary['best_start']), transform, solution, cube_from_obj, timestamps, source_indices, keypoints_obj, link_from_obj, joint_names, active_indices, point_weights)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        rt_write_compact_pkl(pkl_path_out, pkl_path, mode, str(summary['best_start']), summary, transform, solution, timestamps, source_indices, joint_names, active_indices, point_weights, source_validation, bool(args.overwrite))
    report = {**run_config, 'schema': 'consens.left_wuji_three_finger_cube_experiments.v1', 'results': {mode: results[mode] for mode in modes_to_write}, 'best_optimized_mode': min([mode for mode in modes_to_write if mode != 'fixed'], key=lambda mode: results[mode]['objective_rmse_mm']) if any((mode != 'fixed' for mode in modes_to_write)) else None}
    report_path = output_dir / 'experiment_summary.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('\n[RESULT] three-finger objective RMSE')
    for mode in modes_to_write:
        print(f"  {mode:12s} {results[mode]['objective_rmse_mm']:.6f} mm start={results[mode]['best_start']}")
    print(f'[RESULT] summary: {report_path}')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Embedded xArm7 + Wuji-left full-body IK and hardware-safe trajectory processing.
#
# The backend and synchronized multi-camera adapter are namespace-prefixed from
# their former scripts. No script module is imported at runtime.
# -----------------------------------------------------------------------------
import argparse
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import viser
import yourdfpy
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp
from viser.extras import ViserUrdf
fb_FILE_PATH = Path(__file__).resolve()
fb_REPO_ROOT = APRILCUBE_ROOT.parent.parent
fb_DEFAULT_HAND_PKL = fb_REPO_ROOT / 'thirdparty/aprilcube/recordings/021_hand_back_sync_raw_frames_20260712_233831.pkl'
fb_DEFAULT_RS_PKL = fb_REPO_ROOT / 'thirdparty/aprilcube/recordings/012_rs_raw_frames_20260715_192635_with_current_020_postprocessed_pose.pkl'
fb_DEFAULT_URDF = fb_REPO_ROOT / 'thirdparty/xarm7_wuji_left_description/xarm7_wuji_left_w_fingereye_v2.urdf'
fb_LEGACY_HAND_URDF = fb_REPO_ROOT / 'thirdparty/wuji-description/hand/body-with-soft/urdf/left_simplified_w_fingereye.urdf'
fb_DEFAULT_OUTPUT_DIR = fb_REPO_ROOT / 'outputs/retargeting/rs_xarm7_wuji'
fb_DEFAULT_MERGED_PKL = fb_DEFAULT_OUTPUT_DIR / 'rs_hand_three_finger_merged.pkl'
fb_DEFAULT_IK_PKL = fb_DEFAULT_OUTPUT_DIR / 'xarm7_wuji_full_qpos.pkl'
fb_HAND_FORMAT = 'aprilcube_hand_back_software_synced_raw_v1'
fb_RS_FORMAT = 'aprilcube_raw_with_020_postprocessed_pose_stream_v1'
fb_MERGED_FORMAT = 'consens.rs_hand_three_finger_merged.v1'
fb_IK_FORMAT = 'consens.xarm7_wuji_rs_virtual_base_ik.v2'
fb_HAND_POSE_FIELD = 'hand_back_cube_obj_poses'
fb_HAND_QPOS_FIELD = 'left_wuji_three_finger_se3_retargeting_temporal_smoothed'
fb_HAND_METADATA_FIELD = 'left_wuji_three_finger_se3_retargeting'
fb_PALM_CUBE_KEY = 'T_left_palm_link_hand_back_cube_6d_optimized'
fb_FINGER_NAMES = ('thumb', 'index', 'middle')
fb_ARM_JOINT_NAMES = tuple((f'joint{i}' for i in range(1, 8)))
fb_THREE_FINGER_JOINT_NAMES = tuple((f'left_finger{finger}_joint{joint}' for finger in (1, 2, 3) for joint in range(1, 5)))

@dataclass
class fb_HandTrajectory:
    timestamps: np.ndarray
    poses_hand_back_obj: np.ndarray
    qpos: np.ndarray
    qpos_joint_names: list[str]
    palm_from_hand_back: np.ndarray
    quality: list[dict[str, Any]]
    header: dict[str, Any]

@dataclass
class fb_RSTrajectory:
    timestamps: np.ndarray
    poses_camera_hand_back: np.ndarray
    quality: list[dict[str, Any]]
    header: dict[str, Any]

@dataclass
class fb_MergedTrajectory:
    header: dict[str, Any]
    timestamps: np.ndarray
    phase: np.ndarray
    poses_camera_hand_back: np.ndarray
    poses_hand_back_obj: np.ndarray
    poses_camera_obj: np.ndarray
    poses_camera_palm: np.ndarray
    hand_qpos: np.ndarray
    hand_qpos_joint_names: list[str]
    palm_from_hand_back: np.ndarray
    hand_bracket_indices: np.ndarray
    hand_interpolation_alpha: np.ndarray
    rs_quality: list[dict[str, Any]]
    hand_quality: list[dict[str, Any]]

@dataclass
class fb_CandidateResult:
    name: str
    base_from_camera: np.ndarray
    qpos: np.ndarray
    fk_base_palm: np.ndarray
    target_base_palm: np.ndarray
    metrics: dict[str, Any]

@dataclass
class fb_HardwareSafeResult:
    qpos: np.ndarray
    fk_base_palm: np.ndarray
    trajectory_time_s: np.ndarray
    playback_fps: float
    metrics: dict[str, Any]

def fb_validate_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f'{label} must be a finite 4x4 transform')
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-08):
        raise ValueError(f'{label} has an invalid homogeneous bottom row')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-05):
        raise ValueError(f'{label} rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-05):
        raise ValueError(f'{label} rotation determinant is not +1')
    return transform

def fb_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

def fb_matrix_to_wxyz_xyz(transform: np.ndarray) -> np.ndarray:
    transform = fb_validate_transform(transform, 'matrix_to_wxyz_xyz input')
    return np.concatenate([fb_matrix_to_wxyz(transform[:3, :3]), transform[:3, 3]])

def fb_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)), 'p95': float(np.percentile(values, 95.0)), 'max': float(np.max(values))}

def fb_load_hand_trajectory(path: Path) -> fb_HandTrajectory:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    quality: list[dict[str, Any]] = []
    with path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('format') != fb_HAND_FORMAT:
            raise ValueError(f"Unsupported hand PKL format: {header.get('format')}")
        retarget_metadata = header.get(fb_HAND_METADATA_FIELD, {})
        palm_from_hand_back = fb_validate_transform(retarget_metadata[fb_PALM_CUBE_KEY], f'hand header {fb_HAND_METADATA_FIELD}.{fb_PALM_CUBE_KEY}')
        smooth_metadata = header.get(fb_HAND_QPOS_FIELD, {})
        joint_names = [str(name) for name in smooth_metadata.get('active_joint_names', [])]
        if joint_names != list(fb_THREE_FINGER_JOINT_NAMES):
            raise ValueError(f'Unexpected embedded hand qpos order: {joint_names}; expected {list(fb_THREE_FINGER_JOINT_NAMES)}')
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict) or record.get('type') != 'frame_pair':
                continue
            pose_payload = record.get(fb_HAND_POSE_FIELD, {}).get('objects', {})
            frame_poses: list[np.ndarray] = []
            frame_quality: dict[str, Any] = {}
            for name in fb_FINGER_NAMES:
                entry = pose_payload.get(name, {})
                if not entry.get('success', False):
                    raise ValueError(f'Hand frame {len(timestamps)} has no {name} pose')
                frame_poses.append(fb_validate_transform(entry['T_hand_back_cube_obj'], f'hand frame {len(timestamps)} {name}'))
                frame_quality[name] = {'predicted': bool(entry.get('predicted', False)), 'pose_source': str(entry.get('pose_source', '')), 'reproj_error_px': float(entry.get('reproj_error_px', float('nan')))}
            q_entry = record.get(fb_HAND_QPOS_FIELD, {})
            frame_qpos = np.asarray(q_entry.get('wujihand_qpos'), dtype=np.float64)
            if frame_qpos.shape != (len(fb_THREE_FINGER_JOINT_NAMES),):
                raise ValueError(f'Hand frame {len(timestamps)} qpos shape is {frame_qpos.shape}')
            timestamps.append(float(record['pair_timestamp']))
            poses.append(np.stack(frame_poses, axis=0))
            qpos.append(frame_qpos)
            quality.append(frame_quality)
    if len(timestamps) < 2:
        raise ValueError('Hand trajectory must contain at least two frames')
    return fb_HandTrajectory(timestamps=np.asarray(timestamps, dtype=np.float64), poses_hand_back_obj=np.asarray(poses, dtype=np.float64), qpos=np.asarray(qpos, dtype=np.float64), qpos_joint_names=joint_names, palm_from_hand_back=palm_from_hand_back, quality=quality, header=header)

def fb_load_rs_trajectory(path: Path) -> fb_RSTrajectory:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    quality: list[dict[str, Any]] = []
    with path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('format') != fb_RS_FORMAT:
            raise ValueError(f"Unsupported RS PKL format: {header.get('format')}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict) or record.get('type') != 'frame':
                continue
            pose = record.get('pose', {})
            if not pose.get('success', False) or pose.get('T') is None:
                raise ValueError(f'RS frame {len(timestamps)} has no pose')
            transform = fb_validate_transform(pose['T'], f'RS frame {len(timestamps)}')
            transform = transform.copy()
            transform[:3, 3] /= 1000.0
            timestamps.append(float(record['capture_timestamp']))
            poses.append(transform)
            quality.append({'pose_source': str(pose.get('pose_source', '')), 'quality_level': str(pose.get('quality_level', '')), 'quality_reason': str(pose.get('quality_reason', '')), 'pose_filled': bool(pose.get('pose_filled', False)), 'reproj_error_px': float(pose.get('reproj_error', float('nan'))), 'temporal_smoothed': bool(pose.get('temporal_smoothed', False))})
    if len(timestamps) < 2:
        raise ValueError('RS trajectory must contain at least two frames')
    return fb_RSTrajectory(timestamps=np.asarray(timestamps, dtype=np.float64), poses_camera_hand_back=np.asarray(poses, dtype=np.float64), quality=quality, header=header)

def fb_normalized_time(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0.0 or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError('Timestamps must be strictly increasing')
    return (timestamps - timestamps[0]) / duration

def fb_interpolation_brackets(source_phase: np.ndarray, target_phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(source_phase, target_phase, side='right')
    right = np.clip(right, 1, len(source_phase) - 1)
    left = right - 1
    denominator = source_phase[right] - source_phase[left]
    alpha = np.divide(target_phase - source_phase[left], denominator, out=np.zeros_like(target_phase), where=denominator > 0.0)
    alpha[target_phase <= source_phase[0]] = 0.0
    left[target_phase <= source_phase[0]] = 0
    right[target_phase <= source_phase[0]] = 0
    alpha[target_phase >= source_phase[-1]] = 0.0
    left[target_phase >= source_phase[-1]] = len(source_phase) - 1
    right[target_phase >= source_phase[-1]] = len(source_phase) - 1
    return (np.stack([left, right], axis=-1), alpha)

def fb_interpolate_transforms(source_phase: np.ndarray, transforms: np.ndarray, target_phase: np.ndarray) -> np.ndarray:
    transforms = np.asarray(transforms, dtype=np.float64)
    flat = transforms.reshape(len(source_phase), -1, 4, 4)
    output = np.tile(np.eye(4, dtype=np.float64), (len(target_phase), flat.shape[1], 1, 1))
    for item_index in range(flat.shape[1]):
        for axis in range(3):
            output[:, item_index, axis, 3] = np.interp(target_phase, source_phase, flat[:, item_index, axis, 3])
        rotations = Rotation.from_matrix(flat[:, item_index, :3, :3])
        output[:, item_index, :3, :3] = Slerp(source_phase, rotations)(target_phase).as_matrix()
    return output.reshape(len(target_phase), *transforms.shape[1:])

def fb_interpolate_rows(source_phase: np.ndarray, values: np.ndarray, target_phase: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(target_phase, source_phase, values[:, column]) for column in range(values.shape[1])], axis=-1)

def fb_build_merged_trajectory(hand: fb_HandTrajectory, rs: fb_RSTrajectory) -> fb_MergedTrajectory:
    hand_phase = fb_normalized_time(hand.timestamps)
    rs_source_phase = fb_normalized_time(rs.timestamps)
    output_timestamps = np.linspace(rs.timestamps[0], rs.timestamps[-1], len(rs.timestamps), dtype=np.float64)
    output_phase = fb_normalized_time(output_timestamps)
    (brackets, alpha) = fb_interpolation_brackets(hand_phase, output_phase)
    (rs_brackets, rs_alpha) = fb_interpolation_brackets(rs_source_phase, output_phase)
    poses_camera_hand_back = fb_interpolate_transforms(rs_source_phase, rs.poses_camera_hand_back, output_phase)
    poses_hand_back_obj = fb_interpolate_transforms(hand_phase, hand.poses_hand_back_obj, output_phase)
    hand_qpos = fb_interpolate_rows(hand_phase, hand.qpos, output_phase)
    poses_camera_obj = np.einsum('tij,tfjk->tfik', poses_camera_hand_back, poses_hand_back_obj)
    hand_back_from_palm = np.linalg.inv(hand.palm_from_hand_back)
    poses_camera_palm = np.einsum('tij,jk->tik', poses_camera_hand_back, hand_back_from_palm)
    hand_quality = [{'left_source_index': int(pair[0]), 'right_source_index': int(pair[1]), 'alpha': float(weight), 'left': hand.quality[int(pair[0])], 'right': hand.quality[int(pair[1])]} for (pair, weight) in zip(brackets, alpha, strict=True)]
    rs_quality = [{'left_source_index': int(pair[0]), 'right_source_index': int(pair[1]), 'alpha': float(weight), 'left': rs.quality[int(pair[0])], 'right': rs.quality[int(pair[1])]} for (pair, weight) in zip(rs_brackets, rs_alpha, strict=True)]
    header = {'type': 'header', 'format': fb_MERGED_FORMAT, 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'frame_convention': 'A_T_B maps coordinates from frame B into frame A', 'translation_unit': 'm', 'rotation_storage': '4x4 rotation matrix', 'master_timeline': 'uniform samples spanning the RS capture duration', 'temporal_alignment': 'normalized endpoint alignment', 'rs_translation_interpolation': 'linear', 'rs_rotation_interpolation': 'quaternion SLERP', 'hand_translation_interpolation': 'linear', 'hand_rotation_interpolation': 'quaternion SLERP', 'hand_qpos_interpolation': 'linear', 'hand_source_frame_count': int(len(hand.timestamps)), 'hand_source_duration_s': float(hand.timestamps[-1] - hand.timestamps[0]), 'rs_source_frame_count': int(len(rs.timestamps)), 'rs_source_duration_s': float(rs.timestamps[-1] - rs.timestamps[0]), 'output_frame_count': int(len(rs.timestamps)), 'output_fps': float((len(rs.timestamps) - 1) / (rs.timestamps[-1] - rs.timestamps[0])), 'hand_qpos_joint_names': hand.qpos_joint_names, 'T_left_palm_link_hand_back_cube': hand.palm_from_hand_back}
    return fb_MergedTrajectory(header=header, timestamps=output_timestamps, phase=output_phase, poses_camera_hand_back=poses_camera_hand_back, poses_hand_back_obj=poses_hand_back_obj, poses_camera_obj=poses_camera_obj, poses_camera_palm=poses_camera_palm, hand_qpos=hand_qpos, hand_qpos_joint_names=hand.qpos_joint_names, palm_from_hand_back=hand.palm_from_hand_back, hand_bracket_indices=brackets, hand_interpolation_alpha=alpha, rs_quality=rs_quality, hand_quality=hand_quality)

def fb_write_merged_pkl(path: Path, merged: fb_MergedTrajectory, hand_path: Path, rs_path: Path, overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    header = {**merged.header, 'source_hand_pkl': str(hand_path), 'source_rs_pkl': str(rs_path)}
    try:
        with temporary.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame_index in range(len(merged.timestamps)):
                pickle.dump({'type': 'frame', 'frame_index': frame_index, 'rs_capture_timestamp': float(merged.timestamps[frame_index]), 'normalized_phase': float(merged.phase[frame_index]), 'hand_source_bracket_indices': merged.hand_bracket_indices[frame_index], 'hand_interpolation_alpha': float(merged.hand_interpolation_alpha[frame_index]), 'T_rs_camera_hand_back_cube': merged.poses_camera_hand_back[frame_index], 'T_hand_back_cube_obj': merged.poses_hand_back_obj[frame_index], 'T_rs_camera_obj': merged.poses_camera_obj[frame_index], 'T_rs_camera_left_palm_link': merged.poses_camera_palm[frame_index], 'finger_names': list(fb_FINGER_NAMES), 'wujihand_three_finger_qpos': merged.hand_qpos[frame_index], 'rs_pose_quality': merged.rs_quality[frame_index], 'hand_pose_quality': merged.hand_quality[frame_index]}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump({'type': 'footer', 'frame_count': int(len(merged.timestamps))}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def fb_load_merged_pkl(path: Path) -> fb_MergedTrajectory:
    records: list[dict[str, Any]] = []
    with path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('format') != fb_MERGED_FORMAT:
            raise ValueError(f"Unsupported merged PKL format: {header.get('format')}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if isinstance(record, dict) and record.get('type') == 'frame':
                records.append(record)
    if not records:
        raise ValueError(f'No frames in {path}')
    return fb_MergedTrajectory(header=header, timestamps=np.asarray([r['rs_capture_timestamp'] for r in records], dtype=np.float64), phase=np.asarray([r['normalized_phase'] for r in records], dtype=np.float64), poses_camera_hand_back=np.asarray([r['T_rs_camera_hand_back_cube'] for r in records], dtype=np.float64), poses_hand_back_obj=np.asarray([r['T_hand_back_cube_obj'] for r in records], dtype=np.float64), poses_camera_obj=np.asarray([r['T_rs_camera_obj'] for r in records], dtype=np.float64), poses_camera_palm=np.asarray([r['T_rs_camera_left_palm_link'] for r in records], dtype=np.float64), hand_qpos=np.asarray([r['wujihand_three_finger_qpos'] for r in records], dtype=np.float64), hand_qpos_joint_names=[str(v) for v in header['hand_qpos_joint_names']], palm_from_hand_back=np.asarray(header['T_left_palm_link_hand_back_cube'], dtype=np.float64), hand_bracket_indices=np.asarray([r['hand_source_bracket_indices'] for r in records], dtype=np.int32), hand_interpolation_alpha=np.asarray([r['hand_interpolation_alpha'] for r in records], dtype=np.float64), rs_quality=[r['rs_pose_quality'] for r in records], hand_quality=[r['hand_pose_quality'] for r in records])

def fb_load_urdf(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(str(path), filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent))

def fb_joint_signature(root: ET.Element, name: str) -> tuple[Any, ...]:
    node = next((j for j in root.findall('joint') if j.attrib.get('name') == name), None)
    if node is None:
        raise KeyError(name)
    origin = node.find('origin')
    axis = node.find('axis')
    limit = node.find('limit')
    return (node.attrib.get('type'), node.find('parent').attrib.get('link'), node.find('child').attrib.get('link'), None if origin is None else origin.attrib.get('xyz', '0 0 0'), None if origin is None else origin.attrib.get('rpy', '0 0 0'), None if axis is None else axis.attrib.get('xyz'), None if limit is None else limit.attrib.get('lower'), None if limit is None else limit.attrib.get('upper'))

def fb_validate_legacy_qpos_compatibility(combined_path: Path) -> None:
    combined = ET.parse(combined_path).getroot()
    legacy = ET.parse(fb_LEGACY_HAND_URDF).getroot()
    for name in fb_THREE_FINGER_JOINT_NAMES:
        if fb_joint_signature(combined, name) != fb_joint_signature(legacy, name):
            raise ValueError(f'{combined_path} is not legacy-qpos-compatible at {name}; use the combined xArm7 + Wuji-left URDF whose hand joint definitions match the retargeting URDF')

class fb_SequentialArmIKSolver:

    def __init__(self, robot: pk.Robot, palm_link_index: int, lower: np.ndarray, upper: np.ndarray, arm_indices: np.ndarray, max_iterations: int, arm_rest_weight: float=0.0) -> None:
        self.robot = robot
        self.palm_link_index = int(palm_link_index)
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.arm_indices = np.asarray(arm_indices, dtype=np.int32)
        self.max_iterations = int(max_iterations)
        self.arm_rest_weight = float(arm_rest_weight)
        if self.arm_rest_weight < 0.0:
            raise ValueError('arm_rest_weight must be nonnegative')
        self._build()

    def _build(self) -> None:
        num_joints = self.robot.joints.num_actuated_joints

        class TargetVar(jaxls.Var[jax.Array], default_factory=lambda : jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...

        class PreviousVar(jaxls.Var[jax.Array], default_factory=lambda : jnp.zeros((num_joints,), dtype=jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...

        class RestVar(jaxls.Var[jax.Array], default_factory=lambda : jnp.zeros((num_joints,), dtype=jnp.float32), tangent_dim=0, retract_fn=lambda value, delta: value):
            ...
        joint_var = self.robot.joint_var_cls(0)
        target_var = TargetVar(0)
        previous_var = PreviousVar(0)
        rest_var = RestVar(0)
        robot = self.robot
        palm_link_index = self.palm_link_index
        arm_mask = np.zeros(num_joints, dtype=np.float32)
        arm_mask[self.arm_indices] = 1.0
        seed_weights = jnp.asarray(arm_mask * 0.08 + (1.0 - arm_mask) * 80.0, dtype=jnp.float32)
        rest_weights = jnp.asarray(arm_mask * self.arm_rest_weight, dtype=jnp.float32)

        @jaxls.Cost.factory
        def palm_pose_cost(values: jaxls.VarValues, var_q: Any, var_target: Any) -> jax.Array:
            predicted = jaxlie.SE3(robot.forward_kinematics(cfg=values[var_q])[palm_link_index])
            target = jaxlie.SE3(values[var_target])
            error = (predicted.inverse() @ target).log()
            return jnp.concatenate([error[:3] * 80.0, error[3:] * 8.0])

        @jaxls.Cost.factory
        def seed_cost(values: jaxls.VarValues, var_q: Any, var_previous: Any) -> jax.Array:
            return (values[var_q] - values[var_previous]) * seed_weights

        @jaxls.Cost.factory
        def rest_cost(values: jaxls.VarValues, var_q: Any, var_rest: Any) -> jax.Array:
            return (values[var_q] - values[var_rest]) * rest_weights
        self.joint_var = joint_var
        self.target_var = target_var
        self.previous_var = previous_var
        self.rest_var = rest_var
        self.problem = jaxls.LeastSquaresProblem(costs=[palm_pose_cost(joint_var, target_var), seed_cost(joint_var, previous_var), rest_cost(joint_var, rest_var), pk.costs.limit_constraint(robot, joint_var)], variables=[joint_var, target_var, previous_var, rest_var]).analyze(use_onp=True)

        def fk_one(qpos: jax.Array) -> jax.Array:
            return jaxlie.SE3(robot.forward_kinematics(cfg=qpos)[palm_link_index]).as_matrix()
        self._fk_batch = jax.jit(jax.vmap(fk_one))

    def solve_frame(self, target: np.ndarray, seed: np.ndarray, rest: np.ndarray | None=None) -> np.ndarray:
        if rest is None:
            rest = seed
        values = jaxls.VarValues.make([self.joint_var.with_value(jnp.asarray(seed, dtype=jnp.float32)), self.target_var.with_value(jnp.asarray(fb_matrix_to_wxyz_xyz(target), dtype=jnp.float32)), self.previous_var.with_value(jnp.asarray(seed, dtype=jnp.float32)), self.rest_var.with_value(jnp.asarray(rest, dtype=jnp.float32))])
        solution = self.problem.solve(initial_vals=values, verbose=False, linear_solver='dense_cholesky', trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0), termination=jaxls.TerminationConfig(max_iterations=self.max_iterations))
        qpos = np.asarray(solution[self.joint_var], dtype=np.float32)
        return np.clip(qpos, self.lower, self.upper)

    def fk_batch(self, qpos: np.ndarray) -> np.ndarray:
        result = self._fk_batch(jnp.asarray(qpos, dtype=jnp.float32))
        jax.block_until_ready(result)
        return np.asarray(result, dtype=np.float64)

def fb_nominal_arm_postures() -> list[tuple[str, np.ndarray]]:
    return [('center', np.asarray([0.0, -0.65, 0.0, 1.35, 0.0, 0.7, 0.0])), ('left_bias', np.asarray([0.45, -0.8, 0.2, 1.5, 0.0, 0.75, -0.35])), ('right_bias', np.asarray([-0.45, -0.8, -0.2, 1.5, 0.0, 0.75, 0.35])), ('compact', np.asarray([0.0, -0.35, 0.0, 1.05, 0.0, 0.55, 0.0]))]

def fb_make_full_hand_targets(merged: fb_MergedTrajectory, joint_names: list[str], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    natural = np.clip(np.zeros(len(joint_names), dtype=np.float64), lower, upper)
    output = np.tile(natural[None, :], (len(merged.timestamps), 1))
    for (column, name) in enumerate(merged.hand_qpos_joint_names):
        output[:, joint_names.index(name)] = merged.hand_qpos[:, column]
    return np.clip(output, lower[None, :], upper[None, :])

def fb_solve_path_given_extrinsic(solver: fb_SequentialArmIKSolver, camera_palm: np.ndarray, base_from_camera: np.ndarray, hand_targets: np.ndarray, arm_indices: np.ndarray, nominal_arm: np.ndarray, initial_path: np.ndarray | None=None, anchor_index: int | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = np.einsum('ij,tjk->tik', base_from_camera, camera_palm)
    frame_count = len(targets)
    anchor = frame_count // 2 if anchor_index is None else int(anchor_index)
    if not 0 <= anchor < frame_count:
        raise ValueError(f'anchor_index is outside [0, {frame_count}): {anchor}')
    if initial_path is None:
        qpos = hand_targets.copy()
        qpos[:, arm_indices] = nominal_arm[None, :]
    else:
        qpos = np.asarray(initial_path, dtype=np.float64).copy()
        qpos[:, len(fb_ARM_JOINT_NAMES):] = hand_targets[:, len(fb_ARM_JOINT_NAMES):]
    rest = hand_targets[anchor].copy()
    rest[arm_indices] = nominal_arm
    qpos[anchor] = solver.solve_frame(targets[anchor], qpos[anchor], rest)
    qpos[anchor, len(fb_ARM_JOINT_NAMES):] = hand_targets[anchor, len(fb_ARM_JOINT_NAMES):]
    for frame_index in range(anchor + 1, frame_count):
        seed = qpos[frame_index - 1].copy()
        seed[len(fb_ARM_JOINT_NAMES):] = hand_targets[frame_index, len(fb_ARM_JOINT_NAMES):]
        rest = hand_targets[frame_index].copy()
        rest[arm_indices] = nominal_arm
        qpos[frame_index] = solver.solve_frame(targets[frame_index], seed, rest)
        qpos[frame_index, len(fb_ARM_JOINT_NAMES):] = hand_targets[frame_index, len(fb_ARM_JOINT_NAMES):]
    for frame_index in range(anchor - 1, -1, -1):
        seed = qpos[frame_index + 1].copy()
        seed[len(fb_ARM_JOINT_NAMES):] = hand_targets[frame_index, len(fb_ARM_JOINT_NAMES):]
        rest = hand_targets[frame_index].copy()
        rest[arm_indices] = nominal_arm
        qpos[frame_index] = solver.solve_frame(targets[frame_index], seed, rest)
        qpos[frame_index, len(fb_ARM_JOINT_NAMES):] = hand_targets[frame_index, len(fb_ARM_JOINT_NAMES):]
    fk = solver.fk_batch(qpos)
    return (qpos, fk, targets)

def fb_fit_base_from_camera(reference: np.ndarray, camera_palm: np.ndarray, fk_base_palm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:

    def delta_transform(parameters: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        transform[:3, 3] = parameters[3:]
        return transform

    def residual(parameters: np.ndarray) -> np.ndarray:
        candidate = delta_transform(parameters) @ reference
        target = np.einsum('ij,tjk->tik', candidate, camera_palm)
        position = (target[:, :3, 3] - fk_base_palm[:, :3, 3]) * 80.0
        rotation = np.stack([Rotation.from_matrix(fk_base_palm[index, :3, :3].T @ target[index, :3, :3]).as_rotvec() for index in range(len(target))], axis=0) * 8.0
        prior = np.concatenate([parameters[:3] / 0.35, parameters[3:] / 0.12]) * 0.1
        return np.concatenate([position.reshape(-1), rotation.reshape(-1), prior])
    rotation_bound = np.deg2rad(25.0)
    result = least_squares(residual, np.zeros(6, dtype=np.float64), bounds=(np.asarray([-rotation_bound] * 3 + [-0.12] * 3), np.asarray([rotation_bound] * 3 + [0.12] * 3)), loss='soft_l1', f_scale=0.5, max_nfev=80)
    fitted = delta_transform(result.x) @ reference
    return (fitted, {'success': bool(result.success), 'message': str(result.message), 'nfev': int(result.nfev), 'delta_rotation_deg': np.rad2deg(result.x[:3]).tolist(), 'delta_translation_m': result.x[3:].tolist(), 'cost': float(result.cost)})

def fb_trajectory_metrics(qpos: np.ndarray, fk: np.ndarray, target: np.ndarray, timestamps: np.ndarray, arm_indices: np.ndarray, nominal_arm: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    position_error = np.linalg.norm(fk[:, :3, 3] - target[:, :3, 3], axis=-1)
    orientation_error = np.asarray([np.linalg.norm(Rotation.from_matrix(fk[index, :3, :3].T @ target[index, :3, :3]).as_rotvec()) for index in range(len(fk))], dtype=np.float64)
    dt = np.diff(timestamps)
    arm = qpos[:, arm_indices]
    velocity = np.diff(arm, axis=0) / dt[:, None]
    acceleration = np.diff(velocity, axis=0) / (0.5 * (dt[1:] + dt[:-1])[:, None])
    margin = np.minimum(arm - lower[arm_indices], upper[arm_indices] - arm)
    rest_rms = float(np.sqrt(np.mean((arm - nominal_arm[None, :]) ** 2)))
    position_rmse_m = float(np.sqrt(np.mean(position_error ** 2)))
    orientation_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(orientation_error ** 2))))
    velocity_rms = float(np.sqrt(np.mean(velocity ** 2)))
    acceleration_rms = float(np.sqrt(np.mean(acceleration ** 2)))
    min_margin = float(np.min(margin))
    score = position_rmse_m * 1000.0 + orientation_rmse_deg * 0.05 + velocity_rms * 0.1 + acceleration_rms * 0.002 + rest_rms * 0.2 + max(0.0, 0.08 - min_margin) * 20.0
    return {'selection_score': float(score), 'position_error_m': fb_stats(position_error), 'position_rmse_m': position_rmse_m, 'orientation_error_deg': fb_stats(np.rad2deg(orientation_error)), 'orientation_rmse_deg': orientation_rmse_deg, 'arm_velocity_rms_rad_s': velocity_rms, 'arm_acceleration_rms_rad_s2': acceleration_rms, 'arm_rest_deviation_rms_rad': rest_rms, 'minimum_joint_limit_margin_rad': min_margin}

def fb_solve_virtual_base_ik(merged: fb_MergedTrajectory, urdf_path: Path, max_iterations: int, outer_iterations: int, nominal_postures: list[tuple[str, np.ndarray]] | None=None, anchor_index: int | None=None, arm_rest_weight: float=0.0) -> tuple[fb_CandidateResult, list[dict[str, Any]], list[str], np.ndarray, np.ndarray]:
    fb_validate_legacy_qpos_compatibility(urdf_path)
    urdf = fb_load_urdf(urdf_path)
    joint_names = list(urdf.actuated_joint_names)
    if joint_names[:7] != list(fb_ARM_JOINT_NAMES):
        raise ValueError(f'Unexpected xArm joint order: {joint_names[:7]}')
    lower = np.asarray([float(urdf.joint_map[name].limit.lower) for name in joint_names], dtype=np.float64)
    upper = np.asarray([float(urdf.joint_map[name].limit.upper) for name in joint_names], dtype=np.float64)
    arm_indices = np.asarray([joint_names.index(name) for name in fb_ARM_JOINT_NAMES])
    hand_targets = fb_make_full_hand_targets(merged, joint_names, lower, upper)
    default_qpos = np.clip(np.zeros(len(joint_names), dtype=np.float32), lower, upper)
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=jnp.asarray(default_qpos))
    palm_link_index = robot.links.names.index('left_palm_link')
    solver = fb_SequentialArmIKSolver(robot, palm_link_index, lower, upper, arm_indices, max_iterations, arm_rest_weight)
    anchor = len(merged.timestamps) // 2 if anchor_index is None else int(anchor_index)
    if not 0 <= anchor < len(merged.timestamps):
        raise ValueError(f'anchor_index is outside [0, {len(merged.timestamps)}): {anchor}')
    candidates: list[fb_CandidateResult] = []
    candidate_reports: list[dict[str, Any]] = []
    postures = fb_nominal_arm_postures() if nominal_postures is None else nominal_postures
    if not postures:
        raise ValueError('At least one nominal arm posture is required')
    for (name, nominal_arm_raw) in postures:
        nominal_arm = np.clip(nominal_arm_raw, lower[arm_indices] + 0.0001, upper[arm_indices] - 0.0001)
        nominal_full = hand_targets[anchor].copy()
        nominal_full[arm_indices] = nominal_arm
        nominal_fk = solver.fk_batch(nominal_full[None, :])[0]
        base_from_camera = nominal_fk @ np.linalg.inv(merged.poses_camera_palm[anchor])
        qpos: np.ndarray | None = None
        fit_history: list[dict[str, Any]] = []
        for outer in range(max(outer_iterations, 1)):
            (qpos, fk, target) = fb_solve_path_given_extrinsic(solver, merged.poses_camera_palm, base_from_camera, hand_targets, arm_indices, nominal_arm, qpos, anchor)
            if outer + 1 < max(outer_iterations, 1):
                (base_from_camera, fit_report) = fb_fit_base_from_camera(base_from_camera, merged.poses_camera_palm, fk)
                fit_history.append({'outer_iteration': outer + 1, **fit_report})
        assert qpos is not None
        fk = solver.fk_batch(qpos)
        target = np.einsum('ij,tjk->tik', base_from_camera, merged.poses_camera_palm)
        metrics = fb_trajectory_metrics(qpos, fk, target, merged.timestamps, arm_indices, nominal_arm, lower, upper)
        metrics['fit_history'] = fit_history
        candidate = fb_CandidateResult(name=name, base_from_camera=base_from_camera, qpos=qpos, fk_base_palm=fk, target_base_palm=target, metrics=metrics)
        candidates.append(candidate)
        candidate_reports.append({'name': name, 'nominal_arm_qpos': nominal_arm.tolist(), 'anchor_frame_index': anchor, 'arm_rest_weight': float(arm_rest_weight), 'T_link_base_rs_camera': base_from_camera.tolist(), **metrics})
        print(f"[INFO] candidate={name} score={metrics['selection_score']:.4f} pos_rmse={metrics['position_rmse_m'] * 1000.0:.3f}mm ori_rmse={metrics['orientation_rmse_deg']:.3f}deg")
    best = min(candidates, key=lambda item: float(item.metrics['selection_score']))
    print(f'[INFO] Selected virtual extrinsic candidate: {best.name}')
    return (best, candidate_reports, joint_names, lower, upper)

def fb_postprocess_qpos_for_hardware(result: fb_CandidateResult, merged: fb_MergedTrajectory, urdf_path: Path, joint_names: list[str], *, arm_cutoff_hz: float, hand_cutoff_hz: float, velocity_limit_fraction: float, acceleration_limit_rad_s2: float, jerk_limit_rad_s3: float, joint_limit_margin_rad: float) -> fb_HardwareSafeResult:
    """Low-pass qpos and uniformly slow the path to satisfy hardware limits."""
    if not 0.0 < velocity_limit_fraction < 1.0:
        raise ValueError('velocity_limit_fraction must lie in (0, 1)')
    if acceleration_limit_rad_s2 <= 0.0:
        raise ValueError('acceleration_limit_rad_s2 must be positive')
    if jerk_limit_rad_s3 <= 0.0:
        raise ValueError('jerk_limit_rad_s3 must be positive')
    if joint_limit_margin_rad < 0.0:
        raise ValueError('joint_limit_margin_rad must be nonnegative')
    source_fps = float((len(merged.timestamps) - 1) / (merged.timestamps[-1] - merged.timestamps[0]))
    nyquist = 0.5 * source_fps
    if not 0.0 < arm_cutoff_hz < nyquist:
        raise ValueError(f'arm_cutoff_hz must lie in (0, {nyquist})')
    if not 0.0 < hand_cutoff_hz < nyquist:
        raise ValueError(f'hand_cutoff_hz must lie in (0, {nyquist})')
    urdf = fb_load_urdf(urdf_path)
    lower = np.asarray([float(urdf.joint_map[name].limit.lower) for name in joint_names], dtype=np.float64)
    upper = np.asarray([float(urdf.joint_map[name].limit.upper) for name in joint_names], dtype=np.float64)
    velocity_limits = np.asarray([float(urdf.joint_map[name].limit.velocity) for name in joint_names], dtype=np.float64)
    safe_lower = lower + joint_limit_margin_rad
    safe_upper = upper - joint_limit_margin_rad
    if np.any(safe_lower >= safe_upper):
        raise ValueError('joint_limit_margin_rad leaves an empty joint range')
    raw_qpos = np.asarray(result.qpos, dtype=np.float64)
    safe_qpos = raw_qpos.copy()
    safe_qpos[:, :7] = sosfiltfilt(butter(2, arm_cutoff_hz, fs=source_fps, output='sos'), raw_qpos[:, :7], axis=0)
    safe_qpos[:, 7:19] = sosfiltfilt(butter(2, hand_cutoff_hz, fs=source_fps, output='sos'), raw_qpos[:, 7:19], axis=0)
    safe_qpos = np.clip(safe_qpos, safe_lower[None, :], safe_upper[None, :])
    source_dt = 1.0 / source_fps
    source_velocity = np.diff(safe_qpos, axis=0) / source_dt
    source_acceleration = np.diff(source_velocity, axis=0) / source_dt
    source_jerk = np.diff(source_acceleration, axis=0) / source_dt
    velocity_scale = float(np.max(np.abs(source_velocity) / (velocity_limit_fraction * velocity_limits[None, :])))
    acceleration_scale = float(np.sqrt(np.max(np.abs(source_acceleration)) / acceleration_limit_rad_s2))
    jerk_scale = float(np.cbrt(np.max(np.abs(source_jerk)) / jerk_limit_rad_s3))
    time_scale = max(1.0, velocity_scale, acceleration_scale, jerk_scale) * 1.001
    safe_dt = source_dt * time_scale
    safe_fps = 1.0 / safe_dt
    trajectory_time_s = np.arange(len(safe_qpos), dtype=np.float64) * safe_dt
    safe_velocity = np.diff(safe_qpos, axis=0) / safe_dt
    safe_acceleration = np.diff(safe_velocity, axis=0) / safe_dt
    safe_jerk = np.diff(safe_acceleration, axis=0) / safe_dt
    fk_values: list[np.ndarray] = []
    for row in safe_qpos:
        urdf.update_cfg(dict(zip(joint_names, row, strict=True)))
        fk_values.append(urdf.get_transform('left_palm_link', 'link_base'))
    fk_base_palm = np.asarray(fk_values, dtype=np.float64)
    position_error = np.linalg.norm(fk_base_palm[:, :3, 3] - result.target_base_palm[:, :3, 3], axis=-1)
    orientation_error = np.asarray([np.linalg.norm(Rotation.from_matrix(fk_base_palm[index, :3, :3].T @ result.target_base_palm[index, :3, :3]).as_rotvec()) for index in range(len(fk_base_palm))], dtype=np.float64)
    qpos_adjustment_deg = np.rad2deg(np.abs(safe_qpos - raw_qpos))
    velocity_ratio = np.abs(safe_velocity) / velocity_limits[None, :]
    limit_margin = np.minimum(safe_qpos - lower[None, :], upper[None, :] - safe_qpos)
    per_joint_max_velocity = {name: float(np.max(np.abs(safe_velocity[:, index]))) for (index, name) in enumerate(joint_names)}
    metrics = {'schema': 'consens.xarm7_wuji.hardware_safe_postprocess.v1', 'method': 'zero_phase_butterworth_qpos_then_uniform_velocity_acceleration_jerk_time_scaling', 'filter_order': 2, 'arm_cutoff_hz_on_source_timeline': float(arm_cutoff_hz), 'three_finger_cutoff_hz_on_source_timeline': float(hand_cutoff_hz), 'ring_little_policy': 'unchanged zero posture', 'source_fps': source_fps, 'source_duration_s': float(merged.timestamps[-1] - merged.timestamps[0]), 'time_scale': time_scale, 'safe_playback_fps': safe_fps, 'safe_duration_s': float(trajectory_time_s[-1]), 'velocity_limit_fraction': float(velocity_limit_fraction), 'acceleration_limit_rad_s2': float(acceleration_limit_rad_s2), 'jerk_limit_rad_s3': float(jerk_limit_rad_s3), 'joint_limit_margin_rad': float(joint_limit_margin_rad), 'maximum_velocity_limit_ratio': float(np.max(velocity_ratio)), 'velocity_violation_count': int(np.sum(velocity_ratio > 1.0 + 1e-07)), 'maximum_acceleration_rad_s2': float(np.max(np.abs(safe_acceleration))), 'acceleration_violation_count': int(np.sum(np.abs(safe_acceleration) > acceleration_limit_rad_s2 + 1e-07)), 'maximum_jerk_rad_s3': float(np.max(np.abs(safe_jerk))), 'jerk_violation_count': int(np.sum(np.abs(safe_jerk) > jerk_limit_rad_s3 + 1e-07)), 'per_joint_max_velocity_rad_s': per_joint_max_velocity, 'minimum_joint_limit_margin_rad': float(np.min(limit_margin)), 'joint_limit_violation_count': int(np.sum((safe_qpos < lower[None, :] - 1e-07) | (safe_qpos > upper[None, :] + 1e-07))), 'arm_qpos_adjustment_deg': fb_stats(qpos_adjustment_deg[:, :7]), 'three_finger_qpos_adjustment_deg': fb_stats(qpos_adjustment_deg[:, 7:19]), 'safe_palm_position_error_m': fb_stats(position_error), 'safe_palm_position_rmse_m': float(np.sqrt(np.mean(position_error ** 2))), 'safe_palm_orientation_error_deg': fb_stats(np.rad2deg(orientation_error)), 'safe_palm_orientation_rmse_deg': float(np.rad2deg(np.sqrt(np.mean(orientation_error ** 2))))}
    return fb_HardwareSafeResult(qpos=safe_qpos, fk_base_palm=fk_base_palm, trajectory_time_s=trajectory_time_s, playback_fps=safe_fps, metrics=metrics)

def fb_write_ik_pkl(path: Path, merged_path: Path, merged: fb_MergedTrajectory, result: fb_CandidateResult, safe: fb_HardwareSafeResult, candidates: list[dict[str, Any]], urdf_path: Path, joint_names: list[str], overwrite: bool) -> None:
    if path.exists() and (not overwrite):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    position_error = np.linalg.norm(safe.fk_base_palm[:, :3, 3] - result.target_base_palm[:, :3, 3], axis=-1)
    orientation_error = np.asarray([np.linalg.norm(Rotation.from_matrix(safe.fk_base_palm[index, :3, :3].T @ result.target_base_palm[index, :3, :3]).as_rotvec()) for index in range(len(result.qpos))], dtype=np.float64)
    header = {'type': 'header', 'format': fb_IK_FORMAT, 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_merged_pkl': str(merged_path), 'urdf': str(urdf_path), 'frame_convention': 'A_T_B maps coordinates from frame B into frame A', 'translation_unit': 'm', 'qpos_unit': 'rad', 'qpos_policy': 'hardware-safe filtered qpos on a uniformly time-scaled trajectory', 'qpos_joint_names': joint_names, 'qpos_dimension': len(joint_names), 'arm_joint_names': list(fb_ARM_JOINT_NAMES), 'three_finger_joint_names': merged.hand_qpos_joint_names, 'inactive_ring_little_policy': 'zero clipped to URDF joint limits', 'virtual_extrinsic_warning': 'T_link_base_rs_camera is gauge-selected for reachability/continuity and Viser; it is not a physical camera-to-robot calibration', 'selected_candidate': result.name, 'T_link_base_rs_camera': result.base_from_camera, 'T_left_palm_link_hand_back_cube': merged.palm_from_hand_back, 'selection_metrics': result.metrics, 'candidate_reports': candidates, 'hardware_safety': safe.metrics}
    try:
        with temporary.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame_index in range(len(result.qpos)):
                pickle.dump({'type': 'frame', 'frame_index': frame_index, 'rs_capture_timestamp': float(merged.timestamps[frame_index]), 'trajectory_time_s': float(safe.trajectory_time_s[frame_index]), 'normalized_phase': float(merged.phase[frame_index]), 'qpos': safe.qpos[frame_index].astype(np.float32), 'raw_ik_qpos': result.qpos[frame_index].astype(np.float32), 'arm_qpos': safe.qpos[frame_index, :7].astype(np.float32), 'wujihand_qpos': safe.qpos[frame_index, 7:].astype(np.float32), 'wujihand_three_finger_qpos': safe.qpos[frame_index, 7:19].astype(np.float32), 'source_interpolated_three_finger_qpos': merged.hand_qpos[frame_index].astype(np.float32), 'T_rs_camera_hand_back_cube': merged.poses_camera_hand_back[frame_index], 'T_hand_back_cube_obj': merged.poses_hand_back_obj[frame_index], 'T_rs_camera_obj': merged.poses_camera_obj[frame_index], 'target_T_link_base_left_palm_link': result.target_base_palm[frame_index], 'fk_T_link_base_left_palm_link': safe.fk_base_palm[frame_index], 'raw_ik_fk_T_link_base_left_palm_link': result.fk_base_palm[frame_index], 'palm_position_error_m': float(position_error[frame_index]), 'palm_orientation_error_deg': float(np.rad2deg(orientation_error[frame_index])), 'rs_pose_quality': merged.rs_quality[frame_index], 'hand_pose_quality': merged.hand_quality[frame_index]}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump({'type': 'footer', 'frame_count': len(result.qpos), 'safe_playback_fps': float(safe.playback_fps), 'safe_duration_s': float(safe.trajectory_time_s[-1]), 'position_error_m': fb_stats(position_error), 'orientation_error_deg': fb_stats(np.rad2deg(orientation_error))}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def fb_load_ik_pkl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    with path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('format') != fb_IK_FORMAT:
            raise ValueError(f"Unsupported IK PKL format: {header.get('format')}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if isinstance(record, dict) and record.get('type') == 'frame':
                frames.append(record)
    return (header, frames)

def fb_run_viser(path: Path, host: str, port: int, fps: float) -> None:
    (header, frames) = fb_load_ik_pkl(path)
    urdf_path = Path(header['urdf'])
    urdf = fb_load_urdf(urdf_path)
    safe_fps = float(header.get('hardware_safety', {}).get('safe_playback_fps', fps))
    server = viser.ViserServer(host=host, port=port)
    print(f'[INFO] Viser URL: http://127.0.0.1:{server.get_port()}')
    server.scene.set_up_direction('+z')
    server.scene.world_axes.visible = True
    server.gui.set_panel_label('Hardware-safe RS + xArm7 + Wuji IK')
    server.initial_camera.position = (1.1, -1.1, 0.9)
    server.initial_camera.look_at = (0.25, 0.0, 0.45)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    server.scene.add_grid('/ground', width=2.0, height=2.0, plane='xy', cell_size=0.05, section_size=0.25)
    urdf_vis = ViserUrdf(server, urdf, root_node_name='/robot')
    base_from_camera = np.asarray(header['T_link_base_rs_camera'], dtype=np.float64)
    server.scene.add_frame('/rs_camera', position=base_from_camera[:3, 3], wxyz=fb_matrix_to_wxyz(base_from_camera[:3, :3]), axes_length=0.12, axes_radius=0.003)
    target_palm = server.scene.add_frame('/targets/left_palm_link', axes_length=0.08, axes_radius=0.002)
    actual_palm = server.scene.add_frame('/fk/left_palm_link', axes_length=0.07, axes_radius=0.0015)
    hand_back = server.scene.add_frame('/targets/hand_back_cube', axes_length=0.07, axes_radius=0.002)
    server.scene.add_box('/targets/hand_back_cube/box', dimensions=(0.0625, 0.0625, 0.0625), color=(80, 120, 255), opacity=0.18)
    finger_frames = {name: server.scene.add_frame(f'/targets/fingers/{name}', axes_length=0.035, axes_radius=0.0012) for name in fb_FINGER_NAMES}
    with server.gui.add_folder('Playback'):
        slider = server.gui.add_slider('Frame', min=0, max=len(frames) - 1, step=1, initial_value=0)
        autoplay = server.gui.add_checkbox('Auto play', initial_value=False)
        loop = server.gui.add_checkbox('Loop', initial_value=True)
        speed = server.gui.add_slider('FPS', min=1.0, max=30.0, step=0.1, initial_value=max(min(float(fps), safe_fps), 1.0))
        server.gui.add_markdown(f'Hardware-safe real-time playback: `{safe_fps:.3f} FPS`. Higher GUI speeds are visualization-only.')
    with server.gui.add_folder('Virtual camera-to-base transform'):
        server.gui.add_markdown('This is an optimized Viser gauge, not physical calibration.\n\n```text\n' + np.array2string(base_from_camera, precision=8, suppress_small=True) + '\n```')
    with server.gui.add_folder('Current frame'):
        frame_text = server.gui.add_markdown('')
    rendered = -1
    last_step = time.monotonic()

    def render(index: int) -> None:
        nonlocal rendered
        record = frames[index]
        urdf_vis.update_cfg(np.asarray(record['qpos'], dtype=np.float64))
        target = np.asarray(record['target_T_link_base_left_palm_link'])
        actual = np.asarray(record['fk_T_link_base_left_palm_link'])
        cube = base_from_camera @ np.asarray(record['T_rs_camera_hand_back_cube'])
        target_palm.position = target[:3, 3]
        target_palm.wxyz = fb_matrix_to_wxyz(target[:3, :3])
        actual_palm.position = actual[:3, 3]
        actual_palm.wxyz = fb_matrix_to_wxyz(actual[:3, :3])
        hand_back.position = cube[:3, 3]
        hand_back.wxyz = fb_matrix_to_wxyz(cube[:3, :3])
        for (finger_index, name) in enumerate(fb_FINGER_NAMES):
            pose = base_from_camera @ np.asarray(record['T_rs_camera_obj'])[finger_index]
            finger_frames[name].position = pose[:3, 3]
            finger_frames[name].wxyz = fb_matrix_to_wxyz(pose[:3, :3])
        frame_text.content = f"Frame `{index}/{len(frames) - 1}`  \nSafe trajectory time: `{record['trajectory_time_s']:.3f} s`  \nPalm position error: `{record['palm_position_error_m'] * 1000.0:.3f} mm`  \nPalm orientation error: `{record['palm_orientation_error_deg']:.3f} deg`"
        rendered = index
    while True:
        selected = int(slider.value)
        if selected != rendered:
            render(selected)
        if bool(autoplay.value):
            now = time.monotonic()
            if now - last_step >= 1.0 / max(float(speed.value), 1.0):
                next_frame = rendered + 1
                if next_frame >= len(frames):
                    if bool(loop.value):
                        next_frame = 0
                    else:
                        next_frame = len(frames) - 1
                        autoplay.value = False
                slider.value = next_frame
                last_step = now
        else:
            last_step = time.monotonic()
        time.sleep(0.01)

def fb_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hand-pkl', type=Path, default=fb_DEFAULT_HAND_PKL)
    parser.add_argument('--rs-pkl', type=Path, default=fb_DEFAULT_RS_PKL)
    parser.add_argument('--urdf', type=Path, default=fb_DEFAULT_URDF)
    parser.add_argument('--merged-pkl', type=Path, default=fb_DEFAULT_MERGED_PKL)
    parser.add_argument('--ik-pkl', type=Path, default=fb_DEFAULT_IK_PKL)
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--solve-only', action='store_true')
    parser.add_argument('--viser-only', action='store_true', help='Replay an existing --ik-pkl without rebuilding or solving it.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--ik-iterations', type=int, default=60)
    parser.add_argument('--outer-iterations', type=int, default=2)
    parser.add_argument('--arm-smoothing-cutoff-hz', type=float, default=4.0)
    parser.add_argument('--hand-smoothing-cutoff-hz', type=float, default=2.0)
    parser.add_argument('--velocity-limit-fraction', type=float, default=0.75)
    parser.add_argument('--acceleration-limit-rad-s2', type=float, default=20.0)
    parser.add_argument('--jerk-limit-rad-s3', type=float, default=60.0)
    parser.add_argument('--joint-limit-margin-rad', type=float, default=0.005)
    parser.add_argument('--viser', action='store_true')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8127)
    parser.add_argument('--fps', type=float, default=12.0)
    return parser.parse_args()

def fb_main() -> None:
    args = fb_parse_args()
    mode_count = sum((bool(value) for value in (args.merge_only, args.solve_only, args.viser_only)))
    if mode_count > 1:
        raise ValueError('--merge-only, --solve-only, and --viser-only are mutually exclusive')
    hand_path = args.hand_pkl.expanduser().resolve()
    rs_path = args.rs_pkl.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    merged_path = args.merged_pkl.expanduser().resolve()
    ik_path = args.ik_pkl.expanduser().resolve()
    if args.viser_only:
        if not ik_path.is_file():
            raise FileNotFoundError(ik_path)
        fb_run_viser(ik_path, str(args.host), int(args.port), float(args.fps))
        return
    if not args.solve_only:
        for path in (hand_path, rs_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        print('[INFO] Loading hand trajectory')
        hand = fb_load_hand_trajectory(hand_path)
        print('[INFO] Loading RS trajectory')
        rs = fb_load_rs_trajectory(rs_path)
        merged = fb_build_merged_trajectory(hand, rs)
        fb_write_merged_pkl(merged_path, merged, hand_path, rs_path, bool(args.overwrite))
        print(f'[INFO] Saved merged PKL: {merged_path} frames={len(merged.timestamps)} size={merged_path.stat().st_size / 1024 ** 2:.2f}MiB')
        if args.merge_only:
            return
    else:
        if not merged_path.is_file():
            raise FileNotFoundError(merged_path)
        merged = fb_load_merged_pkl(merged_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    (result, candidates, joint_names, _, _) = fb_solve_virtual_base_ik(merged, urdf_path, max(int(args.ik_iterations), 1), max(int(args.outer_iterations), 1))
    safe = fb_postprocess_qpos_for_hardware(result, merged, urdf_path, joint_names, arm_cutoff_hz=float(args.arm_smoothing_cutoff_hz), hand_cutoff_hz=float(args.hand_smoothing_cutoff_hz), velocity_limit_fraction=float(args.velocity_limit_fraction), acceleration_limit_rad_s2=float(args.acceleration_limit_rad_s2), jerk_limit_rad_s3=float(args.jerk_limit_rad_s3), joint_limit_margin_rad=float(args.joint_limit_margin_rad))
    fb_write_ik_pkl(ik_path, merged_path, merged, result, safe, candidates, urdf_path, joint_names, bool(args.overwrite))
    print(f'[INFO] Saved IK PKL: {ik_path} frames={len(result.qpos)} size={ik_path.stat().st_size / 1024 ** 2:.2f}MiB')
    print('[RESULT] T_link_base_rs_camera (virtual Viser gauge):')
    print(np.array2string(result.base_from_camera, precision=9, suppress_small=True))
    print('[RESULT] metrics:')
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print('[RESULT] hardware-safe postprocess:')
    print(json.dumps(safe.metrics, indent=2, ensure_ascii=False))
    if args.viser:
        fb_run_viser(ik_path, str(args.host), int(args.port), float(args.fps))

full_body = SimpleNamespace(
    DEFAULT_URDF=fb_DEFAULT_URDF,
    FINGER_NAMES=fb_FINGER_NAMES,
    MERGED_FORMAT=fb_MERGED_FORMAT,
    MergedTrajectory=fb_MergedTrajectory,
    PALM_CUBE_KEY=fb_PALM_CUBE_KEY,
    THREE_FINGER_JOINT_NAMES=fb_THREE_FINGER_JOINT_NAMES,
    normalized_time=fb_normalized_time,
    postprocess_qpos_for_hardware=fb_postprocess_qpos_for_hardware,
    solve_virtual_base_ik=fb_solve_virtual_base_ik,
    validate_transform=fb_validate_transform,
    write_ik_pkl=fb_write_ik_pkl,
    write_merged_pkl=fb_write_merged_pkl,
)

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
solve_FILE_PATH = Path(__file__).resolve()
solve_REPO_ROOT = APRILCUBE_ROOT.parent.parent
solve_DEFAULT_RAW = solve_REPO_ROOT / 'recordings/multi_cam_record_0717_010151.pkl'
solve_DEFAULT_SIDECAR = solve_REPO_ROOT / 'recordings/multi_cam_record_0717_010151_020_pose_sidecar.pkl'
solve_DEFAULT_RETARGET_NPZ = solve_REPO_ROOT / 'outputs/retargeting/multi_cam_record_0717_010151/three_finger_joint_temporal_optimal/three_finger_se3.npz'
solve_DEFAULT_RETARGET_PKL = solve_DEFAULT_RETARGET_NPZ.with_suffix('.pkl')
solve_DEFAULT_OUTPUT_DIR = solve_REPO_ROOT / 'outputs/retargeting/multi_cam_record_0717_010151/xarm7_wuji_left_optimal'
solve_DEFAULT_MERGED = solve_DEFAULT_OUTPUT_DIR / 'full_body_merged.pkl'
solve_DEFAULT_IK = solve_DEFAULT_OUTPUT_DIR / 'xarm7_wuji_left_full_qpos.pkl'
solve_RAW_FORMAT = 'consens_multi_camera_sync_stream'
solve_SIDECAR_FORMAT = 'consensv2_multi_cam_020_pose_sidecar_v1'
solve_RETARGET_FORMAT = 'consens.left_wuji_three_finger_retarget.compact.v2'
solve_DEFAULT_NOMINAL_ARM_DEG = (0.0, -59.1, 0.0, 29.1, 0.0, 89.7, 0.0)

def solve_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-pkl', type=Path, default=solve_DEFAULT_RAW)
    parser.add_argument('--sidecar-pkl', type=Path, default=solve_DEFAULT_SIDECAR)
    parser.add_argument('--retarget-npz', type=Path, default=solve_DEFAULT_RETARGET_NPZ)
    parser.add_argument('--retarget-pkl', type=Path, default=solve_DEFAULT_RETARGET_PKL)
    parser.add_argument('--urdf', type=Path, default=full_body.DEFAULT_URDF)
    parser.add_argument('--merged-pkl', type=Path, default=solve_DEFAULT_MERGED)
    parser.add_argument('--ik-pkl', type=Path, default=solve_DEFAULT_IK)
    parser.add_argument('--ik-iterations', type=int, default=60)
    parser.add_argument('--outer-iterations', type=int, default=2)
    parser.add_argument('--arm-smoothing-cutoff-hz', type=float, default=4.0)
    parser.add_argument('--hand-smoothing-cutoff-hz', type=float, default=2.0)
    parser.add_argument('--velocity-limit-fraction', type=float, default=0.75)
    parser.add_argument('--acceleration-limit-rad-s2', type=float, default=20.0)
    parser.add_argument('--jerk-limit-rad-s3', type=float, default=60.0)
    parser.add_argument('--joint-limit-margin-rad', type=float, default=0.005)
    parser.add_argument('--nominal-arm-deg', type=float, nargs=7, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'), default=solve_DEFAULT_NOMINAL_ARM_DEG, help='Desired center posture used to place the virtual RS trajectory.')
    parser.add_argument('--anchor-frame', type=int, default=-1, help='Frame mapped to the nominal arm posture; negative selects an SE(3) medoid.')
    parser.add_argument('--arm-rest-weight', type=float, default=0.15, help='Light arm-joint centering weight around --nominal-arm-deg.')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def solve_load_raw_header(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        header = pickle.load(stream)
    if not isinstance(header, dict) or header.get('format') != solve_RAW_FORMAT:
        raise ValueError(f"Unsupported raw PKL format: {header.get('format')}")
    return header

def solve_load_retarget(npz_path: Path, pkl_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        finger_names = [str(value) for value in archive['finger_names']]
        timestamps = np.asarray(archive['timestamps'], dtype=np.float64)
        poses_hand_back_obj = np.asarray(archive['T_hand_back_cube_obj'], dtype=np.float64)
    if finger_names != list(full_body.FINGER_NAMES):
        raise ValueError(f'Unexpected retarget finger order: {finger_names}')
    qpos: list[np.ndarray] = []
    frame_indices: list[int] = []
    with pkl_path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('format') != solve_RETARGET_FORMAT:
            raise ValueError(f"Unsupported retarget PKL format: {header.get('format')}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if isinstance(record, dict) and record.get('type') == 'frame':
                frame_indices.append(int(record['frame_index']))
                qpos.append(np.asarray(record['wujihand_qpos'], dtype=np.float64))
    expected_indices = list(range(len(qpos)))
    if frame_indices != expected_indices:
        raise ValueError('Retarget frame indices are not contiguous from zero')
    hand_qpos = np.asarray(qpos, dtype=np.float64)
    frame_count = len(timestamps)
    if poses_hand_back_obj.shape != (frame_count, 3, 4, 4):
        raise ValueError(f'Unexpected T_hand_back_cube_obj shape: {poses_hand_back_obj.shape}')
    if hand_qpos.shape != (frame_count, len(full_body.THREE_FINGER_JOINT_NAMES)):
        raise ValueError(f'Unexpected hand qpos shape: {hand_qpos.shape}')
    return (header, timestamps, poses_hand_back_obj, hand_qpos)

def solve_load_wrist_poses(path: Path, raw_path: Path, frame_count: int) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    poses: list[np.ndarray] = []
    quality: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get('format') != solve_SIDECAR_FORMAT:
            raise ValueError(f"Unsupported sidecar format: {header.get('format')}")
        final_smoothing = header.get('metadata', {}).get('final_global_smoothing', {})
        required_targets = {'wrist_Q', 'index_Q', 'thumb_Q', 'middle_Q'}
        applied_counts = final_smoothing.get('applied_counts', {}) or {}
        if not bool(final_smoothing.get('complete', False)) or not bool(final_smoothing.get('completion_barrier_passed', False)) or int(final_smoothing.get('frame_count', -1)) != frame_count or (set(final_smoothing.get('targets', [])) != required_targets) or any((int(applied_counts.get(name, -1)) != frame_count for name in required_targets)):
            raise ValueError('Full-body IK requires a complete stage13 globally-smoothed multi-camera sidecar')
        raw_stat = raw_path.stat()
        expected_identity = {'size': int(raw_stat.st_size), 'mtime_ns': int(raw_stat.st_mtime_ns)}
        if header.get('source_multi_cam_identity') != expected_identity:
            raise ValueError('Sidecar source identity does not match raw PKL')
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get('type') == 'footer':
                footer = record
                continue
            if record.get('type') not in ('frame', 'pose_frame'):
                continue
            frame_index = len(poses)
            if int(record.get('sample_index', frame_index)) != frame_index:
                raise ValueError('Sidecar sample indices are not contiguous from zero')
            result = record.get('pose_results', {}).get('wrist_Q', {})
            pose = result.get('pose', {}) or {}
            if not pose.get('success', False) or pose.get('T') is None:
                raise ValueError(f'Sidecar wrist_Q frame {frame_index} has no pose')
            if not bool(pose.get('final_global_smoothing_applied', False)):
                raise ValueError(f'Sidecar wrist_Q frame {frame_index} bypassed mandatory final global smoothing')
            transform = full_body.validate_transform(pose['T'], f'sidecar wrist_Q frame {frame_index}').copy()
            transform[:3, 3] /= 1000.0
            poses.append(transform)
            quality.append({'pose_source': str(pose.get('pose_source', '')), 'quality_level': str(pose.get('quality_level', '')), 'quality_reason': str(pose.get('quality_reason', '')), 'pose_filled': bool(pose.get('pose_filled', False)), 'reproj_error_px': float(pose.get('reproj_error', float('nan'))), 'temporal_smoothed': bool(pose.get('temporal_smoothed', False))})
    if footer is None:
        raise ValueError('Sidecar is missing its footer')
    if len(poses) != frame_count:
        raise ValueError(f'Sidecar/retarget frame count mismatch: {len(poses)} != {frame_count}')
    required_targets = {'wrist_Q', 'index_Q', 'thumb_Q', 'middle_Q'}
    success_counts = footer.get('success_counts', {}) or {}
    applied_counts = footer.get('final_global_smoothing_applied_counts', {}) or {}
    if not bool(footer.get('final_global_smoothing_complete', False)) or any((int(success_counts.get(name, -1)) != frame_count for name in required_targets)) or any((int(applied_counts.get(name, -1)) != frame_count for name in required_targets)):
        raise ValueError('Sidecar footer does not certify complete stage13 smoothing for every target/frame')
    return (header, np.asarray(poses, dtype=np.float64), quality)

def solve_build_merged(raw_path: Path, sidecar_path: Path, retarget_npz_path: Path, retarget_pkl_path: Path) -> full_body.MergedTrajectory:
    solve_load_raw_header(raw_path)
    (retarget_header, timestamps, poses_hand_back_obj, hand_qpos) = solve_load_retarget(retarget_npz_path, retarget_pkl_path)
    (sidecar_header, poses_camera_hand_back, rs_quality) = solve_load_wrist_poses(sidecar_path, raw_path, len(timestamps))
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError('Retarget timestamps must be strictly increasing')
    palm_from_hand_back = full_body.validate_transform(retarget_header[full_body.PALM_CUBE_KEY], f'retarget header {full_body.PALM_CUBE_KEY}')
    poses_camera_obj = np.einsum('tij,tfjk->tfik', poses_camera_hand_back, poses_hand_back_obj)
    poses_camera_palm = np.einsum('tij,jk->tik', poses_camera_hand_back, np.linalg.inv(palm_from_hand_back))
    frame_count = len(timestamps)
    source_indices = np.arange(frame_count, dtype=np.int32)
    source_validation = retarget_header.get('source_validation', {})
    expected_sidecar = Path(str(source_validation.get('pose_sidecar', ''))).resolve()
    if expected_sidecar != sidecar_path.resolve():
        raise ValueError(f'Retarget references a different sidecar: {expected_sidecar}')
    metadata = {'type': 'header', 'format': full_body.MERGED_FORMAT, 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'frame_convention': 'A_T_B maps coordinates from frame B into frame A', 'translation_unit': 'm', 'rotation_storage': '4x4 rotation matrix', 'master_timeline': 'synchronized multi-camera sample timeline', 'temporal_alignment': 'exact sample_index correspondence; no interpolation', 'output_frame_count': frame_count, 'output_fps': float((frame_count - 1) / (timestamps[-1] - timestamps[0])), 'hand_qpos_joint_names': list(full_body.THREE_FINGER_JOINT_NAMES), 'T_left_palm_link_hand_back_cube': palm_from_hand_back, 'source_raw_pkl': str(raw_path), 'source_sidecar_pkl': str(sidecar_path), 'source_retarget_npz': str(retarget_npz_path), 'source_retarget_pkl': str(retarget_pkl_path), 'source_sidecar_created_wall_time': sidecar_header.get('created_wall_time')}
    hand_quality = [{'source_frame_index': int(index), 'exact_sample_match': True} for index in source_indices]
    return full_body.MergedTrajectory(header=metadata, timestamps=timestamps, phase=full_body.normalized_time(timestamps), poses_camera_hand_back=poses_camera_hand_back, poses_hand_back_obj=poses_hand_back_obj, poses_camera_obj=poses_camera_obj, poses_camera_palm=poses_camera_palm, hand_qpos=hand_qpos, hand_qpos_joint_names=list(full_body.THREE_FINGER_JOINT_NAMES), palm_from_hand_back=palm_from_hand_back, hand_bracket_indices=np.stack([source_indices, source_indices], axis=-1), hand_interpolation_alpha=np.zeros(frame_count, dtype=np.float64), rs_quality=rs_quality, hand_quality=hand_quality)

def solve_choose_se3_medoid(poses: np.ndarray) -> int:
    """Return the observed palm pose closest to the trajectory's SE(3) center."""
    poses = np.asarray(poses, dtype=np.float64)
    translations = poses[:, :3, 3]
    position_distance = np.linalg.norm(translations[:, None, :] - translations[None, :, :], axis=-1)
    relative_rotations = np.einsum('aji,bjk->abik', poses[:, :3, :3], poses[:, :3, :3])
    rotation_distance = Rotation.from_matrix(relative_rotations.reshape(-1, 3, 3)).magnitude().reshape(len(poses), len(poses))
    distance = position_distance / 0.1 + rotation_distance / 0.5
    return int(np.argmin(np.sum(distance, axis=1)))

def solve_main() -> None:
    args = solve_parse_args()
    raw_path = args.raw_pkl.expanduser().resolve()
    sidecar_path = args.sidecar_pkl.expanduser().resolve()
    retarget_npz_path = args.retarget_npz.expanduser().resolve()
    retarget_pkl_path = args.retarget_pkl.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    merged_path = args.merged_pkl.expanduser().resolve()
    ik_path = args.ik_pkl.expanduser().resolve()
    for path in (raw_path, sidecar_path, retarget_npz_path, retarget_pkl_path, urdf_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    print('[INFO] Loading synchronized multi-camera retarget')
    merged = solve_build_merged(raw_path, sidecar_path, retarget_npz_path, retarget_pkl_path)
    full_body.write_merged_pkl(merged_path, merged, retarget_pkl_path, sidecar_path, bool(args.overwrite))
    nominal_arm_deg = np.asarray(args.nominal_arm_deg, dtype=np.float64)
    nominal_arm = np.deg2rad(nominal_arm_deg)
    anchor_index = int(args.anchor_frame)
    if anchor_index < 0:
        anchor_index = solve_choose_se3_medoid(merged.poses_camera_palm)
    if not 0 <= anchor_index < len(merged.timestamps):
        raise ValueError(f'--anchor-frame is outside [0, {len(merged.timestamps)}): {anchor_index}')
    print(f'[INFO] Solving {len(merged.timestamps)} full-body IK frames')
    print(f'[INFO] Virtual gauge center: anchor_frame={anchor_index} nominal_arm_deg={nominal_arm_deg.tolist()} arm_rest_weight={float(args.arm_rest_weight):.4f}')
    (result, candidates, joint_names, _, _) = full_body.solve_virtual_base_ik(merged, urdf_path, max(int(args.ik_iterations), 1), max(int(args.outer_iterations), 1), nominal_postures=[('requested_initial_position', nominal_arm)], anchor_index=anchor_index, arm_rest_weight=float(args.arm_rest_weight))
    safe = full_body.postprocess_qpos_for_hardware(result, merged, urdf_path, joint_names, arm_cutoff_hz=float(args.arm_smoothing_cutoff_hz), hand_cutoff_hz=float(args.hand_smoothing_cutoff_hz), velocity_limit_fraction=float(args.velocity_limit_fraction), acceleration_limit_rad_s2=float(args.acceleration_limit_rad_s2), jerk_limit_rad_s3=float(args.jerk_limit_rad_s3), joint_limit_margin_rad=float(args.joint_limit_margin_rad))
    full_body.write_ik_pkl(ik_path, merged_path, merged, result, safe, candidates, urdf_path, joint_names, bool(args.overwrite))
    report = {'ik_pkl': str(ik_path), 'frames': len(merged.timestamps), 'selected_candidate': result.name, 'requested_nominal_arm_deg': nominal_arm_deg.tolist(), 'anchor_frame_index': anchor_index, 'arm_rest_weight': float(args.arm_rest_weight), 'T_link_base_rs_camera': result.base_from_camera.tolist(), 'selection_metrics': result.metrics, 'hardware_safety': safe.metrics}
    print(json.dumps(report, indent=2, ensure_ascii=False))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Embedded complete-pose recovery patches used before stage13.
#
# These are the exact middle temporal/flow, strict index, and wrist
# bidirectional-flow implementations that produced the frozen reference
# sidecar, namespace-prefixed and detached from their former script modules.
# -----------------------------------------------------------------------------
import argparse
import copy
import importlib.util
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
rgb_FILE_PATH = Path(__file__).resolve()
rgb_PROJECT_ROOT = APRILCUBE_ROOT.parent.parent
if str(rgb_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(rgb_PROJECT_ROOT))
rgb_FINALIZE_020_PATH = rgb_PROJECT_ROOT / 'thirdparty' / 'aprilcube' / 'src' / '020_finalize_pose_postprocess.py'
rgb_DEFAULT_SOURCE = rgb_PROJECT_ROOT / 'recordings' / 'multi_cam_record_0717_010151.pkl'
rgb_DEFAULT_SIDECAR = rgb_PROJECT_ROOT / 'recordings' / 'multi_cam_record_0717_010151_020_pose_sidecar.pkl'
rgb_DEFAULT_OUTPUT = rgb_PROJECT_ROOT / 'recordings' / 'multi_cam_record_0717_010151_020_pose_sidecar_middle_rgb_flow_candidate.pkl'
rgb_DEFAULT_QA_DIR = rgb_PROJECT_ROOT / 'recordings' / 'qa_multi_cam_record_0717_010151_020'
rgb_DEFAULT_TARGET_NAME = 'middle_Q'

def rgb_load_finalize_020() -> Any:
    spec = importlib.util.spec_from_file_location('middle_rgb_flow_finalize_020', rgb_FINALIZE_020_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(rgb_FINALIZE_020_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def rgb_load_sidecar(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as stream:
        header = pickle.load(stream)
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get('type') == 'frame':
                frames.append(record)
            elif record.get('type') == 'footer':
                footer = record
                break
    if not isinstance(header, dict) or header.get('type') != 'header':
        raise ValueError(f'Invalid sidecar header: {path}')
    if footer is None:
        raise ValueError(f'Sidecar has no complete footer: {path}')
    if int(footer.get('frame_count', -1)) != len(frames):
        raise ValueError('Sidecar footer frame_count mismatch')
    if any((int(frame.get('sample_index', -1)) != index for (index, frame) in enumerate(frames))):
        raise ValueError('Sidecar sample_index is not contiguous')
    return (header, frames, footer)

def rgb_load_source_samples(path: Path) -> Any:
    with path.open('rb') as stream:
        first_record = pickle.load(stream)
    if mcstream_is_stream_header(first_record):
        index = mcstream_scan_stream_index(path)
        if not index.complete:
            raise ValueError(f'Streaming source has no complete footer: {path}')
        return mcstream_StreamingSampleSequence(index)
    if not isinstance(first_record, dict) or not isinstance(first_record.get('samples'), list):
        raise ValueError(f'Unsupported source recording: {path}')
    return first_record['samples']

def rgb_detection_runtime(finalize020: Any, target_metadata: dict[str, Any]) -> dict[str, Any]:
    intrinsics = finalize020.realsense_load_intrinsics_yaml(target_metadata['intrinsics_yaml'])
    image_size = tuple((int(value) for value in target_metadata['image_size']))
    if tuple(intrinsics['image_size']) != image_size:
        raise ValueError(f"Target intrinsics/image-size mismatch: {intrinsics['image_size']} != {image_size}")
    undistort_pack = finalize020.realsense_create_undistort_maps(intrinsics, image_size)
    camera_matrix = np.asarray(undistort_pack[2], dtype=np.float64).reshape(3, 3) if undistort_pack is not None else np.asarray(intrinsics['K'], dtype=np.float64).reshape(3, 3)
    cube_cfg = Path(target_metadata['cube_cfg']).expanduser().resolve()
    (config, face_id_sets) = finalize020.aprilcube.load_cube_config(str(cube_cfg / 'config.json'))
    return {'intrinsics': intrinsics, 'image_size': image_size, 'undistort_pack': undistort_pack, 'camera_matrix': camera_matrix, 'dist_coeffs': np.zeros(5, dtype=np.float64), 'cube_cfg': cube_cfg, 'config': config, 'face_id_sets': face_id_sets, 'tag_corner_map': finalize020.aprilcube.build_tag_corner_map(config)}

def rgb_detection_frame(finalize020: Any, image: np.ndarray, runtime: dict[str, Any]) -> np.ndarray:
    (target_width, target_height) = runtime['image_size']
    if image.shape[:2] != (target_height, target_width):
        raise ValueError(f'Target raw image is {image.shape[1]}x{image.shape[0]}, but the selected calibration is {target_width}x{target_height}')
    return finalize020.realsense_undistort_frame(image, runtime['undistort_pack'])

def rgb_rotation_delta_deg(rvec_a: Any, rvec_b: Any) -> float:
    (rotation_a, _) = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
    (rotation_b, _) = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
    relative = rotation_b @ rotation_a.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))

def rgb_best_quad_corner_agreement(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4, 2)
    second = np.asarray(second, dtype=np.float64).reshape(4, 2)
    scores: list[float] = []
    for candidate in (second, second[::-1]):
        for shift in range(4):
            shifted = np.roll(candidate, shift, axis=0)
            scores.append(float(np.mean(np.linalg.norm(first - shifted, axis=1))))
    return min(scores)

def rgb_recover_pose(*, finalize020: Any, anchor_image: np.ndarray, target_image: np.ndarray, anchor_pose: dict[str, Any], runtime: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor_tag_ids = [int(value) for value in anchor_pose.get('tag_ids', []) or []]
    if len(anchor_tag_ids) != 1:
        raise ValueError(f'Anchor must contain exactly one tag, got {anchor_tag_ids}')
    tag_id = anchor_tag_ids[0]
    if tag_id not in runtime['tag_corner_map']:
        raise ValueError(f'Anchor tag {tag_id} is not in the target cube config')
    object_points = np.asarray(runtime['tag_corner_map'][tag_id], dtype=np.float64).reshape(4, 3)
    camera_matrix = runtime['camera_matrix']
    dist_coeffs = runtime['dist_coeffs']
    anchor_rvec = np.asarray(anchor_pose['rvec'], dtype=np.float64).reshape(3, 1)
    anchor_tvec = np.asarray(anchor_pose['tvec'], dtype=np.float64).reshape(3, 1)
    (anchor_quad, _) = cv2.projectPoints(object_points, anchor_rvec, anchor_tvec, camera_matrix, dist_coeffs)
    anchor_quad = anchor_quad.reshape(4, 2).astype(np.float32)
    anchor_gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    feature_mask = np.zeros_like(anchor_gray)
    feature_mask_scale = float(getattr(args, 'feature_mask_scale', 1.0))
    if not 0.0 < feature_mask_scale <= 1.0:
        raise ValueError(f'feature_mask_scale must be in (0, 1], got {feature_mask_scale}')
    feature_center = anchor_quad.mean(axis=0, keepdims=True)
    feature_quad = feature_center + feature_mask_scale * (anchor_quad - feature_center)
    cv2.fillConvexPoly(feature_mask, np.rint(feature_quad).astype(np.int32), 255)
    anchor_points = cv2.goodFeaturesToTrack(anchor_gray, maxCorners=int(args.max_features), qualityLevel=float(args.feature_quality), minDistance=float(args.feature_min_distance), mask=feature_mask, blockSize=5)
    if anchor_points is None or len(anchor_points) < int(args.min_features):
        count = 0 if anchor_points is None else len(anchor_points)
        raise RuntimeError(f'Not enough anchor features: {count} < {args.min_features}')
    lk_parameters = {'winSize': (int(args.lk_window), int(args.lk_window)), 'maxLevel': int(args.lk_levels), 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)}
    (target_points, forward_status, _forward_error) = cv2.calcOpticalFlowPyrLK(anchor_gray, target_gray, anchor_points, None, **lk_parameters)
    if target_points is None or forward_status is None:
        raise RuntimeError('Forward LK optical flow failed')
    (backward_points, backward_status, _backward_error) = cv2.calcOpticalFlowPyrLK(target_gray, anchor_gray, target_points, None, **lk_parameters)
    if backward_points is None or backward_status is None:
        raise RuntimeError('Backward LK optical flow failed')
    anchor_xy = anchor_points.reshape(-1, 2)
    target_xy = target_points.reshape(-1, 2)
    backward_xy = backward_points.reshape(-1, 2)
    fb_error = np.linalg.norm(anchor_xy - backward_xy, axis=1)
    finite_points = np.isfinite(target_xy).all(axis=1) & np.isfinite(fb_error)
    good = (forward_status.reshape(-1) > 0) & (backward_status.reshape(-1) > 0) & finite_points & (fb_error <= float(args.max_fb_error))
    good_count = int(good.sum())
    if good_count < int(args.min_good_tracks):
        raise RuntimeError(f'Not enough FB-consistent tracks: {good_count} < {args.min_good_tracks}')
    (homography, homography_mask) = cv2.findHomography(anchor_xy[good], target_xy[good], cv2.RANSAC, float(args.homography_ransac_px))
    if homography is None or homography_mask is None:
        raise RuntimeError('RANSAC homography failed')
    inlier_mask = homography_mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / max(good_count, 1))
    if inlier_count < int(args.min_homography_inliers):
        raise RuntimeError(f'Not enough homography inliers: {inlier_count} < {args.min_homography_inliers}')
    if inlier_ratio < float(args.min_homography_inlier_ratio):
        raise RuntimeError(f'Homography inlier ratio too small: {inlier_ratio:.3f} < {args.min_homography_inlier_ratio:.3f}')
    predicted_tracks = cv2.perspectiveTransform(anchor_xy[good].reshape(-1, 1, 2), homography).reshape(-1, 2)
    homography_errors = np.linalg.norm(predicted_tracks - target_xy[good], axis=1)
    homography_median_px = float(np.median(homography_errors[inlier_mask]))
    fb_median_px = float(np.median(fb_error[good]))
    if homography_median_px > float(args.max_homography_median_px):
        raise RuntimeError(f'Homography residual too high: {homography_median_px:.3f} > {args.max_homography_median_px:.3f}')
    if fb_median_px > float(args.max_fb_median_px):
        raise RuntimeError(f'FB residual too high: {fb_median_px:.3f} > {args.max_fb_median_px:.3f}')
    target_quad = cv2.perspectiveTransform(anchor_quad.reshape(-1, 1, 2), homography).reshape(4, 2)
    (height, width) = target_gray.shape[:2]
    corners_inside = int(np.sum((target_quad[:, 0] >= 0.0) & (target_quad[:, 0] < width) & (target_quad[:, 1] >= 0.0) & (target_quad[:, 1] < height)))
    min_corners_inside = int(getattr(args, 'min_tag_corners_inside', 4))
    if not 0 <= min_corners_inside <= 4:
        raise ValueError(f'min_tag_corners_inside must be in [0, 4], got {min_corners_inside}')
    if corners_inside < min_corners_inside:
        raise RuntimeError(f'Propagated tag has too few in-image corners: {corners_inside}/4 < {min_corners_inside}/4')
    detected = [] if bool(getattr(args, 'skip_current_tag_detection', False)) else finalize020.pose_recovery_detect_sweep(target_gray, config=runtime['config'], valid_ids=set((int(value) for value in runtime['tag_corner_map'])))
    detected_by_id = {int(detected_id): np.asarray(corners, dtype=np.float64).reshape(4, 2) for (detected_id, corners) in detected}
    current_tag_detected = tag_id in detected_by_id
    if not current_tag_detected and (not bool(args.allow_missing_current_tag)):
        raise RuntimeError(f'Current frame did not independently decode anchor tag {tag_id}')
    current_tag_agreement_px = rgb_best_quad_corner_agreement(target_quad, detected_by_id[tag_id]) if current_tag_detected else float('nan')
    if current_tag_detected and current_tag_agreement_px > float(args.max_current_tag_agreement_px):
        raise RuntimeError(f'Current-frame tag/flow disagreement too high: {current_tag_agreement_px:.2f} > {args.max_current_tag_agreement_px:.2f} px')
    (ok, rvec, tvec) = cv2.solvePnP(object_points, target_quad.astype(np.float64), camera_matrix, dist_coeffs, anchor_rvec.copy(), anchor_tvec.copy(), True, cv2.SOLVEPNP_ITERATIVE)
    if not ok or float(np.asarray(tvec).reshape(3)[2]) <= 0.0:
        raise RuntimeError('Tracked-corner PnP failed')
    try:
        (rvec, tvec) = cv2.solvePnPRefineLM(object_points, target_quad.astype(np.float64), camera_matrix, dist_coeffs, rvec, tvec)
    except cv2.error:
        pass
    (projected_quad, _) = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    corner_errors = np.linalg.norm(projected_quad.reshape(4, 2) - target_quad, axis=1)
    flow_corner_reproj_px = float(np.mean(corner_errors))
    translation_delta_mm = float(np.linalg.norm(np.asarray(tvec).reshape(3) - anchor_tvec.reshape(3)))
    rotation_delta = rgb_rotation_delta_deg(anchor_rvec, rvec)
    provisional_pose = {'success': True, 'rvec': rvec, 'tvec': tvec}
    edge_score = float(finalize020.pose_recovery_edge_alignment_score(target_gray, provisional_pose, config=runtime['config'], camera_matrix=camera_matrix, dist_coeffs=dist_coeffs))
    gates = {'flow_corner_reproj_px': (flow_corner_reproj_px, float(args.max_flow_corner_reproj_px)), 'translation_delta_mm': (translation_delta_mm, float(args.max_translation_delta_mm)), 'rotation_delta_deg': (rotation_delta, float(args.max_rotation_delta_deg))}
    for (name, (value, maximum)) in gates.items():
        if not math.isfinite(value) or value > maximum:
            raise RuntimeError(f'{name} too high: {value:.3f} > {maximum:.3f}')
    if edge_score < float(args.min_edge_score):
        raise RuntimeError(f'RGB cube-edge score too low: {edge_score:.3f} < {args.min_edge_score:.3f}')
    faces = sorted((str(face) for (face, ids) in runtime['face_id_sets'].items() if tag_id in {int(value) for value in ids}))
    quality_reason = f'anchor:{args.anchor_frame};tag:{tag_id};tracks:{good_count};inliers:{inlier_count}/{good_count};fb:{fb_median_px:.3f}px;H:{homography_median_px:.3f}px;flow_reproj:{flow_corner_reproj_px:.3f}px;current_tag_agreement:{current_tag_agreement_px:.2f}px;edge:{edge_score:.3f};dt:{translation_delta_mm:.3f}mm;dr:{rotation_delta:.3f}deg'
    pose = {'success': True, 'failure_reason': '', 'pose_source': 'stage12_backward_rgb_flow_tag_pnp', 'quality_level': 'T', 'quality_reason': quality_reason, 'pose_filled': True, 'predicted': True, 'temporal_recovery': True, 'single_frame_only': False, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': finalize020.pose_recovery_pose_transform(rvec, tvec), 'reproj_error': flow_corner_reproj_px, 'reproj_metric': 'optical_flow_propagated_tag_corner_mean_px', 'n_tags': 1, 'tag_ids': [tag_id], 'visible_faces': faces, 'edge_score': edge_score, 'flow_anchor_frame': int(args.anchor_frame), 'flow_target_frame': int(args.target_frame), 'flow_anchor_pose_source': str(anchor_pose.get('pose_source', '')), 'flow_anchor_tag_id': tag_id, 'flow_feature_count': int(len(anchor_points)), 'flow_good_track_count': good_count, 'flow_homography_inlier_count': inlier_count, 'flow_homography_inlier_ratio': inlier_ratio, 'flow_fb_median_px': fb_median_px, 'flow_homography_median_px': homography_median_px, 'flow_corner_reproj_error': flow_corner_reproj_px, 'flow_current_tag_corner_agreement_px': current_tag_agreement_px, 'flow_translation_delta_mm': translation_delta_mm, 'flow_rotation_delta_deg': rotation_delta, 'flow_target_tag_corners': target_quad.astype(np.float64), 'detected_tag_ids': sorted(detected_by_id), 'current_frame_tag_anchor_used': current_tag_detected}
    metrics = {'anchor_frame': int(args.anchor_frame), 'target_frame': int(args.target_frame), 'tag_id': tag_id, 'feature_count': int(len(anchor_points)), 'good_track_count': good_count, 'homography_inlier_count': inlier_count, 'homography_inlier_ratio': inlier_ratio, 'fb_median_px': fb_median_px, 'homography_median_px': homography_median_px, 'flow_corner_reproj_px': flow_corner_reproj_px, 'current_tag_corner_agreement_px': current_tag_agreement_px, 'edge_score': edge_score, 'translation_delta_mm': translation_delta_mm, 'rotation_delta_deg': rotation_delta, 'depth_mm': float(np.asarray(tvec).reshape(3)[2]), 'tag_corners_inside': corners_inside, 'detected_tag_ids': sorted(detected_by_id), 'current_tag_detected': current_tag_detected, 'quality_reason': quality_reason}
    return (pose, metrics)

def rgb_make_overlay(*, finalize020: Any, image: np.ndarray, pose: dict[str, Any], runtime: dict[str, Any], metrics: dict[str, Any], target_name: str) -> np.ndarray:
    output = image.copy()
    (corners, _) = cv2.projectPoints(finalize020.pose_recovery_cube_corners(runtime['config']), np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1), np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1), runtime['camera_matrix'], runtime['dist_coeffs'])
    corners = corners.reshape(-1, 2)
    for (first, second) in finalize020.POSE_RECOVERY_CUBE_EDGES:
        point_a = tuple((int(round(value)) for value in corners[first]))
        point_b = tuple((int(round(value)) for value in corners[second]))
        cv2.line(output, point_a, point_b, (255, 0, 255), 8, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 118), (0, 0, 0), -1)
    lines = [f"{target_name} frame {metrics['target_frame']}: backward RGB-flow recovery (predicted/filled)", f"inliers={metrics['homography_inlier_count']}/{metrics['good_track_count']} FB={metrics['fb_median_px']:.3f}px H={metrics['homography_median_px']:.3f}px flow-reproj={metrics['flow_corner_reproj_px']:.3f}px", f"edge={metrics['edge_score']:.3f} current-tag-agreement={metrics['current_tag_corner_agreement_px']:.2f}px dt={metrics['translation_delta_mm']:.3f}mm dr={metrics['rotation_delta_deg']:.3f}deg"]
    for (line_index, line) in enumerate(lines):
        cv2.putText(output, line, (18, 34 + line_index * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
    return output

def rgb_encode_jpeg(image: np.ndarray, quality: int=95) -> bytes:
    (ok, encoded) = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def rgb_recompute_footer(frames: list[dict[str, Any]], old_footer: dict[str, Any], *, recovered_frame: int, target_name: str) -> dict[str, Any]:
    target_names = list(frames[0].get('pose_results', {}))
    success_counts = {name: 0 for name in target_names}
    source_counts: dict[str, dict[str, int]] = {name: {} for name in target_names}
    for frame in frames:
        for name in target_names:
            pose = frame['pose_results'][name].get('pose', {}) or {}
            success_counts[name] += int(bool(pose.get('success', False)))
            source = str(pose.get('pose_source', ''))
            source_counts[name][source] = source_counts[name].get(source, 0) + 1
    reprocessed = [str(value) for value in old_footer.get('reprocessed_targets', []) or []]
    if target_name not in reprocessed:
        reprocessed.append(target_name)
    return {**copy.deepcopy(old_footer), 'type': 'footer', 'frame_count': len(frames), 'success_counts': success_counts, 'pose_source_counts': source_counts, 'reprocessed_targets': reprocessed, 'leading_rgb_flow_recovered_frames': {target_name: [int(recovered_frame)]}, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def rgb_write_sidecar_atomic(path: Path, header: dict[str, Any], frames: list[dict[str, Any]], footer: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame in frames:
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

def rgb_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def rgb_run(args: argparse.Namespace) -> None:
    source_path = args.source.expanduser().resolve()
    sidecar_path = args.sidecar.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    qa_dir = args.qa_dir.expanduser().resolve()
    target_name = str(args.target_name)
    if int(args.target_frame) != int(args.anchor_frame) - 1:
        raise ValueError('This conservative recovery supports only the frame immediately before anchor')
    finalize020 = rgb_load_finalize_020()
    (header, frames, old_footer) = rgb_load_sidecar(sidecar_path)
    samples = rgb_load_source_samples(source_path)
    if len(samples) != len(frames):
        raise ValueError(f'Source/sidecar frame mismatch: {len(samples)} != {len(frames)}')
    source_identity = header.get('source_multi_cam_identity', {}) or {}
    source_stat = source_path.stat()
    if int(source_identity.get('size', -1)) != int(source_stat.st_size):
        raise ValueError('Sidecar source identity does not match the raw PKL size')
    if int(source_identity.get('mtime_ns', -1)) != int(source_stat.st_mtime_ns):
        raise ValueError('Sidecar source identity does not match the raw PKL mtime')
    target_metadata = header['metadata']['targets'][target_name]
    runtime = rgb_detection_runtime(finalize020, target_metadata)
    anchor_result = frames[int(args.anchor_frame)]['pose_results'][target_name]
    target_result = frames[int(args.target_frame)]['pose_results'][target_name]
    anchor_pose = anchor_result.get('pose', {}) or {}
    target_old_pose = target_result.get('pose', {}) or {}
    if not bool(anchor_pose.get('success', False)) or bool(anchor_pose.get('pose_filled', False)):
        raise ValueError('Anchor pose must be a successful, directly measured pose')
    if bool(target_old_pose.get('success', False)):
        raise ValueError('Target frame already has a successful pose; refusing to overwrite it')
    worker_name = str(target_metadata['worker_name'])
    camera_name = str(target_metadata['camera_name'])
    anchor_raw = samples[int(args.anchor_frame)]['worker_raw_frames'][worker_name][camera_name]
    target_raw = samples[int(args.target_frame)]['worker_raw_frames'][worker_name][camera_name]
    anchor_image = rgb_detection_frame(finalize020, np.asarray(anchor_raw), runtime)
    target_image = rgb_detection_frame(finalize020, np.asarray(target_raw), runtime)
    (recovered_pose, metrics) = rgb_recover_pose(finalize020=finalize020, anchor_image=anchor_image, target_image=target_image, anchor_pose=anchor_pose, runtime=runtime, args=args)
    recovered_pose['fill_original_failure_reason'] = str(target_old_pose.get('failure_reason', ''))
    overlay = rgb_make_overlay(finalize020=finalize020, image=target_image, pose=recovered_pose, runtime=runtime, metrics=metrics, target_name=target_name)
    target_result['pose_before_backward_rgb_flow'] = copy.deepcopy(target_old_pose)
    target_result['pose'] = recovered_pose
    target_result['selected_stage'] = 'stage12_backward_rgb_flow_tag_pnp'
    target_result['overlay_shape'] = tuple((int(value) for value in overlay.shape))
    target_result['overlay_format'] = 'jpeg_bgr'
    target_result['overlay_jpeg'] = rgb_encode_jpeg(overlay)
    target_result.setdefault('pose_candidates', {})['stage12_backward_rgb_flow_tag_pnp'] = copy.deepcopy(recovered_pose)
    frames[int(args.target_frame)]['poses'][target_name] = copy.deepcopy(recovered_pose)
    output_header = copy.deepcopy(header)
    output_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    metadata = output_header.setdefault('metadata', {})
    update_history = list(metadata.get('update_history', []) or [])
    update_history.append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'script': str(rgb_FILE_PATH), 'method': 'one-frame backward RGB LK flow + homography + tag-plane PnP', 'target': target_name, 'recovered_frames': [int(args.target_frame)], 'anchor_frame': int(args.anchor_frame), 'pose_marking': {'predicted': True, 'pose_filled': True}, 'metrics': copy.deepcopy(metrics)})
    metadata['update_history'] = update_history
    recoveries = metadata.setdefault('leading_rgb_flow_recoveries', {})
    recoveries[target_name] = {'script': str(rgb_FILE_PATH), 'source_sidecar': str(sidecar_path), 'recovered_frames': [int(args.target_frame)], 'metrics': copy.deepcopy(metrics)}
    output_footer = rgb_recompute_footer(frames, old_footer, recovered_frame=int(args.target_frame), target_name=target_name)
    rgb_write_sidecar_atomic(output_path, output_header, frames, output_footer)
    (verified_header, verified_frames, verified_footer) = rgb_load_sidecar(output_path)
    verified_pose = verified_frames[int(args.target_frame)]['pose_results'][target_name]['pose']
    if not bool(verified_pose.get('success', False)):
        raise RuntimeError('Written sidecar did not retain the recovered pose')
    if verified_footer['success_counts'].get(target_name) != old_footer['success_counts'].get(target_name, 0) + 1:
        raise RuntimeError('Written sidecar success count did not increase by exactly one')
    if verified_header.get('source_multi_cam_identity') != header.get('source_multi_cam_identity'):
        raise RuntimeError('Written sidecar changed the raw source identity')
    qa_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = qa_dir / f'{target_name}_frame{int(args.target_frame)}_backward_rgb_flow_overlay.png'
    if not cv2.imwrite(str(diagnostic_path), overlay):
        raise RuntimeError(f'Could not write {diagnostic_path}')
    report = {'source_pkl': str(source_path), 'input_sidecar': str(sidecar_path), 'output_sidecar': str(output_path), 'target': target_name, 'status': 'accepted', 'recovered_frames': [int(args.target_frame)], 'remaining_failed_frames': list(range(0, int(args.target_frame))), 'pose_marking': {'predicted': True, 'pose_filled': True}, 'metrics': metrics, 'diagnostic_overlay': str(diagnostic_path), 'footer': verified_footer}
    report_path = qa_dir / f'{target_name}_leading_rgb_flow_recovery.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=rgb_json_default) + '\n', encoding='utf-8')
    print(f'[OK] recovered {target_name} frame {args.target_frame} from frame {args.anchor_frame}')
    print(f'[OK] output: {output_path}')
    print(f'[OK] diagnostic: {diagnostic_path}')
    print(f'[OK] report: {report_path}')
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=rgb_json_default))

def rgb_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=rgb_DEFAULT_SOURCE)
    parser.add_argument('--sidecar', type=Path, default=rgb_DEFAULT_SIDECAR)
    parser.add_argument('--output', type=Path, default=rgb_DEFAULT_OUTPUT)
    parser.add_argument('--qa-dir', type=Path, default=rgb_DEFAULT_QA_DIR)
    parser.add_argument('--target-name', default=rgb_DEFAULT_TARGET_NAME)
    parser.add_argument('--anchor-frame', type=int, default=1)
    parser.add_argument('--target-frame', type=int, default=0)
    parser.add_argument('--max-features', type=int, default=500)
    parser.add_argument('--feature-quality', type=float, default=0.005)
    parser.add_argument('--feature-min-distance', type=float, default=4.0)
    parser.add_argument('--min-features', type=int, default=300)
    parser.add_argument('--lk-window', type=int, default=41)
    parser.add_argument('--lk-levels', type=int, default=5)
    parser.add_argument('--max-fb-error', type=float, default=1.5)
    parser.add_argument('--max-fb-median-px', type=float, default=0.25)
    parser.add_argument('--min-good-tracks', type=int, default=250)
    parser.add_argument('--homography-ransac-px', type=float, default=2.5)
    parser.add_argument('--min-homography-inliers', type=int, default=200)
    parser.add_argument('--min-homography-inlier-ratio', type=float, default=0.7)
    parser.add_argument('--max-homography-median-px', type=float, default=1.0)
    parser.add_argument('--max-current-tag-agreement-px', type=float, default=80.0)
    parser.add_argument('--allow-missing-current-tag', action='store_true', help='Allow recovery when the target frame does not independently decode the anchor tag; all optical-flow, motion, PnP, and edge gates still apply.')
    parser.add_argument('--max-flow-corner-reproj-px', type=float, default=0.5)
    parser.add_argument('--max-translation-delta-mm', type=float, default=2.0)
    parser.add_argument('--max-rotation-delta-deg', type=float, default=5.0)
    parser.add_argument('--min-edge-score', type=float, default=0.04)
    return parser

rgb_flow = SimpleNamespace(
    detection_frame=rgb_detection_frame,
    detection_runtime=rgb_detection_runtime,
    load_finalize_020=rgb_load_finalize_020,
    load_sidecar=rgb_load_sidecar,
    load_source_samples=rgb_load_source_samples,
    recover_pose=rgb_recover_pose,
)

import argparse
import copy
import importlib.util
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
mid_FILE_PATH = Path(__file__).resolve()
mid_PROJECT_ROOT = APRILCUBE_ROOT.parent.parent
if str(mid_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(mid_PROJECT_ROOT))
mid_FINALIZE_020_PATH = mid_PROJECT_ROOT / 'thirdparty/aprilcube/src/020_finalize_pose_postprocess.py'
mid_RGB_FLOW_PATH = Path(__file__).resolve()
mid_DEFAULT_SOURCE = mid_PROJECT_ROOT / 'recordings/multi_cam_record_0717_010151.pkl'
mid_DEFAULT_SIDECAR = mid_PROJECT_ROOT / 'recordings/multi_cam_record_0717_010151_020_pose_sidecar.pkl'
mid_DEFAULT_OUTPUT = mid_PROJECT_ROOT / 'recordings/multi_cam_record_0717_010151_020_pose_sidecar_middle_candidate.pkl'
mid_DEFAULT_QA = mid_PROJECT_ROOT / 'outputs/diagnostics/middle_finger_cam_jitter_0717/reprocess_middle_q.json'

def mid_load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def mid_valid_pose(pose: dict[str, Any]) -> bool:
    if not bool(pose.get('success', False)):
        return False
    try:
        values = np.r_[np.asarray(pose['rvec'], dtype=np.float64).reshape(3), np.asarray(pose['tvec'], dtype=np.float64).reshape(3)]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))

def mid_rotation_delta_deg(first: Any, second: Any) -> float:
    first_rotation = Rotation.from_rotvec(np.asarray(first, dtype=np.float64).reshape(3))
    second_rotation = Rotation.from_rotvec(np.asarray(second, dtype=np.float64).reshape(3))
    return float(np.degrees((first_rotation.inv() * second_rotation).magnitude()))

def mid_interpolate_pose(before: dict[str, Any], after: dict[str, Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    before_t = np.asarray(before['tvec'], dtype=np.float64).reshape(3)
    after_t = np.asarray(after['tvec'], dtype=np.float64).reshape(3)
    tvec = ((1.0 - alpha) * before_t + alpha * after_t).reshape(3, 1)
    rotations = Rotation.from_rotvec(np.stack([np.asarray(before['rvec'], dtype=np.float64).reshape(3), np.asarray(after['rvec'], dtype=np.float64).reshape(3)]))
    rvec = Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0].reshape(3, 1)
    return (rvec, tvec)

def mid_frame_timestamp(frame: dict[str, Any], target_name: str) -> float:
    result = frame['pose_results'][target_name]
    value = result.get('capture_timestamp')
    if value is None:
        value = frame.get('time_monotonic')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite timestamp at sample {frame.get('sample_index')}")
    return value

def mid_failure_pose(reason: str, **metadata: Any) -> dict[str, Any]:
    return {'success': False, 'pose_source': 'fused_failed', 'quality_level': 'Z', 'quality_reason': reason, 'failure_reason': reason, 'pose_filled': False, 'single_frame_only': False, 'reproj_error': float('inf'), **metadata}

def mid_set_pose(frame: dict[str, Any], target_name: str, pose: dict[str, Any], stage: str) -> None:
    result = frame['pose_results'][target_name]
    result['pose'] = pose
    result['selected_stage'] = stage
    frame['poses'][target_name] = copy.deepcopy(pose)

def mid_reset_temporal_results(frames: list[dict[str, Any]], target_name: str) -> list[int]:
    removed: list[int] = []
    for (idx, frame) in enumerate(frames):
        result = frame['pose_results'][target_name]
        pose = result.get('pose', {}) or {}
        if bool(pose.get('temporal_smoothed', False)) and mid_valid_pose(result.get('pose_before_temporal_smoothing', {}) or {}):
            pose = copy.deepcopy(result['pose_before_temporal_smoothing'])
            mid_set_pose(frame, target_name, pose, result.get('selected_stage_before_temporal_smoothing', ''))
        source = str(pose.get('pose_source', ''))
        is_temporal = bool(pose.get('pose_filled', False)) or bool(pose.get('predicted', False)) or 'temporal' in source or ('rgb_flow' in source) or ('interpolation' in source)
        if is_temporal:
            result['pose_before_middle_q_reset'] = copy.deepcopy(pose)
            mid_set_pose(frame, target_name, mid_failure_pose('removed_old_temporal_pose'), 'middle_q_remove_old_temporal_pose')
            removed.append(idx)
    return removed

def mid_gate_stage8(frames: list[dict[str, Any]], target_name: str, *, max_neighbor_seconds: float, min_translation_threshold_mm: float, min_rotation_threshold_deg: float) -> dict[int, dict[str, float]]:
    poses = [frame['pose_results'][target_name].get('pose', {}) or {} for frame in frames]
    timestamps = [mid_frame_timestamp(frame, target_name) for frame in frames]
    valid_indices = [idx for (idx, pose) in enumerate(poses) if mid_valid_pose(pose)]
    rejected: dict[int, dict[str, float]] = {}
    for (position, idx) in enumerate(valid_indices):
        pose = poses[idx]
        if 'stage8_apriltag_single_tag_cfg_edge' not in str(pose.get('pose_source', '')):
            continue
        if position == 0 or position + 1 >= len(valid_indices):
            pose['stage8_temporal_consistency_verified'] = False
            pose['stage8_temporal_consistency_reason'] = 'missing_two_sided_neighbor'
            continue
        before_idx = valid_indices[position - 1]
        after_idx = valid_indices[position + 1]
        before_dt = timestamps[idx] - timestamps[before_idx]
        after_dt = timestamps[after_idx] - timestamps[idx]
        if before_dt <= 0.0 or after_dt <= 0.0 or before_dt > float(max_neighbor_seconds) or (after_dt > float(max_neighbor_seconds)):
            pose['stage8_temporal_consistency_verified'] = False
            pose['stage8_temporal_consistency_reason'] = 'neighbor_time_gap_too_large'
            continue
        alpha = (timestamps[idx] - timestamps[before_idx]) / (timestamps[after_idx] - timestamps[before_idx])
        (predicted_rvec, predicted_tvec) = mid_interpolate_pose(poses[before_idx], poses[after_idx], alpha)
        translation_residual = float(np.linalg.norm(np.asarray(pose['tvec'], dtype=np.float64).reshape(3) - predicted_tvec.reshape(3)))
        rotation_residual = mid_rotation_delta_deg(pose['rvec'], predicted_rvec)
        bracket_translation = float(np.linalg.norm(np.asarray(poses[before_idx]['tvec'], dtype=np.float64).reshape(3) - np.asarray(poses[after_idx]['tvec'], dtype=np.float64).reshape(3)))
        bracket_rotation = mid_rotation_delta_deg(poses[before_idx]['rvec'], poses[after_idx]['rvec'])
        translation_threshold = max(float(min_translation_threshold_mm), 0.75 * bracket_translation + 0.3)
        rotation_threshold = max(float(min_rotation_threshold_deg), 0.75 * bracket_rotation + 0.5)
        metrics = {'before_frame': int(before_idx), 'after_frame': int(after_idx), 'alpha': float(alpha), 'translation_residual_mm': translation_residual, 'rotation_residual_deg': rotation_residual, 'translation_threshold_mm': translation_threshold, 'rotation_threshold_deg': rotation_threshold}
        if translation_residual > translation_threshold or rotation_residual > rotation_threshold:
            rejected[idx] = metrics
        else:
            pose['stage8_temporal_consistency_verified'] = True
            pose['stage8_temporal_consistency_metrics'] = metrics
    for (idx, metrics) in rejected.items():
        result = frames[idx]['pose_results'][target_name]
        original = copy.deepcopy(result.get('pose', {}))
        result.setdefault('pose_candidates', {})['stage8_temporal_consistency_rejected'] = {**copy.deepcopy(original), **metrics}
        mid_set_pose(frames[idx], target_name, mid_failure_pose('stage8_temporal_inconsistent', stage8_temporal_consistency_metrics=metrics), 'stage8_temporal_consistency_rejected')
    return rejected

def mid_failed_runs(frames: list[dict[str, Any]], target_name: str) -> list[list[int]]:
    output: list[list[int]] = []
    current: list[int] = []
    for (idx, frame) in enumerate(frames):
        pose = frame['pose_results'][target_name].get('pose', {}) or {}
        if not mid_valid_pose(pose):
            current.append(idx)
        elif current:
            output.append(current)
            current = []
    if current:
        output.append(current)
    return output

def mid_flow_args(anchor_frame: int, target_frame: int, args: argparse.Namespace) -> Any:
    return SimpleNamespace(anchor_frame=int(anchor_frame), target_frame=int(target_frame), max_features=int(args.flow_max_features), feature_quality=float(args.flow_feature_quality), feature_min_distance=float(args.flow_feature_min_distance), min_features=int(args.flow_min_features), lk_window=int(args.flow_lk_window), lk_levels=int(args.flow_lk_levels), max_fb_error=float(args.flow_max_fb_error), max_fb_median_px=float(args.flow_max_fb_median_px), min_good_tracks=int(args.flow_min_good_tracks), homography_ransac_px=float(args.flow_homography_ransac_px), min_homography_inliers=int(args.flow_min_homography_inliers), min_homography_inlier_ratio=float(args.flow_min_homography_inlier_ratio), max_homography_median_px=float(args.flow_max_homography_median_px), max_current_tag_agreement_px=float(args.flow_max_current_tag_agreement_px), allow_missing_current_tag=True, max_flow_corner_reproj_px=float(args.flow_max_corner_reproj_px), max_translation_delta_mm=float(args.flow_max_translation_delta_mm), max_rotation_delta_deg=float(args.flow_max_rotation_delta_deg), min_edge_score=float(args.flow_min_edge_score), min_tag_corners_inside=int(args.flow_min_tag_corners_inside), skip_current_tag_detection=bool(args.flow_skip_current_tag_detection), feature_mask_scale=float(args.flow_feature_mask_scale))

def mid_recover_one_flow(*, finalize020: Any, rgb_flow: Any, samples: Any, frames: list[dict[str, Any]], runtime: dict[str, Any], target_name: str, preferred_tag_id: int, anchor_idx: int, target_idx: int, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor_pose = copy.deepcopy(frames[anchor_idx]['pose_results'][target_name].get('pose', {}) or {})
    if not mid_valid_pose(anchor_pose):
        raise RuntimeError('invalid anchor pose')
    anchor_tag_ids = [int(value) for value in anchor_pose.get('tag_ids', []) or []]
    if int(preferred_tag_id) in anchor_tag_ids:
        flow_tag_id = int(preferred_tag_id)
    elif anchor_tag_ids:
        flow_tag_id = int(anchor_tag_ids[0])
    else:
        raise RuntimeError('anchor pose has no measured tag id')
    anchor_pose['tag_ids'] = [flow_tag_id]
    worker_name = str(runtime['worker_name'])
    camera_name = str(runtime['camera_name'])
    anchor_raw = samples[anchor_idx]['worker_raw_frames'][worker_name][camera_name]
    target_raw = samples[target_idx]['worker_raw_frames'][worker_name][camera_name]
    anchor_image = rgb_flow.detection_frame(finalize020, np.asarray(anchor_raw), runtime)
    target_image = rgb_flow.detection_frame(finalize020, np.asarray(target_raw), runtime)
    (pose, metrics) = rgb_flow.recover_pose(finalize020=finalize020, anchor_image=anchor_image, target_image=target_image, anchor_pose=anchor_pose, runtime=runtime, args=mid_flow_args(anchor_idx, target_idx, args))
    pose['pose_source'] = 'stage10_adjacent_rgb_flow_tag_pnp'
    pose['quality_level'] = 'T'
    pose['flow_anchor_is_filled'] = bool(anchor_pose.get('pose_filled', False))
    return (pose, metrics)

def mid_recover_gap_edges_with_flow(*, finalize020: Any, rgb_flow: Any, samples: Any, frames: list[dict[str, Any]], runtime: dict[str, Any], target_name: str, preferred_tag_id: int, args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    recovered: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    for run in mid_failed_runs(frames, target_name):
        if run[0] == 0 and len(run) <= int(args.max_flow_gap_frames) and (run[-1] + 1 < len(frames)):
            right_anchor = run[-1] + 1
            for target_idx in reversed(run):
                try:
                    (pose, metrics) = mid_recover_one_flow(finalize020=finalize020, rgb_flow=rgb_flow, samples=samples, frames=frames, runtime=runtime, target_name=target_name, preferred_tag_id=preferred_tag_id, anchor_idx=right_anchor, target_idx=target_idx, args=args)
                    pose['flow_metrics'] = metrics
                    mid_set_pose(frames[target_idx], target_name, pose, 'stage10_adjacent_rgb_flow_tag_pnp')
                    recovered[target_idx] = metrics
                    right_anchor = target_idx
                except Exception as exc:
                    failures[target_idx] = f'leading_flow:{type(exc).__name__}:{exc}'
                    break
            continue
        if run[-1] + 1 == len(frames) and len(run) <= int(args.max_flow_gap_frames) and (run[0] > 0):
            left_anchor = run[0] - 1
            for target_idx in run:
                try:
                    (pose, metrics) = mid_recover_one_flow(finalize020=finalize020, rgb_flow=rgb_flow, samples=samples, frames=frames, runtime=runtime, target_name=target_name, preferred_tag_id=preferred_tag_id, anchor_idx=left_anchor, target_idx=target_idx, args=args)
                    pose['flow_metrics'] = metrics
                    mid_set_pose(frames[target_idx], target_name, pose, 'stage10_adjacent_rgb_flow_tag_pnp')
                    recovered[target_idx] = metrics
                    left_anchor = target_idx
                except Exception as exc:
                    failures[target_idx] = f'trailing_flow:{type(exc).__name__}:{exc}'
                    break
            continue
        if len(run) < 2 or len(run) > int(args.max_flow_gap_frames):
            continue
        if run[0] == 0 or run[-1] + 1 >= len(frames):
            continue
        left_anchor = run[0] - 1
        right_anchor = run[-1] + 1
        edge_count = len(run) // 2
        for target_idx in run[:edge_count]:
            try:
                (pose, metrics) = mid_recover_one_flow(finalize020=finalize020, rgb_flow=rgb_flow, samples=samples, frames=frames, runtime=runtime, target_name=target_name, preferred_tag_id=preferred_tag_id, anchor_idx=left_anchor, target_idx=target_idx, args=args)
                pose['flow_metrics'] = metrics
                mid_set_pose(frames[target_idx], target_name, pose, 'stage10_adjacent_rgb_flow_tag_pnp')
                recovered[target_idx] = metrics
                left_anchor = target_idx
            except Exception as exc:
                failures[target_idx] = f'left_flow:{type(exc).__name__}:{exc}'
                break
        for target_idx in reversed(run[-edge_count:]):
            try:
                (pose, metrics) = mid_recover_one_flow(finalize020=finalize020, rgb_flow=rgb_flow, samples=samples, frames=frames, runtime=runtime, target_name=target_name, preferred_tag_id=preferred_tag_id, anchor_idx=right_anchor, target_idx=target_idx, args=args)
                pose['flow_metrics'] = metrics
                mid_set_pose(frames[target_idx], target_name, pose, 'stage10_adjacent_rgb_flow_tag_pnp')
                recovered[target_idx] = metrics
                right_anchor = target_idx
            except Exception as exc:
                failures[target_idx] = f'right_flow:{type(exc).__name__}:{exc}'
                break
    return (recovered, failures)

def mid_fill_short_local_gaps(frames: list[dict[str, Any]], target_name: str, finalize020: Any, *, max_bracket_gap: int) -> list[int]:
    filled: list[int] = []
    timestamps = [mid_frame_timestamp(frame, target_name) for frame in frames]
    for run in mid_failed_runs(frames, target_name):
        if run[0] == 0 or run[-1] + 1 >= len(frames):
            continue
        before_idx = run[0] - 1
        after_idx = run[-1] + 1
        if after_idx - before_idx > int(max_bracket_gap):
            continue
        before_pose = frames[before_idx]['pose_results'][target_name]['pose']
        after_pose = frames[after_idx]['pose_results'][target_name]['pose']
        for idx in run:
            alpha = (timestamps[idx] - timestamps[before_idx]) / (timestamps[after_idx] - timestamps[before_idx])
            (rvec, tvec) = mid_interpolate_pose(before_pose, after_pose, alpha)
            pose = {'success': True, 'pose_source': 'stage11_local_timestamp_se3_interpolation', 'quality_level': 'F', 'quality_reason': f'local_timestamp_se3_interpolation;bracket:{before_idx}-{after_idx};alpha:{alpha:.6f}', 'pose_filled': True, 'predicted': True, 'local_temporal_interpolation': True, 'single_frame_only': False, 'rvec': rvec, 'tvec': tvec, 'T': finalize020.temporal_completion_make_pose_transform(rvec, tvec), 'reproj_error': float('nan'), 'n_tags': 0, 'tag_ids': [], 'visible_faces': [], 'prev_success_frame': int(before_idx), 'next_success_frame': int(after_idx), 'interpolation_alpha': float(alpha), 'interpolation_clock': 'capture_timestamp'}
            mid_set_pose(frames[idx], target_name, pose, 'stage11_local_timestamp_se3_interpolation')
            filled.append(idx)
    return filled

def mid_frame_image(finalize020: Any, rgb_flow: Any, samples: Any, runtime: dict[str, Any], idx: int) -> np.ndarray:
    raw = samples[idx]['worker_raw_frames'][runtime['worker_name']][runtime['camera_name']]
    return rgb_flow.detection_frame(finalize020, np.asarray(raw), runtime)

def mid_smooth_by_timestamp(*, finalize020: Any, rgb_flow: Any, samples: Any, frames: list[dict[str, Any]], runtime: dict[str, Any], target_name: str, args: argparse.Namespace) -> dict[str, Any]:
    timestamps = [mid_frame_timestamp(frame, target_name) for frame in frames]
    source_poses = [copy.deepcopy(frame['pose_results'][target_name]['pose']) for frame in frames]
    output: list[dict[str, Any]] = []
    edge_rejected: list[int] = []
    translation_deltas: list[float] = []
    rotation_deltas: list[float] = []
    for (idx, source_pose) in enumerate(source_poses):
        neighbors = [(neighbor_idx, timestamps[neighbor_idx], source_poses[neighbor_idx]) for neighbor_idx in range(max(0, idx - int(args.smoothing_window_radius)), min(len(frames), idx + int(args.smoothing_window_radius) + 1)) if abs(timestamps[neighbor_idx] - timestamps[idx]) <= float(args.smoothing_window_seconds)]
        (rvec, tvec, source_count) = finalize020.temporal_smoothing_weighted_pose(neighbors, target_idx=idx, target_timestamp=timestamps[idx], sigma_frames=1.0, sigma_seconds=float(args.smoothing_sigma_seconds))
        filled = bool(source_pose.get('pose_filled', False))
        (rvec, tvec, _translation_delta, _rotation_delta) = finalize020.temporal_smoothing_limit_pose_delta(source_pose, rvec, tvec, max_translation_delta_mm=float(args.smoothing_max_filled_translation_mm) if filled else float(args.smoothing_max_measured_translation_mm), max_rotation_delta_deg=float(args.smoothing_max_filled_rotation_deg) if filled else float(args.smoothing_max_measured_rotation_deg))
        image = mid_frame_image(finalize020, rgb_flow, samples, runtime, idx)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source_edge = finalize020.pose_recovery_edge_alignment_score(gray, source_pose, config=runtime['config'], camera_matrix=runtime['camera_matrix'], dist_coeffs=runtime['dist_coeffs'])
        candidate_edge = finalize020.pose_recovery_edge_alignment_score(gray, {'success': True, 'rvec': rvec, 'tvec': tvec}, config=runtime['config'], camera_matrix=runtime['camera_matrix'], dist_coeffs=runtime['dist_coeffs'])
        blend = 1.0
        while candidate_edge < source_edge - float(args.smoothing_max_edge_drop) and blend > 0.125:
            blend *= 0.5
            (rvec, tvec) = finalize020.temporal_smoothing_blend_from_source(source_pose, rvec, tvec, 0.5)
            candidate_edge = finalize020.pose_recovery_edge_alignment_score(gray, {'success': True, 'rvec': rvec, 'tvec': tvec}, config=runtime['config'], camera_matrix=runtime['camera_matrix'], dist_coeffs=runtime['dist_coeffs'])
        if candidate_edge < source_edge - float(args.smoothing_max_edge_drop):
            rvec = np.asarray(source_pose['rvec'], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(source_pose['tvec'], dtype=np.float64).reshape(3, 1)
            candidate_edge = source_edge
            edge_rejected.append(idx)
        translation_delta = float(np.linalg.norm(np.asarray(source_pose['tvec'], dtype=np.float64).reshape(3) - tvec.reshape(3)))
        rotation_delta = mid_rotation_delta_deg(source_pose['rvec'], rvec)
        smoothed = copy.deepcopy(source_pose)
        smoothed.update({'rvec': rvec, 'tvec': tvec, 'T': finalize020.temporal_completion_make_pose_transform(rvec, tvec), 'temporal_smoothed': bool(translation_delta > 1e-09 or rotation_delta > 1e-09), 'temporal_smoothing_clock': 'capture_timestamp', 'temporal_smoothing_source_count': int(source_count), 'temporal_smoothing_window_seconds': float(args.smoothing_window_seconds), 'temporal_smoothing_sigma_seconds': float(args.smoothing_sigma_seconds), 'temporal_smoothing_translation_delta_mm': translation_delta, 'temporal_smoothing_rotation_delta_deg': rotation_delta, 'temporal_smoothing_edge_before': float(source_edge), 'temporal_smoothing_edge_after': float(candidate_edge), 'temporal_smoothing_edge_rejected': idx in edge_rejected})
        output.append(smoothed)
        translation_deltas.append(translation_delta)
        rotation_deltas.append(rotation_delta)
    for (idx, pose) in enumerate(output):
        result = frames[idx]['pose_results'][target_name]
        result['pose_before_temporal_smoothing'] = source_poses[idx]
        result['selected_stage_before_temporal_smoothing'] = result.get('selected_stage', '')
        mid_set_pose(frames[idx], target_name, pose, 'stage12_timestamp_se3_smoothing')
    return {'smoothed_frames': int(sum((bool(pose.get('temporal_smoothed')) for pose in output))), 'edge_rejected_frames': edge_rejected, 'translation_delta_mm_max': float(max(translation_deltas, default=0.0)), 'rotation_delta_deg_max': float(max(rotation_deltas, default=0.0))}

def mid_draw_overlay(*, finalize020: Any, image: np.ndarray, pose: dict[str, Any], runtime: dict[str, Any], idx: int) -> bytes:
    output = image.copy()
    (corners, _) = cv2.projectPoints(finalize020.pose_recovery_cube_corners(runtime['config']), np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1), np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1), runtime['camera_matrix'], runtime['dist_coeffs'])
    corners = corners.reshape(-1, 2)
    for (first, second) in finalize020.POSE_RECOVERY_CUBE_EDGES:
        point_a = tuple((int(round(value)) for value in corners[first]))
        point_b = tuple((int(round(value)) for value in corners[second]))
        cv2.line(output, point_a, point_b, (0, 255, 255), 7, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 82), (0, 0, 0), -1)
    lines = [f"middle_Q frame {idx}: {pose.get('pose_source', '')}", f"timestamp smoothing dt={float(pose.get('temporal_smoothing_translation_delta_mm', 0.0)):.3f}mm dr={float(pose.get('temporal_smoothing_rotation_delta_deg', 0.0)):.3f}deg"]
    for (line_idx, line) in enumerate(lines):
        cv2.putText(output, line, (18, 32 + line_idx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    (ok, encoded) = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def mid_temporal_residuals(frames: list[dict[str, Any]], target_name: str) -> dict[str, float]:
    translations: list[float] = []
    rotations: list[float] = []
    timestamps = [mid_frame_timestamp(frame, target_name) for frame in frames]
    poses = [frame['pose_results'][target_name]['pose'] for frame in frames]
    for idx in range(1, len(frames) - 1):
        alpha = (timestamps[idx] - timestamps[idx - 1]) / (timestamps[idx + 1] - timestamps[idx - 1])
        (rvec, tvec) = mid_interpolate_pose(poses[idx - 1], poses[idx + 1], alpha)
        translations.append(float(np.linalg.norm(np.asarray(poses[idx]['tvec'], dtype=np.float64).reshape(3) - tvec.reshape(3))))
        rotations.append(mid_rotation_delta_deg(poses[idx]['rvec'], rvec))
    return {'translation_median_mm': float(np.median(translations)), 'translation_p95_mm': float(np.percentile(translations, 95)), 'translation_max_mm': float(np.max(translations)), 'rotation_median_deg': float(np.median(rotations)), 'rotation_p95_deg': float(np.percentile(rotations, 95)), 'rotation_max_deg': float(np.max(rotations))}

def mid_source_counts(frames: list[dict[str, Any]], target_name: str) -> dict[str, int]:
    return dict(Counter((str(frame['pose_results'][target_name]['pose'].get('pose_source', '')) for frame in frames)))

def mid_recompute_footer(frames: list[dict[str, Any]], old_footer: dict[str, Any]) -> dict[str, Any]:
    target_names = list(frames[0]['pose_results'])
    success_counts = {name: 0 for name in target_names}
    pose_source_counts = {name: {} for name in target_names}
    for frame in frames:
        for name in target_names:
            pose = frame['pose_results'][name].get('pose', {}) or {}
            success_counts[name] += int(mid_valid_pose(pose))
            source = str(pose.get('pose_source', ''))
            counts = pose_source_counts[name]
            counts[source] = counts.get(source, 0) + 1
    reprocessed = list(old_footer.get('reprocessed_targets', []) or [])
    if 'middle_Q' not in reprocessed:
        reprocessed.append('middle_Q')
    return {'type': 'footer', 'frame_count': len(frames), 'success_counts': success_counts, 'pose_source_counts': pose_source_counts, 'reprocessed_targets': reprocessed, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def mid_write_sidecar(path: Path, header: dict[str, Any], frames: list[dict[str, Any]], footer: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame in frames:
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def mid_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def mid_run(args: argparse.Namespace) -> None:
    source_path = args.source.expanduser().resolve()
    sidecar_path = args.sidecar.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    qa_path = args.qa.expanduser().resolve()
    target_name = str(args.target_name)
    finalize020 = mid_load_module('middle_q_temporal_finalize020', mid_FINALIZE_020_PATH)
    rgb_flow = mid_load_module('middle_q_temporal_rgb_flow', mid_RGB_FLOW_PATH)
    (header, frames, old_footer) = rgb_flow.load_sidecar(sidecar_path)
    samples = rgb_flow.load_source_samples(source_path)
    if len(samples) != len(frames):
        raise ValueError(f'Source/sidecar frame mismatch: {len(samples)} != {len(frames)}')
    target_metadata = header['metadata']['targets'][target_name]
    runtime = rgb_flow.detection_runtime(finalize020, target_metadata)
    runtime['worker_name'] = target_metadata['worker_name']
    runtime['camera_name'] = target_metadata['camera_name']
    counts_before = mid_source_counts(frames, target_name)
    removed = mid_reset_temporal_results(frames, target_name)
    stage8_rejected = mid_gate_stage8(frames, target_name, max_neighbor_seconds=float(args.stage8_max_neighbor_seconds), min_translation_threshold_mm=float(args.stage8_translation_threshold_mm), min_rotation_threshold_deg=float(args.stage8_rotation_threshold_deg))
    (flow_recovered, flow_failures) = mid_recover_gap_edges_with_flow(finalize020=finalize020, rgb_flow=rgb_flow, samples=samples, frames=frames, runtime=runtime, target_name=target_name, preferred_tag_id=int(args.preferred_tag_id), args=args)
    locally_filled = mid_fill_short_local_gaps(frames, target_name, finalize020, max_bracket_gap=int(args.local_max_bracket_gap))
    remaining_failed = [run for run in mid_failed_runs(frames, target_name)]
    if remaining_failed:
        raise RuntimeError(f'middle_Q remains incomplete after conservative recovery: {remaining_failed}; flow_failures={flow_failures}')
    residuals_before_smoothing = mid_temporal_residuals(frames, target_name)
    smoothing = mid_smooth_by_timestamp(finalize020=finalize020, rgb_flow=rgb_flow, samples=samples, frames=frames, runtime=runtime, target_name=target_name, args=args)
    for (idx, frame) in enumerate(frames):
        pose = frame['pose_results'][target_name]['pose']
        image = mid_frame_image(finalize020, rgb_flow, samples, runtime, idx)
        result = frame['pose_results'][target_name]
        result['overlay_jpeg'] = mid_draw_overlay(finalize020=finalize020, image=image, pose=pose, runtime=runtime, idx=idx)
        result['overlay_format'] = 'jpeg_bgr'
        result['overlay_shape'] = tuple((int(value) for value in image.shape))
    residuals_after_smoothing = mid_temporal_residuals(frames, target_name)
    counts_after = mid_source_counts(frames, target_name)
    if any(('global_temporal' in source for source in counts_after)):
        raise RuntimeError('Old global temporal data survived middle_Q cleanup')
    if any((not mid_valid_pose(frame['pose_results'][target_name]['pose']) for frame in frames)):
        raise RuntimeError('Output contains an invalid middle_Q pose')
    output_header = copy.deepcopy(header)
    output_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    history = output_header.setdefault('metadata', {}).setdefault('update_history', [])
    history.append({'updated_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'script': str(mid_FILE_PATH), 'method': 'stage8 timestamp consistency gate; adjacent tag-2 RGB flow; short local timestamp interpolation; timestamp-domain SE(3) smoothing', 'target': target_name, 'removed_old_temporal_frames': removed, 'stage8_rejected_frames': sorted(stage8_rejected), 'rgb_flow_recovered_frames': sorted(flow_recovered), 'local_interpolated_frames': locally_filled})
    footer = mid_recompute_footer(frames, old_footer)
    mid_write_sidecar(output_path, output_header, frames, footer)
    report = {'source_pkl': source_path, 'input_sidecar': sidecar_path, 'output_sidecar': output_path, 'frame_count': len(frames), 'preferred_tag_id': int(args.preferred_tag_id), 'source_counts_before': counts_before, 'removed_old_temporal_frames': removed, 'stage8_rejected_frames': sorted(stage8_rejected), 'stage8_rejection_metrics': stage8_rejected, 'rgb_flow_recovered_frames': sorted(flow_recovered), 'rgb_flow_metrics': flow_recovered, 'rgb_flow_failures': flow_failures, 'local_interpolated_frames': locally_filled, 'source_counts_after': counts_after, 'residuals_before_smoothing': residuals_before_smoothing, 'smoothing': smoothing, 'residuals_after_smoothing': residuals_after_smoothing, 'footer': footer}
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=mid_json_default) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False, default=mid_json_default))

def mid_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=mid_DEFAULT_SOURCE)
    parser.add_argument('--sidecar', type=Path, default=mid_DEFAULT_SIDECAR)
    parser.add_argument('--output', type=Path, default=mid_DEFAULT_OUTPUT)
    parser.add_argument('--qa', type=Path, default=mid_DEFAULT_QA)
    parser.add_argument('--target-name', default='middle_Q')
    parser.add_argument('--preferred-tag-id', type=int, default=2)
    parser.add_argument('--stage8-max-neighbor-seconds', type=float, default=0.25)
    parser.add_argument('--stage8-translation-threshold-mm', type=float, default=1.5)
    parser.add_argument('--stage8-rotation-threshold-deg', type=float, default=2.5)
    parser.add_argument('--max-flow-gap-frames', type=int, default=5)
    parser.add_argument('--local-max-bracket-gap', type=int, default=3)
    parser.add_argument('--flow-max-features', type=int, default=500)
    parser.add_argument('--flow-feature-quality', type=float, default=0.005)
    parser.add_argument('--flow-feature-min-distance', type=float, default=4.0)
    parser.add_argument('--flow-min-features', type=int, default=80)
    parser.add_argument('--flow-lk-window', type=int, default=41)
    parser.add_argument('--flow-lk-levels', type=int, default=5)
    parser.add_argument('--flow-max-fb-error', type=float, default=1.5)
    parser.add_argument('--flow-max-fb-median-px', type=float, default=0.5)
    parser.add_argument('--flow-min-good-tracks', type=int, default=60)
    parser.add_argument('--flow-homography-ransac-px', type=float, default=2.5)
    parser.add_argument('--flow-min-homography-inliers', type=int, default=40)
    parser.add_argument('--flow-min-homography-inlier-ratio', type=float, default=0.2)
    parser.add_argument('--flow-max-homography-median-px', type=float, default=1.5)
    parser.add_argument('--flow-max-current-tag-agreement-px', type=float, default=80.0)
    parser.add_argument('--flow-max-corner-reproj-px', type=float, default=5.5)
    parser.add_argument('--flow-max-translation-delta-mm', type=float, default=6.0)
    parser.add_argument('--flow-max-rotation-delta-deg', type=float, default=10.0)
    parser.add_argument('--flow-min-edge-score', type=float, default=0.04)
    parser.add_argument('--flow-min-tag-corners-inside', type=int, default=4)
    parser.add_argument('--flow-skip-current-tag-detection', action='store_true')
    parser.add_argument('--flow-feature-mask-scale', type=float, default=1.0)
    parser.add_argument('--smoothing-window-radius', type=int, default=4)
    parser.add_argument('--smoothing-window-seconds', type=float, default=0.18)
    parser.add_argument('--smoothing-sigma-seconds', type=float, default=0.075)
    parser.add_argument('--smoothing-max-measured-translation-mm', type=float, default=1.5)
    parser.add_argument('--smoothing-max-measured-rotation-deg', type=float, default=2.0)
    parser.add_argument('--smoothing-max-filled-translation-mm', type=float, default=3.0)
    parser.add_argument('--smoothing-max-filled-rotation-deg', type=float, default=4.0)
    parser.add_argument('--smoothing-max-edge-drop', type=float, default=0.04)
    return parser

import argparse
import copy
import importlib.util
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
idx_FILE_PATH = Path(__file__).resolve()
idx_PROJECT_ROOT = APRILCUBE_ROOT.parent.parent
idx_FINALIZE_020_PATH = idx_PROJECT_ROOT / 'thirdparty/aprilcube/src/020_finalize_pose_postprocess.py'

def idx_load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def idx_load_stream(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    with path.open('rb') as stream:
        header = pickle.load(stream)
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get('type') == 'frame':
                frames.append(record)
            elif record.get('type') == 'footer':
                footer = record
                break
    if not isinstance(header, dict) or header.get('type') != 'header':
        raise ValueError(f'Invalid stream header: {path}')
    if footer is None or int(footer.get('frame_count', -1)) != len(frames):
        raise ValueError(f'Incomplete stream: {path}')
    return (header, frames, footer)

def idx_valid_pose(pose: dict[str, Any]) -> bool:
    if not bool(pose.get('success', False)):
        return False
    try:
        values = np.r_[np.asarray(pose['rvec'], dtype=np.float64).reshape(3), np.asarray(pose['tvec'], dtype=np.float64).reshape(3)]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))

def idx_pose_transform(rvec: Any, tvec: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform

def idx_interpolate_pose(before: dict[str, Any], after: dict[str, Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    before_t = np.asarray(before['tvec'], dtype=np.float64).reshape(3)
    after_t = np.asarray(after['tvec'], dtype=np.float64).reshape(3)
    tvec = ((1.0 - alpha) * before_t + alpha * after_t).reshape(3, 1)
    rotations = Rotation.from_rotvec(np.stack([np.asarray(before['rvec'], dtype=np.float64).reshape(3), np.asarray(after['rvec'], dtype=np.float64).reshape(3)]))
    rvec = Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0].reshape(3, 1)
    return (rvec, tvec)

def idx_rotation_delta_deg(first: Any, second: Any) -> float:
    a = Rotation.from_rotvec(np.asarray(first, dtype=np.float64).reshape(3))
    b = Rotation.from_rotvec(np.asarray(second, dtype=np.float64).reshape(3))
    return float(np.degrees((a.inv() * b).magnitude()))

def idx_pose_brief(pose: dict[str, Any]) -> dict[str, Any]:
    reproj_error = pose.get('reproj_error')
    try:
        reproj_error = float(reproj_error)
    except (TypeError, ValueError):
        reproj_error = None
    if reproj_error is not None and not np.isfinite(reproj_error):
        reproj_error = None
    return {
        'pose_source': str(pose.get('pose_source', '')),
        'strict_original_pose_source': str(pose.get('strict_original_pose_source', '')),
        'tag_ids': [int(value) for value in pose.get('tag_ids', []) or []],
        'reproj_error_px': reproj_error,
        'pose_filled': bool(pose.get('pose_filled', False)),
        'predicted': bool(pose.get('predicted', False)),
        'quality_level': str(pose.get('quality_level', '')),
        'tvec_mm': np.asarray(pose['tvec'], dtype=np.float64).reshape(3).tolist(),
    }


def idx_pose_deltas(
    poses: list[dict[str, Any]],
    timestamps: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(poses) < 2:
        raise ValueError(f'At least two poses are required for step diagnostics, got {len(poses)}')
    if len(timestamps) != len(poses):
        raise ValueError(
            f'Pose/timestamp length mismatch for step diagnostics: '
            f'{len(poses)} != {len(timestamps)}'
        )
    steps: list[dict[str, Any]] = []
    for before_frame, (before, after) in enumerate(zip(poses, poses[1:])):
        after_frame = before_frame + 1
        before_t = np.asarray(before['tvec'], dtype=np.float64).reshape(3)
        after_t = np.asarray(after['tvec'], dtype=np.float64).reshape(3)
        translation_delta = after_t - before_t
        translation_mm = float(np.linalg.norm(translation_delta))
        rotation_deg = idx_rotation_delta_deg(before['rvec'], after['rvec'])
        before_timestamp = float(timestamps[before_frame])
        after_timestamp = float(timestamps[after_frame])
        dt_s = after_timestamp - before_timestamp
        steps.append(
            {
                'before_frame': int(before_frame),
                'after_frame': int(after_frame),
                'before_capture_timestamp': before_timestamp,
                'after_capture_timestamp': after_timestamp,
                'dt_s': float(dt_s),
                'translation_delta_mm_xyz': translation_delta.tolist(),
                'translation_mm': translation_mm,
                'translation_speed_mm_s': (
                    float(translation_mm / dt_s) if dt_s > 0.0 else None
                ),
                'rotation_deg': rotation_deg,
                'rotation_speed_deg_s': (
                    float(rotation_deg / dt_s) if dt_s > 0.0 else None
                ),
                'before_pose': idx_pose_brief(before),
                'after_pose': idx_pose_brief(after),
            }
        )
    translation = [float(step['translation_mm']) for step in steps]
    rotation = [float(step['rotation_deg']) for step in steps]
    summary = {
        'translation_mm_median': float(np.median(translation)),
        'translation_mm_p95': float(np.percentile(translation, 95)),
        'translation_mm_max': float(np.max(translation)),
        'rotation_deg_median': float(np.median(rotation)),
        'rotation_deg_p95': float(np.percentile(rotation, 95)),
        'rotation_deg_max': float(np.max(rotation)),
        'translation_mm_max_step': max(steps, key=lambda step: float(step['translation_mm'])),
        'rotation_deg_max_step': max(steps, key=lambda step: float(step['rotation_deg'])),
        'top_translation_steps': sorted(
            steps,
            key=lambda step: float(step['translation_mm']),
            reverse=True,
        )[:5],
        'top_rotation_steps': sorted(
            steps,
            key=lambda step: float(step['rotation_deg']),
            reverse=True,
        )[:5],
    }
    return summary, steps


def idx_step_gate_error(
    *,
    target_name: str,
    metric: str,
    threshold: float,
    summary: dict[str, Any],
    steps: list[dict[str, Any]],
) -> RuntimeError:
    if metric == 'translation_mm':
        unit = 'mm'
        gate_name = 'adjacent_translation'
    elif metric == 'rotation_deg':
        unit = 'deg'
        gate_name = 'adjacent_rotation'
    else:
        raise ValueError(f'Unsupported step-gate metric: {metric}')
    violations = [step for step in steps if float(step[metric]) > float(threshold)]
    violations.sort(key=lambda step: float(step[metric]), reverse=True)
    diagnostic = {
        'target_name': target_name,
        'gate': gate_name,
        'metric': metric,
        'unit': unit,
        'threshold': float(threshold),
        'violation_count': len(violations),
        'worst_value': float(violations[0][metric]) if violations else None,
        'worst_excess': (
            float(violations[0][metric]) - float(threshold) if violations else None
        ),
        'violating_frame_pairs': violations,
        'trajectory_summary': summary,
        'interpretation': (
            'This is an index pose data-quality gate before Wuji retargeting/xArm IK; '
            'it is not a robot joint velocity-limit failure.'
        ),
    }
    return RuntimeError(
        f'{target_name} {gate_name} step gate failed:\n'
        + json.dumps(diagnostic, indent=2, ensure_ascii=False, default=idx_json_default)
    )

def idx_fill_short_gaps(poses: list[dict[str, Any]], timestamps: list[float], max_gap_frames: int) -> list[int]:
    filled: list[int] = []
    idx = 0
    while idx < len(poses):
        if idx_valid_pose(poses[idx]):
            idx += 1
            continue
        start = idx
        while idx < len(poses) and (not idx_valid_pose(poses[idx])):
            idx += 1
        end = idx - 1
        gap_length = end - start + 1
        if start == 0 or idx >= len(poses) or gap_length > int(max_gap_frames):
            before_index = start - 1 if start > 0 else None
            after_index = idx if idx < len(poses) else None
            if before_index is None:
                reason = 'leading_gap_has_no_left_anchor'
            elif after_index is None:
                reason = 'trailing_gap_has_no_right_anchor'
            else:
                reason = 'gap_exceeds_local_interpolation_limit'
            diagnostic = {
                'target_name': 'index_Q',
                'gate': 'strict_pose_gap_completion',
                'reason': reason,
                'gap_start_frame': int(start),
                'gap_end_frame': int(end),
                'gap_length_frames': int(gap_length),
                'max_gap_frames': int(max_gap_frames),
                'gap_start_capture_timestamp': float(timestamps[start]),
                'gap_end_capture_timestamp': float(timestamps[end]),
                'gap_duration_s': float(timestamps[end] - timestamps[start]),
                'before_anchor_frame': before_index,
                'after_anchor_frame': after_index,
                'anchor_span_s': (
                    float(timestamps[after_index] - timestamps[before_index])
                    if before_index is not None and after_index is not None
                    else None
                ),
                'before_anchor_pose': (
                    idx_pose_brief(poses[before_index]) if before_index is not None else None
                ),
                'after_anchor_pose': (
                    idx_pose_brief(poses[after_index]) if after_index is not None else None
                ),
                'failed_pose_sources': [
                    str((poses[frame_index] or {}).get('pose_source', ''))
                    for frame_index in range(start, end + 1)
                ],
                'failed_reasons': [
                    str((poses[frame_index] or {}).get('failure_reason', ''))
                    for frame_index in range(start, end + 1)
                ],
                'interpretation': (
                    'The strict current-frame index observations are incomplete. '
                    'The pipeline refuses long or unanchored interpolation before '
                    'Wuji retargeting/xArm IK.'
                ),
            }
            raise RuntimeError(
                'index_Q strict-pose gap is not locally fillable:\n'
                + json.dumps(
                    diagnostic,
                    indent=2,
                    ensure_ascii=False,
                    default=idx_json_default,
                )
            )
        before_index = start - 1
        after_index = idx
        for frame_index in range(start, end + 1):
            alpha = (timestamps[frame_index] - timestamps[before_index]) / (timestamps[after_index] - timestamps[before_index])
            (rvec, tvec) = idx_interpolate_pose(poses[before_index], poses[after_index], alpha)
            poses[frame_index] = {'success': True, 'failure_reason': '', 'pose_source': 'strict_single_tag_local_timestamp_se3_interpolation', 'quality_level': 'F', 'quality_reason': f'isolated_strict_gap;anchors:{before_index}-{after_index};alpha:{alpha:.6f}', 'pose_filled': True, 'predicted': True, 'single_frame_only': False, 'rvec': rvec, 'tvec': tvec, 'T': idx_pose_transform(rvec, tvec), 'reproj_error': float('nan'), 'n_tags': 0, 'tag_ids': [], 'visible_faces': [], 'prev_success_frame': before_index, 'next_success_frame': after_index, 'interpolation_alpha': float(alpha), 'interpolation_clock': 'capture_timestamp'}
            filled.append(frame_index)
    return filled

def idx_draw_cube_overlay(finalize020: Any, jpeg: bytes, pose: dict[str, Any], config: Any, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not decode strict overlay for frame {frame_index}')
    (corners, _) = cv2.projectPoints(finalize020.pose_recovery_cube_corners(config), np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1), np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1), camera_matrix, dist_coeffs)
    corners = corners.reshape(-1, 2)
    for (first, second) in finalize020.POSE_RECOVERY_CUBE_EDGES:
        a = tuple((int(round(v)) for v in corners[first]))
        b = tuple((int(round(v)) for v in corners[second]))
        cv2.line(image, a, b, (0, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(image, f'index_Q frame {frame_index}: local timestamp interpolation', (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    (ok, encoded) = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def idx_recompute_footer(frames: list[dict[str, Any]], old_footer: dict[str, Any], target_name: str) -> dict[str, Any]:
    target_names = list(frames[0]['pose_results'])
    success_counts = {name: 0 for name in target_names}
    source_counts: dict[str, Counter[str]] = {name: Counter() for name in target_names}
    for frame in frames:
        for name in target_names:
            pose = frame['pose_results'][name].get('pose', {}) or {}
            success_counts[name] += int(idx_valid_pose(pose))
            source_counts[name][str(pose.get('pose_source', ''))] += 1
    reprocessed = list(old_footer.get('reprocessed_targets', []) or [])
    if target_name not in reprocessed:
        reprocessed.append(target_name)
    return {'type': 'footer', 'frame_count': len(frames), 'success_counts': success_counts, 'pose_source_counts': {name: dict(counts) for (name, counts) in source_counts.items()}, 'reprocessed_targets': reprocessed, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def idx_write_sidecar(path: Path, header: dict[str, Any], frames: list[dict[str, Any]], footer: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame in frames:
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def idx_json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def idx_run(args: argparse.Namespace) -> None:
    sidecar_path = args.sidecar.expanduser().resolve()
    strict_path = args.strict.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    qa_path = args.qa.expanduser().resolve()
    target_name = str(args.target_name)
    finalize020 = idx_load_module('strict_sidecar_finalize020', idx_FINALIZE_020_PATH)
    (header, frames, old_footer) = idx_load_stream(sidecar_path)
    (strict_header, strict_frames, _strict_footer) = idx_load_stream(strict_path)
    if len(frames) != len(strict_frames):
        raise ValueError(f'Sidecar/strict frame mismatch: {len(frames)} != {len(strict_frames)}')
    source_target = strict_header.get('source_metadata', {}).get('pose_target')
    if source_target != target_name:
        raise ValueError(f'Strict stream target mismatch: {source_target} != {target_name}')
    timestamps = [float(frame['pose_results'][target_name]['capture_timestamp']) for frame in frames]
    if any((right <= left for (left, right) in zip(timestamps, timestamps[1:]))):
        raise ValueError('Target timestamps are not strictly increasing')
    poses: list[dict[str, Any]] = []
    for (frame_index, strict_frame) in enumerate(strict_frames):
        if int(strict_frame.get('frame_index', -1)) != frame_index:
            raise ValueError(f'Strict frame index mismatch at {frame_index}')
        strict_pose = copy.deepcopy(strict_frame.get('pose', {}) or {})
        if idx_valid_pose(strict_pose):
            strict_pose.update({'pose_source': 'strict_aprilcube_single_tag_observation', 'strict_original_pose_source': strict_pose.get('pose_source', ''), 'quality_level': 'A', 'quality_reason': 'current_frame_decoded_apriltag_single_face', 'pose_filled': False, 'predicted': False, 'single_frame_only': True})
        poses.append(strict_pose)
    direct_count = sum((idx_valid_pose(pose) for pose in poses))
    filled = idx_fill_short_gaps(poses, timestamps, int(args.max_gap_frames))
    if any((not idx_valid_pose(pose) for pose in poses)):
        raise RuntimeError('Output pose list remains incomplete')
    deltas, step_diagnostics = idx_pose_deltas(poses, timestamps)
    if deltas['translation_mm_max'] > float(args.max_step_translation_mm):
        raise idx_step_gate_error(
            target_name=target_name,
            metric='translation_mm',
            threshold=float(args.max_step_translation_mm),
            summary=deltas,
            steps=step_diagnostics,
        )
    if deltas['rotation_deg_max'] > float(args.max_step_rotation_deg):
        raise idx_step_gate_error(
            target_name=target_name,
            metric='rotation_deg',
            threshold=float(args.max_step_rotation_deg),
            summary=deltas,
            steps=step_diagnostics,
        )
    (config, _face_sets) = finalize020.aprilcube.load_cube_config(str(Path(strict_header['metadata']['cube_cfg']) / 'config.json'))
    camera_matrix = np.asarray(strict_header['metadata']['detection_camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(strict_header['metadata']['detector_dist_coeffs'], dtype=np.float64).reshape(-1)
    for (frame_index, (frame, strict_frame, pose)) in enumerate(zip(frames, strict_frames, poses)):
        result = frame['pose_results'][target_name]
        result['pose_before_strict_single_tag_merge'] = copy.deepcopy(result.get('pose', {}) or {})
        result['pose'] = pose
        result['selected_stage'] = str(pose['pose_source'])
        result['overlay_jpeg'] = idx_draw_cube_overlay(finalize020, strict_frame['overlay_jpeg'], pose, config, camera_matrix, dist_coeffs, frame_index) if frame_index in filled else strict_frame['overlay_jpeg']
        result['overlay_format'] = strict_frame.get('overlay_format', 'jpeg_bgr')
        result['overlay_shape'] = strict_frame.get('overlay_shape')
        frame['poses'][target_name] = copy.deepcopy(pose)
    output_header = copy.deepcopy(header)
    output_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    history = output_header.setdefault('metadata', {}).setdefault('update_history', [])
    history.append({'updated_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'script': str(idx_FILE_PATH), 'target': target_name, 'strict_pose_stream': str(strict_path), 'method': 'direct strict single-tag observations; isolated timestamp SE(3) fill', 'direct_observation_count': direct_count, 'locally_filled_frames': filled})
    footer = idx_recompute_footer(frames, old_footer, target_name)
    idx_write_sidecar(output_path, output_header, frames, footer)
    report = {'input_sidecar': sidecar_path, 'strict_pose_stream': strict_path, 'output_sidecar': output_path, 'target_name': target_name, 'frame_count': len(frames), 'direct_observation_count': direct_count, 'locally_filled_frames': filled, 'pose_deltas': deltas, 'footer': footer}
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=idx_json_default) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False, default=idx_json_default))

def idx_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--strict', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--target-name', default='index_Q')
    parser.add_argument('--max-gap-frames', type=int, default=2)
    parser.add_argument('--max-step-translation-mm', type=float, default=8.0)
    parser.add_argument('--max-step-rotation-deg', type=float, default=20.0)
    return parser

import argparse
import copy
import importlib.util
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
wrist_FILE_PATH = Path(__file__).resolve()
wrist_PROJECT_ROOT = APRILCUBE_ROOT.parent.parent
wrist_RGB_FLOW_PATH = Path(__file__).resolve()

def wrist_load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def wrist_valid_pose(pose: dict[str, Any]) -> bool:
    if not bool(pose.get('success', False)):
        return False
    try:
        values = np.r_[np.asarray(pose['rvec'], dtype=np.float64).reshape(3), np.asarray(pose['tvec'], dtype=np.float64).reshape(3)]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))

def wrist_rotation_delta_deg(first: Any, second: Any) -> float:
    a = Rotation.from_rotvec(np.asarray(first, dtype=np.float64).reshape(3))
    b = Rotation.from_rotvec(np.asarray(second, dtype=np.float64).reshape(3))
    return float(np.degrees((a.inv() * b).magnitude()))

def wrist_pose_transform(rvec: Any, tvec: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix()
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform

def wrist_blend_poses(left: dict[str, Any], right: dict[str, Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    left_t = np.asarray(left['tvec'], dtype=np.float64).reshape(3)
    right_t = np.asarray(right['tvec'], dtype=np.float64).reshape(3)
    tvec = ((1.0 - alpha) * left_t + alpha * right_t).reshape(3, 1)
    rotations = Rotation.from_rotvec(np.stack([np.asarray(left['rvec'], dtype=np.float64).reshape(3), np.asarray(right['rvec'], dtype=np.float64).reshape(3)]))
    rvec = Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0].reshape(3, 1)
    return (rvec, tvec)

def wrist_flow_args(args: argparse.Namespace) -> Any:
    return SimpleNamespace(anchor_frame=0, target_frame=0, max_features=int(args.flow_max_features), feature_quality=float(args.flow_feature_quality), feature_min_distance=float(args.flow_feature_min_distance), min_features=int(args.flow_min_features), lk_window=int(args.flow_lk_window), lk_levels=int(args.flow_lk_levels), max_fb_error=float(args.flow_max_fb_error), max_fb_median_px=float(args.flow_max_fb_median_px), min_good_tracks=int(args.flow_min_good_tracks), homography_ransac_px=float(args.flow_homography_ransac_px), min_homography_inliers=int(args.flow_min_homography_inliers), min_homography_inlier_ratio=float(args.flow_min_homography_inlier_ratio), max_homography_median_px=float(args.flow_max_homography_median_px), max_current_tag_agreement_px=80.0, allow_missing_current_tag=True, max_flow_corner_reproj_px=float(args.flow_max_corner_reproj_px), max_translation_delta_mm=float(args.flow_max_translation_delta_mm), max_rotation_delta_deg=float(args.flow_max_rotation_delta_deg), min_edge_score=float(args.flow_min_edge_score), min_tag_corners_inside=4, skip_current_tag_detection=True, feature_mask_scale=1.0)

def wrist_propagate(*, rgb_flow: Any, finalize020: Any, runtime: dict[str, Any], images: dict[int, np.ndarray], anchor_pose: dict[str, Any], start: int, end: int, step: int, tag_id: int, args: argparse.Namespace) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    poses = {start: copy.deepcopy(anchor_pose)}
    metrics: dict[int, dict[str, Any]] = {}
    settings = wrist_flow_args(args)
    for frame_index in range(start + step, end + step, step):
        anchor_index = frame_index - step
        settings.anchor_frame = anchor_index
        settings.target_frame = frame_index
        flow_anchor = copy.deepcopy(poses[anchor_index])
        flow_anchor['tag_ids'] = [int(tag_id)]
        (pose, frame_metrics) = rgb_flow.recover_pose(finalize020=finalize020, anchor_image=images[anchor_index], target_image=images[frame_index], anchor_pose=flow_anchor, runtime=runtime, args=settings)
        poses[frame_index] = pose
        metrics[frame_index] = frame_metrics
    return (poses, metrics)

def wrist_draw_overlay(finalize020: Any, image: np.ndarray, pose: dict[str, Any], runtime: dict[str, Any], frame_index: int) -> bytes:
    output = image.copy()
    (corners, _) = cv2.projectPoints(finalize020.pose_recovery_cube_corners(runtime['config']), np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1), np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1), runtime['camera_matrix'], runtime['dist_coeffs'])
    corners = corners.reshape(-1, 2)
    for (first, second) in finalize020.POSE_RECOVERY_CUBE_EDGES:
        a = tuple((int(round(value)) for value in corners[first]))
        b = tuple((int(round(value)) for value in corners[second]))
        cv2.line(output, a, b, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(output, f'wrist_Q frame {frame_index}: bidirectional RGB-flow fusion', (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
    (ok, encoded) = cv2.imencode('.jpg', output, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return encoded.tobytes()

def wrist_sequence_deltas(poses: list[dict[str, Any]]) -> dict[str, float]:
    translation: list[float] = []
    rotation: list[float] = []
    for (before, after) in zip(poses, poses[1:]):
        translation.append(float(np.linalg.norm(np.asarray(after['tvec'], dtype=np.float64).reshape(3) - np.asarray(before['tvec'], dtype=np.float64).reshape(3))))
        rotation.append(wrist_rotation_delta_deg(before['rvec'], after['rvec']))
    return {'translation_mm_median': float(np.median(translation)), 'translation_mm_p95': float(np.percentile(translation, 95)), 'translation_mm_max': float(np.max(translation)), 'rotation_deg_median': float(np.median(rotation)), 'rotation_deg_p95': float(np.percentile(rotation, 95)), 'rotation_deg_max': float(np.max(rotation))}

def wrist_recompute_footer(frames: list[dict[str, Any]], old_footer: dict[str, Any], target_name: str) -> dict[str, Any]:
    target_names = list(frames[0]['pose_results'])
    success_counts = {name: 0 for name in target_names}
    source_counts: dict[str, Counter[str]] = {name: Counter() for name in target_names}
    for frame in frames:
        for name in target_names:
            pose = frame['pose_results'][name].get('pose', {}) or {}
            success_counts[name] += int(wrist_valid_pose(pose))
            source_counts[name][str(pose.get('pose_source', ''))] += 1
    reprocessed = list(old_footer.get('reprocessed_targets', []) or [])
    if target_name not in reprocessed:
        reprocessed.append(target_name)
    return {'type': 'footer', 'frame_count': len(frames), 'success_counts': success_counts, 'pose_source_counts': {name: dict(counts) for (name, counts) in source_counts.items()}, 'reprocessed_targets': reprocessed, 'stopped_wall_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def wrist_write_sidecar(path: Path, header: dict[str, Any], frames: list[dict[str, Any]], footer: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open('wb') as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame in frames:
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def wrist_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def wrist_run(args: argparse.Namespace) -> None:
    source_path = args.source.expanduser().resolve()
    sidecar_path = args.sidecar.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    qa_path = args.qa.expanduser().resolve()
    target_name = str(args.target_name)
    start = int(args.start_anchor)
    end = int(args.end_anchor)
    if end <= start + 1:
        raise ValueError('The anchor interval must contain at least one inner frame')
    rgb_flow = wrist_load_module('wrist_bidirectional_rgb_flow', wrist_RGB_FLOW_PATH)
    finalize020 = rgb_flow.load_finalize_020()
    (header, frames, old_footer) = rgb_flow.load_sidecar(sidecar_path)
    samples = rgb_flow.load_source_samples(source_path)
    if len(samples) != len(frames):
        raise ValueError(f'Source/sidecar mismatch: {len(samples)} != {len(frames)}')
    target_metadata = header['metadata']['targets'][target_name]
    runtime = rgb_flow.detection_runtime(finalize020, target_metadata)
    worker_name = str(target_metadata['worker_name'])
    camera_name = str(target_metadata['camera_name'])
    start_pose = copy.deepcopy(frames[start]['poses'][target_name])
    end_pose = copy.deepcopy(frames[end]['poses'][target_name])
    if not wrist_valid_pose(start_pose) or not wrist_valid_pose(end_pose):
        raise ValueError('Both wrist anchors must have finite poses')
    if not start_pose.get('tag_ids') or not end_pose.get('tag_ids'):
        raise ValueError('Both wrist anchors must be direct tag observations')
    images: dict[int, np.ndarray] = {}
    timestamps: dict[int, float] = {}
    for frame_index in range(start, end + 1):
        raw = samples[frame_index]['worker_raw_frames'][worker_name][camera_name]
        images[frame_index] = rgb_flow.detection_frame(finalize020, np.asarray(raw), runtime)
        timestamps[frame_index] = float(frames[frame_index]['pose_results'][target_name]['capture_timestamp'])
    (left, left_metrics) = wrist_propagate(rgb_flow=rgb_flow, finalize020=finalize020, runtime=runtime, images=images, anchor_pose=start_pose, start=start, end=end - 1, step=1, tag_id=int(args.tag_id), args=args)
    (right, right_metrics) = wrist_propagate(rgb_flow=rgb_flow, finalize020=finalize020, runtime=runtime, images=images, anchor_pose=end_pose, start=end, end=start + 1, step=-1, tag_id=int(args.tag_id), args=args)
    disagreements: dict[int, dict[str, float]] = {}
    fused: dict[int, dict[str, Any]] = {}
    duration = timestamps[end] - timestamps[start]
    for frame_index in range(start + 1, end):
        alpha = (timestamps[frame_index] - timestamps[start]) / duration
        left_pose = left[frame_index]
        right_pose = right[frame_index]
        translation_disagreement = float(np.linalg.norm(np.asarray(left_pose['tvec'], dtype=np.float64).reshape(3) - np.asarray(right_pose['tvec'], dtype=np.float64).reshape(3)))
        rotation_disagreement = wrist_rotation_delta_deg(left_pose['rvec'], right_pose['rvec'])
        disagreements[frame_index] = {'translation_mm': translation_disagreement, 'rotation_deg': rotation_disagreement}
        if translation_disagreement > float(args.max_bidirectional_translation_mm):
            raise RuntimeError(f'Bidirectional translation gate failed at {frame_index}')
        if rotation_disagreement > float(args.max_bidirectional_rotation_deg):
            raise RuntimeError(f'Bidirectional rotation gate failed at {frame_index}')
        (rvec, tvec) = wrist_blend_poses(left_pose, right_pose, alpha)
        fused[frame_index] = {'success': True, 'failure_reason': '', 'pose_source': 'wrist_bidirectional_adjacent_rgb_flow_se3_fusion', 'quality_level': 'T', 'quality_reason': f'direct_anchors:{start}-{end};tag:{int(args.tag_id)};alpha:{alpha:.6f};left_right_dt:{translation_disagreement:.3f}mm;left_right_dr:{rotation_disagreement:.3f}deg', 'pose_filled': True, 'predicted': True, 'temporal_recovery': True, 'single_frame_only': False, 'rvec': rvec, 'tvec': tvec, 'T': wrist_pose_transform(rvec, tvec), 'reproj_error': float('nan'), 'reproj_metric': 'bidirectional_adjacent_rgb_flow_fusion', 'n_tags': 1, 'tag_ids': [int(args.tag_id)], 'visible_faces': [], 'left_anchor_frame': start, 'right_anchor_frame': end, 'fusion_alpha': float(alpha), 'left_flow_metrics': left_metrics[frame_index], 'right_flow_metrics': right_metrics[frame_index], 'bidirectional_translation_disagreement_mm': translation_disagreement, 'bidirectional_rotation_disagreement_deg': rotation_disagreement}
    for (frame_index, pose) in fused.items():
        result = frames[frame_index]['pose_results'][target_name]
        result['pose_before_wrist_bidirectional_recovery'] = copy.deepcopy(result.get('pose', {}) or {})
        result['pose'] = pose
        result['selected_stage'] = str(pose['pose_source'])
        result['overlay_jpeg'] = wrist_draw_overlay(finalize020, images[frame_index], pose, runtime, frame_index)
        result['overlay_format'] = 'jpeg_bgr'
        result['overlay_shape'] = tuple((int(value) for value in images[frame_index].shape))
        frames[frame_index]['poses'][target_name] = copy.deepcopy(pose)
    all_poses = [frame['poses'][target_name] for frame in frames]
    if any((not wrist_valid_pose(pose) for pose in all_poses)):
        raise RuntimeError('wrist_Q is still incomplete after recovery')
    deltas = wrist_sequence_deltas(all_poses)
    output_header = copy.deepcopy(header)
    output_header['created_wall_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    output_header.setdefault('metadata', {}).setdefault('update_history', []).append({'updated_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'script': str(wrist_FILE_PATH), 'target': target_name, 'method': 'two direct anchors; adjacent RGB flow; timestamp SE(3) fusion', 'start_anchor': start, 'end_anchor': end, 'tag_id': int(args.tag_id), 'recovered_frames': sorted(fused)})
    footer = wrist_recompute_footer(frames, old_footer, target_name)
    wrist_write_sidecar(output_path, output_header, frames, footer)
    report = {'source_pkl': source_path, 'input_sidecar': sidecar_path, 'output_sidecar': output_path, 'target_name': target_name, 'anchors': [start, end], 'tag_id': int(args.tag_id), 'recovered_frames': sorted(fused), 'bidirectional_disagreements': disagreements, 'maximum_bidirectional_translation_mm': max((value['translation_mm'] for value in disagreements.values())), 'maximum_bidirectional_rotation_deg': max((value['rotation_deg'] for value in disagreements.values())), 'full_sequence_pose_deltas': deltas, 'footer': footer}
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=wrist_json_default) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False, default=wrist_json_default))

def wrist_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--target-name', default='wrist_Q')
    parser.add_argument('--start-anchor', type=int, default=31)
    parser.add_argument('--end-anchor', type=int, default=50)
    parser.add_argument('--tag-id', type=int, default=118)
    parser.add_argument('--max-bidirectional-translation-mm', type=float, default=12.0)
    parser.add_argument('--max-bidirectional-rotation-deg', type=float, default=8.0)
    parser.add_argument('--flow-max-features', type=int, default=250)
    parser.add_argument('--flow-feature-quality', type=float, default=0.001)
    parser.add_argument('--flow-feature-min-distance', type=float, default=2.0)
    parser.add_argument('--flow-min-features', type=int, default=10)
    parser.add_argument('--flow-lk-window', type=int, default=31)
    parser.add_argument('--flow-lk-levels', type=int, default=4)
    parser.add_argument('--flow-max-fb-error', type=float, default=2.0)
    parser.add_argument('--flow-max-fb-median-px', type=float, default=1.0)
    parser.add_argument('--flow-min-good-tracks', type=int, default=8)
    parser.add_argument('--flow-homography-ransac-px', type=float, default=2.5)
    parser.add_argument('--flow-min-homography-inliers', type=int, default=6)
    parser.add_argument('--flow-min-homography-inlier-ratio', type=float, default=0.15)
    parser.add_argument('--flow-max-homography-median-px', type=float, default=2.0)
    parser.add_argument('--flow-max-corner-reproj-px', type=float, default=6.0)
    parser.add_argument('--flow-max-translation-delta-mm', type=float, default=15.0)
    parser.add_argument('--flow-max-rotation-delta-deg', type=float, default=20.0)
    parser.add_argument('--flow-min-edge-score', type=float, default=0.0)
    return parser

def rgb_load_finalize_020() -> Any:
    return sys.modules[__name__]

def mid_load_module(name: str, path: Path) -> Any:
    return rgb_flow if "rgb_flow" in name or "rgb_flow" in str(path) else sys.modules[__name__]

def idx_load_module(name: str, path: Path) -> Any:
    return sys.modules[__name__]

def wrist_load_module(name: str, path: Path) -> Any:
    return rgb_flow


def embedded_wrist_anchor_interval(
    sidecar_path: Path, target_name: str = "wrist_Q"
) -> tuple[int, int, int] | None:
    """Find direct tag anchors around the longest remaining wrist gap."""
    _header, frames, _footer = rgb_load_sidecar(sidecar_path)
    missing = [
        index
        for index, frame in enumerate(frames)
        if not wrist_valid_pose(frame["pose_results"][target_name].get("pose", {}) or {})
    ]
    if not missing:
        return None
    start_missing, end_missing = min(missing), max(missing)
    left_candidates = range(start_missing - 1, -1, -1)
    right_candidates = range(end_missing + 1, len(frames))
    for left in left_candidates:
        left_pose = frames[left]["pose_results"][target_name].get("pose", {}) or {}
        left_tags = {int(value) for value in left_pose.get("tag_ids", [])}
        if not wrist_valid_pose(left_pose) or not left_tags or bool(left_pose.get("pose_filled", False)):
            continue
        for right in right_candidates:
            right_pose = frames[right]["pose_results"][target_name].get("pose", {}) or {}
            right_tags = {int(value) for value in right_pose.get("tag_ids", [])}
            common = sorted(left_tags & right_tags)
            if wrist_valid_pose(right_pose) and common and not bool(right_pose.get("pose_filled", False)):
                # The dorsal wrist cube's tag 118 is the validated tracking
                # face for bidirectional recovery.  Keep the anchor search
                # data-driven, but prefer that face when both anchors see it.
                tag_id = 118 if 118 in common else common[0]
                return left, right, tag_id
    raise RuntimeError(
        f"Could not find two direct wrist tag anchors around frames {start_missing}-{end_missing}"
    )


def run_embedded_pose_recovery_patches(
    *,
    source_path: Path,
    initial_sidecar: Path,
    strict_index_stream: Path,
    output_sidecar: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Apply the frozen-reference recovery sequence without external scripts."""
    work_dir.mkdir(parents=True, exist_ok=True)
    middle_sidecar = work_dir / "middle_recovered.pkl"
    index_middle_sidecar = work_dir / "index_middle_recovered.pkl"
    middle_args = mid_build_parser().parse_args([])
    middle_args.source = source_path
    middle_args.sidecar = initial_sidecar
    middle_args.output = middle_sidecar
    middle_args.qa = work_dir / "middle_recovery_qa.json"
    # Parameters used by the validated full-sequence middle recovery.  They
    # affect the LK feature set (900 points) and permit adjacent propagation
    # through the long trailing occlusion while keeping motion/PnP gates.
    middle_args.max_flow_gap_frames = 303
    middle_args.flow_max_features = 900
    middle_args.flow_feature_quality = 0.0025
    middle_args.flow_feature_min_distance = 3.0
    middle_args.flow_min_features = 50
    middle_args.flow_lk_window = 45
    middle_args.flow_lk_levels = 5
    middle_args.flow_max_fb_error = 3.0
    middle_args.flow_max_fb_median_px = 1.5
    middle_args.flow_min_good_tracks = 35
    middle_args.flow_homography_ransac_px = 3.0
    middle_args.flow_min_homography_inliers = 22
    middle_args.flow_min_homography_inlier_ratio = 0.12
    middle_args.flow_max_homography_median_px = 2.5
    middle_args.flow_max_corner_reproj_px = 20.0
    middle_args.flow_max_translation_delta_mm = 10.0
    middle_args.flow_max_rotation_delta_deg = 20.0
    middle_args.flow_min_edge_score = 0.0
    middle_args.flow_min_tag_corners_inside = 2
    middle_args.flow_skip_current_tag_detection = True
    print("[INFO] Embedded complete recovery: middle_Q", flush=True)
    mid_run(middle_args)

    index_args = idx_build_parser().parse_args(
        [
            "--sidecar",
            str(middle_sidecar),
            "--strict",
            str(strict_index_stream),
            "--output",
            str(index_middle_sidecar),
            "--qa",
            str(work_dir / "index_recovery_qa.json"),
        ]
    )
    print("[INFO] Embedded complete recovery: index_Q", flush=True)
    idx_run(index_args)

    anchor = embedded_wrist_anchor_interval(index_middle_sidecar)
    if anchor is None:
        shutil.copy2(index_middle_sidecar, output_sidecar)
    else:
        start, end, tag_id = anchor
        wrist_args = wrist_build_parser().parse_args(
            [
                "--source",
                str(source_path),
                "--sidecar",
                str(index_middle_sidecar),
                "--output",
                str(output_sidecar),
                "--qa",
                str(work_dir / "wrist_recovery_qa.json"),
                "--start-anchor",
                str(start),
                "--end-anchor",
                str(end),
                "--tag-id",
                str(tag_id),
            ]
        )
        print(
            "[INFO] Embedded complete recovery: wrist_Q "
            f"anchors={start}-{end} tag={tag_id}",
            flush=True,
        )
        wrist_run(wrist_args)
    header, frames, footer = stage13_load_sidecar(output_sidecar)
    del header
    stage13_assert_complete(frames, STAGE13_TARGETS)
    return {
        "frame_count": len(frames),
        "success_counts": copy.deepcopy(footer.get("success_counts", {})),
        "pose_source_counts": copy.deepcopy(footer.get("pose_source_counts", {})),
    }
# -----------------------------------------------------------------------------
# Embedded stage13: completion barrier + timestamp SE(3) smoothing with RGB gate.
# -----------------------------------------------------------------------------

STAGE13_TARGETS = ("wrist_Q", "index_Q", "thumb_Q", "middle_Q")


def stage13_valid_pose(pose: dict[str, Any]) -> bool:
    if not bool(pose.get("success", False)):
        return False
    try:
        values = np.r_[
            np.asarray(pose["rvec"], dtype=np.float64).reshape(3),
            np.asarray(pose["tvec"], dtype=np.float64).reshape(3),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def stage13_load_sidecar(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    with path.open("rb") as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get("format") != mc_POSE_SIDECAR_FORMAT:
            raise ValueError(f"Unsupported sidecar format: {header.get('format')}")
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get("type") == "frame":
                frames.append(record)
            elif record.get("type") == "footer":
                footer = record
    if footer is None:
        raise ValueError(f"Sidecar has no footer: {path}")
    if len(frames) != int(footer.get("frame_count", -1)):
        raise ValueError("Sidecar frame/footer count mismatch")
    return header, frames, footer


def stage13_source_samples(path: Path) -> Sequence[dict[str, Any]]:
    data = mc_load_source_recording(path)
    samples = data.get("samples")
    if not isinstance(samples, Sequence):
        raise ValueError(f"Raw recording has no sample sequence: {path}")
    return samples


def stage13_source_pose(result: dict[str, Any]) -> dict[str, Any]:
    current = result.get("pose", {}) or {}
    before = result.get("pose_before_temporal_smoothing", {}) or {}
    if bool(current.get("temporal_smoothed", False)) and stage13_valid_pose(before):
        return copy.deepcopy(before)
    return copy.deepcopy(current)


def stage13_assert_complete(
    frames: list[dict[str, Any]], target_names: Sequence[str]
) -> None:
    failures: dict[str, list[int]] = {}
    for target_name in target_names:
        missing = [
            index
            for index, frame in enumerate(frames)
            if not stage13_valid_pose(
                frame.get("pose_results", {})
                .get(target_name, {})
                .get("pose", {})
                or {}
            )
        ]
        if missing:
            failures[target_name] = missing
    if failures:
        compact = {
            name: {"count": len(indices), "frames": indices[:20]}
            for name, indices in failures.items()
        }
        raise RuntimeError(
            "Final global smoothing requires every cube pose on every frame; "
            f"incomplete targets: {compact}"
        )


def stage13_frame_timestamp(frame: dict[str, Any], target_name: str) -> float:
    result = frame["pose_results"][target_name]
    value = result.get("capture_timestamp", frame.get("time_monotonic"))
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite timestamp at sample {frame.get('sample_index')}")
    return value


def stage13_rotation_delta_deg(first: Any, second: Any) -> float:
    first_rotation = Rotation.from_rotvec(np.asarray(first, dtype=np.float64).reshape(3))
    second_rotation = Rotation.from_rotvec(np.asarray(second, dtype=np.float64).reshape(3))
    return float(np.degrees((first_rotation.inv() * second_rotation).magnitude()))


def stage13_interpolate_pose(
    before: dict[str, Any], after: dict[str, Any], alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    before_t = np.asarray(before["tvec"], dtype=np.float64).reshape(3)
    after_t = np.asarray(after["tvec"], dtype=np.float64).reshape(3)
    tvec = ((1.0 - alpha) * before_t + alpha * after_t).reshape(3, 1)
    rotations = Rotation.from_rotvec(
        np.stack(
            [
                np.asarray(before["rvec"], dtype=np.float64).reshape(3),
                np.asarray(after["rvec"], dtype=np.float64).reshape(3),
            ]
        )
    )
    rvec = Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0].reshape(3, 1)
    return rvec, tvec


def stage13_detection_runtime(target_metadata: dict[str, Any]) -> dict[str, Any]:
    intrinsics = realsense_load_intrinsics_yaml(target_metadata["intrinsics_yaml"])
    image_size = tuple(int(value) for value in target_metadata["image_size"])
    if tuple(intrinsics["image_size"]) != image_size:
        raise ValueError(
            "Target intrinsics/image-size mismatch: "
            f"{intrinsics['image_size']} != {image_size}"
        )
    undistort_pack = realsense_create_undistort_maps(intrinsics, image_size)
    camera_matrix = (
        np.asarray(undistort_pack[2], dtype=np.float64).reshape(3, 3)
        if undistort_pack is not None
        else np.asarray(intrinsics["K"], dtype=np.float64).reshape(3, 3)
    )
    cube_cfg = Path(target_metadata["cube_cfg"]).expanduser().resolve()
    config, face_id_sets = aprilcube.load_cube_config(str(cube_cfg / "config.json"))
    return {
        "intrinsics": intrinsics,
        "image_size": image_size,
        "undistort_pack": undistort_pack,
        "camera_matrix": camera_matrix,
        "dist_coeffs": np.zeros(5, dtype=np.float64),
        "cube_cfg": cube_cfg,
        "config": config,
        "face_id_sets": face_id_sets,
        "tag_corner_map": aprilcube.build_tag_corner_map(config),
        "worker_name": target_metadata["worker_name"],
        "camera_name": target_metadata["camera_name"],
    }


def stage13_detection_frame(image: np.ndarray, runtime: dict[str, Any]) -> np.ndarray:
    target_width, target_height = runtime["image_size"]
    if image.shape[:2] != (target_height, target_width):
        raise ValueError(
            f"Target raw image is {image.shape[1]}x{image.shape[0]}, "
            f"but calibration is {target_width}x{target_height}"
        )
    return realsense_undistort_frame(image, runtime["undistort_pack"])


def stage13_pose_step_metrics(poses: list[dict[str, Any]]) -> dict[str, float]:
    translation: list[float] = []
    rotation: list[float] = []
    for before, after in zip(poses, poses[1:]):
        translation.append(
            float(
                np.linalg.norm(
                    np.asarray(after["tvec"], dtype=np.float64).reshape(3)
                    - np.asarray(before["tvec"], dtype=np.float64).reshape(3)
                )
            )
        )
        rotation.append(stage13_rotation_delta_deg(before["rvec"], after["rvec"]))
    return {
        "translation_step_median_mm": float(np.median(translation)),
        "translation_step_p95_mm": float(np.percentile(translation, 95)),
        "translation_step_max_mm": float(np.max(translation)),
        "rotation_step_median_deg": float(np.median(rotation)),
        "rotation_step_p95_deg": float(np.percentile(rotation, 95)),
        "rotation_step_max_deg": float(np.max(rotation)),
    }


def stage13_temporal_residuals(
    frames: list[dict[str, Any]], target_name: str
) -> dict[str, float]:
    translations: list[float] = []
    rotations: list[float] = []
    timestamps = [stage13_frame_timestamp(frame, target_name) for frame in frames]
    poses = [frame["pose_results"][target_name]["pose"] for frame in frames]
    for index in range(1, len(frames) - 1):
        alpha = (timestamps[index] - timestamps[index - 1]) / (
            timestamps[index + 1] - timestamps[index - 1]
        )
        rvec, tvec = stage13_interpolate_pose(poses[index - 1], poses[index + 1], alpha)
        translations.append(
            float(
                np.linalg.norm(
                    np.asarray(poses[index]["tvec"], dtype=np.float64).reshape(3)
                    - tvec.reshape(3)
                )
            )
        )
        rotations.append(stage13_rotation_delta_deg(poses[index]["rvec"], rvec))
    return {
        "translation_median_mm": float(np.median(translations)),
        "translation_p95_mm": float(np.percentile(translations, 95)),
        "translation_max_mm": float(np.max(translations)),
        "rotation_median_deg": float(np.median(rotations)),
        "rotation_p95_deg": float(np.percentile(rotations, 95)),
        "rotation_max_deg": float(np.max(rotations)),
    }


def stage13_smooth_target(
    *,
    samples: Sequence[dict[str, Any]],
    frames: list[dict[str, Any]],
    runtime: dict[str, Any],
    target_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    timestamps = [stage13_frame_timestamp(frame, target_name) for frame in frames]
    source_poses = [
        copy.deepcopy(frame["pose_results"][target_name]["pose"]) for frame in frames
    ]
    output: list[dict[str, Any]] = []
    edge_rejected: list[int] = []
    translation_deltas: list[float] = []
    rotation_deltas: list[float] = []
    for index, source_pose in enumerate(source_poses):
        neighbors = [
            (neighbor_index, timestamps[neighbor_index], source_poses[neighbor_index])
            for neighbor_index in range(
                max(0, index - int(args.window_radius)),
                min(len(frames), index + int(args.window_radius) + 1),
            )
            if abs(timestamps[neighbor_index] - timestamps[index])
            <= float(args.window_seconds)
        ]
        rvec, tvec, source_count = temporal_smoothing_weighted_pose(
            neighbors,
            target_idx=index,
            target_timestamp=timestamps[index],
            sigma_frames=1.0,
            sigma_seconds=float(args.sigma_seconds),
        )
        filled = bool(source_pose.get("pose_filled", False))
        rvec, tvec, _, _ = temporal_smoothing_limit_pose_delta(
            source_pose,
            rvec,
            tvec,
            max_translation_delta_mm=(
                float(args.max_filled_translation_mm)
                if filled
                else float(args.max_measured_translation_mm)
            ),
            max_rotation_delta_deg=(
                float(args.max_filled_rotation_deg)
                if filled
                else float(args.max_measured_rotation_deg)
            ),
        )
        raw = samples[index]["worker_raw_frames"][runtime["worker_name"]][
            runtime["camera_name"]
        ]
        image = stage13_detection_frame(np.asarray(raw), runtime)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source_edge = pose_recovery_edge_alignment_score(
            gray,
            source_pose,
            config=runtime["config"],
            camera_matrix=runtime["camera_matrix"],
            dist_coeffs=runtime["dist_coeffs"],
        )
        candidate_edge = pose_recovery_edge_alignment_score(
            gray,
            {"success": True, "rvec": rvec, "tvec": tvec},
            config=runtime["config"],
            camera_matrix=runtime["camera_matrix"],
            dist_coeffs=runtime["dist_coeffs"],
        )
        blend = 1.0
        while candidate_edge < source_edge - float(args.max_edge_score_drop) and blend > 0.125:
            blend *= 0.5
            rvec, tvec = temporal_smoothing_blend_from_source(source_pose, rvec, tvec, 0.5)
            candidate_edge = pose_recovery_edge_alignment_score(
                gray,
                {"success": True, "rvec": rvec, "tvec": tvec},
                config=runtime["config"],
                camera_matrix=runtime["camera_matrix"],
                dist_coeffs=runtime["dist_coeffs"],
            )
        if candidate_edge < source_edge - float(args.max_edge_score_drop):
            rvec = np.asarray(source_pose["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(source_pose["tvec"], dtype=np.float64).reshape(3, 1)
            candidate_edge = source_edge
            edge_rejected.append(index)
        translation_delta = float(
            np.linalg.norm(
                np.asarray(source_pose["tvec"], dtype=np.float64).reshape(3)
                - tvec.reshape(3)
            )
        )
        rotation_delta = stage13_rotation_delta_deg(source_pose["rvec"], rvec)
        smoothed = copy.deepcopy(source_pose)
        smoothed.update(
            {
                "rvec": rvec,
                "tvec": tvec,
                "T": temporal_completion_make_pose_transform(rvec, tvec),
                "temporal_smoothed": bool(
                    translation_delta > 1e-9 or rotation_delta > 1e-9
                ),
                "temporal_smoothing_clock": "capture_timestamp",
                "temporal_smoothing_source_count": int(source_count),
                "temporal_smoothing_window_seconds": float(args.window_seconds),
                "temporal_smoothing_sigma_seconds": float(args.sigma_seconds),
                "temporal_smoothing_translation_delta_mm": translation_delta,
                "temporal_smoothing_rotation_delta_deg": rotation_delta,
                "temporal_smoothing_edge_before": float(source_edge),
                "temporal_smoothing_edge_after": float(candidate_edge),
                "temporal_smoothing_edge_rejected": index in edge_rejected,
            }
        )
        output.append(smoothed)
        translation_deltas.append(translation_delta)
        rotation_deltas.append(rotation_delta)
    for index, pose in enumerate(output):
        result = frames[index]["pose_results"][target_name]
        result["pose_before_temporal_smoothing"] = source_poses[index]
        result["selected_stage_before_temporal_smoothing"] = result.get(
            "selected_stage", ""
        )
        result["pose"] = pose
        result["selected_stage"] = "stage12_timestamp_se3_smoothing"
        frames[index]["poses"][target_name] = copy.deepcopy(pose)
    return {
        "smoothed_frames": int(sum(bool(pose.get("temporal_smoothed")) for pose in output)),
        "edge_rejected_frames": edge_rejected,
        "translation_delta_mm_max": float(max(translation_deltas, default=0.0)),
        "rotation_delta_deg_max": float(max(rotation_deltas, default=0.0)),
    }


def stage13_footer(
    frames: list[dict[str, Any]], old_footer: dict[str, Any], target_names: Sequence[str]
) -> dict[str, Any]:
    all_names = list(frames[0]["pose_results"])
    success_counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    smoothed_counts: dict[str, int] = {}
    applied_counts: dict[str, int] = {}
    for name in all_names:
        poses = [frame["pose_results"][name].get("pose", {}) or {} for frame in frames]
        success_counts[name] = sum(stage13_valid_pose(pose) for pose in poses)
        source_counts[name] = {}
        for pose in poses:
            source = str(pose.get("pose_source", ""))
            source_counts[name][source] = source_counts[name].get(source, 0) + 1
        smoothed_counts[name] = sum(
            bool(pose.get("final_global_temporal_smoothed", False)) for pose in poses
        )
        applied_counts[name] = sum(
            bool(pose.get("final_global_smoothing_applied", False)) for pose in poses
        )
    reprocessed = list(old_footer.get("reprocessed_targets", []) or [])
    for target_name in target_names:
        if target_name not in reprocessed:
            reprocessed.append(target_name)
    return {
        "type": "footer",
        "frame_count": len(frames),
        "success_counts": success_counts,
        "pose_source_counts": source_counts,
        "final_global_smoothed_counts": smoothed_counts,
        "final_global_smoothing_applied_counts": applied_counts,
        "final_global_smoothing_complete": all(
            applied_counts[name] == len(frames) for name in target_names
        ),
        "reprocessed_targets": reprocessed,
        "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def stage13_write_sidecar(
    path: Path,
    header: dict[str, Any],
    frames: list[dict[str, Any]],
    footer: dict[str, Any],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as stream:
            pickle.dump(header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for frame in frames:
                pickle.dump(frame, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(footer, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def stage13_run(args: argparse.Namespace) -> None:
    source_path = Path(args.source).expanduser().resolve()
    sidecar_path = Path(args.sidecar).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    target_names = [str(value) for value in args.targets]
    header, frames, old_footer = stage13_load_sidecar(sidecar_path)
    missing_targets = sorted(
        set(target_names) - set(header.get("metadata", {}).get("targets", {}))
    )
    if missing_targets:
        raise KeyError(f"Targets absent from sidecar metadata: {missing_targets}")
    raw_stat = source_path.stat()
    expected_identity = {"size": int(raw_stat.st_size), "mtime_ns": int(raw_stat.st_mtime_ns)}
    if header.get("source_multi_cam_identity") != expected_identity:
        raise ValueError("Sidecar source identity does not match raw input")
    stage13_assert_complete(frames, target_names)
    samples = stage13_source_samples(source_path)
    if len(samples) != len(frames):
        raise ValueError(f"Raw/sidecar frame mismatch: {len(samples)} != {len(frames)}")
    for target_name in target_names:
        for frame in frames:
            result = frame["pose_results"][target_name]
            old_pose = copy.deepcopy(result.get("pose", {}) or {})
            source_pose = stage13_source_pose(result)
            result["pose_before_final_global_smoothing"] = source_pose
            result["pose_before_previous_smoothing_reset"] = old_pose
            result["pose"] = copy.deepcopy(source_pose)
            result["selected_stage_before_final_global_smoothing"] = result.get(
                "selected_stage", ""
            )
            result["selected_stage"] = str(
                source_pose.get("pose_source", "recovered_pose")
            )
            frame["poses"][target_name] = copy.deepcopy(source_pose)
    stage13_assert_complete(frames, target_names)
    reports: dict[str, Any] = {}
    for target_name in target_names:
        print(f"[INFO] Embedded stage13 smoothing: {target_name}", flush=True)
        runtime = stage13_detection_runtime(header["metadata"]["targets"][target_name])
        before = stage13_temporal_residuals(frames, target_name)
        before_steps = stage13_pose_step_metrics(
            [frame["pose_results"][target_name]["pose"] for frame in frames]
        )
        smoothing = stage13_smooth_target(
            samples=samples,
            frames=frames,
            runtime=runtime,
            target_name=target_name,
            args=args,
        )
        for frame in frames:
            result = frame["pose_results"][target_name]
            pose = result["pose"]
            pose["final_global_temporal_smoothed"] = bool(
                pose.get("temporal_smoothed", False)
            )
            pose["final_global_smoothing_version"] = 1
            pose["final_global_smoothing_applied"] = True
            pose["final_global_smoothing_completion_barrier"] = True
            result["selected_stage"] = "stage13_final_complete_sidecar_se3_smoothing"
            frame["poses"][target_name] = copy.deepcopy(pose)
        reports[target_name] = {
            "temporal_residuals_before": before,
            "temporal_residuals_after": stage13_temporal_residuals(frames, target_name),
            "pose_steps_before": before_steps,
            "pose_steps_after": stage13_pose_step_metrics(
                [frame["pose_results"][target_name]["pose"] for frame in frames]
            ),
            "smoothing": smoothing,
        }
    stage13_assert_complete(frames, target_names)
    output_header = copy.deepcopy(header)
    output_header["created_wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    output_metadata = output_header.setdefault("metadata", {})
    output_metadata["final_global_smoothing"] = {
        "schema": "consensv2.final_complete_sidecar_smoothing.v1",
        "complete": True,
        "completion_barrier_passed": True,
        "frame_count": len(frames),
        "targets": target_names,
        "applied_counts": {name: len(frames) for name in target_names},
        "input_sidecar": str(sidecar_path),
        "script": str(Path(__file__).resolve()),
        "window_seconds": float(args.window_seconds),
        "sigma_seconds": float(args.sigma_seconds),
        "rgb_edge_guard": True,
        "target_reports": reports,
    }
    output_metadata.setdefault("update_history", []).append(
        {
            "updated_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": str(Path(__file__).resolve()),
            "method": (
                "completion barrier followed by symmetric capture-timestamp SE(3) "
                "smoothing with current-frame RGB cube-edge guard"
            ),
            "input_sidecar": str(sidecar_path),
            "targets": target_names,
            "frame_count": len(frames),
            "window_seconds": float(args.window_seconds),
            "sigma_seconds": float(args.sigma_seconds),
            "completion_barrier_passed": True,
        }
    )
    footer = stage13_footer(frames, old_footer, target_names)
    stage13_write_sidecar(output_path, output_header, frames, footer, bool(args.overwrite))
    print(f"[INFO] Embedded stage13 output: {output_path}")


# -----------------------------------------------------------------------------
# Monolithic pose -> Wuji-left retarget -> xArm7 IK orchestration and embedding.
# -----------------------------------------------------------------------------

MONOLITHIC_QPOS_SCHEMA = "consensv2.multi_camera_wuji_left_xarm7_qpos.v1"
DEFAULT_WUJI_LEFT_URDF = (
    PROJECT_ROOT
    / "thirdparty/wuji-description/hand/body-with-soft/urdf/left_simplified_w_fingereye.urdf"
)
DEFAULT_WUJI_CONTACT_KEYPOINTS = (
    PROJECT_ROOT / "configs/retarget/left_wuji_fingertip_contact_keypoints_v2.yaml"
)
DEFAULT_XARM7_WUJI_LEFT_URDF = (
    PROJECT_ROOT
    / "thirdparty/xarm7_wuji_left_description/xarm7_wuji_left_w_fingereye_v2.urdf"
)


@contextlib.contextmanager
def monolithic_argv(arguments: Sequence[str]):
    previous = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *[str(value) for value in arguments]]
    try:
        yield
    finally:
        sys.argv = previous


def monolithic_load_pickle_stream(
    path: Path, expected_format: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    frames: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    with path.open("rb") as stream:
        header = pickle.load(stream)
        if not isinstance(header, dict) or header.get("format") != expected_format:
            raise ValueError(
                f"Unexpected format for {path}: {header.get('format')} != {expected_format}"
            )
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get("type") == "frame":
                frames.append(record)
            elif record.get("type") == "footer":
                footer = record
    return header, frames, footer


def monolithic_attach_qpos(
    *,
    pose_sidecar: Path,
    output_path: Path,
    retarget_pkl: Path,
    ik_pkl: Path,
    raw_path: Path,
    overwrite: bool,
) -> None:
    pose_header, pose_frames, pose_footer = monolithic_load_pickle_stream(
        pose_sidecar, mc_POSE_SIDECAR_FORMAT
    )
    retarget_header, retarget_frames, _ = monolithic_load_pickle_stream(
        retarget_pkl, rt_COMPACT_PKL_FORMAT
    )
    ik_header, ik_frames, ik_footer = monolithic_load_pickle_stream(
        ik_pkl, fb_IK_FORMAT
    )
    frame_count = len(pose_frames)
    if len(retarget_frames) != frame_count or len(ik_frames) != frame_count:
        raise ValueError(
            "Pose/retarget/IK frame mismatch: "
            f"{frame_count}/{len(retarget_frames)}/{len(ik_frames)}"
        )
    for index, (pose_frame, retarget_frame, ik_frame) in enumerate(
        zip(pose_frames, retarget_frames, ik_frames)
    ):
        if int(pose_frame.get("sample_index", -1)) != index:
            raise ValueError(f"Non-contiguous pose sample_index at {index}")
        if int(retarget_frame.get("frame_index", -1)) != index:
            raise ValueError(f"Non-contiguous retarget frame_index at {index}")
        if int(ik_frame.get("frame_index", -1)) != index:
            raise ValueError(f"Non-contiguous IK frame_index at {index}")
        three_finger = np.asarray(
            retarget_frame["wujihand_qpos"], dtype=np.float32
        ).reshape(12)
        safe_full = np.asarray(ik_frame["qpos"], dtype=np.float32).reshape(27)
        raw_full = np.asarray(ik_frame["raw_ik_qpos"], dtype=np.float32).reshape(27)
        if not np.all(np.isfinite(np.r_[three_finger, safe_full, raw_full])):
            raise ValueError(f"Non-finite qpos at frame {index}")
        qpos_result = {
            "schema": MONOLITHIC_QPOS_SCHEMA,
            "trajectory_time_s": float(ik_frame["trajectory_time_s"]),
            "wuji_left_three_finger_retarget_qpos": three_finger,
            "wuji_left_three_finger_safe_qpos": np.asarray(
                ik_frame["wujihand_three_finger_qpos"], dtype=np.float32
            ).reshape(12),
            "wuji_left_qpos": np.asarray(
                ik_frame["wujihand_qpos"], dtype=np.float32
            ).reshape(20),
            "xarm7_qpos": np.asarray(ik_frame["arm_qpos"], dtype=np.float32).reshape(7),
            "xarm7_wuji_left_qpos": safe_full,
            "xarm7_wuji_left_raw_ik_qpos": raw_full,
            "palm_position_error_m": float(ik_frame["palm_position_error_m"]),
            "palm_orientation_error_deg": float(
                ik_frame["palm_orientation_error_deg"]
            ),
            "target_T_link_base_left_palm_link": np.asarray(
                ik_frame["target_T_link_base_left_palm_link"], dtype=np.float64
            ),
            "fk_T_link_base_left_palm_link": np.asarray(
                ik_frame["fk_T_link_base_left_palm_link"], dtype=np.float64
            ),
        }
        pose_frame["qpos_results"] = qpos_result
        # Direct aliases keep deployment and inspection code simple while the
        # structured qpos_results field remains the canonical representation.
        pose_frame["wujihand_qpos"] = qpos_result["wuji_left_qpos"]
        pose_frame["xarm_qpos"] = qpos_result["xarm7_qpos"]
        pose_frame["qpos"] = qpos_result["xarm7_wuji_left_qpos"]
        pose_frame["raw_ik_qpos"] = qpos_result["xarm7_wuji_left_raw_ik_qpos"]
    output_header = copy.deepcopy(pose_header)
    output_header["created_wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata = output_header.setdefault("metadata", {})
    metadata["robot_qpos"] = {
        "schema": MONOLITHIC_QPOS_SCHEMA,
        "complete": True,
        "frame_count": frame_count,
        "handedness": "left",
        "source_raw_pkl": str(raw_path),
        "wuji_left_retarget": {
            "format": retarget_header.get("format"),
            "mode": retarget_header.get("mode"),
            "best_start": retarget_header.get("best_start"),
            "joint_names": list(retarget_header.get("wujihand_qpos_joint_order", [])),
            "summary": copy.deepcopy(retarget_header.get("summary", {})),
        },
        "full_body_ik": {
            "format": ik_header.get("format"),
            "qpos_joint_names": list(ik_header.get("qpos_joint_names", [])),
            "arm_joint_names": list(ik_header.get("arm_joint_names", [])),
            "three_finger_joint_names": list(
                ik_header.get("three_finger_joint_names", [])
            ),
            "T_left_palm_link_hand_back_cube": copy.deepcopy(
                ik_header.get("T_left_palm_link_hand_back_cube")
            ),
            "T_link_base_rs_camera": copy.deepcopy(
                ik_header.get("T_link_base_rs_camera")
            ),
            "selected_candidate": ik_header.get("selected_candidate"),
            "selection_metrics": copy.deepcopy(ik_header.get("selection_metrics", {})),
            "hardware_safety": copy.deepcopy(ik_header.get("hardware_safety", {})),
            "urdf": str(DEFAULT_XARM7_WUJI_LEFT_URDF),
        },
    }
    metadata.setdefault("update_history", []).append(
        {
            "updated_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": str(Path(__file__).resolve()),
            "method": (
                "embedded Wuji-left three-finger four-point retargeting, virtual "
                "RS-to-xArm gauge optimization, full-body IK, and hardware-safe "
                "velocity/acceleration/jerk postprocessing"
            ),
            "frame_count": frame_count,
        }
    )
    output_footer = copy.deepcopy(pose_footer or {})
    output_footer.update(
        {
            "type": "footer",
            "frame_count": frame_count,
            "robot_qpos_complete": True,
            "robot_qpos_schema": MONOLITHIC_QPOS_SCHEMA,
            "safe_playback_fps": None if ik_footer is None else ik_footer.get("safe_playback_fps"),
            "safe_duration_s": None if ik_footer is None else ik_footer.get("safe_duration_s"),
            "stopped_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    stage13_write_sidecar(
        output_path,
        output_header,
        pose_frames,
        output_footer,
        overwrite=overwrite or output_path == pose_sidecar,
    )


def run_monolithic_qpos_pipeline(
    *,
    raw_path: Path,
    pose_sidecar: Path,
    output_path: Path,
    temp_root: Path,
    overwrite: bool,
    keep_temp: bool,
) -> None:
    for required in (
        raw_path,
        pose_sidecar,
        DEFAULT_WUJI_LEFT_URDF,
        DEFAULT_WUJI_CONTACT_KEYPOINTS,
        DEFAULT_XARM7_WUJI_LEFT_URDF,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if temp_root.exists() and any(temp_root.iterdir()):
        raise FileExistsError(f"Qpos temporary root is not empty: {temp_root}")
    retarget_dir = temp_root / "wuji_left_retarget"
    full_body_dir = temp_root / "xarm7_wuji_left"
    retarget_pkl = retarget_dir / "three_finger_se3.pkl"
    retarget_npz = retarget_dir / "three_finger_se3.npz"
    merged_pkl = full_body_dir / "full_body_merged.pkl"
    ik_pkl = full_body_dir / "xarm7_wuji_left_full_qpos.pkl"
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        print("[INFO] Embedded Wuji-left three-finger retargeting", flush=True)
        with monolithic_argv(
            [
                raw_path,
                "--pose-sidecar",
                pose_sidecar,
                "--wrist-extrinsics",
                rt_DEFAULT_MULTI_CAM_WRIST_EXTRINSICS,
                "--urdf",
                DEFAULT_WUJI_LEFT_URDF,
                "--fingertip-geometry-urdf",
                DEFAULT_XARM7_WUJI_LEFT_URDF,
                "--contact-keypoints",
                DEFAULT_WUJI_CONTACT_KEYPOINTS,
                "--output-dir",
                retarget_dir,
                "--modes",
                "se3",
                "--joint-temporal-optimize",
                "--overwrite",
            ]
        ):
            rt_main()
        print("[INFO] Embedded xArm7 + Wuji-left full-body IK", flush=True)
        with monolithic_argv(
            [
                "--raw-pkl",
                raw_path,
                "--sidecar-pkl",
                pose_sidecar,
                "--retarget-npz",
                retarget_npz,
                "--retarget-pkl",
                retarget_pkl,
                "--urdf",
                DEFAULT_XARM7_WUJI_LEFT_URDF,
                "--merged-pkl",
                merged_pkl,
                "--ik-pkl",
                ik_pkl,
                "--overwrite",
            ]
        ):
            solve_main()
        monolithic_attach_qpos(
            pose_sidecar=pose_sidecar,
            output_path=output_path,
            retarget_pkl=retarget_pkl,
            ik_pkl=ik_pkl,
            raw_path=raw_path,
            overwrite=overwrite,
        )
        print(f"[INFO] Embedded robot qpos into final PKL: {output_path}")
    finally:
        if keep_temp:
            print(f"[INFO] Keeping qpos temporary files: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f"[INFO] Removed qpos temporary files: {temp_root}")

def build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process one synchronized raw multi-camera recording into exactly one "
            "post_progress PKL containing raw images, four cube poses, Wuji-left "
            "qpos, and xArm7 qpos."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Raw PKL recorded by scripts/drafts/"
            "020_visualize_multi_av_cv2_cameras.py."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input-stem>_post_progress.pkl beside the input.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the raw recording schema and required calibration files only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_main_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    fmt = inspect_pkl_format(input_path)
    print(f"[INFO] Auto detected input format: {fmt}")
    if fmt != mcstream_MULTI_CAMERA_STREAM_FORMAT:
        recorder_path = (
            PROJECT_ROOT
            / "scripts/drafts/020_visualize_multi_av_cv2_cameras.py"
        )
        raise ValueError(
            f"Unsupported input PKL format {fmt!r}. This CLI only accepts "
            f"{mcstream_MULTI_CAMERA_STREAM_FORMAT!r} raw recordings produced by "
            f"{recorder_path}."
        )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_post_progress.pkl")
    )
    if output_path == input_path:
        raise ValueError("--output must not overwrite the raw input PKL")
    if output_path.exists() and not args.overwrite and not args.validate_only:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite")

    pose_temp_root = Path("/dev/shm") / f"consensv2lab_020_{input_path.stem}"
    intermediate = pose_temp_root / f"{input_path.stem}_pose_sidecar_intermediate.pkl"
    mc_run(
        argparse.Namespace(
            input=input_path,
            output=intermediate,
            temp_root=pose_temp_root,
            max_frames=None,
            targets=[target.name for target in mc_TARGETS],
            merge_existing=None,
            validate_only=bool(args.validate_only),
            keep_temp=False,
            final_output=output_path,
            final_qa=None,
            skip_final_global_smoothing=False,
        )
    )
    if args.validate_only:
        print(f"[INFO] Validation passed: {input_path}")
        return

    qpos_temp_root = pose_temp_root.with_name(pose_temp_root.name + "_qpos")
    run_monolithic_qpos_pipeline(
        raw_path=input_path,
        pose_sidecar=output_path,
        output_path=output_path,
        temp_root=qpos_temp_root,
        overwrite=bool(args.overwrite),
        keep_temp=False,
    )
    print(f"[INFO] Done: {output_path}")


if __name__ == "__main__":
    main()
