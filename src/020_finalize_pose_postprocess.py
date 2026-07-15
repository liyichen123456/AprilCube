#!/usr/bin/env python3
"""One-file offline AprilCube pose postprocess pipeline.

The readable pipeline lives at the top of this file:

1. Convert 008 multi-cube raw recordings to 012-style single-cube streams.
2. Estimate strict AprilCube poses.
3. Estimate DeepTag dense-keypoint poses.
4. Fuse single-frame candidates using reprojection and edge gates.
5. Recover hard frames with conservative RGB outline refinement.
6. Fill the final gaps temporally and merge poses back into the raw stream.

The long copied helper implementations are kept at the bottom so this file can
run by itself without launching or importing the old numbered stage scripts.
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
import viser
import yaml
from PIL import Image
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

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
from recorder_cv2_cam import CV2CameraManager
from stag_decode.pose_estimator import get_fine_grid_points_anno

preprocess_tag_image = _preprocess

# Set these paths and the optional 008 cube filter before running. The pipeline
# intentionally has no command-line options, so the configuration is reproducible.
# INPUT_PKL = RECORDINGS_DIR / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
INPUT_PKL = RECORDINGS_DIR / "008_raw_frames_20260715_000555.pkl"
OUTPUT_PKL = RECORDINGS_DIR / "012_rs_raw_frames_20260710_214336_with_final_postprocessed_pose.pkl"
# The recording header also contains a thumb cfg, but this capture only contains
# the index cube. Set to None to process every cube listed in the 008 header.
PROCESS_008_CUBE_NAMES: tuple[str, ...] | None = (
    "cube_april_36h11_6_11_1x1x1_15mm",
)

RAW_008_PKL_FORMAT = "aprilcube_raw_frame_stream_v1"
RAW_012_PKL_FORMAT = "aprilcube_rs_raw_frame_stream_v1"
RAW_012_WITH_POSE_PKL_FORMAT = "aprilcube_012_raw_with_pose_stream_v1"
SUPPORTED_012_INPUT_PKL_FORMATS = {
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
    max_translation_delta_mm: float = 35.0
    max_rotation_delta_deg: float = 12.0
    reject_loose_input: bool = True
    jpeg_quality: int = 90


@dataclass(frozen=True)
class TemporalPoseCompletionConfig:
    input_pkl: Path
    raw_pkl: Path
    output_pkl: Path
    translation_smooth: float = 2400.0
    max_bracket_gap: int = 40
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


def visualize_008_recording(pkl_path: Path) -> None:
    print("[STAGE] 008 pose replay and visualization", flush=True)
    replay_008_main(Replay008ViewerConfig(
        pkl_path=str(pkl_path),
        host="0.0.0.0",
        port=8091,
        max_width=960,
        slow=True,
        no_undistort=False,
        shared_detect_tags=True,
        precompute_only=False,
    ))


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
    ))


def fuse_single_frame_pose_candidates(
    raw_pkl: Path,
    deeptag_raw_pkl: Path,
    deeptag_pose_pkl: Path,
    april_strict_pkl: Path,
    loose_deeptag_pkl: Path,
    old_april_pkl: Path,
    output_pkl: Path,
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


def fill_remaining_poses_from_trajectory(
    input_pkl: Path,
    raw_pkl: Path,
    output_pkl: Path,
) -> None:
    print("[STAGE] global temporal fill", flush=True)
    temporal_completion_main(TemporalPoseCompletionConfig(
        input_pkl=input_pkl,
        raw_pkl=raw_pkl,
        output_pkl=output_pkl,
    ))


def visualize_012_pose_stream(pkl_path: Path) -> None:
    print("[STAGE] Viser pose visualization", flush=True)
    pose_viewer_main(PoseVisualizationConfig(
        pkl_path=str(pkl_path),
        host="0.0.0.0",
        port=8095,
        max_width=960,
    ))


def process_008_recording() -> Path:
    pkl_path = INPUT_PKL.expanduser().resolve()
    fmt = inspect_pkl_format(pkl_path)
    if fmt != RAW_008_PKL_FORMAT:
        raise ValueError(f"Expected 008 raw pkl format, got {fmt}: {pkl_path}")

    work_dir = (RECORDINGS_DIR / "020_work" / f"008_multistage_{pkl_path.stem}").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cube_streams = split_008_recording_into_cube_streams(pkl_path, work_dir)
    final_pose_by_cube: dict[str, Path] = {}
    for cube_name, cube_raw_pkl in cube_streams:
        cube_work_dir = work_dir / cube_name
        cube_output_pkl = cube_work_dir / f"{cube_raw_pkl.stem}_020_final.pkl"
        final_pose_by_cube[cube_name] = process_012_recording(
            raw_pkl=cube_raw_pkl,
            output_pkl=cube_output_pkl,
            work_dir=cube_work_dir,
        )

    merge_cube_pose_streams_into_008(
        raw_008_pkl=pkl_path,
        final_pose_by_cube=final_pose_by_cube,
    )
    summarize_008_pose_cache(pkl_path)
    shutil.rmtree(work_dir)
    print(f"[INFO] Removed work dir: {work_dir}")
    visualize_008_recording(pkl_path)
    return pkl_path


def process_configured_recording() -> Path:
    work_dir = (RECORDINGS_DIR / "020_work").resolve()
    output_pkl = process_012_recording(
        raw_pkl=INPUT_PKL.expanduser().resolve(),
        output_pkl=OUTPUT_PKL.expanduser().resolve(),
        work_dir=work_dir,
    )
    if work_dir.exists():
        shutil.rmtree(work_dir)
        print(f"[INFO] Removed work dir: {work_dir}")
    visualize_012_pose_stream(output_pkl)
    return output_pkl


def process_012_recording(
    *,
    raw_pkl: Path,
    output_pkl: Path,
    work_dir: Path,
) -> Path:
    raw_pkl = raw_pkl.expanduser().resolve()
    fmt = inspect_pkl_format(raw_pkl)
    if fmt not in SUPPORTED_012_INPUT_PKL_FORMATS:
        raise ValueError(
            "The 012 pipeline must start from a 012 stream with raw images "
            f"(format={SUPPORTED_012_INPUT_PKL_FORMATS}), got {fmt}: {raw_pkl}"
        )

    output_pkl = output_pkl.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if processed_output_matches_input(output_pkl, raw_pkl):
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
    outline_refine_pkl = work_dir / f"outline_recovered_pose_{raw_pkl.stem}.pkl"
    final_pose_pkl = work_dir / f"temporally_completed_pose_{raw_pkl.stem}.pkl"

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
    )
    recover_poses_from_outlines(
        input_pkl=fused_single_frame_pkl,
        raw_pkl=april_merged_pkl,
        output_pkl=outline_refine_pkl,
    )
    fill_remaining_poses_from_trajectory(
        input_pkl=outline_refine_pkl,
        raw_pkl=april_merged_pkl,
        output_pkl=final_pose_pkl,
    )

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
    if PROCESS_008_CUBE_NAMES is not None:
        requested_cube_names = set(PROCESS_008_CUBE_NAMES)
        cube_path_by_name = {cube_name_from_path(path): path for path in cube_paths}
        missing_cube_names = requested_cube_names - cube_path_by_name.keys()
        if missing_cube_names:
            raise ValueError(
                "Requested 008 cubes are missing from metadata.cube_paths: "
                f"{sorted(missing_cube_names)}"
            )
        cube_paths = [
            cube_path_by_name[name]
            for name in PROCESS_008_CUBE_NAMES
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
REALSENSE_DEFAULT_INTRINSICS_YAML = Path('/home/ps/RobotCamCalib1/outputs/intrinsics_realsense_1280x720_0707_171032.yaml')
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

def dense_deeptag_dense_points_for_frame(frame: dict[str, Any], *, tag_corner_map: dict[int, np.ndarray], min_tags: int) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int], dict[str, Any]]:
    cluster_orders = frame.get('cluster_stats', {}).get('cluster_corner_orders', {}) or {}
    cluster_orders = {int(k): str(v) for k, v in cluster_orders.items()}
    order_votes: dict[str, int] = {}
    for order in cluster_orders.values():
        if order in DENSE_DEEPTAG_CORNER_ORDER_TRANSFORMS:
            order_votes[order] = order_votes.get(order, 0) + 1
    dominant_order = max(order_votes.items(), key=lambda item: item[1])[0] if order_votes else 'id'
    decoded_by_id: dict[int, dict[str, Any]] = {}
    for decoded in frame.get('decoded_tags', []) or []:
        if not decoded.get('is_valid', False):
            continue
        tag_id = int(decoded.get('tag_id', -1))
        if tag_id in tag_corner_map:
            decoded_by_id[tag_id] = decoded
    obj_chunks: list[np.ndarray] = []
    img_chunks: list[np.ndarray] = []
    tag_ids: list[int] = []
    point_counts: dict[int, int] = {}
    for tag_id in sorted(decoded_by_id):
        decoded = decoded_by_id[tag_id]
        image_points = np.asarray(decoded.get('keypoints_in_images', []), dtype=np.float64).reshape(-1, 2)
        if image_points.shape[0] < 4:
            continue
        local_xy = dense_deeptag_dense_local_annotations(image_points.shape[0])
        corner_order = cluster_orders.get(int(tag_id), dominant_order)
        affine_t = dense_deeptag_local_to_cube_affine(tag_corner_map[tag_id], corner_order)
        object_points = np.c_[local_xy, np.ones(local_xy.shape[0], dtype=np.float64)] @ affine_t
        obj_chunks.append(object_points.astype(np.float64))
        img_chunks.append(image_points.astype(np.float64))
        tag_ids.append(int(tag_id))
        point_counts[int(tag_id)] = int(image_points.shape[0])
    if len(tag_ids) < int(min_tags):
        return (np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), tag_ids, point_counts, {'reason': f'dense_tags_too_small:{len(tag_ids)}<{int(min_tags)}'})
    return (np.vstack(obj_chunks), np.vstack(img_chunks), tag_ids, point_counts, {'cluster_corner_order_count': int(len(cluster_orders)), 'corner_order_fallback': dominant_order, 'used_fallback_order_tag_ids': [int(tag_id) for tag_id in tag_ids if int(tag_id) not in cluster_orders]})

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

def dense_deeptag_solve_single_face_dense_pose(object_points: np.ndarray, image_points: np.ndarray, tag_ids: list[int], point_counts: dict[int, int], *, cube_config: Any, face_name: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, max_reproj: float, point_reject_px: float, tag_reject_px: float, min_tags: int, min_inlier_tag_fraction: float, coverage_check_min_raw_tags: int, max_required_inlier_tags: int) -> dict[str, Any]:
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
            if int(tag_active.sum()) >= 4:
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
    if not dense_deeptag_face_normals_ok(rvec, {face_name}):
        return {'success': False, 'failure_reason': 'dense_single_face_normal_away', 'reproj_error': float('inf'), 'raw_reproj_error': reproj}
    if len(used_ids) >= 2:
        quality_level = 'B'
        quality_reason = f'dense_singleface_face_frame:{len(used_ids)}tags'
    else:
        quality_level = 'C'
        quality_reason = 'dense_singletag_face_frame'
    return {'success': True, 'failure_reason': '', 'pose_source': 'deeptag_dense_keypoints_single_face_ippe_cfg_transform', 'quality_level': quality_level, 'quality_reason': quality_reason, 'pose_filled': False, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': dense_deeptag_transform_from_rvec_tvec(rvec, tvec), 'reproj_error': reproj, 'n_points': int(active.sum()), 'n_points_raw': int(object_points.shape[0]), 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'visible_faces': {face_name}, 'single_face_name': face_name, 'single_face_ippe_candidates': int(candidate_count), 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}

def dense_deeptag_solve_dense_pose(object_points: np.ndarray, image_points: np.ndarray, tag_ids: list[int], point_counts: dict[int, int], *, cube_config: Any, face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, ransac_reproj: float, max_reproj: float, point_reject_px: float, tag_reject_px: float, min_tags: int, min_inlier_tag_fraction: float, coverage_check_min_raw_tags: int, max_required_inlier_tags: int) -> dict[str, Any]:
    if object_points.shape[0] < 4:
        return {'success': False, 'failure_reason': 'dense_no_points', 'reproj_error': float('inf')}
    raw_visible_faces = dense_deeptag_visible_faces_for_ids(face_id_sets, tag_ids)
    if len(raw_visible_faces) == 1:
        return dense_deeptag_solve_single_face_dense_pose(object_points, image_points, tag_ids, point_counts, cube_config=cube_config, face_name=next(iter(raw_visible_faces)), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, max_reproj=max_reproj, point_reject_px=point_reject_px, tag_reject_px=tag_reject_px, min_tags=min_tags, min_inlier_tag_fraction=min_inlier_tag_fraction, coverage_check_min_raw_tags=coverage_check_min_raw_tags, max_required_inlier_tags=max_required_inlier_tags)
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
            if int(tag_active.sum()) >= 4:
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
    visible_faces = dense_deeptag_visible_faces_for_ids(face_id_sets, used_ids)
    if not dense_deeptag_face_normals_ok(rvec, visible_faces):
        return {'success': False, 'failure_reason': 'dense_face_normal_away', 'reproj_error': float('inf'), 'raw_reproj_error': reproj}
    if len(visible_faces) >= 2:
        quality_level = 'A'
        quality_reason = f'dense_multiface:{len(visible_faces)}faces/{len(used_ids)}tags'
    elif len(used_ids) >= 2:
        quality_level = 'B'
        quality_reason = f'dense_multitag_singleface:{len(used_ids)}tags'
    else:
        quality_level = 'C'
        quality_reason = 'dense_single_tag_planar'
    return {'success': True, 'failure_reason': '', 'pose_source': 'deeptag_dense_keypoints_all_point_pnp', 'quality_level': quality_level, 'quality_reason': quality_reason, 'pose_filled': False, 'rvec': np.asarray(rvec, dtype=np.float64).reshape(3, 1), 'tvec': np.asarray(tvec, dtype=np.float64).reshape(3, 1), 'T': dense_deeptag_transform_from_rvec_tvec(rvec, tvec), 'reproj_error': reproj, 'n_points': int(active.sum()), 'n_points_raw': int(object_points.shape[0]), 'n_tags': int(len(used_ids)), 'tag_ids': used_ids, 'visible_faces': visible_faces, 'per_tag_reproj_error': per_tag_reproj, 'per_tag_inlier_points': per_tag_inliers, 'rejected_points': int(rejected_points), 'rejected_tags': rejected_tags}

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
    source_path, source_offsets, script012, undistort_pack = dense_deeptag_make_source_frame_loader(header)
    output_pkl = Path(args.output_pkl).expanduser().resolve()
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    total_points = 0
    t0 = time.perf_counter()
    with output_pkl.open('wb') as f:
        pickle.dump({'type': 'header', 'format': 'deeptag_012_offline_stream_v1', 'source_pkl': str(input_pkl), 'source_footer': footer, 'metadata': {'script': str(DENSE_DEEPTAG_THIS_FILE), 'method': 'DeepTag dense keypoints; single-face frames use cfg face-frame IPPE then fixed face-to-cube transform; multiface frames use cube-frame all-point PnP; no temporal filter', 'cube_cfg': str(runtime['cube_cfg']), 'camera_matrix': runtime['camera_matrix'].tolist(), 'dist_coeffs': runtime['dist_coeffs'].tolist(), 'frame_count': int(len(offsets)), 'min_tags': int(args.min_tags), 'ransac_reproj': float(args.ransac_reproj), 'max_reproj': float(args.max_reproj), 'point_reject_px': float(args.point_reject_px), 'tag_reject_px': float(args.tag_reject_px), 'min_inlier_tag_fraction': float(args.min_inlier_tag_fraction), 'coverage_check_min_raw_tags': int(args.coverage_check_min_raw_tags), 'max_required_inlier_tags': int(args.max_required_inlier_tags), 'input_header': dense_deeptag_to_json_compatible(header)}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for out_idx, offset in enumerate(offsets):
            frame = dense_deeptag_load_at(input_pkl, offset)
            object_points, image_points, tag_ids, point_counts, dense_stats = dense_deeptag_dense_points_for_frame(frame, tag_corner_map=runtime['tag_corner_map'], min_tags=int(args.min_tags))
            if object_points.shape[0] >= 4:
                pose = dense_deeptag_solve_dense_pose(object_points, image_points, tag_ids, point_counts, cube_config=runtime['cube_config'], face_id_sets=runtime['face_id_sets'], camera_matrix=runtime['camera_matrix'], dist_coeffs=runtime['dist_coeffs'], ransac_reproj=float(args.ransac_reproj), max_reproj=float(args.max_reproj), point_reject_px=float(args.point_reject_px), tag_reject_px=float(args.tag_reject_px), min_tags=int(args.min_tags), min_inlier_tag_fraction=float(args.min_inlier_tag_fraction), coverage_check_min_raw_tags=int(args.coverage_check_min_raw_tags), max_required_inlier_tags=int(args.max_required_inlier_tags))
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

def pose_recovery_edge_alignment_score(gray: np.ndarray, pose: dict[str, Any], *, config: Any, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> float:
    if not pose.get('success', False):
        return 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    rvec = np.asarray(pose['rvec'], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(pose['tvec'], dtype=np.float64).reshape(3, 1)
    corners_2d, _ = cv2.projectPoints(pose_recovery_cube_corners(config), rvec, tvec, camera_matrix, dist_coeffs)
    corners_2d = corners_2d.reshape(-1, 2)
    h, w = gray.shape[:2]
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
    keys = {'success', 'failure_reason', 'n_tags', 'n_points', 'n_inliers', 'reproj_error', 'tag_ids', 'visible_faces', 'pose_source', 'pose_filled', 'quality_level', 'quality_reason', 'edge_score', 'rvec', 'tvec', 'T'}
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

def single_frame_fusion_apriltag_single_tag_pose(bm: Any, detections: list[tuple[int, np.ndarray]], gray: np.ndarray, *, config: Any, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, max_reproj: float) -> Any:
    best: Any | None = None
    for tag_id, corners in detections:
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
        if best is None or (candidate.edge_score, -candidate.reproj_error) > (best.edge_score or 0.0, -best.reproj_error):
            best = candidate
    if best is None:
        return RecoveryPoseCandidate(False, 'apriltag_single_tag_cfg_pose', frame_index, {}, [], float('inf'), failure_reason='no_single_tag_cfg_candidate')
    return best

def single_frame_fusion_deeptag_single_tag_dense_pose(bm: Any, dense020: Any, deeptag_frame: dict[str, Any], gray: np.ndarray, *, config: Any, tag_corner_map: dict[int, np.ndarray], face_id_sets: dict[str, set[int]], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, frame_index: int, max_reproj: float) -> Any:
    object_points, image_points, tag_ids, point_counts, dense_stats = dense_deeptag_dense_points_for_frame(deeptag_frame, tag_corner_map=tag_corner_map, min_tags=1)
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
        pickle.dump({'type': 'header', 'format': 'aprilcube_deeptag_fused_stream_v1', 'created_wall_time': time.strftime('%Y-%m-%d %H:%M:%S'), 'source_raw_pkl': str(args.raw_pkl.resolve()), 'source_deeptag_pose_pkl': str(args.deeptag_pose_pkl.resolve()), 'source_april_strict_pkl': str(args.april_strict_pkl.resolve()), 'source_deeptag_raw_pkl': str(args.deeptag_raw_pkl.resolve()), 'source_loose_deeptag_pkl': str(args.loose_deeptag_pkl.resolve()), 'source_old_april_pkl': str(args.old_april_pkl.resolve()), 'raw_footer': raw_footer, 'deeptag_footer': dt_footer, 'april_footer': ap_footer, 'deeptag_raw_footer': deeptag_raw_footer, 'loose_footer': loose_footer, 'old_footer': old_footer, 'metadata': {'script': str(SINGLE_FRAME_FUSION_THIS_FILE), 'method': 'single-frame cascade: DeepTag coverage/min-tag2, strict AprilCube, tag-center multiface PnP, face board, AprilTag preprocessing sweep, DeepTag-AprilTag cross validation, edge-checked loose candidates; no temporal filter or fill', 'frame_count': len(frame_indices), 'min_tags': int(args.min_tags), 'max_reproj': float(args.max_reproj), 'edge_threshold': float(args.edge_threshold), 'single_tag_edge_threshold': float(args.single_tag_edge_threshold), 'single_tag_max_reproj': float(args.single_tag_max_reproj)}}, f, protocol=pickle.HIGHEST_PROTOCOL)
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
                fused_pose = single_frame_fusion_copy_pose_with_stage(dt_pose, source='stage1_deeptag_dense_coverage_mintag2', quality_level='A', quality_reason=f"deeptag_dense_reproj:{float(dt_pose.get('reproj_error', float('inf'))):.2f}")
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
                stage_candidates.extend([('stage7_edge_checked_loose_candidate', 'G', best_edge or RecoveryPoseCandidate(False, 'edge_checked_loose_candidates', int(frame_index), {}, [], float('inf'), failure_reason='no_edge_accepted_candidate')), ('stage8_apriltag_single_tag_cfg_edge', 'H', single_frame_fusion_apriltag_single_tag_pose(bm, detections, gray, config=config, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), max_reproj=float(args.single_tag_max_reproj))), ('stage9_deeptag_single_tag_dense_edge', 'I', single_frame_fusion_deeptag_single_tag_dense_pose(bm, dense020, deeptag_raw, gray, config=config, tag_corner_map=tag_corner_map, face_id_sets=face_id_sets, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, frame_index=int(frame_index), max_reproj=float(args.single_tag_max_reproj)))])
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


# ---- Global temporal fill ----
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
    cv2.putText(vis, f"Global temporal fill: {pose.get('pose_source', '')}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
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
    if len(valid_indices) < 4:
        args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.input_pkl, args.output_pkl)
        print(
            '[WARN] Skipping global temporal fill: '
            f'need at least 4 valid poses, got {len(valid_indices)}; '
            f'copied input stream to {args.output_pkl}'
        )
        return
    x = np.asarray(valid_indices, dtype=np.float64)
    translations = np.vstack([np.asarray(frames[idx]['pose']['tvec'], dtype=np.float64).reshape(3) for idx in valid_indices])
    splines = [UnivariateSpline(x, translations[:, dim], k=3, s=float(args.translation_smooth)) for dim in range(3)]
    rotations = Rotation.from_matrix(np.stack([temporal_completion_rotation_from_rvec(frames[idx]['pose']['rvec']) for idx in valid_indices], axis=0))
    slerp = Slerp(x, rotations)
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
        tvec = np.array([spline(float(idx)) for spline in splines], dtype=np.float64).reshape(3, 1)
        rvec = slerp([float(idx)]).as_rotvec()[0].reshape(3, 1).astype(np.float64)
        raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
        image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
        detect_frame = realsense_undistort_frame(image, undistort_pack)
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
        edge_score = pose_recovery_edge_alignment_score(gray, {'success': True, 'rvec': rvec, 'tvec': tvec}, config=config, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        filled[idx] = {'success': True, 'pose_source': 'stage11_global_temporal_filter_fill', 'quality_level': 'F', 'quality_reason': f'global_temporal_fill;bracket:{prev_idx}-{next_idx};gap:{gap};edge:{edge_score:.2f}', 'pose_filled': True, 'temporal_filter_fill': True, 'single_frame_only': False, 'rvec': rvec, 'tvec': tvec, 'T': temporal_completion_make_pose_transform(rvec, tvec), 'reproj_error': float('nan'), 'n_tags': 0, 'tag_ids': [], 'visible_faces': [], 'edge_score': float(edge_score), 'prev_success_frame': int(prev_idx), 'next_success_frame': int(next_idx), 'bracket_gap': int(gap), 'translation_smooth': float(args.translation_smooth)}
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
        out_header['metadata'] = {**(out_header.get('metadata', {}) or {}), 'script': str(TEMPORAL_COMPLETION_THIS_FILE), 'method': 'fill only remaining failed frames from whole-sequence temporal pose trajectory', 'translation_smooth': float(args.translation_smooth), 'max_bracket_gap': int(args.max_bracket_gap)}
        pickle.dump(out_header, f, protocol=pickle.HIGHEST_PROTOCOL)
        for idx in indices:
            frame = copy.deepcopy(frames[idx])
            if idx in filled:
                raw_record = pose_recovery_load_at(args.raw_pkl, raw_offset_by_frame[int(idx)])
                image = np.asarray(raw_record['image_bgr'], dtype=np.uint8)
                detect_frame = realsense_undistort_frame(image, undistort_pack)
                frame['pose_original'] = copy.deepcopy(frame.get('pose', {}))
                frame['pose'] = filled[idx]
                frame['selected_stage'] = 'stage11_global_temporal_filter_fill'
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


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("020_finalize_pose_postprocess.py does not accept CLI args; edit constants at the top.")

    fmt = inspect_pkl_format(INPUT_PKL.expanduser().resolve())
    print(f"[INFO] Auto detected input format: {fmt}")
    if fmt == RAW_008_PKL_FORMAT:
        output = process_008_recording()
    elif fmt in SUPPORTED_012_INPUT_PKL_FORMATS:
        output = process_configured_recording()
    else:
        raise ValueError(f"Unsupported input pkl format: {fmt}")
    print(f"[INFO] Done: {output}")


if __name__ == "__main__":
    main()
