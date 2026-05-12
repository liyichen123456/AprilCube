from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from aprilcube_pose_benchmark.common_io import (
    clahe_gray_from_bgr,
    get_camera_record,
    get_detect_frame_bgr,
    get_single_camera_name,
    load_recording,
)
from aprilcube_pose_benchmark.common_plot import (
    compute_metrics,
    save_metrics,
    save_npz,
    save_pose_curve,
    save_reprojection_curve,
)
from aprilcube_pose_benchmark.common_pose import (
    build_detector_from_record,
    create_native_detector,
    detect_pupil_tags,
    empty_result,
    pose_xyz_rpy_from_result,
)
from aprilcube_pose_benchmark.common_viser import draw_gray_overlay, play_results_in_viser


@dataclass
class BenchmarkConfig:
    algorithm_name: str
    pkl_path: str
    output_root: str
    enable_viser: bool
    viser_host: str
    viser_port: int
    playback_fps: float
    loop_playback: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: tuple[int, int]
    estimate_tag_pose: bool


AlgorithmFn = Callable[[Any, Any, list[Any], np.ndarray, dict[str, Any]], dict[str, Any]]


def run_benchmark(config: BenchmarkConfig, algorithm_fn: AlgorithmFn) -> dict[str, Any]:
    meta, frames = load_recording(config.pkl_path)
    if not frames:
        raise ValueError(f"No frames in recording: {config.pkl_path}")
    camera_name = get_single_camera_name(frames)
    first_record = None
    for frame in frames:
        first_record = get_camera_record(frame, camera_name)
        if first_record is not None:
            break
    if first_record is None:
        raise ValueError(f"No camera record found for {camera_name}")

    cube_path = Path(str(meta.get("cube_path", ""))).expanduser().resolve()
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube path from pkl does not exist: {cube_path}")

    detector = build_detector_from_record(cube_path, first_record)
    native_detector = create_native_detector(detector)
    context: dict[str, Any] = {
        "camera_name": camera_name,
        "meta": meta,
        "config": config,
        "detector": detector,
    }

    output_dir = Path(config.output_root).expanduser().resolve() / Path(config.pkl_path).stem / config.algorithm_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices: list[int] = []
    success_list: list[bool] = []
    xyz_mm: list[np.ndarray] = []
    rpy_deg: list[np.ndarray] = []
    reproj_errors: list[float] = []
    visible_face_count: list[int] = []
    visible_tag_count: list[int] = []
    frames_out: list[dict[str, Any]] = []

    for seq_idx, frame in enumerate(frames):
        record = get_camera_record(frame, camera_name)
        if record is None:
            continue
        frame_idx = int(frame.get("frame_idx", seq_idx))
        image_bgr = get_detect_frame_bgr(record)
        gray = clahe_gray_from_bgr(
            image_bgr,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
        tags = detect_pupil_tags(
            detector,
            native_detector,
            gray,
            estimate_tag_pose=config.estimate_tag_pose,
        )
        try:
            result = algorithm_fn(detector, native_detector, tags, gray, context)
        except Exception as exc:
            print(f"[WARNING] {config.algorithm_name} frame={frame_idx} failed: {type(exc).__name__}: {exc}")
            result = empty_result()

        if not isinstance(result.get("detections", None), list):
            result["detections"] = [(int(tag.tag_id), np.asarray(tag.corners, dtype=np.float64).reshape(4, 2)) for tag in tags]
        result.setdefault("n_tags", len(result["detections"]))
        result.setdefault("tag_ids", [int(tag_id) for tag_id, _corners in result["detections"]])

        pose = pose_xyz_rpy_from_result(result)
        if pose is None:
            xyz = np.full(3, np.nan, dtype=np.float64)
            rpy = np.full(3, np.nan, dtype=np.float64)
        else:
            xyz, rpy = pose
        frame_indices.append(frame_idx)
        success_list.append(bool(result.get("success", False)))
        xyz_mm.append(xyz)
        rpy_deg.append(rpy)
        reproj_errors.append(float(result.get("reproj_error", np.nan)))
        visible_faces = result.get("visible_faces", set())
        visible_face_count.append(len(visible_faces) if visible_faces is not None else 0)
        visible_tag_count.append(int(result.get("n_tags", 0)))

        overlay = draw_gray_overlay(detector, gray, result)
        frames_out.append(
            {
                "frame_idx": frame_idx,
                "result": result,
                "gray_overlay_rgb": overlay,
            }
        )

    frame_indices_arr = np.asarray(frame_indices, dtype=np.int64)
    success_arr = np.asarray(success_list, dtype=bool)
    xyz_arr = np.vstack(xyz_mm) if xyz_mm else np.empty((0, 3), dtype=np.float64)
    rpy_arr = np.vstack(rpy_deg) if rpy_deg else np.empty((0, 3), dtype=np.float64)
    reproj_arr = np.asarray(reproj_errors, dtype=np.float64)
    face_count_arr = np.asarray(visible_face_count, dtype=np.int64)
    tag_count_arr = np.asarray(visible_tag_count, dtype=np.int64)

    save_npz(
        output_dir,
        frame_indices=frame_indices_arr,
        success=success_arr,
        xyz_mm=xyz_arr,
        rpy_deg=rpy_arr,
        reproj_error_px=reproj_arr,
        visible_face_count=face_count_arr,
        visible_tag_count=tag_count_arr,
    )
    pose_plot = save_pose_curve(output_dir, config.algorithm_name, frame_indices_arr, xyz_arr, rpy_arr)
    reproj_plot = save_reprojection_curve(output_dir, config.algorithm_name, frame_indices_arr, reproj_arr)
    metrics = compute_metrics(success_arr, xyz_arr, rpy_arr, reproj_arr)
    metrics.update(
        {
            "algorithm_name": config.algorithm_name,
            "pkl_path": str(Path(config.pkl_path).expanduser().resolve()),
            "camera_name": camera_name,
            "cube_path": str(cube_path),
            "clahe_clip_limit": float(config.clahe_clip_limit),
            "clahe_tile_grid_size": list(config.clahe_tile_grid_size),
            "pose_plot": str(pose_plot),
            "reproj_plot": str(reproj_plot),
        }
    )
    save_metrics(output_dir, metrics)
    print(
        f"[RESULT] {config.algorithm_name}: success={metrics['num_success']}/{metrics['num_frames']} "
        f"rate={metrics['success_rate']:.3f} output={output_dir}"
    )

    if config.enable_viser:
        play_results_in_viser(
            algorithm_name=config.algorithm_name,
            output_dir=output_dir,
            frames_out=frames_out,
            playback_fps=config.playback_fps,
            host=config.viser_host,
            port=config.viser_port,
            loop=config.loop_playback,
        )

    return metrics


def result_from_pnp_tuple(detector: Any, detections: list[tuple[int, np.ndarray]], pose_tuple: Any) -> dict[str, Any]:
    from aprilcube_pose_benchmark.common_pose import result_from_pose

    if pose_tuple is None:
        result = empty_result()
        result["detections"] = detections
        result["n_tags"] = len(detections)
        return result
    rvec, tvec, reproj, n_inliers = pose_tuple
    return result_from_pose(
        detector,
        detections,
        rvec,
        tvec,
        reproj_error=float(reproj),
        n_inliers=int(n_inliers),
    )

