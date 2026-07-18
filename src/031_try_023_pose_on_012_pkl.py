#!/usr/bin/env python3
"""Run the 023 DeepTag-first pose estimator on a single-camera 012 PKL.

This adapter deliberately writes a separate compact pose stream.  It never
rewrites the source 012 recording in place, unlike the original 023 script's
021 workflow.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pickle
import sys
import time
from collections import Counter, deque
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import numpy as np


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    APRILCUBE_ROOT
    / "recordings/012_rs_raw_frames_20260710_214336_with_final_postprocessed_pose.pkl"
)
DEFAULT_OUTPUT = (
    APRILCUBE_ROOT
    / "recordings/023_trial_deeptag_internal_grid_012_rs_raw_frames_20260710_214336.pkl"
)
SUPPORTED_INPUT_FORMATS = {
    "aprilcube_rs_raw_frame_stream_v1",
    "aprilcube_012_raw_with_pose_stream_v1",
    "aprilcube_012_raw_with_final_postprocessed_pose_stream_v1",
    "aprilcube_raw_with_020_postprocessed_pose_stream_v1",
}
OUTPUT_FORMAT = "deeptag_012_offline_stream_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try the 023 DeepTag-first estimator on a 012 pose PKL."
    )
    parser.add_argument("input_pkl", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--cube-cfg", type=Path)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_023_module() -> ModuleType:
    path = Path(__file__).with_name("023_offline_pos_estimate_021_pkl.py")
    spec = importlib.util.spec_from_file_location("aprilcube_offline_023", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load 023 estimator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nested_dicts(root: dict[str, Any]) -> Iterator[dict[str, Any]]:
    queue: deque[dict[str, Any]] = deque([root])
    while queue:
        item = queue.popleft()
        yield item
        for value in item.values():
            if isinstance(value, dict):
                queue.append(value)


def infer_calibration_and_cube(header: dict[str, Any]) -> tuple[Path, Path]:
    for item in nested_dicts(header):
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            continue
        intrinsics = metadata.get("intrinsics_yaml")
        cube_cfg = metadata.get("cube_cfg")
        if intrinsics and cube_cfg:
            return Path(intrinsics).expanduser().resolve(), Path(cube_cfg).expanduser().resolve()
    raise ValueError("Could not infer intrinsics_yaml and cube_cfg from the 012 header")


def as_pose_dict(result: dict[str, Any]) -> dict[str, Any]:
    pose = dict(result)
    pose["pose_source"] = (
        "023_deeptag_internal_grid"
        if pose.get("pose_backend") == "deeptag_internal_grid"
        else "023_cv2_fallback"
        if pose.get("pose_backend") == "cv2_fallback"
        else "023_failed"
    )
    pose["pose_filled"] = False
    return pose


def rotation_step_deg(previous: np.ndarray, current: np.ndarray) -> float:
    def matrix(rotvec: np.ndarray) -> np.ndarray:
        vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
        theta = float(np.linalg.norm(vector))
        if theta < 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z = vector / theta
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)

    relative = matrix(previous).T @ matrix(current)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def percentile_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    input_pkl = args.input_pkl.expanduser().resolve()
    output_pkl = args.output_pkl.expanduser().resolve()
    if not input_pkl.is_file():
        raise FileNotFoundError(input_pkl)
    if output_pkl.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_pkl}")

    with input_pkl.open("rb") as source:
        source_header = pickle.load(source)
    if not isinstance(source_header, dict) or source_header.get("format") not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported 012 format: {source_header.get('format') if isinstance(source_header, dict) else type(source_header).__name__}")

    inferred_intrinsics, inferred_cube = infer_calibration_and_cube(source_header)
    intrinsics = (args.intrinsics or inferred_intrinsics).expanduser().resolve()
    cube_cfg = (args.cube_cfg or inferred_cube).expanduser().resolve()
    if not intrinsics.is_file():
        raise FileNotFoundError(f"Intrinsics YAML not found: {intrinsics}")
    if not cube_cfg.is_dir():
        raise FileNotFoundError(f"Cube config directory not found: {cube_cfg}")

    module_023 = load_023_module()
    estimator = module_023.CameraPoseEstimator("012_camera", intrinsics, [cube_cfg])
    print(f"[INFO] input={input_pkl}", flush=True)
    print(f"[INFO] output={output_pkl}", flush=True)
    print(f"[INFO] intrinsics={intrinsics}", flush=True)
    print(f"[INFO] cube_cfg={cube_cfg}", flush=True)
    print(f"[INFO] deeptag_device={estimator.deeptag_backend.device}", flush=True)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_pkl.with_name(f".{output_pkl.name}.tmp")
    temporary.unlink(missing_ok=True)
    success_count = 0
    backend_counts: Counter[str] = Counter()
    reprojection_errors: list[float] = []
    translation_steps: list[float] = []
    rotation_steps: list[float] = []
    previous_success: tuple[int, np.ndarray, np.ndarray] | None = None
    frame_count = 0
    started = time.perf_counter()

    try:
        with input_pkl.open("rb") as source, temporary.open("wb") as destination:
            pickle.load(source)
            output_header = {
                "type": "header",
                "format": OUTPUT_FORMAT,
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_pkl": str(input_pkl),
                "metadata": {
                    "script": str(Path(__file__).resolve()),
                    "method": "023 DeepTag internal-grid primary with CV2/CLAHE fallback",
                    "algorithm": module_023.OFFLINE_POS_ALGORITHM,
                    "intrinsics_yaml": str(intrinsics),
                    "cube_cfg": str(cube_cfg),
                    "deeptag_device": str(estimator.deeptag_backend.device),
                    "source_pose_field": "pose",
                    "source_pose_copy_field": "pose_original_final",
                },
            }
            pickle.dump(output_header, destination, protocol=pickle.HIGHEST_PROTOCOL)

            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break
                if not isinstance(record, dict) or record.get("type") != "frame":
                    continue
                if args.max_frames > 0 and frame_count >= int(args.max_frames):
                    break

                frame_index = int(record.get("frame_index", frame_count))
                camera_record = {
                    "image_bgr": np.asarray(record["image_bgr"], dtype=np.uint8),
                    "capture_timestamp": float(record["capture_timestamp"]),
                    "sequence": frame_index,
                }
                offline_pos = estimator.estimate(camera_record)
                estimator.add_visualization_images(camera_record, offline_pos)
                cube_results = offline_pos.get("cube_results", [])
                if len(cube_results) != 1:
                    raise RuntimeError(f"Expected one cube result, got {len(cube_results)}")
                pose = as_pose_dict(cube_results[0].get("result", {}))
                backend = str(pose.get("pose_backend") or "failed")
                backend_counts[backend] += 1

                if pose.get("success", False) and pose.get("rvec") is not None and pose.get("tvec") is not None:
                    success_count += 1
                    reprojection = float(pose.get("reproj_error", float("nan")))
                    if np.isfinite(reprojection):
                        reprojection_errors.append(reprojection)
                    rvec = np.asarray(pose["rvec"], dtype=np.float64).reshape(3)
                    tvec = np.asarray(pose["tvec"], dtype=np.float64).reshape(3)
                    if previous_success is not None and frame_index == previous_success[0] + 1:
                        translation_steps.append(float(np.linalg.norm(tvec - previous_success[2])))
                        rotation_steps.append(rotation_step_deg(previous_success[1], rvec))
                    previous_success = (frame_index, rvec, tvec)
                else:
                    previous_success = None

                output_frame = {
                    "type": "frame",
                    "frame_index": frame_index,
                    "loop_frame_idx": int(record.get("loop_frame_idx", frame_index)),
                    "capture_timestamp": float(record["capture_timestamp"]),
                    "pose": pose,
                    "pose_original_final": record.get("pose", {}),
                    "selected_stage": pose["pose_source"],
                    "overlay_format": "jpeg_bgr",
                    "overlay_jpeg": camera_record[module_023.UNDISTORTED_POSE_OVERLAY_JPEG_FIELD],
                    "overlay_shape": tuple(int(value) for value in estimator.calibration["image_size"][::-1]) + (3,),
                }
                pickle.dump(output_frame, destination, protocol=pickle.HIGHEST_PROTOCOL)
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"\r[INFO] frames={frame_count} success={success_count} "
                        f"speed={frame_count / max(elapsed, 1e-9):.2f}fps",
                        end="",
                        flush=True,
                    )

            statistics = {
                "frame_count": frame_count,
                "success_count": success_count,
                "backend_counts": dict(backend_counts),
                "reprojection_error_px": percentile_summary(reprojection_errors),
                "translation_step_mm": percentile_summary(translation_steps),
                "rotation_step_deg": percentile_summary(rotation_steps),
            }
            pickle.dump({"type": "footer", **statistics}, destination, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(output_pkl)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print()
    print(f"[INFO] saved={output_pkl}")
    print(f"[INFO] statistics={statistics}")


if __name__ == "__main__":
    main()
