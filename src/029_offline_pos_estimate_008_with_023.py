#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_023_PATH = APRILCUBE_ROOT / "src/023_offline_pos_estimate_021_pkl.py"
DEFAULT_PKL_PATH = (
    APRILCUBE_ROOT / "recordings/008_raw_frames_20260715_000555.pkl"
)
DEFAULT_INDEX_CUBE_NAME = "cube_april_36h11_6_11_1x1x1_15mm"

EXPECTED_PKL_FORMAT = "aprilcube_raw_frame_stream_v1"
POSE_CACHE_FORMAT = "aprilcube_023_deeptag_008_pose_v1"
OFFLINE_POS_FIELD = "offline_pos"
OFFLINE_POS_CACHE_KEY_FIELD = "offline_pos_cache_key"
PROGRESS_PRINT_INTERVAL_S = 0.5


def load_script_023() -> Any:
    module_name = "aprilcube_offline_pos_023"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_023_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SCRIPT_023_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_header(pkl_path: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as source:
        header = pickle.load(source)
    if not isinstance(header, dict) or header.get("type") != "header":
        raise ValueError(f"Invalid 008 PKL header: {pkl_path}")
    if header.get("format") != EXPECTED_PKL_FORMAT:
        raise ValueError(
            f"Unsupported PKL format {header.get('format')!r}; "
            f"expected {EXPECTED_PKL_FORMAT!r}"
        )
    return header


def selected_cube_paths(
    header: dict[str, Any], requested_cube_names: list[str] | None
) -> list[Path]:
    configured = [
        Path(value).expanduser().resolve()
        for value in header.get("metadata", {}).get("cube_paths", [])
    ]
    if not configured:
        raise ValueError("008 PKL header contains no cube_paths")
    if not requested_cube_names:
        selected = configured
    else:
        requested = set(requested_cube_names)
        selected = [path for path in configured if path.name in requested]
        missing = sorted(requested - {path.name for path in selected})
        if missing:
            raise ValueError(f"Requested cubes are absent from the PKL header: {missing}")
    for path in selected:
        if not path.exists():
            raise FileNotFoundError(f"Cube config not found: {path}")
    return selected


def build_estimators(
    module_023: Any,
    header: dict[str, Any],
    cube_paths: list[Path],
) -> dict[str, Any]:
    metadata = header.get("metadata", {})
    intrinsics_by_camera = metadata.get("intrinsics_yaml", {})
    camera_names = metadata.get("opened_cameras") or list(intrinsics_by_camera)
    estimators: dict[str, Any] = {}
    for camera_name_value in camera_names:
        camera_name = str(camera_name_value)
        intrinsics_value = intrinsics_by_camera.get(camera_name)
        if not intrinsics_value:
            raise ValueError(f"008 PKL header lacks intrinsics for {camera_name}")
        estimators[camera_name] = module_023.CameraPoseEstimator(
            camera_name,
            Path(intrinsics_value),
            cube_paths,
        )
    if not estimators:
        raise ValueError("008 PKL header contains no active cameras")
    return estimators


def build_cache_key(
    module_023: Any,
    pkl_path: Path,
    estimators: dict[str, Any],
    cube_paths: list[Path],
) -> dict[str, Any]:
    return {
        "format": POSE_CACHE_FORMAT,
        "algorithm": module_023.OFFLINE_POS_ALGORITHM,
        "source_pkl": str(pkl_path),
        "camera_names": list(estimators),
        "intrinsics_yaml": {
            camera_name: str(estimator.intrinsics_yaml)
            for camera_name, estimator in estimators.items()
        },
        "cube_paths": [str(path) for path in cube_paths],
        "pose_backend": module_023.POSE_BACKEND,
        "deeptag_parameters": module_023.backend_parameters(),
        "fallback_backend": "cv2_adaptive_clahe_single_tag_face_frame",
        "undistort_before_detection": True,
        "source_image_field": "image_bgr",
    }


def print_progress(
    completed_bytes: int,
    total_bytes: int,
    frame_count: int,
    started: float,
    success_counts: dict[str, dict[str, int]],
) -> None:
    ratio = min(max(completed_bytes / max(total_bytes, 1), 0.0), 1.0)
    width = 30
    filled = int(round(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.perf_counter() - started, 1e-9)
    summaries = []
    for camera_name, cube_counts in success_counts.items():
        counts = ",".join(f"{name}:{count}" for name, count in cube_counts.items())
        summaries.append(f"{camera_name}={counts}")
    sys.stdout.write(
        f"\r[INFO] 023-on-008 [{bar}] {ratio * 100:5.1f}% "
        f"frames={frame_count} speed={frame_count / elapsed:.2f}frame/s "
        + " ".join(summaries)
    )
    sys.stdout.flush()


def rewrite_with_023_poses(
    module_023: Any,
    pkl_path: Path,
    header: dict[str, Any],
    estimators: dict[str, Any],
    cube_paths: list[Path],
    cache_key: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    module_023.ensure_enough_space_for_safe_rewrite(pkl_path)
    source_size = pkl_path.stat().st_size
    temporary_path = pkl_path.with_name(f".{pkl_path.name}.029-rewrite.tmp")
    temporary_path.unlink(missing_ok=True)

    success_counts = {
        camera_name: {
            cube_name: 0 for cube_name, _detector in estimator.detectors
        }
        for camera_name, estimator in estimators.items()
    }
    statistics = module_023.make_pose_statistics(estimators)
    frame_count = 0
    started = time.perf_counter()
    last_progress = started

    updated_header = copy.deepcopy(header)
    updated_header.setdefault("metadata", {})["offline_pos_estimation_023_008"] = {
        "cache_format": POSE_CACHE_FORMAT,
        "algorithm": module_023.OFFLINE_POS_ALGORITHM,
        "pose_backend": module_023.POSE_BACKEND,
        "fallback_backend": "cv2_adaptive_clahe_single_tag_face_frame",
        "cube_paths": [str(path) for path in cube_paths],
        "camera_names": list(estimators),
        "completed": True,
        "written_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    finalized_statistics: dict[str, Any] = {}
    try:
        with pkl_path.open("rb") as source, temporary_path.open("wb") as destination:
            source_header = pickle.load(source)
            if not module_023.values_equal(source_header, header):
                raise RuntimeError("PKL header changed after estimator initialization")
            pickle.dump(updated_header, destination, protocol=pickle.HIGHEST_PROTOCOL)

            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break

                if isinstance(record, dict) and record.get("type") == "frame":
                    camera_name = str(record.get("camera_name", ""))
                    estimator = estimators.get(camera_name)
                    if estimator is None:
                        raise ValueError(
                            f"Frame {frame_count} uses unconfigured camera {camera_name!r}"
                        )
                    offline_pos = estimator.estimate(record)
                    record[OFFLINE_POS_FIELD] = offline_pos
                    record[OFFLINE_POS_CACHE_KEY_FIELD] = cache_key
                    record["offline_pos_algorithm"] = module_023.OFFLINE_POS_ALGORITHM
                    for cube in offline_pos["cube_results"]:
                        cube_name = str(cube["cube_name"])
                        result = cube["result"]
                        module_023.update_pose_statistics(
                            statistics,
                            camera_name,
                            cube_name,
                            result,
                        )
                        if result.get("success", False):
                            success_counts[camera_name][cube_name] += 1
                    frame_count += 1
                elif isinstance(record, dict) and record.get("type") == "footer":
                    finalized_statistics = module_023.finalize_pose_statistics(statistics)
                    record["offline_pos_estimation_023_008"] = {
                        "cache_format": POSE_CACHE_FORMAT,
                        "algorithm": module_023.OFFLINE_POS_ALGORITHM,
                        "pose_backend": module_023.POSE_BACKEND,
                        "frame_count": frame_count,
                        "success_counts": success_counts,
                        "statistics": finalized_statistics,
                    }

                pickle.dump(record, destination, protocol=pickle.HIGHEST_PROTOCOL)
                now = time.perf_counter()
                if now - last_progress >= PROGRESS_PRINT_INTERVAL_S:
                    print_progress(
                        source.tell(),
                        source_size,
                        frame_count,
                        started,
                        success_counts,
                    )
                    last_progress = now

            destination.flush()
            os.fsync(destination.fileno())

        finalized_statistics = module_023.finalize_pose_statistics(statistics)
        print_progress(source_size, source_size, frame_count, started, success_counts)
        print()
        temporary_path.replace(pkl_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    report_path = (
        APRILCUBE_ROOT
        / "outputs/offline_pose_statistics"
        / f"{pkl_path.stem}_{POSE_CACHE_FORMAT}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(
            {
                "pkl_path": str(pkl_path),
                "cache_key": cache_key,
                "frame_count": frame_count,
                "success_counts": success_counts,
                "statistics": finalized_statistics,
            },
            report_file,
            indent=2,
        )
    print(f"[INFO] Replaced original PKL atomically: {pkl_path}")
    print(f"[INFO] Pose success counts: {success_counts}")
    print(json.dumps(finalized_statistics, indent=2))
    print(f"[INFO] Statistics report: {report_path}")
    return finalized_statistics, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 023 DeepTag-grid-primary pose estimator on an 008 raw-frame PKL "
            "and rewrite offline_pos in place."
        )
    )
    parser.add_argument(
        "pkl_path",
        nargs="?",
        type=Path,
        default=DEFAULT_PKL_PATH,
        help="008_raw_frames_*.pkl to rewrite in place.",
    )
    parser.add_argument(
        "--cube-name",
        action="append",
        dest="cube_names",
        help=(
            "Cube config directory name to process; repeat for multiple cubes. "
            "By default all cube configs recorded in the header are processed."
        ),
    )
    args = parser.parse_args()

    pkl_path = args.pkl_path.expanduser().resolve()
    if not pkl_path.is_file():
        raise FileNotFoundError(f"008 PKL not found: {pkl_path}")
    module_023 = load_script_023()
    header = load_header(pkl_path)
    cube_paths = selected_cube_paths(header, args.cube_names)
    estimators = build_estimators(module_023, header, cube_paths)
    cache_key = build_cache_key(module_023, pkl_path, estimators, cube_paths)

    print(f"[INFO] Input PKL: {pkl_path}")
    print(f"[INFO] Reusing estimator: {SCRIPT_023_PATH}")
    print(f"[INFO] Cubes: {[path.name for path in cube_paths]}")
    for camera_name, estimator in estimators.items():
        print(f"[INFO] [{camera_name}] intrinsics={estimator.intrinsics_yaml}")
    print(
        "[INFO] Pose path: DeepTag internal grid + homography RANSAC + IPPE/LM; "
        "CV2/CLAHE single-tag face-frame fallback."
    )
    rewrite_with_023_poses(
        module_023,
        pkl_path,
        header,
        estimators,
        cube_paths,
        cache_key,
    )


if __name__ == "__main__":
    main()
