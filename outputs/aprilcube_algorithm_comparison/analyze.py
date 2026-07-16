#!/usr/bin/env python3
"""Reproducible comparison of the historical 9-algorithm benchmark and 020.

The script keeps unlike reprojection metrics separate.  It replays the nine
historical algorithms on the same 012 raw stream already used by the saved 020
work products, then profiles the saved 020 stages without recomputing DeepTag.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import aprilcube  # noqa: E402
from aprilcube_pose_benchmark.common_pose import (  # noqa: E402
    apriltag_family_from_dict_name,
    camera_matrix_to_intrinsic_dict,
)


HISTORICAL_ROOT = (
    REPO_ROOT
    / "outputs"
    / "aprilcube_pose_benchmark"
    / "recording_20260511_162011"
)
RAW_012 = REPO_ROOT / "recordings" / "012_rs_raw_frames_20260715_192635.pkl"
WORK_020 = REPO_ROOT / "recordings" / "020_work_current_012_20260715_192635"
QA_020 = (
    REPO_ROOT.parents[1]
    / "recordings"
    / "qa_multi_cam_record_0716_180451_020"
    / "qa_report.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pickle_stream(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    header: dict[str, Any] = {}
    footer: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            if not isinstance(record, dict):
                continue
            if record.get("type") == "header":
                header = record
            elif record.get("type") == "frame":
                frames.append(record)
            elif record.get("type") == "footer":
                footer = record
    return header, frames, footer


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def result_record(frame_index: int, result: dict[str, Any], elapsed_ms: float = 0.0) -> dict[str, Any]:
    success = bool(result.get("success", False))
    rvec = result.get("rvec")
    tvec = result.get("tvec")
    return {
        "frame_index": int(frame_index),
        "success": success,
        "predicted": bool(result.get("predicted", False) or result.get("pose_filled", False)),
        "reproj_error_px": finite_number(result.get("reproj_error")),
        "rvec": None if not success or rvec is None else np.asarray(rvec, dtype=float).reshape(3).tolist(),
        "tvec_mm": None if not success or tvec is None else np.asarray(tvec, dtype=float).reshape(3).tolist(),
        "n_tags": int(result.get("n_tags", 0) or 0),
        "pose_source": str(result.get("pose_source", "")),
        "mode": str((result.get("algorithm_debug") or {}).get("mode", "")),
        "elapsed_ms": float(elapsed_ms),
    }


def longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_count = len(records)
    success = [bool(row["success"]) for row in records]
    predicted = [bool(row["success"] and row["predicted"]) for row in records]
    measured = [
        bool(row["success"] and not row["predicted"] and row["reproj_error_px"] is not None)
        for row in records
    ]
    reproj = np.asarray(
        [row["reproj_error_px"] for row, valid in zip(records, measured) if valid],
        dtype=float,
    )

    translation_steps: list[float] = []
    rotation_steps: list[float] = []
    for previous, current in zip(records[:-1], records[1:]):
        if current["frame_index"] != previous["frame_index"] + 1:
            continue
        if not previous["success"] or not current["success"]:
            continue
        if previous["tvec_mm"] is None or current["tvec_mm"] is None:
            continue
        translation_steps.append(
            float(
                np.linalg.norm(
                    np.asarray(current["tvec_mm"], dtype=float)
                    - np.asarray(previous["tvec_mm"], dtype=float)
                )
            )
        )
        previous_rotation = Rotation.from_rotvec(np.asarray(previous["rvec"], dtype=float))
        current_rotation = Rotation.from_rotvec(np.asarray(current["rvec"], dtype=float))
        rotation_steps.append(
            float(np.degrees((previous_rotation.inv() * current_rotation).magnitude()))
        )

    def stats(values: np.ndarray | list[float]) -> dict[str, float | None]:
        array = np.asarray(values, dtype=float)
        if not len(array):
            return {"mean": None, "median": None, "p95": None, "max": None}
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    return {
        "frame_count": frame_count,
        "success_count": int(sum(success)),
        "success_rate": float(np.mean(success)) if frame_count else 0.0,
        "measured_count": int(sum(measured)),
        "measured_rate": float(np.mean(measured)) if frame_count else 0.0,
        "predicted_or_filled_count": int(sum(predicted)),
        "predicted_or_filled_rate": float(np.mean(predicted)) if frame_count else 0.0,
        "nonmeasured_success_count": int(sum(success) - sum(measured)),
        "longest_predicted_or_filled_run": longest_true_run(predicted),
        "reprojection_px": stats(reproj),
        "adjacent_translation_step_mm": stats(translation_steps),
        "adjacent_rotation_step_deg": stats(rotation_steps),
        "algorithm_elapsed_ms": stats([row["elapsed_ms"] for row in records]),
        "pose_source_counts": dict(Counter(row["pose_source"] for row in records)),
        "mode_counts": dict(Counter(row["mode"] for row in records if row["mode"])),
    }


def historical_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for npz_path in sorted(HISTORICAL_ROOT.glob("alg_*/poses.npz")):
        metrics_path = npz_path.with_name("metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            success = np.asarray(data["success"], dtype=bool)
            reproj = np.asarray(data["reproj_error_px"], dtype=float)
            finite = np.isfinite(reproj)
            inferred_hold = success & ~finite
            tags = np.asarray(data["visible_tag_count"], dtype=int)
            measured_reproj = reproj[finite]
            summary[npz_path.parent.name] = {
                **metrics,
                "measured_count": int(np.count_nonzero(finite)),
                "measured_rate": float(np.mean(finite)),
                "inferred_hold_count": int(np.count_nonzero(inferred_hold)),
                "inferred_hold_rate": float(np.mean(inferred_hold)),
                "hold_with_zero_tags": int(np.count_nonzero(inferred_hold & (tags == 0))),
                "hold_with_detected_tags": int(np.count_nonzero(inferred_hold & (tags > 0))),
                "longest_inferred_hold_run": longest_true_run(inferred_hold.tolist()),
                "reproj_median_px": float(np.median(measured_reproj)),
                "reproj_p95_px": float(np.percentile(measured_reproj, 95)),
            }
    return summary


def make_detector(cube_cfg: Path, camera_matrix: np.ndarray, *, enable_filter: bool) -> Any:
    return aprilcube.detector(
        cube_cfg,
        intrinsic_cfg=camera_matrix_to_intrinsic_dict(camera_matrix),
        dist_coeffs=np.zeros(5, dtype=float),
        enable_filter=enable_filter,
        fast=True,
    )


def load_historical_algorithms() -> dict[str, tuple[Any, Callable[..., dict[str, Any]]]]:
    algorithms: dict[str, tuple[Any, Callable[..., dict[str, Any]]]] = {}
    for index in range(1, 10):
        path = next((SRC_DIR / "aprilcube_pose_benchmark").glob(f"alg_{index:02d}_*.py"))
        module = load_module(f"comparison_alg_{index:02d}", path)
        algorithms[str(module.ALGORITHM_NAME)] = (module, module.algorithm_fn)
    return algorithms


def replay_historical_algorithms(max_frames: int = 0) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    header, raw_frames, footer = load_pickle_stream(RAW_012)
    if max_frames > 0:
        raw_frames = raw_frames[:max_frames]
    metadata = header["metadata"]
    cube_cfg = Path(metadata["cube_cfg"])
    image_size = tuple(int(value) for value in metadata["image_size"])
    raw_camera_matrix = np.asarray(metadata["raw_camera_matrix"], dtype=float)
    raw_dist_coeffs = np.asarray(metadata["raw_dist_coeffs"], dtype=float)
    detection_camera_matrix = np.asarray(metadata["detection_camera_matrix"], dtype=float)
    map_x, map_y = cv2.initUndistortRectifyMap(
        raw_camera_matrix,
        raw_dist_coeffs,
        None,
        detection_camera_matrix,
        image_size,
        cv2.CV_32FC1,
    )

    algorithms = load_historical_algorithms()
    detectors = {
        name: make_detector(cube_cfg, detection_camera_matrix, enable_filter=False)
        for name in algorithms
    }
    contexts = {
        name: {"camera_name": "d435", "detector": detectors[name]}
        for name in algorithms
    }
    family = apriltag_family_from_dict_name(next(iter(detectors.values())).config.dict_name)
    native_clahe = Detector(families=family, quad_decimate=1.0)
    native_plain = Detector(families=family, quad_decimate=1.0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    camera_params = (
        float(detection_camera_matrix[0, 0]),
        float(detection_camera_matrix[1, 1]),
        float(detection_camera_matrix[0, 2]),
        float(detection_camera_matrix[1, 2]),
    )
    tag_size_m = float(next(iter(detectors.values())).config.tag_size_mm) / 1000.0

    module_004 = load_module(
        "comparison_004_alg06",
        SRC_DIR / "004_cv2_alg_06_aprilcube_detect_multi_cube.py",
    )
    detector_004 = make_detector(cube_cfg, detection_camera_matrix, enable_filter=True)
    runtime_004 = module_004.AprilCubePupilCornersPoseRuntime(
        detector=detector_004,
        native_detector=native_plain,
        pose_estimator=None,
    )
    detector_alg09_plain = make_detector(cube_cfg, detection_camera_matrix, enable_filter=False)
    alg09_name = next(name for name in algorithms if name.startswith("alg_09_"))
    alg09_plain_context: dict[str, Any] = {"camera_name": "d435", "detector": detector_alg09_plain}

    output: dict[str, list[dict[str, Any]]] = {name: [] for name in algorithms}
    output["004_cv2_alg_06_runtime"] = []
    output["004_cv2_alg_09_runtime"] = []

    for sequence_index, frame in enumerate(raw_frames):
        image = np.asarray(frame["image_bgr"], dtype=np.uint8)
        undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        gray_clahe = clahe.apply(gray)
        tags_clahe = native_clahe.detect(
            gray_clahe,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size_m,
        )
        tags_plain = native_plain.detect(gray, estimate_tag_pose=False)
        frame_index = sequence_index

        for name, (module, algorithm_fn) in algorithms.items():
            valid_tags = [tag for tag in tags_clahe if int(tag.tag_id) in detectors[name].valid_ids]
            started = time.perf_counter()
            try:
                result = algorithm_fn(
                    detectors[name],
                    native_clahe,
                    valid_tags,
                    gray_clahe,
                    contexts[name],
                )
            except Exception as error:  # preserve failures for a fair denominator
                result = {"success": False, "failure_reason": f"{type(error).__name__}: {error}"}
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            output[name].append(result_record(frame_index, result, elapsed_ms))

        valid_plain_tags = [tag for tag in tags_plain if int(tag.tag_id) in detector_004.valid_ids]
        module_004.time = SimpleNamespace(
            monotonic=lambda timestamp=float(frame.get("capture_timestamp", sequence_index)): timestamp
        )
        started = time.perf_counter()
        result_004 = runtime_004.process_frame(
            camera_name="d435",
            image=undistorted,
            native_tags=valid_plain_tags,
        )
        output["004_cv2_alg_06_runtime"].append(
            result_record(frame_index, result_004, (time.perf_counter() - started) * 1000.0)
        )

        valid_alg09_plain_tags = [
            tag for tag in tags_plain if int(tag.tag_id) in detector_alg09_plain.valid_ids
        ]
        started = time.perf_counter()
        result_alg09_plain = algorithms[alg09_name][1](
            detector_alg09_plain,
            native_plain,
            valid_alg09_plain_tags,
            gray,
            alg09_plain_context,
        )
        output["004_cv2_alg_09_runtime"].append(
            result_record(
                frame_index,
                result_alg09_plain,
                (time.perf_counter() - started) * 1000.0,
            )
        )

    run_context = {
        "raw_stream": str(RAW_012.relative_to(REPO_ROOT)),
        "raw_header": header,
        "raw_footer": footer,
        "evaluated_frames": len(raw_frames),
        "preprocessing": {
            "historical_algorithms": "pinhole undistort then CLAHE clip=3.0 grid=8x8",
            "004_runtimes": "pinhole undistort, no CLAHE",
        },
        "detector": "pupil_apriltags tag36h11; one shared detection pass per preprocessing path",
    }
    return run_context, output


def profile_020_stages() -> dict[str, Any]:
    stage_paths = {
        "strict_aprilcube": WORK_020 / "strict_aprilcube_pose_012_rs_raw_frames_20260715_192635.pkl",
        "strict_deeptag_dense": WORK_020 / "strict_deeptag_dense_pose_012_rs_raw_frames_20260715_192635.pkl",
        "fused_single_frame": WORK_020 / "fused_single_frame_pose_012_rs_raw_frames_20260715_192635.pkl",
        "temporal_outlier_rejected": WORK_020 / "temporal_outlier_rejected_pose_012_rs_raw_frames_20260715_192635.pkl",
        "outline_recovered": WORK_020 / "outline_recovered_pose_012_rs_raw_frames_20260715_192635.pkl",
        "temporally_completed": WORK_020 / "temporally_completed_pose_012_rs_raw_frames_20260715_192635.pkl",
        "temporally_smoothed_final": WORK_020 / "temporally_smoothed_pose_012_rs_raw_frames_20260715_192635.pkl",
    }
    stages: dict[str, Any] = {}
    for stage_name, path in stage_paths.items():
        header, frames, footer = load_pickle_stream(path)
        records = [result_record(index, frame.get("pose") or {}) for index, frame in enumerate(frames)]
        stages[stage_name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "metadata": header.get("metadata", {}),
            "footer": footer,
            "summary": summarize_records(records),
        }
    return stages


def load_multi_camera_qa() -> dict[str, Any]:
    if not QA_020.exists():
        return {"available": False, "path": str(QA_020)}
    report = json.loads(QA_020.read_text(encoding="utf-8"))
    frames = int(report["frame_count"])
    targets = {}
    for target, details in report["targets"].items():
        success = int(details["success_count"])
        targets[target] = {
            "camera_name": details["camera_name"],
            "success_count": success,
            "success_rate": success / frames,
            "measured_final_count": int(details["measured_final_count"]),
            "final_reprojection_mean_px": details["final_measured_reprojection_px"].get("mean"),
            "final_edge_alignment_mean": details["final_edge_alignment"].get("mean"),
        }
    return {
        "available": True,
        "path": str(QA_020.relative_to(REPO_ROOT.parents[1])),
        "frame_count": frames,
        "targets": targets,
        "pose_source_counts": report["sidecar_footer"].get("pose_source_counts", {}),
    }


def run_analysis(max_frames: int = 0) -> dict[str, Any]:
    run_context, replay_records = replay_historical_algorithms(max_frames=max_frames)
    replay_summary = {name: summarize_records(records) for name, records in replay_records.items()}
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "decision_question": "Which historical AprilCube pose algorithm or 020 pipeline is preferable?",
        "historical_benchmark": historical_summary(),
        "same_input_replay": {
            "context": run_context,
            "summary": replay_summary,
        },
        "pipeline_020_stages": profile_020_stages(),
        "pipeline_020_multicamera_qa": load_multi_camera_qa(),
        "comparison_guardrails": [
            "Historical May reprojection uses Pupil AprilTag outer corners; 020 uses DeepTag dense points and stage-specific anchors, so absolute reprojection values are not cross-pipeline comparable.",
            "Success includes held, interpolated, or globally filled poses in alg_09 and 020; measured coverage is reported separately.",
            "No recording has external 6DoF ground truth, so reprojection and temporal smoothness do not establish absolute pose accuracy.",
            "020 is an offline non-causal cascade using future frames; the historical algorithms and 004 runtimes are intended for online/causal use.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "analysis_results.json")
    args = parser.parse_args()
    results = run_analysis(max_frames=max(0, int(args.max_frames)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)
    print(json.dumps(results["same_input_replay"]["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
