#!/usr/bin/env python3
"""Temporally smooth embedded Wuji Hand qpos and rewrite every source frame."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import stat
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yourdfpy
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt


APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
CONSENS_ROOT = APRILCUBE_ROOT.parent.parent
DEFAULT_PKL = (
    APRILCUBE_ROOT
    / "recordings/021_hand_back_sync_raw_frames_20260712_233831.pkl"
)
DEFAULT_RESULT = (
    CONSENS_ROOT
    / "outputs/retargeting/021_new_intrinsics_recovered/three_finger/"
    "three_finger_se3.npz"
)
DEFAULT_OUTPUT = DEFAULT_RESULT.with_name(
    "three_finger_se3_temporal_smoothed_butterworth2_6hz.npz"
)
DEFAULT_URDF = (
    CONSENS_ROOT
    / "thirdparty/wuji-description/hand/body-with-soft/urdf/"
    "left_simplified_w_fingereye.urdf"
)

SOURCE_FIELD = "left_wuji_three_finger_se3_retargeting"
OUTPUT_FIELD = "left_wuji_three_finger_se3_retargeting_temporal_smoothed"
SCHEMA = "consens.left_wuji_three_finger_se3.temporal_smoothed.v1"
DEFAULT_CUTOFF_HZ = 6.0
DEFAULT_FILTER_ORDER = 2
FINGER_LINKS = {
    "thumb": "left_finger1_link4",
    "index": "left_finger2_link4",
    "middle": "left_finger3_link4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a time-aware zero-phase low-pass filter to embedded three-finger "
            "Wuji Hand qpos, recompute FK, and atomically write each frame."
        )
    )
    parser.add_argument("pkl_path", nargs="?", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ)
    parser.add_argument("--filter-order", type=int, default=DEFAULT_FILTER_ORDER)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        "schema",
        "finger_names",
        "qpos",
        "joint_names",
        "active_joint_indices",
        "active_joint_names",
        "timestamps",
        "source_record_indices",
        "target_T_left_palm_link_obj",
        "target_keypoints",
        "keypoints_obj_m",
        "T_robot_link_obj",
        "keypoint_weights",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise KeyError(f"Retargeting NPZ is missing {missing}")
    return payload


def load_robot(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(path),
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
    )


def temporal_lowpass(
    timestamps: np.ndarray,
    values: np.ndarray,
    cutoff_hz: float,
    order: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    if len(timestamps) < 8 or values.shape[0] != len(timestamps):
        raise ValueError("Temporal smoothing needs at least 8 timestamped samples")
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Retargeting timestamps must be strictly increasing")

    median_dt = float(np.median(np.diff(timestamps)))
    sample_rate_hz = 1.0 / median_dt
    if not 0.0 < cutoff_hz < sample_rate_hz / 2.0:
        raise ValueError(
            f"cutoff_hz must be in (0, {sample_rate_hz / 2.0:.3f}), got {cutoff_hz}"
        )
    if order < 1:
        raise ValueError("filter_order must be positive")

    uniform_timestamps = np.arange(
        timestamps[0],
        timestamps[-1] + 0.25 * median_dt,
        median_dt,
        dtype=np.float64,
    )
    uniform_values = PchipInterpolator(timestamps, values, axis=0)(uniform_timestamps)
    sections = butter(order, cutoff_hz, fs=sample_rate_hz, output="sos")
    uniform_smoothed = sosfiltfilt(sections, uniform_values, axis=0)
    smoothed = PchipInterpolator(
        uniform_timestamps,
        uniform_smoothed,
        axis=0,
    )(timestamps)
    return smoothed, {
        "filter_order": int(order),
        "cutoff_hz": float(cutoff_hz),
        "median_source_dt_s": median_dt,
        "uniform_sample_rate_hz": sample_rate_hz,
        "uniform_sample_count": int(len(uniform_timestamps)),
        "zero_phase": True,
        "irregular_timestamp_resampling": "PCHIP",
    }


def joint_limits(
    robot: yourdfpy.URDF,
    joint_names: list[str],
    active_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for index in active_indices:
        name = joint_names[int(index)]
        joint = robot.joint_map[name]
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise ValueError(f"Active joint {name} does not have finite URDF limits")
        lower.append(float(joint.limit.lower))
        upper.append(float(joint.limit.upper))
    return np.asarray(lower), np.asarray(upper)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.asarray(transform[:3, :3], dtype=np.float64)
        @ np.asarray(points, dtype=np.float64).T
    ).T + np.asarray(transform[:3, 3], dtype=np.float64)


def recompute_fk(
    robot: yourdfpy.URDF,
    qpos: np.ndarray,
    joint_names: list[str],
    finger_names: list[str],
    link_from_obj: np.ndarray,
    keypoints_obj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(qpos)
    predicted_poses = np.tile(
        np.eye(4, dtype=np.float64),
        (frame_count, len(finger_names), 1, 1),
    )
    predicted_points = np.zeros(
        (frame_count, len(finger_names), keypoints_obj.shape[1], 3),
        dtype=np.float64,
    )
    for frame_index, frame_qpos in enumerate(qpos):
        robot.update_cfg(dict(zip(joint_names, frame_qpos)))
        for finger_index, finger_name in enumerate(finger_names):
            root_from_link = robot.get_transform(
                FINGER_LINKS[finger_name],
                "left_palm_link",
            )
            root_from_obj = root_from_link @ link_from_obj[finger_index]
            predicted_poses[frame_index, finger_index] = root_from_obj
            predicted_points[frame_index, finger_index] = transform_points(
                root_from_obj,
                keypoints_obj[finger_index],
            )
    return predicted_poses, predicted_points


def weighted_errors(
    predicted_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = np.linalg.norm(predicted_points - target_points, axis=-1)
    weighted_sum = np.sum(errors * weights, axis=(1, 2))
    weight_sum = np.sum(weights, axis=(1, 2))
    mean_error = weighted_sum / np.maximum(weight_sum, 1e-12)
    max_error = np.zeros(len(errors), dtype=np.float64)
    for frame_index in range(len(errors)):
        active = weights[frame_index] > 0.0
        max_error[frame_index] = (
            float(np.max(errors[frame_index][active])) if np.any(active) else np.nan
        )
    return errors, mean_error, max_error


def acceleration_statistics(
    qpos: np.ndarray,
    timestamps: np.ndarray,
    active_indices: np.ndarray,
) -> dict[str, float]:
    active = np.asarray(qpos, dtype=np.float64)[:, active_indices]
    time_delta = np.diff(timestamps)
    velocity = np.diff(active, axis=0) / time_delta[:, None]
    acceleration_dt = (time_delta[:-1] + time_delta[1:]) * 0.5
    acceleration = np.diff(velocity, axis=0) / acceleration_dt[:, None]
    absolute = np.abs(acceleration)
    return {
        "rms_rad_s2": float(np.sqrt(np.mean(acceleration * acceleration))),
        "median_abs_rad_s2": float(np.median(absolute)),
        "p95_abs_rad_s2": float(np.percentile(absolute, 95.0)),
        "max_abs_rad_s2": float(np.max(absolute)),
    }


def distribution(values: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1) * float(scale)
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "max": float(np.max(data)),
    }


def build_smoothed_result(
    payload: dict[str, np.ndarray],
    robot: yourdfpy.URDF,
    cutoff_hz: float,
    order: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    qpos_raw = np.asarray(payload["qpos"], dtype=np.float64)
    timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
    active_indices = np.asarray(payload["active_joint_indices"], dtype=np.int32)
    joint_names = [str(value) for value in payload["joint_names"]]
    finger_names = [str(value) for value in payload["finger_names"]]

    active_smoothed_unclipped, filter_metadata = temporal_lowpass(
        timestamps,
        qpos_raw[:, active_indices],
        cutoff_hz,
        order,
    )
    lower, upper = joint_limits(robot, joint_names, active_indices)
    lower_violations = active_smoothed_unclipped < lower[None, :]
    upper_violations = active_smoothed_unclipped > upper[None, :]
    active_smoothed = np.clip(active_smoothed_unclipped, lower, upper)
    qpos_smoothed = qpos_raw.copy()
    qpos_smoothed[:, active_indices] = active_smoothed

    predicted_poses, predicted_points = recompute_fk(
        robot,
        qpos_smoothed,
        joint_names,
        finger_names,
        np.asarray(payload["T_robot_link_obj"], dtype=np.float64),
        np.asarray(payload["keypoints_obj_m"], dtype=np.float64),
    )
    errors, mean_error, max_error = weighted_errors(
        predicted_points,
        np.asarray(payload["target_keypoints"], dtype=np.float64),
        np.asarray(payload["keypoint_weights"], dtype=np.float64),
    )

    delta = qpos_smoothed[:, active_indices] - qpos_raw[:, active_indices]
    report = {
        "schema": SCHEMA,
        "source_result": str(DEFAULT_RESULT),
        "source_field": SOURCE_FIELD,
        "output_field": OUTPUT_FIELD,
        "method": "irregular_time_pchip_zero_phase_butterworth_lowpass",
        **filter_metadata,
        "frame_count": int(len(qpos_smoothed)),
        "active_joint_count": int(len(active_indices)),
        "active_joint_names": [joint_names[int(index)] for index in active_indices],
        "joint_limit_clipping": {
            "lower_value_count": int(np.count_nonzero(lower_violations)),
            "upper_value_count": int(np.count_nonzero(upper_violations)),
            "post_filter_limit_violations": 0,
        },
        "qpos_adjustment_deg": distribution(np.abs(delta), scale=180.0 / np.pi),
        "joint_acceleration_before": acceleration_statistics(
            qpos_raw, timestamps, active_indices
        ),
        "joint_acceleration_after": acceleration_statistics(
            qpos_smoothed, timestamps, active_indices
        ),
        "contact_error_before_mm": distribution(
            np.asarray(payload["mean_keypoint_error_m"], dtype=np.float64),
            scale=1000.0,
        ),
        "contact_error_after_mm": distribution(mean_error, scale=1000.0),
        "contact_max_error_after_mm": distribution(max_error, scale=1000.0),
    }

    output = dict(payload)
    output.update(
        {
            "result_name": np.asarray(
                f"three_finger_se3_temporal_smoothed_butterworth{order}_{cutoff_hz:g}hz"
            ),
            "qpos": qpos_smoothed.astype(np.float32),
            "predicted_T_left_palm_link_obj": predicted_poses,
            "predicted_keypoints": predicted_points.astype(np.float32),
            "mean_keypoint_error_m": mean_error,
            "max_keypoint_error_m": max_error,
            "temporal_smoothing_schema": np.asarray(SCHEMA),
            "temporal_smoothing_method": np.asarray(report["method"]),
            "temporal_smoothing_cutoff_hz": np.asarray(cutoff_hz),
            "temporal_smoothing_filter_order": np.asarray(order),
            "source_qpos": qpos_raw.astype(np.float32),
            "qpos_delta_from_source": (qpos_smoothed - qpos_raw).astype(np.float32),
        }
    )
    output["per_point_keypoint_error_m"] = errors.astype(np.float32)
    return output, report


def write_npz(path: Path, payload: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    temporary.unlink(missing_ok=True)
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def embedded_header(
    report: dict[str, Any],
    result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    return {
        **report,
        "source_result": str(result_path),
        "smoothed_result": str(output_path),
        "wujihand_qpos_dimension": 12,
        "wujihand_qpos_unit": "rad",
        "frame_result_keys": [
            "wujihand_qpos",
            "qpos_delta_from_source_rad",
            "target_T_left_palm_link_obj",
            "predicted_T_left_palm_link_obj",
            "target_keypoints",
            "predicted_keypoints",
            "keypoint_error_m",
            "keypoint_weight",
        ],
    }


def frame_payload(
    output: dict[str, np.ndarray],
    row: int,
    active_indices: np.ndarray,
) -> dict[str, Any]:
    qpos = np.asarray(output["qpos"][row], dtype=np.float32)
    source_qpos = np.asarray(output["source_qpos"][row], dtype=np.float32)
    predicted = np.asarray(output["predicted_keypoints"][row], dtype=np.float32)
    target = np.asarray(output["target_keypoints"][row], dtype=np.float32)
    return {
        "schema": SCHEMA,
        "source_field": SOURCE_FIELD,
        "wujihand_qpos": qpos[active_indices],
        "qpos_delta_from_source_rad": (qpos - source_qpos)[active_indices],
        "target_T_left_palm_link_obj": output["target_T_left_palm_link_obj"][row],
        "predicted_T_left_palm_link_obj": output[
            "predicted_T_left_palm_link_obj"
        ][row],
        "target_keypoints": target,
        "predicted_keypoints": predicted,
        "keypoint_error_m": np.linalg.norm(predicted - target, axis=-1),
        "keypoint_weight": output["keypoint_weights"][row],
    }


def print_rewrite_progress(done: int, total: int, frames: int) -> None:
    ratio = min(max(done / max(total, 1), 0.0), 1.0)
    width = 30
    filled = int(round(width * ratio))
    print(
        f"\r[WRITE] [{'#' * filled}{'-' * (width - filled)}] "
        f"{ratio * 100:5.1f}% frames={frames}",
        end="",
        flush=True,
    )


def rewrite_source_pkl(
    pkl_path: Path,
    result_path: Path,
    output_path: Path,
    output: dict[str, np.ndarray],
    report: dict[str, Any],
    overwrite: bool,
) -> None:
    source_size = pkl_path.stat().st_size
    available = shutil.disk_usage(pkl_path.parent).free
    required = source_size + 1_000_000_000
    print(
        f"[INFO] Atomic rewrite space: source={source_size / 1024**3:.2f} GiB "
        f"free={available / 1024**3:.2f} GiB required={required / 1024**3:.2f} GiB"
    )
    if available < required:
        raise OSError(
            f"Need {required} free bytes for atomic rewrite; only {available} available"
        )

    source_indices = np.asarray(output["source_record_indices"], dtype=np.int32)
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("source_record_indices contains duplicates")
    row_by_source_index = {
        int(source_index): row for row, source_index in enumerate(source_indices)
    }
    active_indices = np.asarray(output["active_joint_indices"], dtype=np.int32)
    header_value = embedded_header(report, result_path, output_path)
    temporary = pkl_path.with_name(f".{pkl_path.name}.qpos-smoothing.tmp")
    temporary.unlink(missing_ok=True)
    written: set[int] = set()
    footer_count = 0

    try:
        with pkl_path.open("rb") as source, temporary.open("wb") as target:
            source_header = pickle.load(source)
            if source_header.get("type") != "header":
                raise ValueError("Source PKL does not start with a header")
            if OUTPUT_FIELD in source_header and not overwrite:
                raise FileExistsError(
                    f"Source header already contains {OUTPUT_FIELD!r}; pass --overwrite"
                )
            merged_header = dict(source_header)
            merged_header[OUTPUT_FIELD] = header_value
            pickle.dump(merged_header, target, protocol=pickle.HIGHEST_PROTOCOL)

            while True:
                try:
                    record = pickle.load(source)
                except EOFError:
                    break
                if not isinstance(record, dict):
                    pickle.dump(record, target, protocol=pickle.HIGHEST_PROTOCOL)
                    continue
                if record.get("type") == "footer":
                    merged_footer = dict(record)
                    merged_footer[OUTPUT_FIELD] = header_value
                    pickle.dump(merged_footer, target, protocol=pickle.HIGHEST_PROTOCOL)
                    footer_count += 1
                    continue
                if record.get("type") != "frame_pair":
                    pickle.dump(record, target, protocol=pickle.HIGHEST_PROTOCOL)
                    continue

                source_index = int(record["pair_index"])
                row = row_by_source_index.get(source_index)
                if row is None:
                    raise KeyError(f"No smoothed result for pair_index={source_index}")
                source_embedded = record.get(SOURCE_FIELD)
                if not isinstance(source_embedded, dict):
                    raise KeyError(f"Frame {source_index} is missing {SOURCE_FIELD}")
                expected_source_qpos = np.asarray(
                    output["source_qpos"][row, active_indices], dtype=np.float32
                )
                actual_source_qpos = np.asarray(
                    source_embedded["wujihand_qpos"], dtype=np.float32
                )
                if not np.allclose(
                    actual_source_qpos,
                    expected_source_qpos,
                    atol=1e-6,
                    rtol=0.0,
                ):
                    raise ValueError(
                        f"Frame {source_index} embedded qpos differs from source NPZ"
                    )
                if OUTPUT_FIELD in record and not overwrite:
                    raise FileExistsError(
                        f"Frame {source_index} already contains {OUTPUT_FIELD!r}"
                    )
                merged_record = dict(record)
                merged_record[OUTPUT_FIELD] = frame_payload(output, row, active_indices)
                pickle.dump(merged_record, target, protocol=pickle.HIGHEST_PROTOCOL)
                written.add(source_index)
                print_rewrite_progress(source.tell(), source_size, len(written))

            target.flush()
            os.fsync(target.fileno())
        print()
        expected = set(row_by_source_index)
        if written != expected:
            raise ValueError(
                f"Wrote {len(written)} frame results, expected {len(expected)}"
            )
        if footer_count != 1:
            raise ValueError(f"Expected one footer, found {footer_count}")
        os.chmod(temporary, stat.S_IMODE(pkl_path.stat().st_mode))
        os.replace(temporary, pkl_path)
        directory_fd = os.open(pkl_path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    pkl_path = args.pkl_path.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    urdf_path = args.urdf.expanduser().resolve()
    for path in (pkl_path, result_path, urdf_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = load_npz(result_path)
    robot = load_robot(urdf_path)
    output, report = build_smoothed_result(
        payload,
        robot,
        float(args.cutoff_hz),
        int(args.filter_order),
    )
    report.update(
        {
            "source_pkl": str(pkl_path),
            "source_result": str(result_path),
            "smoothed_result": str(output_path),
            "urdf": str(urdf_path),
            "written_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    write_npz(output_path, output, bool(args.overwrite))
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[INFO] Smoothed NPZ: {output_path}")
    print(f"[INFO] Report: {report_path}")
    rewrite_source_pkl(
        pkl_path,
        result_path,
        output_path,
        output,
        report,
        bool(args.overwrite),
    )
    print(f"[INFO] Embedded field: {OUTPUT_FIELD}")
    print(f"[INFO] Rewritten PKL: {pkl_path}")


if __name__ == "__main__":
    main()
