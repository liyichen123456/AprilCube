#!/usr/bin/env python3
from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

APRILCUBE_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = APRILCUBE_ROOT / "src"
RECORDINGS_DIR = APRILCUBE_ROOT / "recordings"

# Edit these constants before running. This script intentionally does not accept
# command-line arguments, so a run is reproducible from the file contents.
RUN_MODE = "012"  # "012", "008", or "MERGE_FINAL"
PYTHON_EXE = Path("/home/ps/miniconda3/envs/foundation_stereo/bin/python")

INPUT_PKL = RECORDINGS_DIR / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
OUTPUT_PKL = RECORDINGS_DIR / "012_rs_raw_frames_20260710_214336_with_final_postprocessed_pose.pkl"
WORK_DIR = RECORDINGS_DIR / "020_work"
KEEP_INTERMEDIATES = False
DRY_RUN = False

RUN_012_SLOW_APRILTAG = True
RUN_012_UNDISTORT = True
RUN_012_FILL_MISSING_POSE = True
RUN_012_FALLBACK_LAYOUT = "cfg"

RUN_008_SHARED_DETECT_TAGS = True
RUN_008_SLOW_APRILTAG = True
RUN_008_UNDISTORT = True

MERGE_RAW_PKL = INPUT_PKL
MERGE_FINAL_POSE_PKL = RECORDINGS_DIR / "025_global_temporal_filter_fill_final.pkl"
MERGE_OUTPUT_PKL = OUTPUT_PKL
MERGE_TIMESTAMP_TOLERANCE = 1e-6
MERGE_KEEP_ORIGINAL_POSE = True
MERGE_KEEP_POSE_CANDIDATES = True

FORMAT_008_RAW = "aprilcube_raw_frame_stream_v1"
FORMAT_012_RAW = "aprilcube_rs_raw_frame_stream_v1"
FORMAT_012_RAW_WITH_POSE = "aprilcube_012_raw_with_pose_stream_v1"
SUPPORTED_012_INPUT_FORMATS = {FORMAT_012_RAW, FORMAT_012_RAW_WITH_POSE}

PIPELINE_STAGES = [
    {
        "stage": "008_offline_replay",
        "summary": "Run src/011_visualize_008_pkl.py over 008 raw image frames and write offline_pose_frame in-place.",
    },
    {
        "stage": "012_aprilcube_offline",
        "summary": "Run src/014_visualize_012_pkl.py over raw image frames and write an offline pose stream.",
    },
    {
        "stage": "merge_final",
        "summary": "Merge the selected pose stream into the raw frame stream by capture_timestamp.",
    },
]


def inspect_pkl_format(path: Path) -> str:
    with path.expanduser().resolve().open("rb") as f:
        header = pickle.load(f)
    if not isinstance(header, dict):
        raise ValueError(f"Unsupported pkl header in {path}: {type(header).__name__}")
    fmt = str(header.get("format", ""))
    if not fmt:
        raise ValueError(f"PKL has no header format: {path}")
    return fmt


def run_command(cmd: list[str]) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    if DRY_RUN:
        return
    subprocess.run(cmd, cwd=str(APRILCUBE_ROOT), check=True)


def python_cmd(script_name: str, *extra: str) -> list[str]:
    return [str(PYTHON_EXE.expanduser()), str(SRC_DIR / script_name), *extra]


def run_008_pipeline() -> Path:
    pkl_path = INPUT_PKL.expanduser().resolve()
    fmt = inspect_pkl_format(pkl_path)
    if fmt != FORMAT_008_RAW:
        raise ValueError(f"Expected 008 raw pkl format, got {fmt}: {pkl_path}")

    cmd = python_cmd("011_visualize_008_pkl.py", str(pkl_path), "--precompute-only")
    if RUN_008_SHARED_DETECT_TAGS:
        cmd.append("--shared-detect-tags")
    if RUN_008_SLOW_APRILTAG:
        cmd.append("--slow")
    if not RUN_008_UNDISTORT:
        cmd.append("--no-undistort")

    run_command(cmd)
    if not DRY_RUN:
        summarize_008_pose_cache(pkl_path)
    return pkl_path


def run_012_pipeline() -> Path:
    raw_pkl = INPUT_PKL.expanduser().resolve()
    fmt = inspect_pkl_format(raw_pkl)
    if fmt not in SUPPORTED_012_INPUT_FORMATS:
        raise ValueError(
            "The 012 pipeline must start from a 012 stream with raw images "
            f"(format={SUPPORTED_012_INPUT_FORMATS}), got {fmt}: {raw_pkl}"
        )

    work_dir = WORK_DIR.expanduser().resolve()
    april_pose_pkl = work_dir / f"014_offline_pose_vis_{raw_pkl.stem}_020_aprilcube.pkl"
    output_pkl = OUTPUT_PKL.expanduser().resolve()

    if not DRY_RUN:
        work_dir.mkdir(parents=True, exist_ok=True)

    cmd014 = python_cmd(
        "014_visualize_012_pkl.py",
        str(raw_pkl),
        "--output-pkl",
        str(april_pose_pkl),
        "--precompute-only",
        "--no-filter",
        "--fallback-layout",
        RUN_012_FALLBACK_LAYOUT,
    )
    if RUN_012_SLOW_APRILTAG:
        cmd014.append("--slow")
    if not RUN_012_UNDISTORT:
        cmd014.append("--no-undistort")
    if not RUN_012_FILL_MISSING_POSE:
        cmd014.append("--no-fill-missing-pose")
    run_command(cmd014)

    if DRY_RUN:
        print(f"[DRY-RUN] merge-final raw={raw_pkl} final_pose={april_pose_pkl} output={output_pkl}")
    else:
        merge_final_pose_stream(
            raw_pkl=raw_pkl,
            final_pose_pkl=april_pose_pkl,
            output_pkl=output_pkl,
            timestamp_tolerance=MERGE_TIMESTAMP_TOLERANCE,
            keep_original_pose=MERGE_KEEP_ORIGINAL_POSE,
            keep_pose_candidates=MERGE_KEEP_POSE_CANDIDATES,
        )
        summarize_pose_stream(output_pkl, "pose")

    if not KEEP_INTERMEDIATES and not DRY_RUN:
        shutil.rmtree(work_dir)
        print(f"[INFO] Removed work dir: {work_dir}")
    else:
        print(f"[INFO] Work dir: {work_dir}")
    return output_pkl


def run_merge_final_only() -> Path:
    return merge_final_pose_stream(
        raw_pkl=MERGE_RAW_PKL.expanduser().resolve(),
        final_pose_pkl=MERGE_FINAL_POSE_PKL.expanduser().resolve(),
        output_pkl=MERGE_OUTPUT_PKL.expanduser().resolve(),
        timestamp_tolerance=MERGE_TIMESTAMP_TOLERANCE,
        keep_original_pose=MERGE_KEEP_ORIGINAL_POSE,
        keep_pose_candidates=MERGE_KEEP_POSE_CANDIDATES,
    )


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
                "format": "aprilcube_raw_with_020_postprocessed_pose_stream_v1",
                "created_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_raw_pkl": str(raw_pkl),
                "source_final_pose_pkl": str(final_pose_pkl),
                "raw_header": raw_header,
                "raw_footer": raw_footer,
                "final_pose_header": final_header,
                "final_pose_footer": final_footer,
                "metadata": {
                    "pipeline_stages": PIPELINE_STAGES,
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


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit("020_finalize_pose_postprocess.py does not accept CLI args; edit constants at the top.")

    if RUN_MODE == "008":
        output = run_008_pipeline()
    elif RUN_MODE == "012":
        output = run_012_pipeline()
    elif RUN_MODE == "MERGE_FINAL":
        output = run_merge_final_only()
    else:
        raise ValueError(f"Unsupported RUN_MODE={RUN_MODE!r}")
    print(f"[INFO] Done: {output}")


if __name__ == "__main__":
    main()
