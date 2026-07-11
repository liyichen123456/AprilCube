#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
HELPER_SOURCE_COMMIT = "515de6d"
HELPER_SCRIPT_NAMES = {
    "020_deeptag_dense_keypoints_pose.py",
    "022_benchmark_single_frame_recovery_methods.py",
    "023_fuse_all_single_frame_recovery_methods.py",
    "024_temporal_outline_refine_recovery.py",
    "025_global_temporal_filter_fill_remaining.py",
}

DEFAULT_012_PKL = RECORDINGS_DIR / "012_rs_raw_frames_20260710_214336_with_aprilcube_pose.pkl"
DEFAULT_RAW_PKL = DEFAULT_012_PKL
DEFAULT_FINAL_POSE_PKL = RECORDINGS_DIR / "025_global_temporal_filter_fill_final.pkl"
DEFAULT_OUTPUT_PKL = RECORDINGS_DIR / "012_rs_raw_frames_20260710_214336_with_final_postprocessed_pose.pkl"

SUPPORTED_008_FORMATS = {"aprilcube_raw_frame_stream_v1"}
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
        "summary": "Run src/014_visualize_012_pkl.py to create AprilCube single-frame/fallback pose pkl.",
    },
    {
        "stage": "012_merge_aprilcube_pose",
        "summary": "Run src/017_merge_012_raw_and_014_pose_pkl.py to attach AprilCube pose to raw frames.",
    },
    {
        "stage": "012_deeptag_robust",
        "summary": "Run src/016_deeptag_012_pkl.py to create DeepTag robust pose candidates.",
    },
    {
        "stage": "012_deeptag_dense_strict",
        "summary": "Run src/020_deeptag_dense_keypoints_pose.py with coverage/min-tag2 gates.",
    },
    {
        "stage": "012_deeptag_dense_loose",
        "summary": "Run src/020_deeptag_dense_keypoints_pose.py in a looser mode for edge-gated fallback candidates.",
    },
    {
        "stage": "012_single_frame_recovery_fusion",
        "summary": "Run src/023_fuse_all_single_frame_recovery_methods.py with no temporal filtering.",
    },
    {
        "stage": "012_temporal_outline_refine",
        "summary": "Run src/024_temporal_outline_refine_recovery.py for conservative RGB-outline recovery.",
    },
    {
        "stage": "012_global_temporal_fill",
        "summary": "Run src/025_global_temporal_filter_fill_remaining.py for final remaining failed frames.",
    },
    {
        "stage": "merge_final",
        "summary": "Merge a final pose stream into the raw frame stream by capture_timestamp.",
    },
]


def rewrite_legacy_argv(argv: list[str]) -> list[str]:
    commands = {"auto", "008", "012", "merge-final"}
    if not argv or argv[0] in {"-h", "--help"}:
        return argv
    if argv and argv[0] in commands:
        return argv
    if any(arg in {"--raw-pkl", "--final-pose-pkl"} for arg in argv):
        return ["merge-final", *argv]
    return ["auto", *argv]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-entry AprilCube offline pose postprocess pipeline. Passing a pkl path directly "
            "uses auto mode. Use merge-final to only merge an already-computed pose stream."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    auto = subparsers.add_parser("auto", help="Inspect pkl format and run the matching available pipeline.")
    add_common_run_args(auto)
    auto.add_argument("pkl_path", nargs="?", type=Path, default=DEFAULT_012_PKL, help="Input raw-frame pkl.")
    auto.add_argument("--output-pkl", type=Path, default=None, help="Output pkl for 012/final merge modes.")
    auto.add_argument("--work-dir", type=Path, default=None, help="Directory for 012 intermediate pkls.")
    auto.add_argument("--keep-intermediates", action="store_true", help="Keep 012 intermediate pkls.")
    auto.add_argument("--run-benchmark", action="store_true", help="Also run 022 benchmark diagnostics.")

    cmd008 = subparsers.add_parser("008", help="Run the 008 offline image-to-pose pipeline in-place.")
    add_common_run_args(cmd008)
    cmd008.add_argument("pkl_path", type=Path, help="008 raw-frame pkl.")

    cmd012 = subparsers.add_parser("012", help="Run the available 012 raw-image to fused-pose pipeline.")
    add_common_run_args(cmd012)
    cmd012.add_argument("pkl_path", nargs="?", type=Path, default=DEFAULT_012_PKL, help="012 raw-frame pkl.")
    cmd012.add_argument("--output-pkl", type=Path, default=None)
    cmd012.add_argument("--work-dir", type=Path, default=None)
    cmd012.add_argument("--keep-intermediates", action="store_true")
    cmd012.add_argument("--run-benchmark", action="store_true", help="Also run 022 benchmark diagnostics.")

    merge = subparsers.add_parser("merge-final", help="Only merge an existing final pose pkl into a raw pkl.")
    merge.add_argument("--raw-pkl", type=Path, default=DEFAULT_RAW_PKL)
    merge.add_argument("--final-pose-pkl", type=Path, default=DEFAULT_FINAL_POSE_PKL)
    merge.add_argument("--output-pkl", type=Path, default=DEFAULT_OUTPUT_PKL)
    merge.add_argument("--timestamp-tolerance", type=float, default=1e-6)
    merge.add_argument("--keep-original-pose", action=argparse.BooleanOptionalAction, default=True)
    merge.add_argument("--keep-pose-candidates", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args(rewrite_legacy_argv(argv))


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable for child scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--fast", action="store_true", help="Use faster detector settings where supported.")
    parser.add_argument("--no-undistort", action="store_true", help="Disable undistortion where supported.")
    parser.add_argument(
        "--shared-detect-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use shared AprilTag detection for 008 replay.",
    )


def inspect_pkl_format(path: Path) -> str:
    with path.expanduser().resolve().open("rb") as f:
        header = pickle.load(f)
    if not isinstance(header, dict):
        raise ValueError(f"Unsupported pkl header in {path}: {type(header).__name__}")
    fmt = str(header.get("format", ""))
    if not fmt:
        raise ValueError(f"PKL has no header format: {path}")
    return fmt


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("[CMD] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(APRILCUBE_ROOT), check=True)


def python_cmd(args: argparse.Namespace, script_name: str, *extra: str) -> list[str]:
    helper_dir = getattr(args, "_helper_script_dir", None)
    if helper_dir is not None and script_name in HELPER_SCRIPT_NAMES:
        script_path = Path(helper_dir) / script_name
    else:
        script_path = SRC_DIR / script_name
    return [str(Path(args.python).expanduser()), str(script_path), *extra]


def prepare_helper_scripts(args: argparse.Namespace, work_dir: Path) -> Path:
    helper_dir = work_dir / "_020_stage_helpers"
    args._helper_script_dir = helper_dir
    if bool(args.dry_run):
        return helper_dir

    helper_dir.mkdir(parents=True, exist_ok=True)
    for script_name in sorted(HELPER_SCRIPT_NAMES):
        source_ref = f"{HELPER_SOURCE_COMMIT}:src/{script_name}"
        result = subprocess.run(
            ["git", "show", source_ref],
            cwd=str(APRILCUBE_ROOT),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        script_path = helper_dir / script_name
        script_path.write_text(result.stdout, encoding="utf-8")
        script_path.chmod(0o755)

    for name, target in {
        "012_rs_aprilcube_detect.py": SRC_DIR / "012_rs_aprilcube_detect.py",
        "aprilcube": SRC_DIR / "aprilcube",
    }.items():
        link = helper_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=target.is_dir())

    return helper_dir


def run_008_pipeline(args: argparse.Namespace) -> Path:
    pkl_path = Path(args.pkl_path).expanduser().resolve()
    fmt = inspect_pkl_format(pkl_path)
    if fmt not in SUPPORTED_008_FORMATS:
        raise ValueError(f"Expected 008 raw pkl format, got {fmt}: {pkl_path}")

    cmd = python_cmd(args, "011_visualize_008_pkl.py", str(pkl_path), "--precompute-only")
    if bool(args.shared_detect_tags):
        cmd.append("--shared-detect-tags")
    if not bool(args.fast):
        cmd.append("--slow")
    if bool(args.no_undistort):
        cmd.append("--no-undistort")

    run_command(cmd, dry_run=bool(args.dry_run))
    if not bool(args.dry_run):
        summarize_008_pose_cache(pkl_path)
    return pkl_path


def default_012_output_path(raw_pkl: Path) -> Path:
    return raw_pkl.with_name(f"{raw_pkl.stem}_with_020_postprocessed_pose.pkl")


def default_work_dir(raw_pkl: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return RECORDINGS_DIR / f"020_work_{raw_pkl.stem}_{stamp}"


def run_012_pipeline(args: argparse.Namespace) -> Path:
    raw_pkl = Path(args.pkl_path).expanduser().resolve()
    fmt = inspect_pkl_format(raw_pkl)
    if fmt not in SUPPORTED_012_INPUT_FORMATS:
        raise ValueError(
            "The one-click 012 pipeline must start from a 012 stream with raw images "
            f"(format={SUPPORTED_012_INPUT_FORMATS}), got {fmt}: {raw_pkl}"
        )

    output_pkl = Path(args.output_pkl).expanduser().resolve() if args.output_pkl else default_012_output_path(raw_pkl)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else default_work_dir(raw_pkl).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    helper_dir = prepare_helper_scripts(args, work_dir)
    print(f"[INFO] Stage helper scripts: {helper_dir}")

    april_strict_pkl = work_dir / f"014_offline_pose_vis_{raw_pkl.stem}_aprilcube_style_nofill_notagfix.pkl"
    april_merged_pkl = work_dir / f"017_{raw_pkl.stem}_with_aprilcube_pose.pkl"
    deeptag_raw_pkl = work_dir / f"016_deeptag_robust_cluster_{raw_pkl.stem}.pkl"
    deeptag_dense_strict_pkl = (
        work_dir / f"020_deeptag_dense_keypoints_pose_{raw_pkl.stem}_faceframe_alltags_coverage_mintag2.pkl"
    )
    deeptag_dense_loose_pkl = work_dir / f"020_deeptag_dense_keypoints_pose_{raw_pkl.stem}_faceframe_alltags.pkl"
    benchmark_pkl = work_dir / f"022_recovery_method_benchmark_{raw_pkl.stem}.pkl"
    fused_single_frame_pkl = work_dir / f"023_fused_all_single_frame_recovery_{raw_pkl.stem}.pkl"
    outline_refine_pkl = work_dir / f"024_temporal_outline_refine_recovery_conservative_fixed_{raw_pkl.stem}.pkl"
    final_pose_pkl = work_dir / f"025_global_temporal_filter_fill_final_{raw_pkl.stem}.pkl"

    cmd014 = python_cmd(
        args,
        "014_visualize_012_pkl.py",
        str(raw_pkl),
        "--output-pkl",
        str(april_strict_pkl),
        "--precompute-only",
        "--no-filter",
        "--fallback-layout",
        "cfg",
        "--no-fill-missing-pose",
    )
    if not bool(args.fast):
        cmd014.append("--slow")
    if bool(args.no_undistort):
        cmd014.append("--no-undistort")
    run_command(cmd014, dry_run=bool(args.dry_run))

    if fmt == FORMAT_012_RAW:
        run_command(
            python_cmd(
                args,
                "017_merge_012_raw_and_014_pose_pkl.py",
                "--raw-pkl",
                str(raw_pkl),
                "--pose-pkl",
                str(april_strict_pkl),
                "--output-pkl",
                str(april_merged_pkl),
            ),
            dry_run=bool(args.dry_run),
        )
    else:
        april_merged_pkl = raw_pkl
        print(
            "[INFO] Input already has raw images plus a pose field; "
            "using it as the raw/old-April stream for downstream stages."
        )
    run_command(
        python_cmd(
            args,
            "016_deeptag_012_pkl.py",
            str(april_merged_pkl),
            "--output-pkl",
            str(deeptag_raw_pkl),
            "--quiet-deeptag",
        ),
        dry_run=bool(args.dry_run),
    )

    run_command(
        python_cmd(
            args,
            "020_deeptag_dense_keypoints_pose.py",
            str(deeptag_raw_pkl),
            "--output-pkl",
            str(deeptag_dense_strict_pkl),
            "--min-tags",
            "2",
            "--max-reproj",
            "6.0",
            "--point-reject-px",
            "8.0",
            "--tag-reject-px",
            "8.0",
            "--min-inlier-tag-fraction",
            "0.5",
            "--coverage-check-min-raw-tags",
            "3",
            "--max-required-inlier-tags",
            "4",
        ),
        dry_run=bool(args.dry_run),
    )
    run_command(
        python_cmd(
            args,
            "020_deeptag_dense_keypoints_pose.py",
            str(deeptag_raw_pkl),
            "--output-pkl",
            str(deeptag_dense_loose_pkl),
            "--min-tags",
            "1",
            "--max-reproj",
            "12.0",
            "--point-reject-px",
            "12.0",
            "--tag-reject-px",
            "12.0",
            "--min-inlier-tag-fraction",
            "0.0",
            "--coverage-check-min-raw-tags",
            "1000000",
            "--max-required-inlier-tags",
            "1000000",
        ),
        dry_run=bool(args.dry_run),
    )

    if bool(args.run_benchmark):
        run_command(
            python_cmd(
                args,
                "022_benchmark_single_frame_recovery_methods.py",
                "--raw-pkl",
                str(april_merged_pkl),
                "--deeptag-pkl",
                str(deeptag_raw_pkl),
                "--failed-reference-pkl",
                str(deeptag_dense_strict_pkl),
                "--loose-candidate-pkl",
                str(deeptag_dense_loose_pkl),
                "--april-old-pkl",
                str(april_merged_pkl),
                "--output-pkl",
                str(benchmark_pkl),
            ),
            dry_run=bool(args.dry_run),
        )

    run_command(
        python_cmd(
            args,
            "023_fuse_all_single_frame_recovery_methods.py",
            "--raw-pkl",
            str(april_merged_pkl),
            "--deeptag-raw-pkl",
            str(deeptag_raw_pkl),
            "--deeptag-pose-pkl",
            str(deeptag_dense_strict_pkl),
            "--april-strict-pkl",
            str(april_strict_pkl),
            "--loose-deeptag-pkl",
            str(deeptag_dense_loose_pkl),
            "--old-april-pkl",
            str(april_merged_pkl),
            "--output-pkl",
            str(fused_single_frame_pkl),
            "--min-tags",
            "2",
            "--max-reproj",
            "3.0",
            "--edge-threshold",
            "0.45",
            "--single-tag-edge-threshold",
            "0.60",
            "--single-tag-max-reproj",
            "1.0",
        ),
        dry_run=bool(args.dry_run),
    )
    run_command(
        python_cmd(
            args,
            "024_temporal_outline_refine_recovery.py",
            "--input-pkl",
            str(fused_single_frame_pkl),
            "--raw-pkl",
            str(april_merged_pkl),
            "--output-pkl",
            str(outline_refine_pkl),
            "--reject-loose-input",
        ),
        dry_run=bool(args.dry_run),
    )
    run_command(
        python_cmd(
            args,
            "025_global_temporal_filter_fill_remaining.py",
            "--input-pkl",
            str(outline_refine_pkl),
            "--raw-pkl",
            str(april_merged_pkl),
            "--output-pkl",
            str(final_pose_pkl),
        ),
        dry_run=bool(args.dry_run),
    )

    merge_args = argparse.Namespace(
        raw_pkl=raw_pkl,
        final_pose_pkl=final_pose_pkl,
        output_pkl=output_pkl,
        timestamp_tolerance=1e-6,
        keep_original_pose=True,
        keep_pose_candidates=True,
    )
    if not bool(args.dry_run):
        merge_final_pose_stream(merge_args)
        summarize_pose_stream(output_pkl, "pose")
    else:
        print(
            "[DRY-RUN] merge-final "
            f"--raw-pkl {raw_pkl} --final-pose-pkl {final_pose_pkl} --output-pkl {output_pkl}"
        )

    if not bool(args.keep_intermediates) and not bool(args.dry_run):
        shutil.rmtree(work_dir)
        print(f"[INFO] Removed work dir: {work_dir}")
    else:
        print(f"[INFO] Work dir: {work_dir}")
    return output_pkl


def run_auto(args: argparse.Namespace) -> Path:
    pkl_path = Path(args.pkl_path).expanduser().resolve()
    fmt = inspect_pkl_format(pkl_path)
    print(f"[INFO] Input format: {fmt}")
    if fmt in SUPPORTED_008_FORMATS:
        return run_008_pipeline(args)
    if fmt in SUPPORTED_012_INPUT_FORMATS:
        return run_012_pipeline(args)
    raise ValueError(f"Unsupported input pkl format for auto mode: {fmt}")


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


def merge_final_pose_stream(args: argparse.Namespace) -> Path:
    raw_pkl = Path(args.raw_pkl).expanduser().resolve()
    final_pose_pkl = Path(args.final_pose_pkl).expanduser().resolve()
    output_pkl = Path(args.output_pkl).expanduser().resolve()

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
                    "timestamp_tolerance": float(args.timestamp_tolerance),
                    "keep_original_pose": bool(args.keep_original_pose),
                    "keep_pose_candidates": bool(args.keep_pose_candidates),
                },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        for out_idx, raw_offset in enumerate(raw_offsets):
            raw_frame = load_at(raw_pkl, raw_offset)
            raw_ts = float(raw_frame["capture_timestamp"])
            pose_ts, pose_offset = nearest_timestamp(raw_ts, final_by_timestamp, float(args.timestamp_tolerance))
            pose_frame = load_at(final_pose_pkl, pose_offset)
            if not frame_indices_match(raw_frame, pose_frame):
                raise ValueError(
                    f"Frame index mismatch at timestamp {raw_ts}: raw={raw_frame.get('frame_index')} "
                    f"pose={pose_frame.get('frame_index')}"
                )

            out_frame = dict(raw_frame)
            if bool(args.keep_original_pose):
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
            if bool(args.keep_pose_candidates) and "pose_candidates" in pose_frame:
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
    args = parse_args(sys.argv[1:])
    if args.command == "auto":
        output = run_auto(args)
    elif args.command == "008":
        output = run_008_pipeline(args)
    elif args.command == "012":
        output = run_012_pipeline(args)
    elif args.command == "merge-final":
        output = merge_final_pose_stream(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(f"[INFO] Done: {output}")


if __name__ == "__main__":
    main()
