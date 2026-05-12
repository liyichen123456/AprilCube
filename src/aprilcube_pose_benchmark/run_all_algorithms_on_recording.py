from __future__ import annotations

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aprilcube_pose_benchmark.plot_all_pose_curves import save_compare_plot


# =========================
# 中文参数区
# =========================

# 要批量评测的 002 录制文件。
PKL_PATH = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/logs_002/recording_20260511_162011.pkl"

# 结果保存根目录。
OUTPUT_ROOT = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/outputs/aprilcube_pose_benchmark"

# 批量跑时是否打开 viser。建议保持 False；单独看某个算法时运行对应 alg_*.py。
ENABLE_VISER_WHEN_RUN_ALL = False

# 所有算法统一使用的 CLAHE 参数。
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ALGORITHM_MODULES = [
    "aprilcube_pose_benchmark.alg_01_pupil_tag_pose_to_cube_pose_fuse",
    "aprilcube_pose_benchmark.alg_02_pupil_all_tag_corners_to_cube_pnp_lm",
    "aprilcube_pose_benchmark.alg_03_pupil_all_tag_corners_to_cube_pnp_ransac_lm",
    "aprilcube_pose_benchmark.alg_04_pupil_per_face_corners_to_face_pnp_then_cube_fuse",
    "aprilcube_pose_benchmark.alg_05_pupil_single_face_temporal_else_multiface_cube_pnp",
    "aprilcube_pose_benchmark.alg_06_pupil_tag_pose_candidates_to_cube_consistency_select",
    "aprilcube_pose_benchmark.alg_07_pupil_cube_pnp_lm_then_se3_temporal_filter",
    "aprilcube_pose_benchmark.alg_08_hybrid_multiface_pnp_single_tag_ippe_temporal",
    "aprilcube_pose_benchmark.alg_09_cube_candidate_cluster_ransac_temporal_sanity",
]


def main() -> None:
    metrics_by_algorithm = {}
    for module_name in ALGORITHM_MODULES:
        module = importlib.import_module(module_name)
        config = replace(
            module.CONFIG,
            pkl_path=PKL_PATH,
            output_root=OUTPUT_ROOT,
            enable_viser=ENABLE_VISER_WHEN_RUN_ALL,
            clahe_clip_limit=CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE,
        )
        metrics = module.run_benchmark(config, module.algorithm_fn)
        metrics_by_algorithm[config.algorithm_name] = metrics

    recording_name = Path(PKL_PATH).stem
    recording_output_dir = Path(OUTPUT_ROOT).expanduser().resolve() / recording_name
    compare_path = save_compare_plot(recording_output_dir)
    metrics_path = recording_output_dir / "compare_all_algorithms_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_by_algorithm, f, indent=2, ensure_ascii=False)
    print(f"[RESULT] compare plot: {compare_path}")
    print(f"[RESULT] metrics json: {metrics_path}")


if __name__ == "__main__":
    main()
