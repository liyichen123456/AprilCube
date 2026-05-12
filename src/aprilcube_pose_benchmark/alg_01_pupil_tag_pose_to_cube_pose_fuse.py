from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aprilcube_pose_benchmark.common_pose import native_tag_pose_fusion
from aprilcube_pose_benchmark.common_runner import BenchmarkConfig, run_benchmark


# =========================
# 中文参数区
# =========================

# 要评测的 002 录制文件。
PKL_PATH = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/logs_002/recording_20260511_162011.pkl"

# 结果保存根目录。脚本会自动创建 recording 名称和算法名称子目录。
OUTPUT_ROOT = "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/outputs/aprilcube_pose_benchmark"

# 是否打开 viser 播放每帧结果。批量跑算法时建议设为 False。
ENABLE_VISER = True
VISER_HOST = "0.0.0.0"
VISER_PORT = 8091
PLAYBACK_FPS = 25.0
LOOP_PLAYBACK = True

# 检测前对灰度图做 CLAHE 增强，所有算法统一使用这组参数。
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

ALGORITHM_NAME = "alg_01_pupil_tag_pose_to_cube_pose_fuse"


def algorithm_fn(detector: Any, native_detector: Any, tags: list[Any], gray: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    """每个 tag 先用 pupil_apriltags 原生 pose，再反推 cube pose 并融合。"""
    return native_tag_pose_fusion(detector, tags)


CONFIG = BenchmarkConfig(
    algorithm_name=ALGORITHM_NAME,
    pkl_path=PKL_PATH,
    output_root=OUTPUT_ROOT,
    enable_viser=ENABLE_VISER,
    viser_host=VISER_HOST,
    viser_port=VISER_PORT,
    playback_fps=PLAYBACK_FPS,
    loop_playback=LOOP_PLAYBACK,
    clahe_clip_limit=CLAHE_CLIP_LIMIT,
    clahe_tile_grid_size=CLAHE_TILE_GRID_SIZE,
    estimate_tag_pose=True,
)


def main() -> None:
    run_benchmark(CONFIG, algorithm_fn)


if __name__ == "__main__":
    main()

