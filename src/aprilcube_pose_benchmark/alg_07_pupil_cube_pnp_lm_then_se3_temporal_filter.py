from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aprilcube_pose_benchmark.common_pose import (
    average_rotations,
    detections_from_tags,
    pnp_cube_pose_lm,
    reprojection_error,
    result_from_pose,
)
from aprilcube_pose_benchmark.common_runner import BenchmarkConfig, result_from_pnp_tuple, run_benchmark


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
VISER_PORT = 8097
PLAYBACK_FPS = 25.0
LOOP_PLAYBACK = True

# 检测前对灰度图做 CLAHE 增强，所有算法统一使用这组参数。
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# 最终 cube pose 的时序低通强度。越小越稳但延迟越大；越大越接近原始 PnP。
SE3_FILTER_ALPHA = 0.35

ALGORITHM_NAME = "alg_07_pupil_cube_pnp_lm_then_se3_temporal_filter"


def algorithm_fn(detector: Any, native_detector: Any, tags: list[Any], gray: np.ndarray, context: dict[str, Any]) -> dict[str, Any]:
    """先整体 cube PnP + LM，再对最终 cube frame 做 SE(3) 低通滤波。"""
    detections = detections_from_tags(tags)
    raw_pose = pnp_cube_pose_lm(detector, detections, use_ransac=False)
    raw_result = result_from_pnp_tuple(detector, detections, raw_pose)
    if not raw_result.get("success", False):
        return raw_result

    raw_rot, _ = cv2.Rodrigues(np.asarray(raw_result["rvec"], dtype=np.float64).reshape(3, 1))
    raw_tvec = np.asarray(raw_result["tvec"], dtype=np.float64).reshape(3, 1)
    prev = context.get("filtered_pose", None)
    alpha = float(SE3_FILTER_ALPHA)
    if prev is None:
        filt_rot = raw_rot
        filt_tvec = raw_tvec
    else:
        prev_rot, prev_tvec = prev
        filt_rot = average_rotations([prev_rot, raw_rot], np.array([1.0 - alpha, alpha], dtype=np.float64))
        if filt_rot is None:
            filt_rot = raw_rot
        filt_tvec = (1.0 - alpha) * np.asarray(prev_tvec, dtype=np.float64).reshape(3, 1) + alpha * raw_tvec
    context["filtered_pose"] = (filt_rot, filt_tvec)
    filt_rvec, _ = cv2.Rodrigues(filt_rot)
    reproj = reprojection_error(detector, detections, filt_rvec, filt_tvec)
    result = result_from_pose(
        detector,
        detections,
        filt_rvec,
        filt_tvec,
        reproj_error=reproj,
        n_inliers=len(detections) * 4,
        debug={"raw_reproj_error": raw_result.get("reproj_error", None), "filter_alpha": alpha},
    )
    return result


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
    estimate_tag_pose=False,
)


def main() -> None:
    run_benchmark(CONFIG, algorithm_fn)


if __name__ == "__main__":
    main()

