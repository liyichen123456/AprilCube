from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aprilcube_pose_benchmark.common_plot import save_compare_pose_curve


# =========================
# 中文参数区
# =========================

# 单个 recording 的所有算法输出目录。
RECORDING_OUTPUT_DIR = (
    "/home/ps/project/ConSensV2Lab/thirdparty/aprilcube/"
    "outputs/aprilcube_pose_benchmark/recording_20260511_162011"
)

# 汇总图文件名。
COMPARE_FIG_NAME = "compare_all_algorithms_xyz_rpy.png"


def load_algorithm_outputs(recording_output_dir: str | Path) -> dict[str, dict[str, np.ndarray]]:
    root = Path(recording_output_dir).expanduser().resolve()
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(root.glob("alg_*/poses.npz")):
        data = np.load(npz_path)
        outputs[npz_path.parent.name] = {key: data[key] for key in data.files}
    if not outputs:
        raise FileNotFoundError(f"No alg_*/poses.npz found in {root}")
    return outputs


def save_compare_plot(recording_output_dir: str | Path) -> Path:
    root = Path(recording_output_dir).expanduser().resolve()
    outputs = load_algorithm_outputs(root)
    out_path = root / COMPARE_FIG_NAME
    saved = save_compare_pose_curve(out_path, outputs)
    print(f"[RESULT] saved {saved}")
    return saved


def main() -> None:
    save_compare_plot(RECORDING_OUTPUT_DIR)


if __name__ == "__main__":
    main()

