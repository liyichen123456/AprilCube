# aprilcube

`aprilcube` 用于生成带 ArUco / AprilTag 标记的三维打印 cube / cuboid，并从相机图像中估计其 6DoF 位姿。当前仓库除了原始的生成与检测包，还包含 OpenCV 实时检测脚本、多 cube viser 可视化脚本，以及一组 AprilCube 位姿算法 benchmark。

![printing process](assets/printing_process.gif)

## 功能概览

- 生成可 3D 打印的多色 `cube.3mf`，每个面可排布一个或多个 ArUco / AprilTag。
- 输出与模型配套的 `config.json`，检测端可直接读取 tag ID、面朝向和三维角点。
- 基于 OpenCV / AprilTag 检测结果估计 cube 在相机坐标系下的 6DoF 位姿。
- 支持 Kalman / SE(3) 时序平滑、异步检测、世界坐标外参、viser 三维可视化。
- 提供普通 OpenCV 相机和多 cube 实时检测实验脚本。
- 提供 `aprilcube_pose_benchmark`，用于离线比较多种 cube pose 估计算法。

## 目录结构

```text
.
├── src/aprilcube/                    # 核心 Python 包：生成、检测、CLI
├── src/aprilcube_runtime.py          # AprilTag 原生检测 + cube pose 融合的运行时工具
├── src/aprilcube_pose_benchmark/     # 9 组离线位姿算法与评测/画图工具
├── src/00*.py                        # CV2 实时实验脚本
├── examples/                         # 简单 webcam 示例
├── models/                           # 原有生成模型
├── cube_april_* / aruco_cube_*        # 本地新增 cube 模型与打印文件
├── assets/                           # 图片、相机内参等资源
└── outputs/                          # benchmark 输出结果，按需提交
```

## 安装

基础包只依赖 OpenCV contrib 和 NumPy：

```bash
pip install -e .
```

如果要运行实时检测、viser 可视化或 benchmark，通常还需要安装：

```bash
pip install pupil-apriltags viser trimesh scipy pyyaml
```

CV2 实时脚本还依赖项目上层的相机工具，例如 `scripts/utils/recorder_cv2_cam.py` 和 `april_tag_detector.py`。这些脚本会把上层 `scripts/utils` 加入 `sys.path`。

## 生成 cube

命令行入口：

```bash
aprilcube generate --grid 1x1x1 --dict apriltag_36h11 --ids 0-5 --tag-size 10 -o cube_april_36h11_0_5_1x1x1_10mm
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--grid WxHxD` | cube 三个方向的 tag 网格，例如 `1x1x1`、`2x2x2` |
| `--dict` | ArUco / AprilTag 字典，例如 `4x4_100`、`apriltag_25h9`、`apriltag_36h11` |
| `--ids` | tag ID 范围或列表，例如 `0-5`、`6,7,8,9,10,11` |
| `--tag-size` | 单个 tag 的物理边长，单位 mm |
| `--cell-size` | 以 cell 为单位控制几何尺寸，可替代 `--tag-size` |
| `--margin-cell` | 相邻 tag 间距 |
| `--border-cell` | 外边框宽度 |
| `-o, --output` | 输出目录 |

每个生成模型目录通常包含：

```text
cube.3mf              # 多色 3MF 打印文件
config.json           # 生成参数、tag ID、面布局、三维角点
thumbnail.png         # 模型预览图
README.md             # 单个模型的说明
mujoco/
  cube.obj            # 带 UV 的 mesh
  cube.mtl
  cube_atlas.png
  cube.xml
```

当前仓库新增了多组 1x1x1 / 2x2x2 cube，例如：

- `cube_april_25h9_0_5_1x1x1_10mm`
- `cube_april_36h11_0_5_1x1x1_10mm`
- `cube_april_36h11_6_11_1x1x1_10mm`
- `cube_april_36h11_6_11_1x1x1_15mm`
- `cube_april_36h11_6_11_1x1x1_20mm`
- `cube_april_36h11_12_17_1x1x1_10mm`
- `cube_april_36h11_18_23_1x1x1_10mm`
- `cube_april_36h11_24_29_1x1x1_10mm`
- `cube_april_36h11_30_35_1x1x1_10mm`
- `aruco_cube_4x4_100_2x2x2`
- `aruco_cube_april_25h9_2x2x2_14.7mm`

## Python API

```python
import cv2
import aprilcube

det = aprilcube.detector(
    "cube_april_36h11_0_5_1x1x1_10mm",
    {"fx": 800, "fy": 800, "cx": 640, "cy": 360},
    fast=True,
)

frame = cv2.imread("frame.png")
result = det.process_frame(frame)

if result["success"]:
    print(result["rvec"])          # Rodrigues 旋转向量
    print(result["tvec"])          # 平移，单位 mm
    print(result["T"])             # 4x4 相机坐标系位姿
    print(result["reproj_error"])  # 重投影误差，单位 px
```

`aprilcube.detector()` 支持三种内参输入：

```python
# 1. dict
intrinsics = {"fx": 800, "fy": 800, "cx": 640, "cy": 360}

# 2. 3x3 numpy camera matrix
intrinsics = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=np.float64)

# 3. JSON 文件，包含 camera_matrix 和可选 dist_coeffs
det = aprilcube.detector("cube_dir/config.json", "calib.json")
```

如需世界坐标系位姿，可传入 `extrinsic=T_world_cam`：

```python
det = aprilcube.detector("cube_dir", intrinsics, extrinsic=T_world_cam)
result = det.process_frame(frame)
T_world_obj = det.world_pose(result)
```

## viser 可视化

`CubePoseEstimator.build_viser()` 会启动一个 viser 服务，自动显示相机坐标系、世界坐标系、cube mesh 和实时位姿。

```python
det = aprilcube.detector("cube_dir", intrinsics, fast=True)
server = det.build_viser(port=8080)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    det.process_frame(frame)
```

当前 `src/aprilcube/detect.py` 中的 viser 支持：

- 可配置 object frame 的轴长、轴半径和原点半径。
- 可通过内部开关控制是否显示 mesh、是否显示 object axes。
- OpenCV 叠加显示中，坐标轴和 box 线条更细，box 颜色调整为橙色，画面遮挡更少。

## 实时脚本

根目录下的 `src/00*.py` 是面向实验的脚本，默认参数集中写在文件顶部的 `User macros` 区域。

| 脚本 | 用途 |
| --- | --- |
| `src/004_cv2_aprilcube_detect_multi_cube.py` | 普通 OpenCV 相机多 cube 检测 |
| `src/004_cv2_alg_06_aprilcube_detect_multi_cube.py` | CV2 多 cube，使用 alg 06 |
| `src/004_cv2_alg_09_aprilcube_detect_multi_cube.py` | CV2 多 cube，使用 alg 09 |

运行示例：

```bash
python src/004_cv2_alg_09_aprilcube_detect_multi_cube.py
```

CV2 脚本可读取 `assets/intrinsics_DECXIN_3081V1_USB3_0509.yaml` 这类 YAML 内参文件，字段包含 `image_size`、`K`、`dist`、`fx/fy/cx/cy` 和标定误差。

## benchmark

`src/aprilcube_pose_benchmark` 用于离线评测不同 cube pose 解算策略。入口脚本：

```bash
python src/aprilcube_pose_benchmark/run_all_algorithms_on_recording.py
```

当前包含 9 组算法。它们的主要差异在于：直接使用 `pupil_apriltags` 给出的单 tag pose，还是只使用 tag 角点重新做 cube 级 PnP；是否把每个 face 分开求解；是否显式处理单个平面 tag 的 IPPE 二义性；以及是否使用上一帧位姿做时序筛选或保持。

| 算法 | 核心输入 | 求解策略 | 主要特点 |
| --- | --- | --- | --- |
| `alg_01_pupil_tag_pose_to_cube_pose_fuse` | `pupil_apriltags` 原生 tag pose | 每个 tag 先独立估计 tag pose，再根据 tag 在 cube 上的固定外参反推 cube pose，最后融合多个 cube pose | 实现最直接，能利用 `pupil_apriltags` 自带 pose；但每个 tag 的平面 pose 二义性和噪声会先进入 cube pose，旋转跳变风险较高 |
| `alg_02_pupil_all_tag_corners_to_cube_pnp_lm` | 所有可见 tag 的 2D 角点 | 把所有 tag 角点映射到 cube 坐标系，一次性做整体 cube PnP，并用 LM refine | 不依赖单 tag pose，几何约束统一；多 tag 可见时重投影误差较低，但没有 RANSAC，异常角点会直接影响解 |
| `alg_03_pupil_all_tag_corners_to_cube_pnp_ransac_lm` | 所有可见 tag 的 2D 角点 | 先对整体 cube PnP 做 RANSAC 去异常点，再用 LM refine | 比 alg 02 更抗局部误检/坏角点；如果可见点太少或 RANSAC 剔除过多，成功率可能略低 |
| `alg_04_pupil_per_face_corners_to_face_pnp_then_cube_fuse` | 按 face 分组后的 tag 角点 | 每个可见 face 单独做 PnP 得到 cube pose，再融合多个 face 的结果 | 保留 face 级局部一致性，适合分析不同面的贡献；但单 face 本质仍接近平面问题，旋转稳定性依赖融合质量 |
| `alg_05_pupil_single_face_temporal_else_multiface_cube_pnp` | tag 角点 + 单 face 时序 tag pose | 只有一个 face 可见时走 `TemporalTagPoseEstimator` 的 tag pose 融合；两个及以上 face 可见时回到整体 cube PnP | 针对“只看到一面”的场景增强稳定性；多面时保持 alg 02 的整体几何约束，但模式切换处可能出现位姿风格差异 |
| `alg_06_pupil_tag_pose_candidates_to_cube_consistency_select` | 单 tag 平面 pose 候选 | 每个 tag 保留多个 pose 候选，转换成 cube pose 后，通过 cube 间一致性选择最合理的一组 | 显式处理平面 tag 的多解问题，比直接信任原生 pose 更稳；依赖候选之间的一致性，多 tag 支撑越多越可靠 |
| `alg_07_pupil_cube_pnp_lm_then_se3_temporal_filter` | 整体 cube PnP 结果 | 先运行 alg 02 风格的 cube PnP + LM，再对最终 cube 位姿做 SE(3) 低通滤波，`SE3_FILTER_ALPHA=0.35` | 明显压低帧间抖动；代价是会引入滞后，快速运动时重投影误差和真实位姿响应可能变差 |
| `alg_08_hybrid_multiface_pnp_single_tag_ippe_temporal` | 多 face 角点 / 单 tag IPPE 候选 | 两个及以上 face 可见时用整体 cube PnP；只有单 tag 时用 IPPE 生成两个平面解，并用上一帧 cube pose + 重投影误差消歧 | 专门处理“多面可靠、单 tag 也不断轨”的情况；第一次只有单 tag 且没有上一帧参考时置信度低，不会用它初始化强时序 |
| `alg_09_cube_candidate_cluster_ransac_temporal_sanity` | IPPE 候选 + cube PnP RANSAC 候选 + 上一帧 pose | 为每个 tag 生成 IPPE cube 候选，同时加入整体 RANSAC PnP 候选；对候选做旋转/平移聚类，选最大一致簇并加时序 sanity gate；失败时可 hold 上一帧 | 当前最完整也最保守：同时利用候选聚类、RANSAC 和时序约束，录制数据上成功率最高；参数更多，输出可能包含 `predicted=True` 的保持帧 |

算法之间可以按复杂度理解为四条路线：

- `alg_01`：信任单 tag pose，再做 cube 融合。
- `alg_02` / `alg_03`：只信任角点，在 cube 层统一 PnP；`alg_03` 比 `alg_02` 多 RANSAC。
- `alg_04` / `alg_05`：按可见 face 或单面/多面情况切换策略。
- `alg_06` / `alg_08` / `alg_09`：围绕平面 tag 的多解二义性做候选选择；`alg_09` 额外加入聚类、RANSAC 候选、时序门限和 hold 机制。

当前 `outputs/aprilcube_pose_benchmark/recording_20260511_162011` 中保存了一次 568 帧录制的评测结果。`alg_09` 在该记录上成功率为 100%，平均重投影误差约 `8.71 px`；其他多数算法成功率约 `93.5%`。

## 本次提交建议

建议提交：

- `README.md`：中文项目说明。
- `src/aprilcube/detect.py`：可视化线宽与 viser object frame / mesh / axes 控制更新。
- `src/aprilcube_runtime.py`：AprilTag 原生 pose、时序 pose 与 cube pose 融合工具。
- `src/004*.py`：CV2 多 cube 实验脚本。
- `src/aprilcube_pose_benchmark/`：离线算法评测框架与 9 组算法实现。
- `assets/intrinsics_DECXIN_3081V1_USB3_0509.yaml`：DECXIN USB3 相机内参。
- 新增的 `cube_april_*` 与 `aruco_cube_*` 模型目录：打印文件、配置、预览图与 MuJoCo mesh。
- `outputs/aprilcube_pose_benchmark/...`：如果希望保留此次 benchmark 图表和指标，可以提交；否则建议忽略。

不建议直接提交：

- `logs_002/recording_20260511_162011.pkl`：当前约 8.5G，建议放到数据盘、对象存储或 Git LFS。
- `src/.cache/`：缓存目录，建议加入 `.gitignore`。

推荐提交前检查：

```bash
git status --short
du -sh logs_002 outputs src/.cache
git diff -- src/aprilcube/detect.py README.md
```

如果要避免误提交大文件，可先把下面内容加入 `.gitignore`：

```gitignore
logs_002/
outputs/
src/.cache/
```
