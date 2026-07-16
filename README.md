# aprilcube

`aprilcube` 用于生成带 ArUco / AprilTag 标记的三维打印 cube / cuboid，并从相机图像中估计其 6DoF 位姿。当前仓库除了原始的生成与检测包，还包含 OpenCV 实时检测脚本和多 cube viser 可视化脚本。

![printing process](assets/printing_process.gif)

## 功能概览

- 生成可 3D 打印的多色 `cube.3mf`，每个面可排布一个或多个 ArUco / AprilTag。
- 输出与模型配套的 `config.json`，检测端可直接读取 tag ID、面朝向和三维角点。
- 基于 OpenCV / AprilTag 检测结果估计 cube 在相机坐标系下的 6DoF 位姿。
- 支持 Kalman / SE(3) 时序平滑、异步检测、世界坐标外参、viser 三维可视化。
- 提供普通 OpenCV 相机和多 cube 实时检测实验脚本。

## 目录结构

```text
.
├── src/aprilcube/                    # 核心 Python 包：生成、检测、CLI
├── src/aprilcube_runtime.py          # AprilTag 原生检测 + cube pose 融合的运行时工具
├── src/00*.py                        # CV2 实时实验脚本
├── examples/                         # 简单 webcam 示例
├── models/                           # 原有生成模型
├── cube_april_* / aruco_cube_*        # 本地新增 cube 模型与打印文件
└── assets/                           # 图片、相机内参等资源
```

## 安装

基础包只依赖 OpenCV contrib 和 NumPy：

```bash
pip install -e .
```

如果要运行实时检测或 viser 可视化，通常还需要安装：

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
| `src/008_cv2_naive_aprilcube_detect.py` | 普通 OpenCV / 鱼眼相机多 cube 检测 |

运行示例：

```bash
python src/008_cv2_naive_aprilcube_detect.py
```

CV2 脚本可读取 `assets/intrinsics_DECXIN_3081V1_USB3_0509.yaml` 这类 YAML 内参文件，字段包含 `image_size`、`K`、`dist`、`fx/fy/cx/cy` 和标定误差。

## 本次提交建议

建议提交：

- `README.md`：中文项目说明。
- `src/aprilcube/detect.py`：可视化线宽与 viser object frame / mesh / axes 控制更新。
- `src/aprilcube_runtime.py`：AprilTag 原生 pose、时序 pose 与 cube pose 融合工具。
- `src/008_cv2_naive_aprilcube_detect.py`：CV2 多 cube 实验脚本。
- `assets/intrinsics_DECXIN_3081V1_USB3_0509.yaml`：DECXIN USB3 相机内参。
- 新增的 `cube_april_*` 与 `aruco_cube_*` 模型目录：打印文件、配置、预览图与 MuJoCo mesh。

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
