#!/usr/bin/env python3
"""Build the canonical report artifact from executed comparison results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUTPUT_DIR / "analysis_results.json"
ARTIFACT_PATH = OUTPUT_DIR / "artifact.json"
SOURCE_NOTES_PATH = OUTPUT_DIR / "source_notes.md"


def short_name(method: str) -> str:
    if method.startswith("alg_"):
        return method.split("_", 2)[0] + "_" + method.split("_", 2)[1]
    return method.replace("_runtime", "")


def materialize_with_sql(
    table_name: str,
    rows: list[dict],
    query: str,
) -> list[dict]:
    """Load reviewed Python rows into SQLite and execute the report query."""
    if not rows:
        return []
    columns = list(rows[0])

    def sqlite_type(column: str) -> str:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and all(isinstance(value, (int, bool)) for value in values):
            return "INTEGER"
        if values and all(isinstance(value, (int, float, bool)) for value in values):
            return "REAL"
        return "TEXT"

    connection = sqlite3.connect(":memory:")
    try:
        definitions = ", ".join(
            f'"{column}" {sqlite_type(column)}' for column in columns
        )
        connection.execute(f'CREATE TABLE "{table_name}" ({definitions})')
        placeholders = ", ".join("?" for _ in columns)
        connection.executemany(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',
            [[row.get(column) for column in columns] for row in rows],
        )
        cursor = connection.execute(query)
        output_columns = [item[0] for item in cursor.description]
        return [dict(zip(output_columns, values)) for values in cursor.fetchall()]
    finally:
        connection.close()


results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
generated_at = results["generated_at"]

historical_rows = []
for method, metrics in results["historical_benchmark"].items():
    historical_rows.append(
        {
            "method": short_name(method),
            "full_method": method,
            "output_rate": metrics["success_rate"],
            "measured_rate": metrics["measured_rate"],
            "hold_rate": metrics["inferred_hold_rate"],
            "reproj_mean_px": metrics["reproj_mean_px"],
            "reproj_p95_px": metrics["reproj_p95_px"],
            "translation_step_mean_mm": metrics["translation_step_mean_mm"],
            "rotation_step_mean_deg": metrics["rotation_step_mean_deg"],
        }
    )
historical_rows.sort(key=lambda row: row["method"])

same_input_rows = []
for method, summary in results["same_input_replay"]["summary"].items():
    same_input_rows.append(
        {
            "method": short_name(method),
            "full_method": method,
            "output_rate": summary["success_rate"],
            "measured_rate": summary["measured_rate"],
            "held_or_filled_rate": summary["predicted_or_filled_rate"],
            "failed_rate": max(0.0, 1.0 - summary["success_rate"]),
            "longest_hold_or_fill": summary["longest_predicted_or_filled_run"],
            "reproj_mean_px": summary["reprojection_px"]["mean"],
            "translation_step_p95_mm": summary["adjacent_translation_step_mm"]["p95"],
            "rotation_step_p95_deg": summary["adjacent_rotation_step_deg"]["p95"],
        }
    )
same_input_rows.sort(key=lambda row: (-row["measured_rate"], row["method"]))

stage_rows = []
for order, (stage, payload) in enumerate(results["pipeline_020_stages"].items(), start=1):
    summary = payload["summary"]
    stage_rows.append(
        {
            "order": order,
            "stage": stage,
            "output_rate": summary["success_rate"],
            "measured_rate": summary["measured_rate"],
            "filled_rate": summary["predicted_or_filled_rate"],
            "reproj_mean_px_within_020": summary["reprojection_px"]["mean"],
            "translation_step_p95_mm": summary["adjacent_translation_step_mm"]["p95"],
            "rotation_step_p95_deg": summary["adjacent_rotation_step_deg"]["p95"],
        }
    )

qa_payload = results["pipeline_020_multicamera_qa"]
qa_rows = []
for target, metrics in qa_payload.get("targets", {}).items():
    qa_rows.append(
        {
            "target": target,
            "camera": metrics["camera_name"],
            "output_rate": metrics["success_rate"],
            "measured_count": metrics["measured_final_count"],
            "reproj_mean_px_within_020": metrics["final_reprojection_mean_px"],
            "edge_alignment_mean": metrics["final_edge_alignment_mean"],
        }
    )

selected_current_methods = {
    "alg_03",
    "alg_09",
    "004_cv2_alg_06",
    "004_cv2_alg_09",
}
current_composition = [
    row for row in same_input_rows if row["method"] in selected_current_methods
]
final_020 = next(
    row for row in stage_rows if row["stage"] == "temporally_smoothed_final"
)
fused_020 = next(row for row in stage_rows if row["stage"] == "fused_single_frame")
current_composition.extend(
    [
        {
            "method": "020_fused_single_frame",
            "full_method": "020 fused single-frame stage",
            "output_rate": fused_020["output_rate"],
            "measured_rate": fused_020["measured_rate"],
            "held_or_filled_rate": fused_020["filled_rate"],
            "failed_rate": 1.0 - fused_020["output_rate"],
            "longest_hold_or_fill": 0,
            "reproj_mean_px": fused_020["reproj_mean_px_within_020"],
            "translation_step_p95_mm": fused_020["translation_step_p95_mm"],
            "rotation_step_p95_deg": fused_020["rotation_step_p95_deg"],
        },
        {
            "method": "020_final_offline",
            "full_method": "020 temporally smoothed final stage",
            "output_rate": final_020["output_rate"],
            "measured_rate": final_020["measured_rate"],
            "held_or_filled_rate": final_020["filled_rate"],
            "failed_rate": 1.0 - final_020["output_rate"],
            "longest_hold_or_fill": None,
            "reproj_mean_px": final_020["reproj_mean_px_within_020"],
            "translation_step_p95_mm": final_020["translation_step_p95_mm"],
            "rotation_step_p95_deg": final_020["rotation_step_p95_deg"],
        },
    ]
)
current_composition.sort(key=lambda row: (-row["measured_rate"], row["method"]))

HISTORICAL_SQL = """SELECT method, full_method, output_rate, measured_rate, hold_rate,
       reproj_mean_px, reproj_p95_px, translation_step_mean_mm,
       rotation_step_mean_deg
FROM historical_algorithms
ORDER BY method"""
SAME_INPUT_SQL = """SELECT method, full_method, output_rate, measured_rate,
       held_or_filled_rate, failed_rate, longest_hold_or_fill,
       reproj_mean_px, translation_step_p95_mm, rotation_step_p95_deg
FROM same_input_algorithms
ORDER BY measured_rate DESC, method"""
CURRENT_COMPOSITION_SQL = """SELECT method, full_method, output_rate, measured_rate,
       held_or_filled_rate, failed_rate, longest_hold_or_fill,
       reproj_mean_px, translation_step_p95_mm, rotation_step_p95_deg
FROM current_composition
ORDER BY measured_rate DESC, method"""
PIPELINE_SQL = """SELECT "order", stage, output_rate, measured_rate, filled_rate,
       reproj_mean_px_within_020, translation_step_p95_mm,
       rotation_step_p95_deg
FROM pipeline_020_stages
ORDER BY "order";"""
QA_SQL = """SELECT target, camera, output_rate, measured_count,
       reproj_mean_px_within_020, edge_alignment_mean
FROM pipeline_020_qa
ORDER BY output_rate DESC, target"""

historical_rows = materialize_with_sql(
    "historical_algorithms", historical_rows, HISTORICAL_SQL
)
same_input_rows = materialize_with_sql(
    "same_input_algorithms", same_input_rows, SAME_INPUT_SQL
)
current_composition = materialize_with_sql(
    "current_composition", current_composition, CURRENT_COMPOSITION_SQL
)
stage_rows = materialize_with_sql(
    "pipeline_020_stages", stage_rows, PIPELINE_SQL
)
qa_rows = materialize_with_sql("pipeline_020_qa", qa_rows, QA_SQL)

sources = [
    {
        "id": "historical_benchmark",
        "label": "May 2026 nine-algorithm benchmark outputs",
        "path": "analysis_results.json",
        "query": {
            "engine": "SQLite over reviewed Python-derived rows",
            "language": "sql",
            "sql": HISTORICAL_SQL,
            "description": "The notebook recomputes coverage and hold counts from metrics.json/poses.npz; this executed SQL selects the reviewed report rows.",
            "tables_used": [
                "outputs/aprilcube_pose_benchmark/recording_20260511_162011/compare_all_algorithms_metrics.json",
                "outputs/aprilcube_pose_benchmark/recording_20260511_162011/alg_*/poses.npz",
            ],
            "metric_definitions": [
                "output_rate = success frames / all frames",
                "measured_rate = frames with finite reprojection / all frames",
                "inferred hold = successful frame with non-finite reprojection; exact for alg_09 make_hold outputs",
            ],
        },
    },
    {
        "id": "same_input_replay",
        "label": "July 15 same-input replay of the historical algorithms and 004 runtimes",
        "path": "aprilcube_algorithm_comparison.ipynb",
        "query": {
            "engine": "SQLite over reviewed Python-derived rows",
            "language": "sql",
            "sql": SAME_INPUT_SQL,
            "description": "The notebook executes the 205-frame replay; this executed SQL selects the reviewed per-method report rows.",
            "tables_used": [
                "recordings/012_rs_raw_frames_20260715_192635.pkl",
                "src/aprilcube_pose_benchmark/alg_01..09",
                "src/004_cv2_alg_06_aprilcube_detect_multi_cube.py",
                "src/004_cv2_alg_09_aprilcube_detect_multi_cube.py",
            ],
            "filters": [
                "all 205 frames",
                "historical algorithms: undistort + CLAHE 3.0/8x8",
                "004 runtimes: undistort without CLAHE",
            ],
            "metric_definitions": [
                "measured frame = success, not predicted/filled, finite reprojection",
                "translation step = adjacent-frame Euclidean tvec distance in millimeters",
                "rotation step = adjacent-frame SO(3) geodesic angle in degrees",
            ],
        },
    },
    {
        "id": "current_composition",
        "label": "July 15 same-input output composition including 020 stages",
        "path": "aprilcube_algorithm_comparison.ipynb",
        "query": {
            "engine": "SQLite over reviewed Python-derived rows",
            "language": "sql",
            "sql": CURRENT_COMPOSITION_SQL,
            "description": "Executed selection of measured, held/filled, and failed shares after combining reviewed replay and 020 stage summaries.",
            "tables_used": ["current_composition"],
            "metric_definitions": [
                "measured_rate = fresh finite-reprojection measurements / all frames",
                "held_or_filled_rate = predicted, interpolated, recovered, or trajectory-filled frames / all frames",
                "failed_rate = unsuccessful frames / all frames",
            ],
        },
    },
    {
        "id": "pipeline_020",
        "label": "Saved 020 pipeline stages for the July 15 012 stream",
        "path": "analysis_results.json",
        "query": {
            "engine": "SQLite over reviewed Python-derived rows",
            "language": "sql",
            "sql": PIPELINE_SQL,
            "description": "The notebook profiles each saved 020 stream; this executed SQL selects stage-level report rows.",
            "tables_used": [
                "recordings/020_work_current_012_20260715_192635/*.pkl",
                "src/020_finalize_pose_postprocess.py",
            ],
            "metric_definitions": [
                "filled frame = pose_filled true",
                "measured frame = successful non-filled pose with finite stage reprojection",
            ],
        },
    },
    {
        "id": "pipeline_020_qa",
        "label": "July 16 multi-camera 020 QA report",
        "path": "analysis_results.json",
        "query": {
            "engine": "SQLite over reviewed Python-derived rows",
            "language": "sql",
            "sql": QA_SQL,
            "description": "The notebook extracts per-target QA values; this executed SQL selects the reviewed report rows.",
            "tables_used": [
                "recordings/qa_multi_cam_record_0716_180451_020/qa_report.json"
            ],
        },
    },
]

charts = [
    {
        "id": "historical_measured_coverage",
        "title": "历史 9 算法的实测覆盖率",
        "subtitle": "2026-05-11，568 帧；有限重投影帧占比",
        "showDescription": True,
        "intent": "comparison",
        "question": "历史 benchmark 中哪些算法真正输出了逐帧测量，而非仅维持成功状态？",
        "rationale": "按算法排序的横向柱图最直接显示 alg_09 的成功率与实测率分离。",
        "comparisonContext": {
            "baseline": "同一 568 帧旧录制",
            "denominator": "全部帧",
            "grain": "算法",
            "unit": "比例",
        },
        "type": "horizontalBar",
        "dataset": "historical_algorithms",
        "sourceId": "historical_benchmark",
        "encodings": {
            "x": {"field": "method", "type": "nominal", "label": "算法"},
            "y": {
                "field": "measured_rate",
                "type": "quantitative",
                "format": "percent",
                "label": "实测覆盖率",
            },
        },
        "valueFormat": "percent",
        "layout": "full",
        "labels": {"values": "all"},
        "legend": {"interactive": False, "position": "bottom"},
    },
    {
        "id": "current_output_composition",
        "title": "当前 205 帧录制的输出构成",
        "subtitle": "直接测量、持有/补帧与失败的占比；同一原始图像流",
        "showDescription": True,
        "intent": "composition",
        "question": "在当前 2×2×2 cube 录制上，各方案的输出由多少直接测量和多少持有或补帧组成？",
        "rationale": "100% 堆叠横向柱图能避免把连续输出误读成连续测量。",
        "comparisonContext": {
            "baseline": "同一 205 帧 012 原始录制",
            "denominator": "全部帧",
            "grain": "算法或流水线阶段",
            "unit": "比例",
        },
        "type": "horizontalStackedBar100",
        "dataset": "current_composition",
        "sourceId": "current_composition",
        "encodings": {
            "x": {"field": "method", "type": "nominal", "label": "方案"},
            "y": {
                "fields": ["measured_rate", "held_or_filled_rate", "failed_rate"],
                "type": "quantitative",
                "format": "percent",
                "label": "帧占比",
            },
        },
        "valueFormat": "percent",
        "layout": "full",
        "legend": {"interactive": False, "position": "bottom", "sort": "spec"},
    },
]

tables = [
    {
        "id": "historical_detail",
        "title": "历史 benchmark 明细",
        "subtitle": "旧 1×1×1 cube、568 帧；重投影只在相同 Pupil 角点口径内比较",
        "showDescription": True,
        "dataset": "historical_algorithms",
        "defaultSort": {"field": "measured_rate", "direction": "desc"},
        "density": "dense",
        "sourceId": "historical_benchmark",
        "layout": "full",
        "columns": [
            {"field": "method", "label": "算法", "type": "text"},
            {"field": "output_rate", "label": "输出率", "format": "percent"},
            {"field": "measured_rate", "label": "实测率", "format": "percent"},
            {"field": "hold_rate", "label": "推定 hold", "format": "percent"},
            {"field": "reproj_mean_px", "label": "均值重投影 px", "format": "number"},
            {"field": "translation_step_mean_mm", "label": "平均位移步长 mm", "format": "number"},
            {"field": "rotation_step_mean_deg", "label": "平均 RPY 步长 °", "format": "number"},
        ],
    },
    {
        "id": "current_replay_detail",
        "title": "当前录制的历史算法与 004 runtime 复测",
        "subtitle": "同一 205 帧；重投影仅用于同检测器内部诊断",
        "showDescription": True,
        "dataset": "same_input_algorithms",
        "defaultSort": {"field": "measured_rate", "direction": "desc"},
        "density": "dense",
        "sourceId": "same_input_replay",
        "layout": "full",
        "columns": [
            {"field": "method", "label": "方案", "type": "text"},
            {"field": "output_rate", "label": "输出率", "format": "percent"},
            {"field": "measured_rate", "label": "实测率", "format": "percent"},
            {"field": "held_or_filled_rate", "label": "hold/补帧", "format": "percent"},
            {"field": "longest_hold_or_fill", "label": "最长连续 hold", "format": "number"},
            {"field": "reproj_mean_px", "label": "均值重投影 px", "format": "number"},
            {"field": "translation_step_p95_mm", "label": "P95 位移步长 mm", "format": "number"},
            {"field": "rotation_step_p95_deg", "label": "P95 旋转步长 °", "format": "number"},
        ],
    },
    {
        "id": "pipeline_stage_detail",
        "title": "020 各阶段覆盖率",
        "subtitle": "同一 205 帧；阶段顺序从 strict 检测到最终约束平滑",
        "showDescription": True,
        "dataset": "pipeline_020_stages",
        "defaultSort": {"field": "order", "direction": "asc"},
        "density": "spacious",
        "sourceId": "pipeline_020",
        "layout": "full",
        "columns": [
            {"field": "order", "label": "阶段", "format": "number"},
            {"field": "stage", "label": "名称", "type": "text"},
            {"field": "output_rate", "label": "输出率", "format": "percent"},
            {"field": "measured_rate", "label": "实测率", "format": "percent"},
            {"field": "filled_rate", "label": "补帧率", "format": "percent"},
            {"field": "translation_step_p95_mm", "label": "P95 位移步长 mm", "format": "number"},
            {"field": "rotation_step_p95_deg", "label": "P95 旋转步长 °", "format": "number"},
        ],
    },
    {
        "id": "multicamera_qa_detail",
        "title": "020 多相机、多 cube QA",
        "subtitle": "2026-07-16，203 帧；不同 target 的成功率差异",
        "showDescription": True,
        "dataset": "pipeline_020_qa",
        "defaultSort": {"field": "output_rate", "direction": "desc"},
        "density": "spacious",
        "sourceId": "pipeline_020_qa",
        "layout": "full",
        "columns": [
            {"field": "target", "label": "target", "type": "text"},
            {"field": "camera", "label": "camera", "type": "text"},
            {"field": "output_rate", "label": "输出率", "format": "percent"},
            {"field": "measured_count", "label": "有限重投影帧", "format": "number"},
            {"field": "reproj_mean_px_within_020", "label": "020 内重投影 px", "format": "number"},
            {"field": "edge_alignment_mean", "label": "边缘对齐均值", "format": "number"},
        ],
    },
]

blocks = [
    {
        "id": "title",
        "type": "markdown",
        "body": "# AprilCube 9 种历史算法与 020 的技术比较",
        "layout": "full",
    },
    {
        "id": "technical_summary",
        "type": "markdown",
        "body": """## 技术结论

**离线最终 pose：选 `020`。** 在与 `020` 相同的当前 205 帧 2×2×2 cube 录制上，最终阶段输出 205/205；其中 154 帧（75.1%）有直接测量锚点，51 帧（24.9%）来自轮廓恢复、插值或全局时序补帧。它不是单一 PnP 算法，而是检测、候选融合、异常剔除、恢复和约束平滑的离线级联。

**实时跟踪：不要把历史 `alg_09` 的 100% 当成 100% 检测。** 旧 benchmark 中它有 174/568 帧无有限重投影，属于持有旧 pose；在当前 205 帧复测中，`004_cv2_alg_09` 仅 34 帧（16.6%）是直接测量，142 帧（69.3%）在 hold，最长连续 hold 为 139 帧。

**旧 1×1×1 数据内，`alg_03` 是更好的逐帧测量基线。** 它的均值重投影最低（7.85 px），实测覆盖 92.6%；但没有外部 6DoF 真值，因此不能证明绝对 pose 最准。

**推荐架构：** 离线继续用完整 `020`；实时则抽取 `020` 的单帧 DeepTag dense + 候选门限部分，再接短时、有限 hold 的因果滤波器，而不是直接使用完整非因果 `020` 或无限 hold 的 `alg_09`。""",
        "layout": "full",
    },
    {
        "id": "historical_heading",
        "type": "markdown",
        "body": """## 历史 `alg_09` 赢在连续输出，不是逐帧测量

旧 benchmark 的大多数算法实测覆盖约 93%，`alg_09` 的输出率升到 100%，但实测覆盖反而降至 69.4%。因此原先“`alg_09` 成功率 100%”应改写为“69.4% 测量 + 30.6% hold”。这会显著压低表观运动步长，却可能在物体真实运动时长期输出陈旧 pose。""",
        "layout": "full",
        "sourceId": "historical_benchmark",
    },
    {
        "id": "historical_chart_block",
        "type": "chart",
        "chartId": "historical_measured_coverage",
        "layout": "full",
    },
    {
        "id": "historical_chart_interpretation",
        "type": "markdown",
        "body": """图中只画“有有限重投影的实测帧”。`alg_03` 在旧数据上兼顾较高实测覆盖和最低均值重投影；`alg_09` 则应被理解为带时序保持的连续轨迹策略，而不是更强的逐帧观测器。""",
        "layout": "full",
    },
    {
        "id": "historical_table_block",
        "type": "table",
        "tableId": "historical_detail",
        "layout": "full",
    },
    {
        "id": "current_heading",
        "type": "markdown",
        "body": """## 在当前 2×2×2 cube 上，`020` 的优势来自 DeepTag 和分阶段恢复

同输入复测显示，历史算法多数仍能返回数值，但出现 20–63 px 的均值重投影和大幅姿态跳变，说明它们从旧 1×1×1 配置迁移后并未保持几何一致性。`004_cv2_alg_06` 的严格门限只接受 11 个直接测量；`004_cv2_alg_09` 大部分时间依赖 hold。相比之下，`020` 的单帧融合阶段已有 75.6% 直接测量覆盖，最终再用显式标记的恢复/补帧补足到 100%。""",
        "layout": "full",
    },
    {
        "id": "current_chart_block",
        "type": "chart",
        "chartId": "current_output_composition",
        "layout": "full",
    },
    {
        "id": "current_chart_interpretation",
        "type": "markdown",
        "body": """每根柱都把“直接测量”和“持有/恢复/补帧”拆开。`020_final_offline` 的 100% 中约四分之一依赖未来帧或轨迹恢复；它适合离线后处理，不适合直接作为实时控制算法。`020_fused_single_frame` 则是更合适的实时候选基础。""",
        "layout": "full",
    },
    {
        "id": "current_table_block",
        "type": "table",
        "tableId": "current_replay_detail",
        "layout": "full",
    },
    {
        "id": "scope_definitions",
        "type": "markdown",
        "body": """## 比较范围与指标定义

- **历史 cohort：** 2026-05-11，568 帧，`r_wrist`，旧 1×1×1 10 mm tag cube。
- **同输入 cohort：** 2026-07-15，205 帧，D435，当前 2×2×2 outer-62.5 mm cube；9 个历史函数、两个 `004` runtime 和保存的 `020` 阶段都使用这份原始图像流。
- **输出率：** 返回 `success=True` 的帧数除以全部帧数。
- **实测率：** `success=True`、非 predicted/filled 且有有限重投影值的帧占比。
- **hold/补帧率：** 使用上一 pose、插值、轮廓恢复或全局轨迹填充的帧占比。
- **时序步长：** 相邻成功帧的 tvec 欧氏距离和 SO(3) 测地旋转角。旧 JSON 中的 rotation step 是 RPY 欧氏差，仅作原始 benchmark 审计字段。
- **重投影：** 只在同一 detector/关键点口径内部比较。Pupil tag 外角点误差不能和 DeepTag dense 内部点误差横向排名。""",
        "layout": "full",
    },
    {
        "id": "methodology",
        "type": "markdown",
        "body": """## 方法：保留原始路径，再增加同输入复测

1. 从历史 `metrics.json` 和 `poses.npz` 复算输出、有限重投影覆盖和连续 hold。
2. 将 9 个历史 `algorithm_fn` 在保存的 205 帧 012 原始流上重放；使用原 benchmark 的 undistort + CLAHE 参数。
3. 同时运行 `004_cv2_alg_06` 和 `004_cv2_alg_09` runtime。后者直接导入历史 `alg_09`；前者虽然文件名含 alg06，实际是全 cube RANSAC/LM PnP、tag 异常点剔除、face-normal/temporal gate 和 Kalman filter，并不是历史 benchmark 的 alg06。
4. 读取 `020` 每个已保存 stage 的逐帧 pose 和 footer，分别统计测量、恢复、补帧、跳变与来源。
5. 用另一份 203 帧多相机 QA 检查是否能把单 cube 结论推广到其他 target。""",
        "layout": "full",
    },
    {
        "id": "stage_heading",
        "type": "markdown",
        "body": """## `020` 的 100% 是可审计的逐级覆盖，而非隐式 hold

Strict AprilCube 单独只覆盖 9.8%；DeepTag dense 提升到 73.2%；单帧融合达到 75.6%；时序异常剔除后保留 75.1% 测量；轮廓/插值恢复到 87.3%；全局补帧最终到 100%。这种来源标记比 `alg_09` 的笼统 success 更适合离线数据生产和质量审计。""",
        "layout": "full",
        "sourceId": "pipeline_020",
    },
    {
        "id": "stage_table_block",
        "type": "table",
        "tableId": "pipeline_stage_detail",
        "layout": "full",
    },
    {
        "id": "limitations",
        "type": "markdown",
        "body": """## 限制、稳健性与不能下的结论

**没有外部 ground truth。** 重投影低只表示与当前检测角点自洽；错误的 tag 对应、镜像解或平面 PnP 分支也可能取得低误差。当前结论不能替代平移/旋转真值误差。

**`020` 是非因果离线算法。** Stage 10–12 使用前后帧、全局样条/Slerp 和对称窗口；它的覆盖率不能与实时延迟约束下的算法公平等同。

**跨配置泛化仍不足。** 203 帧多相机 QA 中，`index_Q` 为 100%，`wrist_Q` 为 62.1%，而 `thumb_Q` 与 `middle_Q` 均为 0%。后续 provisional orientation/RGB flow 结果不是等价的 `020` 实测 pose。

**同输入复测存在预处理差异。** 历史函数保留 CLAHE，`004` 保留无 CLAHE；这是对各自默认路径的测试，而不是只隔离 pose solver 的消融实验。""",
        "layout": "full",
    },
    {
        "id": "qa_table_block",
        "type": "table",
        "tableId": "multicamera_qa_detail",
        "layout": "full",
    },
    {
        "id": "recommendations",
        "type": "markdown",
        "body": """## 建议的工程决策

1. **离线录制后处理：继续使用完整 `020`**，并在下游始终保留 `pose_source`、`quality_level`、`pose_filled`，不要把填充帧当成等权真值。
2. **实时路径：以 `020_fused_single_frame` 为起点**，保留 DeepTag dense、cross-validation、face-normal 和 reprojection/edge gate；替换 Stage 10–12 为短时因果滤波。
3. **设置有限 hold：** 最多保持少量帧或固定毫秒数，超限明确输出 invalid；禁止 `alg_09` 式 139 帧无界持有。
4. **旧 1×1×1 系统若必须选一个旧算法：选 `alg_03` 作为测量基线**，再单独设计短时滤波；不要按成功率选择 `alg_09`。
5. **为每个 cube/camera 做独立验收：** `thumb_Q` 和 `middle_Q` 当前是阻断项，不能从 `index_Q` 的 100% 外推。""",
        "layout": "full",
    },
    {
        "id": "further_questions",
        "type": "markdown",
        "body": """## 下一轮需要回答的问题

- 用 motion capture、机器人正运动学或已测量外参生成 6DoF ground truth 后，哪种方案的平移/旋转 P50、P95 和最大误差最低？
- 在相同 detector 角点输入下，只替换 solver，`alg_03`、`004_alg06` 与 `020` strict PnP 的差异还剩多少？
- DeepTag dense 单帧阶段在目标 GPU 上的端到端延迟和吞吐是多少，能否满足实时控制？
- 合理的最大 hold 时长是多少，超过后下游应停止控制、降级还是切换视觉源？
- `thumb_Q`/`middle_Q` 的失败主要来自可见 tag 数、角点顺序、内参、cube cfg 还是视角/遮挡？""",
        "layout": "full",
    },
]

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "AprilCube 9 种历史算法与 020 的技术比较",
        "description": "历史 benchmark、同输入重放、020 分阶段结果与多相机 QA 的技术比较。",
        "generatedAt": generated_at,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    },
    "snapshot": {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "historical_algorithms": historical_rows,
            "same_input_algorithms": same_input_rows,
            "current_composition": current_composition,
            "pipeline_020_stages": stage_rows,
            "pipeline_020_qa": qa_rows,
        },
        "accessIssues": [],
    },
    "sources": sources,
    "package_info": {
        "root": ".",
        "manifestPath": "artifact.json",
        "snapshotPath": "artifact.json",
    },
}

ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

SOURCE_NOTES_PATH.write_text(
    """# Source and chart notes

## Report structure mapping

- Technical summary: `technical_summary`
- Key findings with visuals: historical measured coverage and current output composition
- Scope/data/definitions: `scope_definitions`
- Methodology: `methodology`
- Limitations/robustness: `limitations` plus multi-camera QA table
- Recommended next steps: `recommendations`
- Further questions: `further_questions`

## Chart map

- `historical_measured_coverage`: comparison / horizontal bar; method × measured rate; supports the claim that alg09's 100% output is not 100% fresh measurement; single-root default palette.
- `current_output_composition`: composition / 100% horizontal stacked bar; method × measured/held-or-filled/failed shares; supports the offline-vs-online recommendation; blue/orange/neutral with direct legend.

## Omitted visuals

- 020's seven ordered stages use a table rather than a line because the stages are discrete transformations, not temporal observations.
- Reprojection is not charted across Pupil and DeepTag because the keypoint definitions differ materially.
- Multi-camera QA has only four targets, so a spacious exact table is clearer than a chart.

## Validation caveat

- No source includes external 6DoF ground truth. All accuracy language is restricted to self-consistency, coverage, and temporal behavior.
""",
    encoding="utf-8",
)

print(ARTIFACT_PATH)
print(SOURCE_NOTES_PATH)
