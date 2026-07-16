#!/usr/bin/env python3
"""Build the reviewable notebook companion for the algorithm comparison."""

from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = OUTPUT_DIR / "aprilcube_algorithm_comparison.ipynb"

notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "pyroki",
    "language": "python",
    "name": "pyroki",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# AprilCube 9-algorithm vs 020 comparison

## tl;dr

- For **offline final pose production on the current 2×2×2 cube**, use `020`: on the shared 205-frame recording it produced 205/205 outputs, with 154/205 directly measured anchors and 51/205 temporally recovered or filled.
- Do not interpret historical `alg_09` as 100% measured detection. In the May benchmark, 174/568 outputs had no finite reprojection measurement and were inferred holds; on the current shared recording, `004_cv2_alg_09` measured only 34/205 frames and held a previous pose for 142 frames.
- For the old 1×1×1 benchmark, `alg_03` had the lowest measured mean reprojection (7.85 px at 92.6% coverage). `alg_09` optimized continuity, not independent per-frame measurement.
- No dataset has external 6DoF ground truth. The recommendation is therefore about coverage, self-consistency, and robustness—not proven absolute pose accuracy.
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

### Key assumptions

- The decision is whether to keep using one of the historical algorithms or the current `020` postprocess pipeline.
- The historical May result and July `020` result are not treated as directly comparable because they use different recordings, cubes, cameras, and reprojection definitions.
- The nine historical functions are replayed on the same July 15 raw stream used by the saved `020` stages. Historical algorithms use the original CLAHE preprocessing; the `004` runtimes use their no-CLAHE path.
- “Measured” requires a successful pose, no prediction/fill flag, and a finite reprojection value. Held/interpolated/filled outputs are reported separately.
- Reprojection is compared only within one detector family. Pupil outer-corner error and DeepTag dense-keypoint error are not mixed into one ranking.
"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import sys

import pandas as pd

OUTPUT_DIR = Path.cwd()
if not (OUTPUT_DIR / "analyze.py").exists():
    OUTPUT_DIR = OUTPUT_DIR / "outputs" / "aprilcube_algorithm_comparison"
sys.path.insert(0, str(OUTPUT_DIR))
from analyze import run_analysis

results = run_analysis(max_frames=0)
results_path = OUTPUT_DIR / "analysis_results.json"
results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"saved: {results_path}")
print(f"shared replay frames: {results['same_input_replay']['context']['evaluated_frames']}")
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Data

The first table is the apples-to-apples July 15 replay. It shows output availability separately from independently measured frames. This distinction prevents a long pose hold from being scored as a fresh detection.
"""
    ),
    nbf.v4.new_code_cell(
        """same_rows = []
for method, summary in results["same_input_replay"]["summary"].items():
    same_rows.append({
        "method": method,
        "output_rate_pct": 100 * summary["success_rate"],
        "measured_rate_pct": 100 * summary["measured_rate"],
        "held_or_filled_pct": 100 * summary["predicted_or_filled_rate"],
        "longest_hold_or_fill": summary["longest_predicted_or_filled_run"],
        "reproj_mean_px_same_family_only": summary["reprojection_px"]["mean"],
        "translation_step_p95_mm": summary["adjacent_translation_step_mm"]["p95"],
        "rotation_step_p95_deg": summary["adjacent_rotation_step_deg"]["p95"],
    })
same_df = pd.DataFrame(same_rows).sort_values(
    ["measured_rate_pct", "output_rate_pct"], ascending=False
)
same_df.round(2)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## The historical “100%” result contains substantial pose holding

The May benchmark remains useful for comparing the nine old methods on their original 1×1×1 recording. Its aggregate `success_rate`, however, conflates measurement with temporal hold for `alg_09`. Finite reprojection coverage is a better proxy for fresh measurements in the saved outputs.
"""
    ),
    nbf.v4.new_code_cell(
        """historical_rows = []
for method, metrics in results["historical_benchmark"].items():
    historical_rows.append({
        "method": method,
        "output_rate_pct": 100 * metrics["success_rate"],
        "measured_rate_pct": 100 * metrics["measured_rate"],
        "inferred_hold_pct": 100 * metrics["inferred_hold_rate"],
        "longest_inferred_hold": metrics["longest_inferred_hold_run"],
        "reproj_mean_px": metrics["reproj_mean_px"],
        "reproj_p95_px": metrics["reproj_p95_px"],
        "translation_step_mean_mm": metrics["translation_step_mean_mm"],
        "rotation_step_mean_deg_rpy_metric": metrics["rotation_step_mean_deg"],
    })
historical_df = pd.DataFrame(historical_rows).sort_values(
    ["measured_rate_pct", "reproj_mean_px"], ascending=[False, True]
)
historical_df.round(2)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## 020 earns continuity through a staged offline cascade

On the shared July 15 stream, the strict AprilCube-only stage was insufficient. DeepTag dense points supplied most measured anchors; later stages rejected one temporal spike, recovered some gaps with outline constraints, filled the rest from the full sequence, and applied constrained smoothing. The final 100% is therefore 75.1% measured and 24.9% recovered/filled.
"""
    ),
    nbf.v4.new_code_cell(
        """stage_rows = []
for stage, payload in results["pipeline_020_stages"].items():
    summary = payload["summary"]
    stage_rows.append({
        "stage": stage,
        "output_rate_pct": 100 * summary["success_rate"],
        "measured_rate_pct": 100 * summary["measured_rate"],
        "filled_rate_pct": 100 * summary["predicted_or_filled_rate"],
        "reproj_mean_px_within_020": summary["reprojection_px"]["mean"],
        "translation_step_p95_mm": summary["adjacent_translation_step_mm"]["p95"],
        "rotation_step_p95_deg": summary["adjacent_rotation_step_deg"]["p95"],
    })
stages_df = pd.DataFrame(stage_rows)
stages_df.round(2)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Robustness is cube- and camera-dependent

The separate July 16 multi-camera QA prevents overgeneralizing the single-cube result. The saved `020` sidecar succeeded on all `index_Q` frames, partially on `wrist_Q`, and not at all on `thumb_Q` or `middle_Q`. Later provisional orientation/flow stages are not equivalent to a validated `020` measurement.
"""
    ),
    nbf.v4.new_code_cell(
        """qa = results["pipeline_020_multicamera_qa"]
qa_rows = []
for target, metrics in qa.get("targets", {}).items():
    qa_rows.append({
        "target": target,
        "camera": metrics["camera_name"],
        "output_rate_pct": 100 * metrics["success_rate"],
        "measured_count": metrics["measured_final_count"],
        "reproj_mean_px_within_020": metrics["final_reprojection_mean_px"],
        "edge_alignment_mean": metrics["final_edge_alignment_mean"],
    })
pd.DataFrame(qa_rows).round(3)
"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

1. **Offline finalization:** keep `020`. It has the strongest current evidence because it explicitly combines measurement, rejection, recovery, and constrained smoothing.
2. **Online tracking:** do not use `alg_09`’s reported 100% at face value. A long stale hold can look smooth while being wrong. Cap hold duration and expose measured-vs-predicted state.
3. **Old 1×1×1 setup:** `alg_03` is the best old per-frame measurement candidate by mean reprojection, but it still lacks ground-truth validation.
4. **Current 2×2×2 setup:** reuse the causal part of `020`—especially DeepTag dense single-frame fusion—then add a short causal filter. The full `020` pipeline is non-causal and unsuitable as-is for live control.
5. **Next decisive experiment:** capture a short synchronized sequence with motion-capture or surveyed fiducials, then evaluate translation/rotation error, measured coverage, stale-hold duration, latency, and failure recovery on exactly the same frames.
"""
    ),
]

nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
