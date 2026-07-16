# Source and chart notes

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
