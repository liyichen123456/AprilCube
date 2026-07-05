# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 12.5 x 12.5 x 12.5 mm |
| Tag size | 10 mm (8x8 cells) |
| Cell size | 1.25 mm |
| Margin | 1 cell (1.25 mm) |
| Border | 1 cell (1.25 mm) |
| Total tags | 6 |
| Tag IDs | 48–53 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 48 |
| -X | 49 |
| +Y | 50 |
| -Y | 51 |
| +Z | 52 |
| -Z | 53 |

## Files

| File | Description |
|------|-------------|
| `cube.3mf` | Multi-color 3MF for Bambu Studio |
| `config.json` | Detector config (used by `detect_cube.py`) |
| `thumbnail.png` | 6-view preview |
| `mujoco/cube.xml` | MuJoCo MJCF model |
| `mujoco/cube.obj` | Wavefront OBJ mesh (UV-mapped) |
| `mujoco/cube.mtl` | OBJ material file |
| `mujoco/cube_atlas.png` | Texture atlas |

## Config JSON

```json
{
  "dict": "apriltag_36h11",
  "grid": "1x1x1",
  "tag_ids": [
    48,
    49,
    50,
    51,
    52,
    53
  ],
  "faces": {
    "+X": [
      48
    ],
    "-X": [
      49
    ],
    "+Y": [
      50
    ],
    "-Y": [
      51
    ],
    "+Z": [
      52
    ],
    "-Z": [
      53
    ]
  },
  "tag_size_mm": 10.0,
  "cell_size_mm": 1.25,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    12.5,
    12.5,
    12.5
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 1x1x1 --dict apriltag_36h11 --tag-size 10 --margin-cell 1 --border-cell 1 -o cube_april_36h11_48_53_1x1x1_10mm
```
