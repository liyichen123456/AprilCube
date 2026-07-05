# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 100 x 100 x 100 mm |
| Tag size | 80 mm (8x8 cells) |
| Cell size | 10 mm |
| Margin | 1 cell (10 mm) |
| Border | 1 cell (10 mm) |
| Total tags | 6 |
| Tag IDs | 100–105 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 100 |
| -X | 101 |
| +Y | 102 |
| -Y | 103 |
| +Z | 104 |
| -Z | 105 |

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
    100,
    101,
    102,
    103,
    104,
    105
  ],
  "faces": {
    "+X": [
      100
    ],
    "-X": [
      101
    ],
    "+Y": [
      102
    ],
    "-Y": [
      103
    ],
    "+Z": [
      104
    ],
    "-Z": [
      105
    ]
  },
  "tag_size_mm": 80.0,
  "cell_size_mm": 10.0,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    100.0,
    100.0,
    100.0
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 1x1x1 --dict apriltag_36h11 --tag-size 80 --margin-cell 1 --border-cell 1 -o cube_april_36h11_100_105_1x1x1_80mm
```
