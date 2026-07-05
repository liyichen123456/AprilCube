# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 62.5 x 62.5 x 62.5 mm |
| Tag size | 50 mm (8x8 cells) |
| Cell size | 6.25 mm |
| Margin | 1 cell (6.25 mm) |
| Border | 1 cell (6.25 mm) |
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
  "tag_size_mm": 50.0,
  "cell_size_mm": 6.25,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    62.5,
    62.5,
    62.5
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 1x1x1 --dict apriltag_36h11 --tag-size 50 --margin-cell 1 --border-cell 1 -o cube_april_36h11_100_105_1x1x1_50mm
```
