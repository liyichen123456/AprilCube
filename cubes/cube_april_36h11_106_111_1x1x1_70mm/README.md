# ArUco Cube — 1x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `apriltag_36h11` |
| Grid | 1x1x1 (X x Y x Z tags) |
| Box dimensions | 87.5 x 87.5 x 87.5 mm |
| Tag size | 70 mm (8x8 cells) |
| Cell size | 8.75 mm |
| Margin | 1 cell (8.75 mm) |
| Border | 1 cell (8.75 mm) |
| Total tags | 6 |
| Tag IDs | 106–111 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 106 |
| -X | 107 |
| +Y | 108 |
| -Y | 109 |
| +Z | 110 |
| -Z | 111 |

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
  "schema_version": 1,
  "target": {
    "type": "cuboid",
    "grid": "1x1x1"
  },
  "dict": "apriltag_36h11",
  "grid": "1x1x1",
  "tag_ids": [
    106,
    107,
    108,
    109,
    110,
    111
  ],
  "faces": {
    "+X": [
      106
    ],
    "-X": [
      107
    ],
    "+Y": [
      108
    ],
    "-Y": [
      109
    ],
    "+Z": [
      110
    ],
    "-Z": [
      111
    ]
  },
  "tag_size_mm": 70.0,
  "cell_size_mm": 8.75,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 8,
  "box_dims": [
    87.5,
    87.5,
    87.5
  ]
}
```

## Regenerate

```bash
aprilcube generate --grid 1x1x1 --dict apriltag_36h11 --tag-size 70 --margin-cell 1 --border-cell 1 -o cube_april_36h11_106_111_1x1x1_70mm
```
