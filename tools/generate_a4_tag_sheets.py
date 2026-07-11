#!/usr/bin/env python3
"""Generate A4 print sheets for existing AprilCube marker textures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FACE_DEFS = [
    ("+X", 0, +1, 1, -1, 2, -1),
    ("-X", 0, -1, 1, +1, 2, -1),
    ("+Y", 1, +1, 0, +1, 2, -1),
    ("-Y", 1, -1, 0, -1, 2, -1),
    ("+Z", 2, +1, 0, +1, 1, +1),
    ("-Z", 2, -1, 0, +1, 1, -1),
]

ATLAS_LAYOUT = [
    ("+X", "-X", "+Y"),
    ("-Y", "+Z", "-Z"),
]

A4_MM = (210.0, 297.0)


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def load_font(size_px: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size_px)
    return ImageFont.load_default()


def face_layout_cells(config: dict, face_def: tuple) -> tuple[int, int]:
    _name, _nax, _ns, right_ax, _rs, down_ax, _ds = face_def
    grid = [int(v) for v in config["grid"].split("x")]
    marker_pixels = int(config["marker_pixels"])
    margin_cells = int(config["margin_cells"])
    border_cells = int(config["border_cells"])

    def axis_cells(n_tags: int) -> int:
        return 2 * border_cells + n_tags * marker_pixels + max(0, n_tags - 1) * margin_cells

    cells = [axis_cells(n) for n in grid]
    return cells[down_ax], cells[right_ax]


def infer_atlas_regions(config: dict, atlas: Image.Image) -> dict[str, tuple[int, int, int, int]]:
    cell_sizes: dict[str, tuple[int, int]] = {}
    for face_def in FACE_DEFS:
        name = face_def[0]
        cell_sizes[name] = face_layout_cells(config, face_def)

    col_width_cells = []
    for col_idx in range(3):
        col_width_cells.append(max(cell_sizes[name][1] for name in (row[col_idx] for row in ATLAS_LAYOUT)))
    row_height_cells = []
    for row in ATLAS_LAYOUT:
        row_height_cells.append(max(cell_sizes[name][0] for name in row))

    total_w_cells = sum(col_width_cells)
    total_h_cells = sum(row_height_cells)
    px_per_cell_w = atlas.width / total_w_cells
    px_per_cell_h = atlas.height / total_h_cells
    if abs(px_per_cell_w - px_per_cell_h) > 1e-6:
        raise ValueError(f"atlas cells are not square: {atlas.width}x{atlas.height}")
    px_per_cell = px_per_cell_w
    if abs(px_per_cell - round(px_per_cell)) > 1e-6:
        raise ValueError(f"cannot infer integer atlas pixels-per-cell: {px_per_cell}")
    ppc = int(round(px_per_cell))

    regions: dict[str, tuple[int, int, int, int]] = {}
    y = 0
    for row_idx, row in enumerate(ATLAS_LAYOUT):
        x = 0
        for col_idx, name in enumerate(row):
            h_cells, w_cells = cell_sizes[name]
            regions[name] = (x, y, w_cells * ppc, h_cells * ppc)
            x += col_width_cells[col_idx] * ppc
        y += row_height_cells[row_idx] * ppc
    return regions


def face_physical_size_mm(config: dict, face_def: tuple) -> tuple[float, float]:
    _name, _nax, _ns, right_ax, _rs, down_ax, _ds = face_def
    box_dims = [float(v) for v in config["box_dims"]]
    return box_dims[right_ax], box_dims[down_ax]


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def format_face_id_label(config: dict, face_def: tuple) -> str:
    face = face_def[0]
    ids = [int(v) for v in config["faces"][face]]
    face_rows, face_cols, _down_cells, _right_cells = face_layout_cells_for_tags(config, face_def)
    if len(ids) == 1:
        return f"ID {ids[0]}   {face}"
    rows = []
    for row in range(face_rows):
        row_ids = ids[row * face_cols:(row + 1) * face_cols]
        rows.append(" ".join(str(v) for v in row_ids))
    return f"IDs {face}: " + " / ".join(rows)


def face_layout_cells_for_tags(config: dict, face_def: tuple) -> tuple[int, int, int, int]:
    _name, _nax, _ns, right_ax, _rs, down_ax, _ds = face_def
    grid = [int(v) for v in config["grid"].split("x")]
    marker_pixels = int(config["marker_pixels"])
    margin_cells = int(config["margin_cells"])
    border_cells = int(config["border_cells"])

    def axis_cells(n_tags: int) -> int:
        return 2 * border_cells + n_tags * marker_pixels + max(0, n_tags - 1) * margin_cells

    cells = [axis_cells(n) for n in grid]
    return grid[down_ax], grid[right_ax], cells[down_ax], cells[right_ax]


def make_sheet(
    cube_dir: Path,
    dpi: int,
    frame_mm: float,
    keep_atlas_mirror: bool,
    margin_mm: float,
) -> Image.Image:
    config_path = cube_dir / "config.json"
    atlas_path = cube_dir / "mujoco" / "cube_atlas.png"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    atlas = Image.open(atlas_path).convert("RGB")
    regions = infer_atlas_regions(config, atlas)
    atlas_is_mirrored = bool(config.get("tag_pattern_mirrored", True))

    page_w = mm_to_px(A4_MM[0], dpi)
    page_h = mm_to_px(A4_MM[1], dpi)
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    title_font = load_font(mm_to_px(4.2, dpi), bold=True)
    meta_font = load_font(mm_to_px(2.8, dpi))
    label_font = load_font(mm_to_px(3.2, dpi), bold=True)
    small_font = load_font(mm_to_px(2.3, dpi))

    margin_x = mm_to_px(margin_mm, dpi)
    title_y = mm_to_px(8.0, dpi)
    title = cube_dir.name
    draw.text((margin_x, title_y), title, fill="black", font=title_font)

    ids = config["tag_ids"]
    meta = (
        f"A4 1:1 print | dict {config['dict']} | tag {float(config['tag_size_mm']):g} mm "
        f"| face/cut size {float(config['box_dims'][0]):g} mm | IDs {ids[0]}-{ids[-1]}"
    )
    draw.text((margin_x, title_y + mm_to_px(6.0, dpi)), meta, fill=(60, 60, 60), font=meta_font)
    draw.text(
        (margin_x, title_y + mm_to_px(10.5, dpi)),
        (
            "Atlas orientation, matching thumbnail/config. "
            if keep_atlas_mirror
            else "Non-mirrored AprilTags. "
        )
        + "Black outer rectangle is a cutting guide outside the white quiet border. Print at 100% scale.",
        fill=(60, 60, 60),
        font=meta_font,
    )

    frame_px = max(1, mm_to_px(frame_mm, dpi))
    content_top = mm_to_px(32.0, dpi)
    content_bottom = mm_to_px(278.0, dpi)
    available_w = page_w - 2 * margin_x
    available_h = content_bottom - content_top
    cols, rows = 3, 2
    slot_w = available_w / cols
    slot_h = available_h / rows

    for idx, face_def in enumerate(FACE_DEFS):
        face = face_def[0]
        col = idx % cols
        row = idx // cols
        slot_x = int(round(margin_x + col * slot_w))
        slot_y = int(round(content_top + row * slot_h))
        slot_w_px = int(round(slot_w))
        slot_h_px = int(round(slot_h))

        right_mm, down_mm = face_physical_size_mm(config, face_def)
        tag_w = mm_to_px(right_mm, dpi)
        tag_h = mm_to_px(down_mm, dpi)
        patch_w = tag_w + 2 * frame_px
        patch_h = tag_h + 2 * frame_px

        x0 = slot_x + (slot_w_px - patch_w) // 2
        y0 = slot_y + mm_to_px(8.0, dpi)
        if y0 + patch_h + mm_to_px(15.0, dpi) > slot_y + slot_h_px:
            y0 = slot_y + (slot_h_px - patch_h - mm_to_px(13.0, dpi)) // 2

        rx, ry, rw, rh = regions[face]
        crop = atlas.crop((rx, ry, rx + rw, ry + rh))
        if atlas_is_mirrored and not keep_atlas_mirror:
            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop = crop.resize((tag_w, tag_h), Image.Resampling.NEAREST)
        page.paste(crop, (x0 + frame_px, y0 + frame_px))
        for i in range(frame_px):
            draw.rectangle(
                (x0 + i, y0 + i, x0 + patch_w - 1 - i, y0 + patch_h - 1 - i),
                outline="black",
            )

        label = format_face_id_label(config, face_def)
        active_label_font = label_font
        tw, th = text_size(draw, label, label_font)
        if tw > patch_w:
            active_label_font = small_font
            tw, th = text_size(draw, label, active_label_font)
        label_x = x0 + patch_w // 2 - tw // 2
        label_y = y0 + patch_h + mm_to_px(2.3, dpi)
        draw.text((label_x, label_y), label, fill="black", font=active_label_font)

        size_label = f"paste area {right_mm:g} x {down_mm:g} mm"
        sw, _ = text_size(draw, size_label, small_font)
        draw.text(
            (x0 + patch_w // 2 - sw // 2, label_y + th + mm_to_px(1.4, dpi)),
            size_label,
            fill=(70, 70, 70),
            font=small_font,
        )

    scale_y = mm_to_px(285.0, dpi)
    scale_x = margin_x
    scale_w = mm_to_px(50.0, dpi)
    draw.line((scale_x, scale_y, scale_x + scale_w, scale_y), fill="black", width=max(1, mm_to_px(0.25, dpi)))
    draw.line((scale_x, scale_y - mm_to_px(1.5, dpi), scale_x, scale_y + mm_to_px(1.5, dpi)), fill="black", width=1)
    draw.line(
        (scale_x + scale_w, scale_y - mm_to_px(1.5, dpi), scale_x + scale_w, scale_y + mm_to_px(1.5, dpi)),
        fill="black",
        width=1,
    )
    draw.text((scale_x + scale_w + mm_to_px(3.0, dpi), scale_y - mm_to_px(2.0, dpi)), "50 mm check", fill="black", font=meta_font)
    return page


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cube_dirs", nargs="+", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--frame-mm", type=float, default=0.4)
    parser.add_argument("--margin-mm", type=float, default=12.0)
    parser.add_argument("--combined-pdf", type=Path)
    parser.add_argument(
        "--keep-atlas-mirror",
        action="store_true",
        help="Keep raw atlas orientation instead of converting mirrored atlases to standard non-mirrored tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages: list[Image.Image] = []
    for cube_dir in args.cube_dirs:
        page = make_sheet(
            cube_dir,
            args.dpi,
            args.frame_mm,
            args.keep_atlas_mirror,
            args.margin_mm,
        )
        png_path = cube_dir / "print_a4_tags.png"
        pdf_path = cube_dir / "print_a4_tags.pdf"
        page.save(png_path, dpi=(args.dpi, args.dpi))
        page.save(pdf_path, "PDF", resolution=args.dpi)
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")
        pages.append(page)

    if args.combined_pdf and pages:
        args.combined_pdf.parent.mkdir(parents=True, exist_ok=True)
        first, rest = pages[0], pages[1:]
        first.save(args.combined_pdf, "PDF", resolution=args.dpi, save_all=True, append_images=rest)
        print(f"Wrote {args.combined_pdf}")


if __name__ == "__main__":
    main()
