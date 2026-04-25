#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window
from tqdm import tqdm


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chips (
            chip_id INTEGER PRIMARY KEY,
            source_file TEXT NOT NULL,
            chip_path TEXT NOT NULL UNIQUE,
            x_off INTEGER NOT NULL,
            y_off INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            center_x_src REAL NOT NULL,
            center_y_src REAL NOT NULL,
            lon REAL NOT NULL,
            lat REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_chip(img_array: np.ndarray, out_path: Path) -> None:
    rgb = np.transpose(img_array[:3], (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_path, quality=95)


def process_geotiff(
    tif_path: Path,
    chips_dir: Path,
    db_path: Path,
    chip_size: int,
    stride: int,
    skip_blank_threshold: float,
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    with rasterio.open(tif_path) as src:
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        x_positions = list(range(0, max(1, src.width - chip_size + 1), stride))
        y_positions = list(range(0, max(1, src.height - chip_size + 1), stride))

        for y_off in tqdm(y_positions, desc=f"Rows in {tif_path.name}"):
            for x_off in x_positions:
                arr = src.read(window=Window(x_off, y_off, chip_size, chip_size), boundless=False)
                if arr.shape[1] != chip_size or arr.shape[2] != chip_size:
                    continue
                if float(np.std(arr[:3])) < skip_blank_threshold:
                    continue

                center_col = x_off + chip_size / 2.0
                center_row = y_off + chip_size / 2.0
                center_x_src, center_y_src = src.xy(center_row, center_col)
                lon, lat = transformer.transform(center_x_src, center_y_src)

                chip_path = chips_dir / f"{tif_path.stem}_x{x_off}_y{y_off}.jpg"
                save_chip(arr, chip_path)
                cur.execute(
                    """
                    INSERT OR IGNORE INTO chips (
                        source_file, chip_path, x_off, y_off, width, height,
                        center_x_src, center_y_src, lon, lat
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(tif_path),
                        str(chip_path),
                        x_off,
                        y_off,
                        chip_size,
                        chip_size,
                        center_x_src,
                        center_y_src,
                        lon,
                        lat,
                    ),
                )

    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut GeoTIFF satellite mosaics into searchable chips.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--chips-dir", type=Path, default=Path("data/chips"))
    parser.add_argument("--db-path", type=Path, default=Path("data/index/chips.sqlite"))
    parser.add_argument("--chip-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--skip-blank-threshold", type=float, default=5.0)
    args = parser.parse_args()

    args.chips_dir.mkdir(parents=True, exist_ok=True)
    init_db(args.db_path)
    stride = args.chip_size - args.overlap
    tif_files = sorted(list(args.input_dir.glob("*.tif")) + list(args.input_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"No GeoTIFF files found in {args.input_dir}")

    for tif_path in tif_files:
        process_geotiff(tif_path, args.chips_dir, args.db_path, args.chip_size, stride, args.skip_blank_threshold)

    print(f"Saved chip metadata to {args.db_path}")


if __name__ == "__main__":
    main()
