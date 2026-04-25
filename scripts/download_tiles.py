#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import mercantile
import numpy as np
import requests
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from tqdm import tqdm


def download_xyz_geotiff(
    west: float,
    south: float,
    east: float,
    north: float,
    zoom: int,
    out_path: Path,
    tile_url: str,
) -> None:
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    if not tiles:
        raise RuntimeError("No XYZ tiles found for this bounding box and zoom.")

    min_x = min(tile.x for tile in tiles)
    max_x = max(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    max_y = max(tile.y for tile in tiles)

    width = (max_x - min_x + 1) * 256
    height = (max_y - min_y + 1) * 256
    canvas = Image.new("RGB", (width, height))

    for tile in tqdm(tiles, desc="Downloading XYZ tiles"):
        url = tile_url.format(x=tile.x, y=tile.y, z=tile.z)
        response = requests.get(url, timeout=20, headers={"User-Agent": "sat2geo-aarhus/0.1"})
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        px = (tile.x - min_x) * 256
        py = (tile.y - min_y) * 256
        canvas.paste(image, (px, py))

    top_left = mercantile.xy_bounds(min_x, min_y, zoom)
    bottom_right = mercantile.xy_bounds(max_x, max_y, zoom)
    transform = from_bounds(top_left.left, bottom_right.bottom, bottom_right.right, top_left.top, width, height)

    arr = np.transpose(np.array(canvas), (2, 0, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=arr.dtype,
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(arr)

    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XYZ satellite tiles and mosaic them into a GeoTIFF.")
    parser.add_argument("--west", type=float, default=10.1905)
    parser.add_argument("--south", type=float, default=56.1402)
    parser.add_argument("--east", type=float, default=10.2194)
    parser.add_argument("--north", type=float, default=56.1598)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--out", type=Path, default=Path("data/raw/aarhus_z18.tif"))
    parser.add_argument(
        "--tile-url",
        default="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        help="XYZ tile URL template. Check provider terms before downloading.",
    )
    args = parser.parse_args()
    download_xyz_geotiff(args.west, args.south, args.east, args.north, args.zoom, args.out, args.tile_url)


if __name__ == "__main__":
    main()
