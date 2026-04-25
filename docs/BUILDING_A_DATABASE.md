# Building a Larger Database

Sat2Geo works by searching a local database of satellite-image chips. The Aarhus demo is intentionally small. To locate images across all of Denmark, France, or another large region, you need to build a larger database first.

## Pipeline

1. Download or prepare georeferenced satellite imagery.
2. Cut it into overlapping chips.
3. Embed every chip with DINOv2.
4. Build a Faiss vector index.
5. Query new satellite/map crops against that index.

## 1. Download a GeoTIFF From XYZ Tiles

The included downloader can mosaic any XYZ tile endpoint into a GeoTIFF:

```powershell
python scripts/download_tiles.py `
  --west 10.1905 --south 56.1402 --east 10.2194 --north 56.1598 `
  --zoom 20 `
  --out data/raw/aarhus_z20.tif
```

For larger regions, split the region into smaller bounding boxes. A full country at zoom 20 can become enormous very quickly.

Check the terms of your imagery provider before downloading or redistributing tiles. Public/open imagery is the best fit for open-source datasets.

## 2. Chip the GeoTIFF

```powershell
python scripts/chip_tiles.py `
  --input-dir data/raw `
  --chips-dir data/chips `
  --db-path data/index/chips.sqlite `
  --chip-size 256 `
  --overlap 64
```

Higher overlap improves recall but creates more chips. The bundled Aarhus zoom-20 demo uses 256 px chips with 64 px overlap and produces 15,618 vectors.

## 3. Embed the Chips

```powershell
python scripts/embed_tiles.py `
  --db-path data/index/chips.sqlite `
  --embeddings-path data/embeddings/chip_embeddings.npy `
  --ids-path data/embeddings/chip_ids.npy `
  --batch-size 32
```

Use a CUDA GPU if available. CPU works for small demos but is slow for country-scale builds.

## 4. Build the Faiss Index

```powershell
python scripts/build_faiss_index.py `
  --embeddings-path data/embeddings/chip_embeddings.npy `
  --ids-path data/embeddings/chip_ids.npy `
  --index-path data/index/chips_faiss.index
```

The script chooses a simple index for small datasets and compressed IVF/PQ-style indexes for larger ones. You can override with `--factory`.

## Scaling Notes

Approximate growth depends on zoom, chip size, overlap, and the amount of imagery. For large areas:

- Store raw GeoTIFFs outside Git.
- Build indexes in tiles or regions, then merge or search multiple indexes.
- Prefer GPU embedding.
- Use `OPQ16,IVF4096,PQ16` or similar Faiss factories for millions of chips.
- Keep metadata in SQLite or PostGIS.
- Use open imagery if you plan to publish the data.

The code is intentionally straightforward so researchers can swap DINOv2 for another image encoder or replace Faiss with a vector database.
