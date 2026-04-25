#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def load_rows(db_path: Path) -> list[tuple[int, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT chip_id, chip_path FROM chips ORDER BY chip_id ASC")
    rows = cur.fetchall()
    conn.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DINOv2 embeddings for map chips.")
    parser.add_argument("--db-path", type=Path, default=Path("data/index/chips.sqlite"))
    parser.add_argument("--embeddings-path", type=Path, default=Path("data/embeddings/chip_embeddings.npy"))
    parser.add_argument("--ids-path", type=Path, default=Path("data/embeddings/chip_ids.npy"))
    parser.add_argument("--model-name", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_rows(args.db_path)
    if not rows:
        raise RuntimeError("No chips found. Run scripts/chip_tiles.py first.")

    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    all_ids: list[int] = []
    all_embs: list[np.ndarray] = []
    batch_imgs: list[Image.Image] = []
    batch_ids: list[int] = []

    def flush() -> None:
        nonlocal batch_imgs, batch_ids
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
        all_embs.append(emb)
        all_ids.extend(batch_ids)
        batch_imgs = []
        batch_ids = []

    for chip_id, chip_path in tqdm(rows, desc="Embedding chips"):
        batch_imgs.append(Image.open(chip_path).convert("RGB"))
        batch_ids.append(int(chip_id))
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    embeddings = l2_normalize(np.vstack(all_embs).astype(np.float32))
    ids = np.array(all_ids, dtype=np.int64)
    args.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_path, embeddings)
    np.save(args.ids_path, ids)
    print(f"Saved {embeddings.shape} embeddings to {args.embeddings_path}")


if __name__ == "__main__":
    main()
