#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def choose_factory(n_vectors: int) -> str:
    if n_vectors < 2_000:
        return "Flat"
    if n_vectors < 50_000:
        return "IVF256,Flat"
    return "OPQ16,IVF4096,PQ16"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Faiss index from chip embeddings.")
    parser.add_argument("--embeddings-path", type=Path, default=Path("data/embeddings/chip_embeddings.npy"))
    parser.add_argument("--ids-path", type=Path, default=Path("data/embeddings/chip_ids.npy"))
    parser.add_argument("--index-path", type=Path, default=Path("data/index/chips_faiss.index"))
    parser.add_argument("--factory", default=None, help="Faiss factory string. Defaults by dataset size.")
    parser.add_argument("--train-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    embeddings = l2_normalize(np.load(args.embeddings_path).astype(np.float32))
    ids = np.load(args.ids_path).astype(np.int64)
    if len(embeddings) != len(ids):
        raise ValueError("Embeddings and IDs length mismatch.")

    n, dim = embeddings.shape
    factory = args.factory or choose_factory(n)
    print(f"Building Faiss index: vectors={n}, dim={dim}, factory={factory}")

    base_index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    index = faiss.IndexIDMap2(base_index) if factory == "Flat" else base_index
    train_index = base_index
    if not train_index.is_trained:
        rng = np.random.default_rng(args.seed)
        sample_n = min(n, args.train_samples)
        train_x = embeddings[rng.choice(n, size=sample_n, replace=False)]
        train_index.train(train_x)
    index.add_with_ids(embeddings, ids)

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_path))
    print(f"Saved {args.index_path}")


if __name__ == "__main__":
    main()
