from __future__ import annotations

import argparse
from pathlib import Path

from .locator import DEFAULT_MODEL, locate_image


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Locate an Aarhus satellite/map crop by searching the bundled Sat2Geo vector index."
    )
    parser.add_argument("query_image", help="Path to the satellite/aerial image crop to locate.")
    parser.add_argument("--db-path", default=root / "data" / "index" / "chips.sqlite")
    parser.add_argument("--index-path", default=root / "data" / "index" / "chips_faiss.index")
    parser.add_argument("--embeddings-path", default=root / "data" / "embeddings" / "chip_embeddings.npy")
    parser.add_argument("--ids-path", default=root / "data" / "embeddings" / "chip_ids.npy")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-k", type=int, default=500)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--no-exact", action="store_true", help="Use Faiss retrieval only instead of exact full-vector search.")
    parser.add_argument("--no-geometric", action="store_true", help="Disable OpenCV geometric verification/refinement.")
    parser.add_argument("--geometric-rerank-k", type=int, default=120)
    parser.add_argument("--min-inliers", type=int, default=14)
    parser.add_argument("--template-threshold", type=float, default=0.72)
    parser.add_argument("--source-crs", default="EPSG:3857")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omitted for auto.")
    args = parser.parse_args()

    result = locate_image(
        query_image=Path(args.query_image),
        db_path=Path(args.db_path),
        index_path=Path(args.index_path),
        embeddings_path=Path(args.embeddings_path),
        ids_path=Path(args.ids_path),
        model_name=args.model_name,
        top_k=args.top_k,
        search_k=args.search_k,
        nprobe=args.nprobe,
        exact=not args.no_exact,
        geometric_refine=not args.no_geometric,
        geometric_rerank_k=args.geometric_rerank_k,
        min_inliers=args.min_inliers,
        template_threshold=args.template_threshold,
        source_crs=args.source_crs,
        device=args.device,
    )

    if not result.matches:
        print("No matches found.")
        return

    print("\nTop matches:")
    print(f"Confidence: {result.confidence}")
    for match in result.matches:
        print(
            f"{match.rank:02d}. coords=\"{match.lat:.7f}, {match.lon:.7f}\" "
            f"score={match.score:.6f} inliers={match.inliers} "
            f"method={match.method} chip_id={match.chip_id} "
            f"source={Path(match.source_file).name} x_off={match.x_off} y_off={match.y_off}"
        )

    best = result.matches[0]
    print("\nBest coordinate:")
    print(f'coords="{best.lat:.7f}, {best.lon:.7f}"')
    print(best.google_maps_url)
    if result.confidence == "low":
        print("\nWarning: low confidence. Use a larger crop, remove labels/UI overlays, or build a matching hybrid/labeled index before treating this as precise.")

    if result.fused_lat is not None and result.fused_lon is not None:
        print("\nFused coordinate:")
        print(f'coords="{result.fused_lat:.7f}, {result.fused_lon:.7f}"')
        print(f"https://www.google.com/maps?q={result.fused_lat:.7f},{result.fused_lon:.7f}")
        if result.fused_radius_m is not None:
            print(f"match_spread_m={result.fused_radius_m:.1f}")


if __name__ == "__main__":
    main()
