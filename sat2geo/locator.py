from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from pyproj import Transformer
from transformers import AutoImageProcessor, AutoModel


DEFAULT_MODEL = "facebook/dinov2-base"


@dataclass(frozen=True)
class Match:
    rank: int
    chip_id: int
    score: float
    lat: float
    lon: float
    chip_path: str
    source_file: str
    x_off: int
    y_off: int
    inliers: int = 0
    refined: bool = False
    geometric_score: float = 0.0
    method: str = "embedding"

    @property
    def google_maps_url(self) -> str:
        return f"https://www.google.com/maps?q={self.lat:.7f},{self.lon:.7f}"


@dataclass(frozen=True)
class LocateResult:
    matches: list[Match]
    fused_lat: float | None
    fused_lon: float | None
    fused_radius_m: float | None
    used_geometric_verification: bool = False
    confidence: str = "low"


@dataclass(frozen=True)
class PixelGeoTransform:
    x_slope: float
    x_intercept: float
    y_slope: float
    y_intercept: float
    transformer: Transformer

    def pixel_to_wgs84(self, col: float, row: float) -> tuple[float, float]:
        x_src = self.x_slope * col + self.x_intercept
        y_src = self.y_slope * row + self.y_intercept
        lon, lat = self.transformer.transform(x_src, y_src)
        return float(lat), float(lon)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def load_chip_metadata(db_path: Path, chip_ids: np.ndarray) -> list[tuple | None]:
    if len(chip_ids) == 0:
        return []

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in chip_ids)
    cur.execute(
        f"""
        SELECT chip_id, chip_path, lat, lon, source_file, x_off, y_off
        , width, height, center_x_src, center_y_src
        FROM chips
        WHERE chip_id IN ({placeholders})
        """,
        tuple(int(x) for x in chip_ids),
    )
    rows = cur.fetchall()
    conn.close()

    by_id = {int(row[0]): row for row in rows}
    return [by_id.get(int(chip_id)) for chip_id in chip_ids]


def resolve_chip_path(chip_path: str, db_path: Path) -> Path:
    path = Path(chip_path)
    if path.exists():
        return path
    repo_root = db_path.resolve().parents[2]
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return path


def load_pixel_geo_transform(db_path: Path, source_crs: str) -> PixelGeoTransform:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT x_off, y_off, width, height, center_x_src, center_y_src FROM chips")
    rows = cur.fetchall()
    conn.close()
    if len(rows) < 2:
        raise RuntimeError("Need at least two chips to estimate pixel-to-coordinate transform.")

    arr = np.array(rows, dtype=np.float64)
    center_cols = arr[:, 0] + arr[:, 2] / 2.0
    center_rows = arr[:, 1] + arr[:, 3] / 2.0
    center_x_src = arr[:, 4]
    center_y_src = arr[:, 5]

    x_slope, x_intercept = np.polyfit(center_cols, center_x_src, deg=1)
    y_slope, y_intercept = np.polyfit(center_rows, center_y_src, deg=1)
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    return PixelGeoTransform(
        x_slope=float(x_slope),
        x_intercept=float(x_intercept),
        y_slope=float(y_slope),
        y_intercept=float(y_intercept),
        transformer=transformer,
    )


def embed_image(query_image: Path, model_name: str, device: str) -> np.ndarray:
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    img = Image.open(query_image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)

    return l2_normalize(emb)


def exact_rerank(
    query_embedding: np.ndarray,
    candidate_ids: np.ndarray,
    embeddings_path: Path,
    ids_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    all_emb = l2_normalize(np.load(embeddings_path).astype(np.float32))
    all_ids = np.load(ids_path).astype(np.int64)
    id_to_idx = {int(chip_id): idx for idx, chip_id in enumerate(all_ids)}

    keep: list[tuple[int, int]] = []
    for chip_id in candidate_ids:
        idx = id_to_idx.get(int(chip_id))
        if idx is not None:
            keep.append((int(chip_id), idx))

    if not keep:
        return candidate_ids, np.zeros(len(candidate_ids), dtype=np.float32)

    rerank_ids = np.array([item[0] for item in keep], dtype=np.int64)
    rerank_idx = np.array([item[1] for item in keep], dtype=np.int64)
    scores = all_emb[rerank_idx] @ query_embedding[0]
    order = np.argsort(-scores)
    return rerank_ids[order], scores[order].astype(np.float32)


def exact_search_all(
    query_embedding: np.ndarray,
    embeddings_path: Path,
    ids_path: Path,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_emb = l2_normalize(np.load(embeddings_path).astype(np.float32))
    all_ids = np.load(ids_path).astype(np.int64)
    scores = all_emb @ query_embedding[0]
    top_n = min(max(1, top_n), len(all_ids))
    top_idx = np.argpartition(-scores, top_n - 1)[:top_n]
    order = top_idx[np.argsort(-scores[top_idx])]
    return all_ids[order].astype(np.int64), scores[order].astype(np.float32)


def create_feature_detector():
    if hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    return "orb", cv2.ORB_create(nfeatures=5000)


def compute_query_features(query_image: Path):
    img = Image.open(query_image).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    detector_name, detector = create_feature_detector()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return gray.shape, detector_name, detector, keypoints, descriptors


def geometric_verify(
    query_keypoints,
    query_descriptors,
    query_shape: tuple[int, int],
    detector_name: str,
    detector,
    chip_path: Path,
) -> tuple[int, tuple[float, float] | None, float]:
    if query_descriptors is None or len(query_keypoints) < 4:
        return 0, None, 0.0

    chip_gray = cv2.imread(str(chip_path), cv2.IMREAD_GRAYSCALE)
    if chip_gray is None:
        return 0, None, 0.0
    chip_gray = cv2.equalizeHist(chip_gray)
    chip_keypoints, chip_descriptors = detector.detectAndCompute(chip_gray, None)
    if chip_descriptors is None or len(chip_keypoints) < 4:
        return 0, None, 0.0

    if detector_name == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        ratio = 0.72
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        ratio = 0.78

    try:
        pairs = matcher.knnMatch(query_descriptors, chip_descriptors, k=2)
    except cv2.error:
        return 0, None, 0.0

    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio * second.distance:
            good.append(first)

    if len(good) < 4:
        return 0, None, 0.0

    src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([chip_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    except cv2.error:
        return 0, None, 0.0
    if homography is None or mask is None:
        return 0, None, 0.0

    inliers = int(np.sum(mask))
    if inliers < 4:
        return inliers, None, float(inliers)

    q_h, q_w = query_shape
    query_center = np.array([[[0.5 * (q_w - 1), 0.5 * (q_h - 1)]]], dtype=np.float32)
    try:
        chip_center = cv2.perspectiveTransform(query_center, homography)[0][0]
    except cv2.error:
        return inliers, None, float(inliers)

    cx = float(chip_center[0])
    cy = float(chip_center[1])
    if not np.isfinite(cx) or not np.isfinite(cy):
        return inliers, None, float(inliers)

    c_h, c_w = chip_gray.shape
    valid_center = -0.05 * c_w <= cx <= 1.05 * c_w and -0.05 * c_h <= cy <= 1.05 * c_h
    inlier_ratio = inliers / max(1, len(good))
    score = float(inliers * (0.5 + inlier_ratio))
    return inliers, (cx, cy) if valid_center else None, score


def template_verify(query_image: Path, chip_path: Path) -> tuple[float, tuple[float, float] | None]:
    query_gray = cv2.imread(str(query_image), cv2.IMREAD_GRAYSCALE)
    chip_gray = cv2.imread(str(chip_path), cv2.IMREAD_GRAYSCALE)
    if query_gray is None or chip_gray is None:
        return 0.0, None

    q_h, q_w = query_gray.shape
    c_h, c_w = chip_gray.shape
    if q_h > c_h or q_w > c_w:
        return 0.0, None

    query_gray = cv2.equalizeHist(query_gray)
    chip_gray = cv2.equalizeHist(chip_gray)
    result = cv2.matchTemplate(chip_gray, query_gray, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    cx = float(max_location[0] + q_w / 2.0)
    cy = float(max_location[1] + q_h / 2.0)
    return float(max_value), (cx, cy)


def locate_image(
    query_image: Path,
    db_path: Path,
    index_path: Path,
    embeddings_path: Path | None = None,
    ids_path: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    search_k: int = 100,
    nprobe: int = 32,
    exact: bool = True,
    geometric_refine: bool = True,
    geometric_rerank_k: int = 120,
    min_inliers: int = 14,
    template_threshold: float = 0.72,
    source_crs: str = "EPSG:3857",
    fuse_top_k: int = 5,
    fuse_max_radius_m: float = 250.0,
    fuse_min_score_ratio: float = 0.96,
    device: str | None = None,
) -> LocateResult:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    query_embedding = embed_image(query_image, model_name, device)

    candidate_count = max(top_k, search_k, geometric_rerank_k if geometric_refine else 0)
    if exact and embeddings_path and ids_path and embeddings_path.exists() and ids_path.exists():
        candidate_ids, candidate_scores = exact_search_all(query_embedding, embeddings_path, ids_path, candidate_count)
    else:
        index = faiss.read_index(str(index_path))
        if hasattr(index, "nprobe"):
            index.nprobe = nprobe

        distances, labels = index.search(query_embedding, candidate_count)
        candidate_ids = labels[0]
        valid = candidate_ids >= 0
        candidate_ids = candidate_ids[valid].astype(np.int64)
        candidate_scores = distances[0][valid].astype(np.float32)

        if embeddings_path and ids_path and embeddings_path.exists() and ids_path.exists():
            candidate_ids, candidate_scores = exact_rerank(query_embedding, candidate_ids, embeddings_path, ids_path)

    rerank_n = min(len(candidate_ids), max(top_k, geometric_rerank_k if geometric_refine else top_k))
    ranked_ids = candidate_ids[:rerank_n]
    ranked_scores = candidate_scores[:rerank_n]
    rows = load_chip_metadata(db_path, ranked_ids)

    matches: list[Match] = []
    pixel_geo: PixelGeoTransform | None = None
    query_features = None
    if geometric_refine:
        try:
            pixel_geo = load_pixel_geo_transform(db_path, source_crs)
            query_features = compute_query_features(query_image)
        except Exception:
            pixel_geo = None
            query_features = None

    for rank, (chip_id, score, row) in enumerate(zip(ranked_ids, ranked_scores, rows), start=1):
        if row is None:
            continue
        _, chip_path, lat, lon, source_file, x_off, y_off, _, _, _, _ = row
        refined = False
        inliers = 0
        geometric_score = 0.0
        method = "embedding"
        out_lat = float(lat)
        out_lon = float(lon)
        if geometric_refine and pixel_geo is not None and query_features is not None:
            resolved_chip_path = resolve_chip_path(str(chip_path), db_path)
            template_score, chip_xy = template_verify(query_image, resolved_chip_path)
            if chip_xy is not None and template_score >= template_threshold:
                chip_x, chip_y = chip_xy
                out_lat, out_lon = pixel_geo.pixel_to_wgs84(float(x_off) + chip_x, float(y_off) + chip_y)
                refined = True
                geometric_score = template_score * 100.0
                method = "template"

            query_shape, detector_name, detector, query_keypoints, query_descriptors = query_features
            sift_inliers, sift_xy, sift_score = geometric_verify(
                query_keypoints,
                query_descriptors,
                query_shape,
                detector_name,
                detector,
                resolved_chip_path,
            )
            inliers = sift_inliers
            if sift_xy is not None and sift_inliers >= min_inliers and sift_score > geometric_score:
                chip_x, chip_y = sift_xy
                out_lat, out_lon = pixel_geo.pixel_to_wgs84(float(x_off) + chip_x, float(y_off) + chip_y)
                refined = True
                geometric_score = sift_score
                method = "geometry"

        matches.append(
            Match(
                rank=rank,
                chip_id=int(chip_id),
                score=float(score),
                lat=out_lat,
                lon=out_lon,
                chip_path=str(chip_path),
                source_file=str(source_file),
                x_off=int(x_off),
                y_off=int(y_off),
                inliers=int(inliers),
                refined=refined,
                geometric_score=geometric_score,
                method=method,
            )
        )

    has_good_geometry = any(match.refined and match.inliers >= min_inliers for match in matches)
    if has_good_geometry:
        matches.sort(
            key=lambda match: (
                match.refined,
                match.geometric_score,
                match.inliers,
                match.score,
            ),
            reverse=True,
        )
    else:
        matches.sort(key=lambda match: match.score, reverse=True)

    matches = [
        Match(
            rank=rank,
            chip_id=match.chip_id,
            score=match.score,
            lat=match.lat,
            lon=match.lon,
            chip_path=match.chip_path,
            source_file=match.source_file,
            x_off=match.x_off,
            y_off=match.y_off,
            inliers=match.inliers,
            refined=match.refined,
            geometric_score=match.geometric_score,
            method=match.method,
        )
        for rank, match in enumerate(matches[:top_k], start=1)
    ]

    fused_lat: float | None = None
    fused_lon: float | None = None
    fused_radius_m: float | None = None
    if matches:
        best = matches[0]
        fuse_candidates = [match for match in matches if match.refined] if has_good_geometry else matches
        nearby = [
            match
            for match in fuse_candidates[: max(1, fuse_top_k)]
            if match.score >= best.score * fuse_min_score_ratio
            and haversine_m(best.lat, best.lon, match.lat, match.lon) <= fuse_max_radius_m
        ]
        if nearby:
            weights = np.array(
                [
                    max(0.0, match.score) * max(1.0, match.geometric_score if match.refined else 1.0)
                    for match in nearby
                ],
                dtype=np.float64,
            )
            if np.any(weights > 0):
                lats = np.array([match.lat for match in nearby], dtype=np.float64)
                lons = np.array([match.lon for match in nearby], dtype=np.float64)
                fused_lat = float(np.average(lats, weights=weights))
                fused_lon = float(np.average(lons, weights=weights))
                fused_radius_m = max(haversine_m(fused_lat, fused_lon, m.lat, m.lon) for m in nearby)

    confidence = "low"
    if matches:
        if any(match.method == "template" for match in matches[:3]):
            confidence = "high"
        elif any(match.method == "geometry" and match.inliers >= 25 for match in matches[:3]):
            confidence = "high"
        elif any(match.method == "geometry" and match.inliers >= min_inliers for match in matches[:3]):
            confidence = "medium"
        elif len(matches) == 1:
            confidence = "medium" if matches[0].score >= 0.82 else "low"
        else:
            score_gap = matches[0].score - matches[1].score
            if matches[0].score >= 0.84 and score_gap >= 0.03:
                confidence = "medium"

    return LocateResult(
        matches=matches,
        fused_lat=fused_lat,
        fused_lon=fused_lon,
        fused_radius_m=fused_radius_m,
        used_geometric_verification=has_good_geometry,
        confidence=confidence,
    )
