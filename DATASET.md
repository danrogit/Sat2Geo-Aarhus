# Bundled Aarhus Dataset

This repository currently contains an Aarhus-only zoom-20 demo database.

```text
region: Aarhus, Denmark
zoom: 20
chip_size: 256 px
overlap: 64 px
stride: 192 px
chips/vectors: 15,618
embedding_model: facebook/dinov2-base
embedding_dim: 768
index: Faiss IVF256,Flat
lat_min: 56.1402891
lat_max: 56.1597954
lon_min: 10.1903350
lon_max: 10.2194316
```

Approximate local size:

```text
data/chips/       309.5 MB
data/embeddings/   45.9 MB
data/index/        49.2 MB
```

The raw zoom-20 GeoTIFF used to build the dataset is intentionally not included. It is an intermediate artifact and was about 1.6 GB.

Before publishing the bundled data publicly, confirm that the imagery license allows redistribution of derived chips, embeddings, and indexes. If not, publish the code only and ask users to build their own local database.
