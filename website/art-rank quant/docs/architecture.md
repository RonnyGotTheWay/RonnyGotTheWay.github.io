# Architecture

ART-Rank Quant is a modular monolith. Parquet snapshots are the immutable research record; PostgreSQL is the serving and audit store. HMM and TFT provide lagged market context, PatchTST provides temporal embeddings, and LambdaRank produces cross-sectional stock relevance. Only comparable ranking outputs may be dynamically ensembled.

Services bind to localhost in the first release. Redis, Prefect, TimescaleDB, object storage, Kubernetes and broker execution are explicitly outside the MVP.

