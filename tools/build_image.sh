#!/usr/bin/env bash
# Build the image, export it as a gzip tarball and print its SHA-256 (needed for the Solution Submission form)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-datamate:final}
BASE=${BASE:-vllm/vllm-openai:v0.10.1}
docker build --build-arg BASE="$BASE" -f docker/Dockerfile -t "$TAG" .
docker save "$TAG" | gzip -1 > datamate_final.tar.gz
sha256sum datamate_final.tar.gz | tee datamate_final.sha256
ls -lh datamate_final.tar.gz
