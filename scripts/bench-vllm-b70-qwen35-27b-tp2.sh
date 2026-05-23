#!/usr/bin/env bash
# llama-benchy sweep against a running vllm-b70 TP=2 serve of Qwen3.5-27B.
#
# Requires scripts/start-vllm-b70-qwen35-27b-tp2.sh to be serving on port 8000.

set -euo pipefail

OUT="/home/player1/vllm-b70/results/qwen35-27b-tp2-bench-$(date +%Y%m%d-%H%M%S).md"
mkdir -p "$(dirname "$OUT")"

~/.local/bin/llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key dummy \
  --model /home/player1/models/qwen3.5-27b-bf16 \
  --pp 128 512 1024 \
  --tg 64 128 \
  --concurrency 1 4 \
  --runs 2 \
  --enable-prefix-caching \
  --save-result "$OUT" \
  --format md

echo "Results saved to $OUT"
