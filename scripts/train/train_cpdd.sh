#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${RESUME:?Set RESUME to the CPDD step-38000 or later restart checkpoint}"
: "${DATASET_DIR:?Set DATASET_DIR to the accepted CPDD dataset directory}"
: "${EXCLUDE_GENOMES_FILE:?Set EXCLUDE_GENOMES_FILE to the frozen training exclusion list}"
: "${GPUS:?Set GPUS, for example GPUS=0,1,2,3,4,5}"
NPROC="${NPROC:-6}"
OUT="${OUT:-runs/cpdd_final_anneal}"
MASTER_PORT="${MASTER_PORT:-29500}"
mkdir -p "$OUT"

args=(
  --resume "$RESUME" --output "$OUT" --dataset-dir "$DATASET_DIR" --repo-root "$ROOT"
  --exclude-genomes-file "$EXCLUDE_GENOMES_FILE"
  --stage B --peak 5e-5 --warmup 500 --clip 5.0 --aa-weight 1.0
  --supervision-budget 7000 --mask-fraction 0.25
  --long-probability 1.0 --substitution-max 0.3
  --ckpt-free 8 --ckpt-free-long 8 --max-steps 46000
  --switch-step 38000 --rewarmup-steps 300
  --decay-start 40000 --decay-steps 6000 --keep-steps 40000,46000
  --val-every 500 --long-val-every 2500 --diagnostic-steps 50
  --checkpoint-every 250 --expect-world "$NPROC"
  --metrics "$OUT/metrics.jsonl" --stop-file "$OUT/STOP"
)
if [[ -n "${VAL_PANEL:-}" ]]; then
  args+=(--val-panel "$VAL_PANEL")
fi

CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="$ROOT" \
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  -m experiments.phadf.restart "${args[@]}"
