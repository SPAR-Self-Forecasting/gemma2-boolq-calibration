#!/usr/bin/env bash
# run_training_no_passage.sh — Train calibrated QA on BoolQ WITHOUT passage (Gemma 2 2B IT)
set -euo pipefail

PRIME_RL_DIR="/root/prime-rl"
OUTPUT_DIR="/root/outputs/calibrated-qa-gemma2-2b-no-passage"
CONFIG_SRC="/root/calibrated_qa_gemma2_2b_no_passage.toml"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN with access to google/gemma-2-2b-it}"

for arg in "$@"; do
    case "$arg" in
        --wandb-key=*) export WANDB_API_KEY="${arg#*=}" ;;
    esac
done

cd "$PRIME_RL_DIR"

echo "=== [1/2] Installing calibrated-qa-no-passage environment ==="
rm -rf .venv/lib/python3.12/site-packages/calibrated_qa_no_passage*
rm -rf .venv/lib/python3.12/site-packages/__pycache__/calibrated_qa_no_passage*
uv pip install /root/calibrated_qa_no_passage/
uv run python -c "import calibrated_qa_no_passage; print('calibrated-qa-no-passage environment OK')"

echo ""
echo "=== [2/2] RL training on calibrated-qa-no-passage (Gemma 2 2B IT) ==="

uv run rl @ "$CONFIG_SRC" \
    --output-dir "$OUTPUT_DIR/rl" \
    --wandb.offline true \
    --wandb.shared false

echo "Training complete."
echo "Generations:    $OUTPUT_DIR/rl/generations.jsonl"
echo "Weights:        $OUTPUT_DIR/rl/weights/"
