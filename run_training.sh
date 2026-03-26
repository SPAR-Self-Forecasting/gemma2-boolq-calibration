#!/usr/bin/env bash
# run_training.sh — Train calibrated QA on BoolQ with Gemma 2 2B IT
#
# Model: google/gemma-2-2b-it (gated — needs HF_TOKEN)
# Requires 2 GPUs: GPU 0 for inference, GPU 1 for training.
#
# Usage:
#   bash run_training.sh [--wandb-key=YOUR_KEY]
set -euo pipefail

PRIME_RL_DIR="/root/prime-rl"
OUTPUT_DIR="/root/outputs/calibrated-qa-gemma2-2b"
CONFIG_SRC="/root/calibrated_qa_gemma2_2b.toml"

# REQUIRED: Gemma 2 is gated, set your HuggingFace token
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN with access to google/gemma-2-2b-it}"

for arg in "$@"; do
    case "$arg" in
        --wandb-key=*) export WANDB_API_KEY="${arg#*=}" ;;
    esac
done

cd "$PRIME_RL_DIR"

# ---------- Step 1: Install calibrated-qa environment ----------
echo "=== [1/2] Installing calibrated-qa environment ==="
# CRITICAL: Remove stale files before install. Hatch build caching causes
# the old calibrated_qa.py flat file to persist alongside the package.
# Python imports the flat file first, ignoring the updated package version.
rm -rf .venv/lib/python3.12/site-packages/calibrated_qa*
rm -rf .venv/lib/python3.12/site-packages/__pycache__/calibrated_qa*
uv pip install /root/calibrated_qa/
uv run python -c "import calibrated_qa; print('calibrated-qa environment OK')"

# Verify the correct version is installed (should have INSTRUCTIONS, not SYSTEM_PROMPT)
if uv run python -c "import calibrated_qa; assert hasattr(calibrated_qa, 'INSTRUCTIONS'), 'Wrong version installed!'" 2>/dev/null; then
    echo "Correct calibrated-qa version verified."
else
    echo "ERROR: Wrong calibrated_qa version! The stale flat file may still be present."
    echo "Run: rm -rf .venv/lib/python3.12/site-packages/calibrated_qa*"
    exit 1
fi

# ---------- Step 2: RL training (2 GPUs — standard entrypoint) ----------
echo ""
echo "=== [2/2] RL training on calibrated-qa (Gemma 2 2B IT) ==="

uv run rl @ "$CONFIG_SRC" \
    --output-dir "$OUTPUT_DIR/rl" \
    --wandb.offline true \
    --wandb.shared false

echo "Training complete."
echo "Generations:    $OUTPUT_DIR/rl/generations.jsonl"
echo "Weights:        $OUTPUT_DIR/rl/weights/"
