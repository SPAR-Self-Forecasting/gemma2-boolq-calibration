# CLAUDE.md — Running on RunPod

This file contains instructions for Claude to set up and run this experiment on a fresh RunPod instance.

## Requirements

- RunPod instance with **2× A100 80GB** (or 2× H100)
- SSH access to the instance
- A HuggingFace token with access to `google/gemma-2-2b-it` (gated model)

Set these variables before running any commands:

```bash
SSH_HOST=<IP>
SSH_PORT=<PORT>
SSH="ssh root@$SSH_HOST -p $SSH_PORT -i ~/.ssh/id_ed25519"
SCP="scp -P $SSH_PORT -i ~/.ssh/id_ed25519"
HF_TOKEN=<your_hf_token>
```

## Step 1 — Upload files to RunPod

From the repo root:

```bash
$SCP setup_runpod.sh root@$SSH_HOST:/root/
$SCP run_training.sh root@$SSH_HOST:/root/
$SCP configs/calibrated_qa_gemma2_2b.toml root@$SSH_HOST:/root/calibrated_qa_gemma2_2b.toml
$SCP -r calibrated_qa root@$SSH_HOST:/root/calibrated_qa
```

## Step 2 — Install prime-rl (first time only, ~5–10 min)

```bash
$SSH "bash /root/setup_runpod.sh"
```

This installs Python 3.12, uv, clones prime-rl, and runs `uv sync`. It is idempotent — safe to re-run.

## Step 3 — Run training (~85–110 min)

```bash
$SSH "nohup bash -c 'HF_TOKEN=$HF_TOKEN bash /root/run_training.sh' > /root/training.log 2>&1 &"
```

Monitor progress:

```bash
# Trainer log (loss, entropy, grad norm)
$SSH "tail -f /root/training.log"

# Orchestrator log (reward per step — the important one)
$SSH "grep 'SUCCESS.*Reward' /root/outputs/calibrated-qa-gemma2-2b/rl/logs/orchestrator.stdout"
```

Expected reward trajectory: ~0.38 at step 0, rapid climb to ~0.85 by step 6, plateau at 0.85–0.91 for remaining steps. A brief collapse around steps 10–12 (model tries long outputs, hits length penalty) is normal and self-corrects.

## Step 4 — Download results

```bash
mkdir -p results
$SCP root@$SSH_HOST:/root/outputs/calibrated-qa-gemma2-2b/rl/generations.jsonl results/
$SCP root@$SSH_HOST:/root/outputs/calibrated-qa-gemma2-2b/rl/logs/orchestrator.stdout results/
```

To also download the model weights (~5GB):

```bash
$SCP -r root@$SSH_HOST:/root/outputs/calibrated-qa-gemma2-2b/rl/weights/ results/weights/
```

## Step 5 — Analyze

```bash
pip install matplotlib numpy
python analyze_calibration.py --results-dir results --batch-size 1024 --group-size 5 --max-steps 50
python sample_generations.py --results-dir results --batch-size 1024 --max-steps 50 --n 10
```

Outputs:
- `results/calibration_curves.png` — reliability diagrams per 5-step group
- `results/brier_over_time.png` — reward, Brier, ECE, accuracy over training
- `results/calibration_metrics.csv` — per-generation metrics
- `results/cot_samples.txt` — 10 CoT samples from step 0 vs step 49

## Quirks to watch out for

**Stale calibrated_qa.py**: Hatch sometimes caches the old flat file. `run_training.sh` handles this automatically with `rm -rf .venv/lib/python3.12/site-packages/calibrated_qa*` before reinstalling. If you edit `calibrated_qa.py` and reinstall manually, always do this rm first.

**HF dataset cache**: After changing the prompt or dataset processing, clear the cache:
```bash
$SSH "rm -rf /root/.cache/huggingface/datasets"
```

**Generations file size**: `generations.jsonl` is ~161MB uncompressed. Compress before committing:
```bash
gzip -k results/generations.jsonl
```

## Re-running on an existing RunPod (prime-rl already installed)

Skip step 2 and go straight to step 3. The run script re-installs the `calibrated_qa` package each time to pick up any edits.
