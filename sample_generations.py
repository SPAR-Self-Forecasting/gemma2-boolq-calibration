"""
Sample 10 random generations from the first and last training step
and write them to a text file for qualitative CoT analysis.

Usage:
    python sample_generations.py [--results-dir DIR] [--batch-size 1024] [--n 10] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path


def load_step(path: Path, step: int, batch_size: int) -> list[dict]:
    records = []
    start = step * batch_size
    end = start + batch_size
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_generation(rec: dict, idx: int) -> str:
    gt = str(rec.get("ground_truth", "")).lower() == "true"
    p_yes = rec.get("p_yes")
    reward = rec.get("reward", "")
    parseable = rec.get("parseable", False)
    completion = rec.get("full_completion", "").strip()

    lines = [
        f"{'='*70}",
        f"Sample {idx}",
        f"  Ground truth : {'YES' if gt else 'NO'}",
        f"  p(YES)       : {f'{p_yes:.2f}' if p_yes is not None else 'unparseable'}",
        f"  Reward       : {f'{reward:.4f}' if isinstance(reward, float) else reward}",
        f"  Parseable    : {parseable}",
        f"{'─'*70}",
        "Completion:",
        completion,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--n", type=int, default=10, help="Samples per step")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    gen_path = args.results_dir / "generations.jsonl"
    out_path = args.results_dir / "cot_samples.txt"

    first_step = 0
    last_step = args.max_steps - 1

    first_recs = load_step(gen_path, first_step, args.batch_size)
    last_recs = load_step(gen_path, last_step, args.batch_size)

    first_sample = random.sample(first_recs, min(args.n, len(first_recs)))
    last_sample = random.sample(last_recs, min(args.n, len(last_recs)))

    with open(out_path, "w") as f:
        f.write(f"CoT Style Analysis — {args.n} samples from step {first_step} vs step {last_step}\n")
        f.write(f"Generations file: {gen_path}\n\n")

        f.write(f"{'#'*70}\n")
        f.write(f"# STEP {first_step} (first step)\n")
        f.write(f"{'#'*70}\n\n")
        for i, rec in enumerate(first_sample, 1):
            f.write(format_generation(rec, i))
            f.write("\n\n")

        f.write(f"{'#'*70}\n")
        f.write(f"# STEP {last_step} (last step)\n")
        f.write(f"{'#'*70}\n\n")
        for i, rec in enumerate(last_sample, 1):
            f.write(format_generation(rec, i))
            f.write("\n\n")

    print(f"Saved {out_path}")
    print(f"  Step {first_step}: {len(first_recs)} records, sampled {len(first_sample)}")
    print(f"  Step {last_step}: {len(last_recs)} records, sampled {len(last_sample)}")


if __name__ == "__main__":
    main()
