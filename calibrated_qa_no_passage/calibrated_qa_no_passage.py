import json
import os
import re
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

import verifiers as vf

# Instructions without passage — model answers from its own knowledge
INSTRUCTIONS = (
    "Answer the following question.\n"
    "First, reason about the answer in 1-3 sentences.\n"
    "Then, write your final answer on a new line in this exact format:\n"
    "ANSWER: YES with XX% probability\n"
    "or\n"
    "ANSWER: NO with XX% probability\n"
    "where XX is your confidence from 0 to 100. Only give ONE answer line.\n"
    "\n"
    "Example:\n"
    "Question: Do Iran and Afghanistan speak the same language?\n"
    "Reasoning: Persian/Farsi is spoken in Iran, and Dari — a variety of Persian — is "
    "spoken in Afghanistan. They are closely related but not identical.\n"
    "ANSWER: YES with 80% probability\n"
    "\n"
    "Example:\n"
    "Question: Is Elder Scrolls Online the same as Skyrim?\n"
    "Reasoning: Elder Scrolls Online is an MMO while Skyrim is a single-player RPG. "
    "They are different games in the same franchise.\n"
    "ANSWER: NO with 95% probability\n"
)

# Match "ANSWER: YES with XX% probability" or "YES with XX% probability"
ANSWER_PATTERN = re.compile(
    r"(?:ANSWER:\s*)?(YES|NO)\s+with\s+(\d+)\s*%\s*(?:probability|confidence)?",
    re.IGNORECASE,
)

MAX_TOKENS = 512


def _extract_probability_from_answer(text):
    """Extract P(YES) using format: 'ANSWER: YES/NO with XX% probability'."""
    matches = list(ANSWER_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    label = match.group(1).upper()
    pct = max(0, min(100, int(match.group(2))))
    return pct / 100.0 if label == "YES" else 1.0 - pct / 100.0


def _length_penalty(token_count):
    """Bucket penalty: penalize too-short (<1/5 max) and too-long (>4/5 max) completions."""
    low = MAX_TOKENS * 0.2
    high = MAX_TOKENS * 0.8

    if token_count < low:
        return -1.0 * (1.0 - token_count / low)
    elif token_count > high:
        t = (token_count - high) / (MAX_TOKENS - high)
        return -1.0 * min(t, 1.0)
    return 0.0


def _msg_text(msg):
    if isinstance(msg, dict):
        return msg.get("role", ""), msg.get("content", "") or ""
    return getattr(msg, "role", "") or "", getattr(msg, "content", "") or ""


_WRITE_COUNT = [0]

def _log_generation(path, record):
    import sys
    _WRITE_COUNT[0] += 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = {}
        for k, v in record.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                safe[k] = v
            else:
                safe[k] = str(v)
        with open(path, "a") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
        if _WRITE_COUNT[0] % 1000 == 0:
            print(f"[calibrated_qa_no_passage] Wrote {_WRITE_COUNT[0]} records to {path}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[calibrated_qa_no_passage] LOG ERROR #{_WRITE_COUNT[0]}: {e}", file=sys.stderr, flush=True)


def load_environment(
    dataset_name="google/boolq",
    dataset_split="train",
    system_prompt=None,
    seed=42,
    log_dir=None,
):
    ds = load_dataset(dataset_name, split=dataset_split)
    ds = ds.shuffle(seed=seed)

    # Balance to 50/50 true/false
    true_rows = ds.filter(lambda x: x["answer"] is True)
    false_rows = ds.filter(lambda x: x["answer"] is False)
    n = min(len(true_rows), len(false_rows))
    true_rows = true_rows.select(range(n))
    false_rows = false_rows.select(range(n))
    balanced = concatenate_datasets([true_rows, false_rows]).shuffle(seed=seed)

    # Build user message: instructions + question only (no passage)
    def _build_question(x):
        question = x["question"]
        return (
            f"{INSTRUCTIONS}\n"
            f"Question: {question}?"
        )

    train_dataset = balanced.map(
        lambda x: {
            "question": _build_question(x),
            "answer": "true" if x["answer"] else "false",
            "info": {},
            "task": "calibrated-qa-no-passage",
        },
        remove_columns=balanced.column_names,
    )

    parser = vf.MaybeThinkParser()

    _log_dir = log_dir or os.environ.get("CALIBRATED_QA_LOG_DIR", "/root/outputs/calibrated-qa-no-passage")
    _gen_log_path = Path(_log_dir) / "generations.jsonl"
    _call_count = [0]

    def brier_reward_func(completion, answer, **kwargs):
        raw_text = parser.parse_answer(completion) or ""

        has_unclosed_think = "<think>" in str(completion) and "</think>" not in str(completion)
        if has_unclosed_think:
            raw_text = ""

        extracted_p = _extract_probability_from_answer(raw_text)
        parseable = extracted_p is not None and not has_unclosed_think
        p_yes = extracted_p if parseable else 0.5

        actual = 1.0 if str(answer).lower() == "true" else 0.0
        brier = (p_yes - actual) ** 2
        reward = 1.0 - brier

        if not parseable:
            reward = 0.0

        full_raw = ""
        if isinstance(completion, list):
            for msg in completion:
                role, content = _msg_text(msg)
                full_raw += f"[{role}] {content}\n"
        else:
            full_raw = str(completion)

        completion_text = ""
        if isinstance(completion, list):
            for msg in completion:
                _, content = _msg_text(msg)
                completion_text += content
        else:
            completion_text = str(completion)
        approx_tokens = len(completion_text) / 4.0
        len_pen = _length_penalty(approx_tokens)
        reward += len_pen

        _call_count[0] += 1

        state = kwargs.get("state")
        full_prompt = ""
        if state:
            prompt = state.get("prompt", "")
            if isinstance(prompt, list):
                for msg in prompt:
                    role, content = _msg_text(msg)
                    full_prompt += f"[{role}] {content}\n"
            else:
                full_prompt = str(prompt)

        _log_generation(_gen_log_path, {
            "full_prompt": full_prompt,
            "ground_truth": answer,
            "p_yes": p_yes,
            "p_yes_extracted": extracted_p,
            "parseable": parseable,
            "brier_score": brier,
            "reward": reward,
            "length_penalty": len_pen,
            "approx_tokens": approx_tokens,
            "parsed_answer": str(raw_text)[:500],
            "full_completion": full_raw,
        })

        return reward

    rubric = vf.Rubric(
        funcs=[brier_reward_func],
        weights=[1.0],
    )

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        system_prompt=None,
        parser=parser,
        rubric=rubric,
    )
