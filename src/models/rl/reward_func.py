import os
from pathlib import Path

import requests
import torch

from src.utils.misc import parse_qa_response, load_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PRECISION_JUDGE_PROMPT = load_prompt(PROJECT_ROOT / "prompts/rl/verify_precision.txt")

RECALL_JUDGE_PROMPT = load_prompt(PROJECT_ROOT / "prompts/rl/verify_recall.txt")


JUDGE_ENDPOINT = os.environ.get(
    "JUDGE_ENDPOINT",
    "http://127.0.0.1:5001/judge",
)


def parse_ground_truth(ground_truth: str):
    """Returns (description, qa_pairs) or (None, None) if invalid."""
    if not ground_truth:
        return None, None

    desc_prefix = "Description: "
    qa_prefix = "QA Pairs:\n"

    if not ground_truth.startswith(desc_prefix):
        return None, None

    qa_index = ground_truth.find(qa_prefix)
    if qa_index == -1:
        return None, None

    description = ground_truth[len(desc_prefix) : qa_index].strip()

    qa_pairs = ground_truth[qa_index + len(qa_prefix) :].strip()

    return description, qa_pairs


def build_precision_judge_prompt(summary: str, question: str, answer: str) -> str:
    return PRECISION_JUDGE_PROMPT.format(
        summary=summary,
        question=question,
        answer=answer,
    )


def build_recall_judge_prompt(
    question_answer_pairs: str, question: str, answer: str
) -> str:
    return RECALL_JUDGE_PROMPT.format(
        qa_pairs=question_answer_pairs,
        question=question,
        answer=answer,
    )


def _call_remote_judge(all_judge_prompts):
    """Send judge prompts to the remote GPU server and get back scores."""
    if not all_judge_prompts:
        return []

    payload = {"prompts": all_judge_prompts}
    try:
        resp = requests.post(JUDGE_ENDPOINT, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("scores", [])
    except Exception as e:
        print(f"[reward_func] Remote judge error: {e}")
        return [0.0] * len(all_judge_prompts)


def clean_model_output(text: str) -> str:
    """Remove special tokens from model generation."""
    special_tokens = [
        "<|im_start|>",
        "<|im_end|}",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    return text.strip()


def _has_valid_qa_pairs(qa_pairs):
    """Return whether the parser found any questions."""
    if not qa_pairs or len(qa_pairs) != 4:
        return False
    return len(qa_pairs[0]) > 0


def reward_func(queries, prompts, labels):
    """Compute QA precision and recall, then return their F1 as the reward.

    Precision checks generated answers against the reference description.
    Recall checks reference answers against the generated trace.
    Scores use a percentage scale. Empty predictions receive zero, as in the
    historical training logs. Empty reference QA also receives zero.
    """
    all_prec_prompts = []
    all_rec_prompts = []

    meta = []

    for i, query in enumerate(queries):
        prompt = prompts[i] if prompts is not None else ""
        label = labels[i] if labels is not None else ""

        if prompt and isinstance(query, str) and query.startswith(prompt):
            solution_str = query[len(prompt) :].lstrip()
        else:
            solution_str = query
        solution_str = clean_model_output(solution_str)

        label_summary, label_qa_str = parse_ground_truth(label)

        qa_pairs_pred = parse_qa_response(solution_str)
        has_pred_qas = _has_valid_qa_pairs(qa_pairs_pred)

        if label_qa_str:
            qa_pairs_gt = parse_qa_response(label_qa_str)
            has_gold_qas = _has_valid_qa_pairs(qa_pairs_gt)
        else:
            qa_pairs_gt = None
            has_gold_qas = False

        if not has_pred_qas or not has_gold_qas:
            meta.append((0, 0, 0, 0))
            continue

        # Check generated answers against the reference description.
        p_start = len(all_prec_prompts)
        questions_pred, evidences_pred, answers_pred, types_pred = qa_pairs_pred
        for q_pred, e_pred, a_pred, t_pred in zip(
            questions_pred, evidences_pred, answers_pred, types_pred
        ):
            all_prec_prompts.append(
                build_precision_judge_prompt(label_summary, q_pred, a_pred)
            )
        p_end = len(all_prec_prompts)

        # Check reference answers against the generated trace.
        r_start = len(all_rec_prompts)
        label_qs, label_evs, label_ans, label_types = qa_pairs_gt
        for q_label, e_label, a_label, t_label in zip(
            label_qs, label_evs, label_ans, label_types
        ):
            all_rec_prompts.append(
                build_recall_judge_prompt(solution_str, q_label, a_label)
            )
        r_end = len(all_rec_prompts)

        meta.append((p_start, p_end, r_start, r_end))

    if all_prec_prompts:
        prec_scores_list = _call_remote_judge(all_prec_prompts)
        if len(prec_scores_list) != len(all_prec_prompts):
            print(
                "[reward_func] Warning: precision judge returned "
                f"{len(prec_scores_list)} scores for "
                f"{len(all_prec_prompts)} prompts. Falling back to zeros."
            )
            prec_scores_list = [0.0] * len(all_prec_prompts)
        prec_scores = torch.tensor(prec_scores_list, dtype=torch.float32)
    else:
        prec_scores = torch.tensor([], dtype=torch.float32)

    if all_rec_prompts:
        rec_scores_list = _call_remote_judge(all_rec_prompts)
        if len(rec_scores_list) != len(all_rec_prompts):
            print(
                "[reward_func] Warning: recall judge returned "
                f"{len(rec_scores_list)} scores for "
                f"{len(all_rec_prompts)} prompts. Falling back to zeros."
            )
            rec_scores_list = [0.0] * len(all_rec_prompts)
        rec_scores = torch.tensor(rec_scores_list, dtype=torch.float32)
    else:
        rec_scores = torch.tensor([], dtype=torch.float32)

    batch_prec = []
    batch_rec = []
    batch_f1 = []

    for p_start, p_end, r_start, r_end in meta:
        if p_end > p_start:
            sample_prec = prec_scores[p_start:p_end].mean().item() * 100.0
        else:
            sample_prec = 0.0

        if r_end > r_start:
            sample_rec = rec_scores[r_start:r_end].mean().item() * 100.0
        else:
            sample_rec = 0.0

        if (sample_prec + sample_rec) > 0:
            sample_f1 = 2.0 * sample_prec * sample_rec / (sample_prec + sample_rec)
        else:
            sample_f1 = 0.0

        batch_prec.append(sample_prec)
        batch_rec.append(sample_rec)
        batch_f1.append(sample_f1)

    prec_tensor = torch.tensor(batch_prec, dtype=torch.float32)
    rec_tensor = torch.tensor(batch_rec, dtype=torch.float32)
    f1_tensor = torch.tensor(batch_f1, dtype=torch.float32)

    return {
        "rewards": f1_tensor,
        "scores": f1_tensor,
        "extra_logs": {
            "judge_rewards": f1_tensor,  # for backward-compat
            "precision": prec_tensor,
            "recall": rec_tensor,
            "f1": f1_tensor,
        },
    }
