
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_test_prompts() -> list[str]:
    # Only include prompts that decode to an even number of tokens (needed for sequence parallelism)
    return [
        # "The capital of France is",
        # "Machine learning is",
        "Python is a programming language that",
        "In the year 2024",
        # "Artificial intelligence will be",
        "The weather today is",
        # "Scientists have discovered",
        # "The most important thing in life is",
    ]


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], tokenizer, device: str = "cuda"):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoded = self.tokenizer(prompt, return_tensors="pt")

        input_ids = encoded.input_ids.squeeze(0)  # Remove batch dimension
        attention_mask = encoded.attention_mask.squeeze(0)  # Remove batch dimension
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        return {
            "prompt": prompt,
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


@dataclass
class ModelCheckResult:
    logits: np.array
    input_ids: list[int]
    topk_ids: list[int]
    topk_scores: list[float]
    prompt: str = None


def calculate_logits_stats(hf_logits: np.array, mcore_logits: np.array, eps: float = 1e-8):
    absdiff = np.abs(hf_logits - mcore_logits)
    max_diff = absdiff.max()
    avg_diff = absdiff.mean()
    relative_diff = (absdiff / (np.abs(hf_logits) + eps)).mean() * 100

    return {"max_diff": max_diff, "avg_diff": avg_diff, "rel_diff": relative_diff}


def calculate_topk_stats(
    hf_topk_ids: list[int], mcore_topk_ids: list[int], topk_ranks: list[int] = [1, 3, 5]
):
    matches = {}
    topk_ranks = [k for k in topk_ranks if k <= len(hf_topk_ids)]
    for k in topk_ranks:
        ref = set(hf_topk_ids[:k])
        test = set(mcore_topk_ids[:k])
        matches[f"pass@{k}"] = True if ref == test else False

    return matches


def postprocess_logits(
    hf_results: list[ModelCheckResult],
    mcore_results: list[ModelCheckResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(hf_results) != len(mcore_results):
        raise ValueError("Result lists must be the same length (one per prompt).")

    per_tok_rows = []
    logits_rows = []
    for prompt_idx, (hf, mcore) in enumerate(zip(hf_results, mcore_results)):
        assert hf.input_ids == mcore.input_ids

        logits_stats = calculate_logits_stats(hf.logits, mcore.logits)
        logits_rows.append(
            {
                "prompt_idx": prompt_idx,
                "prompt": hf.prompt,
                "input_ids": hf.input_ids,
                **logits_stats,
            }
        )
        for tok_idx, input_id in enumerate(hf.input_ids):
            hf_topk = hf.topk_ids[tok_idx]
            mc_topk = hf.topk_ids[tok_idx]
            topk_matches = calculate_topk_stats(hf_topk, mc_topk)
            per_tok_rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "token_idx": tok_idx,
                    "input_ids": hf.input_ids,
                    "token_id": input_id,
                    "hf_topk_ids": hf.topk_ids[tok_idx],
                    "mcore_topk_ids": mcore.topk_ids[tok_idx],
                    **topk_matches,
                    "hf_topk_scores": hf.topk_scores[tok_idx],
                    "mcore_topk_scores": mcore.topk_scores[tok_idx],
                }
            )
    logits_df = pd.DataFrame(logits_rows).set_index(["prompt_idx"])
    per_tok_df = pd.DataFrame(per_tok_rows).set_index(["prompt_idx", "token_idx"])
    return logits_df, per_tok_df
