# utils.py
# Adapted from Mustafa Bozdag
# Description: Utility functions for data preprocessing, token masking (for MLM),
#              attention mask creation, and dataset formatting for fine-tuning and evaluation.

import time
import datetime
import numpy as np
import pandas as pd
import torch

from typing import List, Tuple
from keras.utils import pad_sequences
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DocDataset(Dataset):
    """
    PyTorch Dataset class for ECtHR-Gtuned dataset used in LCD debiasing.
    Each item returns tokenized input IDs, segment IDs, attention mask, and binary label.
    """

    def __init__(self, path: str, max_seq_length: int, tokenizer: PreTrainedTokenizer) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sources, self.targets = self._load(path)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source: List[int] = self.sources[idx]
        seq_len = len(source)

        if seq_len > self.max_seq_length:
            input_ids = np.array(source[:self.max_seq_length])
            input_mask = np.array([1] * self.max_seq_length)
        else:
            pad = [0] * (self.max_seq_length - seq_len)
            input_ids = np.array(source + pad)
            input_mask = np.array([1] * seq_len + pad)

        segment_ids = np.array([0] * self.max_seq_length)
        # label = np.array(self.targets[idx])
        label = torch.tensor(self.targets[idx], dtype=torch.long)


        return input_ids, segment_ids, input_mask, label

    def _load(self, path: str) -> Tuple[List[List[int]], List[int]]:
        df = pd.read_csv(path, sep="\t")
        sources, targets = [], []

        for i in range(len(df)):
            text = df["text"][i]
            label = int(df["applicant_gender"][i]) - 1  # Convert 1/2 → 0/1
            tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            sources.append(input_ids)
            targets.append(label)

        return sources, targets


def format_time(seconds: float) -> str:
    """Convert seconds to hh:mm:ss string format."""
    rounded = int(round(seconds))
    return str(datetime.timedelta(seconds=rounded))


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability=0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies standard MLM masking to inputs: 80% [MASK], 10% random, 10% original.
    Returns:
        masked_inputs, labels (with -100 for unmasked tokens)
    """
    if tokenizer.mask_token is None:
        raise ValueError("Tokenizer must have a [MASK] token.")

    labels = inputs.clone()
    labels = labels.to(torch.long)  
    prob_matrix = torch.full(labels.shape, mlm_probability)

    special_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    prob_matrix.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), value=0.0)

    if tokenizer.pad_token_id is not None:
        pad_mask = labels.eq(tokenizer.pad_token_id)
        prob_matrix.masked_fill_(pad_mask, value=0.0)

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # 80% → [MASK]
    replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% → random word
    random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~replaced
    rand_words = torch.randint(len(tokenizer), labels.shape, dtype=inputs.dtype)
    inputs[random] = rand_words[random]

    return inputs, labels


def tokenize_to_id(sentences: List[str], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    """Tokenize and encode a list of sentences with special tokens."""
    return [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]


def attention_mask_creator(input_ids: torch.Tensor) -> torch.Tensor:
    """Generate attention mask: 1 for non-padding tokens, 0 for padding."""
    return torch.tensor([[int(token_id > 0) for token_id in sent] for sent in input_ids])


def input_pipeline(sentences: List[str], tokenizer: PreTrainedTokenizer, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenizes and pads a list of sentences to a uniform length.
    Returns:
        input_ids, attention_mask (both as tensors)
    """
    input_ids = tokenize_to_id(sentences, tokenizer)
    padded = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post", value=tokenizer.mask_token_id)
    input_ids_tensor = torch.tensor(padded)
    attention_mask = attention_mask_creator(input_ids_tensor)
    return input_ids_tensor, attention_mask


def prob_with_prior(pred_TM, pred_TAM, input_ids_TAM, original_ids, tokenizer):
    """
    Compute association score: log(p_target / p_prior)
    For each sentence, finds the masked position, retrieves the predicted probability
    of the original token in both the target-masked and fully-masked inputs.
    """
    pred_TM = pred_TM.cpu()
    pred_TAM = pred_TAM.cpu()
    input_ids_TAM = input_ids_TAM.cpu()

    scores = []
    for i, input_ids in enumerate(input_ids_TAM):
        mask_idx = np.where(input_ids == tokenizer.mask_token_id)[0]
        target_token_id = original_ids[i][mask_idx[0]]

        p_target = pred_TM[i][mask_idx[0]][target_token_id].item()
        p_prior = pred_TAM[i][mask_idx[0]][target_token_id].item()

        scores.append(np.log(p_target / p_prior))

    return scores
