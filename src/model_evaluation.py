# model_evaluation.py
# Adapted from Mustafa Bozdag
# Description: Evaluates gender bias in a BERT-based model using the BEC-Cri or BEC-Pro datasets,
#              following the method described in Section 3.1 of the paper.

import math
import torch
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from utils import input_pipeline, prob_with_prior

def model_evaluation(eval_df, tokenizer, model, device):
    """
    Evaluates bias in a model using MLM predictions on BEC-style sentence templates.

    Args:
        eval_df: A pandas DataFrame containing the columns:
                 - 'Sentence'     (original)
                 - 'Sent_TM'      (target masked)
                 - 'Sent_TAM'     (target + attribute masked)
        tokenizer: The tokenizer used for the model.
        model: A Masked Language Model (e.g., LegalBERT or BERT-base).
        device: 'cuda' or 'cpu'.

    Returns:
        A list of association scores (one per input row), representing gender bias.
    """

    # Compute padded max sequence length (next power of 2 for efficiency)
    max_len = max([len(sent.split()) for sent in eval_df["Sent_TM"]])
    max_len_eval = 2 ** math.ceil(math.log2(max_len))
    print(f"Max sequence length for evaluation: {max_len_eval}")

    # Tokenize input sentence versions
    tokens_TM, attn_TM = input_pipeline(eval_df["Sent_TM"], tokenizer, max_len_eval)
    tokens_TAM, attn_TAM = input_pipeline(eval_df["Sent_TAM"], tokenizer, max_len_eval)
    tokens_original, _ = input_pipeline(eval_df["Sentence"], tokenizer, max_len_eval)

    assert tokens_TM.shape == attn_TM.shape
    assert tokens_TAM.shape == attn_TAM.shape

    # Prepare dataloader
    dataset = TensorDataset(tokens_TM, attn_TM, tokens_TAM, attn_TAM, tokens_original)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=20, sampler=sampler)

    model.to(device)
    model.eval()

    all_association_scores = []

    for step, batch in enumerate(dataloader):
        b_TM = batch[0].to(device)       # Target masked
        b_att_TM = batch[1].to(device)
        b_TAM = batch[2].to(device)      # Target + attribute masked
        b_att_TAM = batch[3].to(device)
        b_original = batch[4]            # Used for recovering true target word

        with torch.no_grad():
            out_TM = model(b_TM, attention_mask=b_att_TM)
            out_TAM = model(b_TAM, attention_mask=b_att_TAM)
            pred_TM = softmax(out_TM[0], dim=2)      # Token-level prediction probs
            pred_TAM = softmax(out_TAM[0], dim=2)

        assert pred_TM.shape == pred_TAM.shape

        # Compute log(p_target / p_prior) scores for each sample
        assoc_scores = prob_with_prior(pred_TM, pred_TAM, b_TAM, b_original, tokenizer)
        all_association_scores.extend(assoc_scores)

    return all_association_scores
