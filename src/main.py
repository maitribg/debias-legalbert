# main.py: Unified Debiasing Script for LCD and GAP (MB version)

import os
import time
import random
import argparse
import copy
import math
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from nltk import sent_tokenize
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification
)

from utils import DocDataset, input_pipeline
from fine_tune import fine_tune
from model_evaluation import model_evaluation

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--tune', help='Dataset for fine-tuning')
    parser.add_argument('--eval', required=True, help='Evaluation dataset')
    parser.add_argument('--out', default='')
    parser.add_argument('--batch', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--devID', required=True, type=int)
    parser.add_argument('--classes', default=2, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device(f'cuda:{args.devID}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"--- Loading evaluation data from: {args.eval}")
    eval_data = pd.read_csv(args.eval, sep='\t')

    print(f"--- Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, output_attentions=False, output_hidden_states=False)

    print("\n--- Calculating pre-association scores...")
    pre_scores = model_evaluation(eval_data, tokenizer, model, device)
    eval_data = eval_data.assign(Pre_Assoc=pre_scores)

    model_name = args.model.split('/')[-1].split('.')[0]
    eval_name = os.path.basename(args.eval).split('.')[0]

    if args.tune:
        print(f"--- Loading fine-tuning data from: {args.tune}")
        tune_name = os.path.basename(args.tune).split('.')[0]

        if 'gap' in args.tune:
            df = pd.read_csv(args.tune, sep='\t')
            sentences = [s for text in df.Text for s in sent_tokenize(text)]
            max_len = 2 ** math.ceil(math.log2(max(len(s.split()) for s in sentences)))
            tune_tokens, tune_masks = input_pipeline(sentences, tokenizer, max_len)
            dataset = TensorDataset(tune_tokens, tune_masks)

        elif 'ecthr' in args.tune:
            max_len = 512
            dataset = DocDataset(args.tune, max_len, tokenizer)
            modelMLM = copy.deepcopy(model)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.classes, output_attentions=False, output_hidden_states=False, return_dict=False)

        else:
            raise ValueError("Unsupported tuning dataset")

        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch)
        model = fine_tune(model, tune_name, dataloader, args.epoch, args.lr, args.eps, tokenizer, device)

        model_path = f"{args.out}models/{model_name}-debiased_{tune_name}_lr{args.lr}_eps{args.eps}_ep{args.epoch}"
        os.makedirs(model_path, exist_ok=True)
        if hasattr(model, 'bert'):
            model.bert.save_pretrained(model_path)
        else:
            model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        if 'ecthr' in args.tune:
            modelMLM.load_state_dict(model.state_dict(), strict=False)
            model = modelMLM

        print("\n--- Calculating post-association scores...")
        post_scores = model_evaluation(eval_data, tokenizer, model, device)
        eval_data = eval_data.assign(Post_Assoc=post_scores)

    out_file = f"{args.out}results/{model_name}_debiased_{tune_name}_{eval_name}.tsv" if args.tune else f"{args.out}results/{model_name}_{eval_name}.tsv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    eval_data.to_csv(out_file, sep='\t', index=False)
    print(f"\n✅ Results saved to: {out_file}")
