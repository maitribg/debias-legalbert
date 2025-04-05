# src/debias_gap.py
# Description: Fine-tunes a model using the GAP-Flipped dataset for MLM-based debiasing.

import argparse
import os
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader, RandomSampler
from nltk import sent_tokenize
import pandas as pd
import math
from utils import input_pipeline
from fine_tune import fine_tune

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased", help="Base model for fine-tuning")
    parser.add_argument("--data", default="data/gap-flipped.tsv", help="Path to GAP-Flipped .tsv")
    parser.add_argument("--output_dir", default="models/", help="Directory to save model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("📚 Loading GAP-Flipped dataset...")
    df = pd.read_csv(args.data, sep="\t")
    sentences = []
    for text in df["Text"]:
        sentences += sent_tokenize(text)

    max_seq_len = max(len(s.split()) for s in sentences)
    max_seq_len = 2 ** math.ceil(math.log2(max_seq_len))
    print(f"📏 Max sequence length: {max_seq_len}")

    input_ids, attention_masks = input_pipeline(sentences, tokenizer, max_seq_len)
    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)

    print("🧠 Loading MLM model...")
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    print("🚀 Starting GAP fine-tuning...")
    model = fine_tune(model, "gap-flipped", dataloader, args.epochs, args.lr, args.eps, tokenizer, args.device)

    # Save the model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"gap-debiased_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"\n✅ GAP-debiased model saved to: {model_path}")

if __name__ == "__main__":
    main()
