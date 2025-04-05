# src/debias_lcd.py
# Description: Fine-tunes LegalBERT-Small using ECtHR-Gtuned dataset (Legal-Context-Debias)
#              and saves the debiased model.

import argparse
import os
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler
from utils import DocDataset
from fine_tune import fine_tune

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nlpaueb/legal-bert-small-uncased", help="Base model to fine-tune")
    parser.add_argument("--data", default="data/ecthr-gtuned.tsv", help="ECtHR-Gtuned .tsv file path")
    parser.add_argument("--output_dir", default="models/", help="Where to save the debiased model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("📚 Loading ECtHR-Gtuned dataset...")
    dataset = DocDataset(args.data, max_seq_length=512, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)

    print("🧠 Loading model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    print("🚀 Starting LCD fine-tuning...")
    model = fine_tune(model, "ecthr-gtuned", dataloader, args.epochs, args.lr, args.eps, tokenizer, args.device)

    # Save model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"lcd-debiased_ep{args.epochs}_lr{args.lr}_bs{args.batch_size}")
    model.bert.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    print(f"\n✅ Debiased model saved to: {model_path}")

if __name__ == "__main__":
    main()
