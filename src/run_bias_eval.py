# src/run_bias_eval.py
# Description: Runs bias evaluation on BEC-Cri or BEC-Pro and saves association scores.

import argparse
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertForMaskedLM
from utils import input_pipeline
from model_evaluation import model_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory or model name (e.g., bert-base-uncased)")
    parser.add_argument("--data", required=True, help="Path to bias evaluation data (BEC-Cri or BEC-Pro .tsv)")
    parser.add_argument("--output", required=True, help="Path to save output .tsv with association scores")
    args = parser.parse_args()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    print(f"📑 Loading evaluation data from: {args.data}")
    df = pd.read_csv(args.data, sep="\t")

    print(f"🧠 Loading model from: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except:
        tokenizer = BertTokenizer.from_pretrained(args.model)

    try:
        model = AutoModelForMaskedLM.from_pretrained(args.model)
    except:
        model = BertForMaskedLM.from_pretrained(args.model)

    # Run evaluation
    print("🔍 Evaluating model bias...")
    associations = model_evaluation(df, tokenizer, model, device)

    # Save results
    df = df.assign(AssociationScore=associations)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep="\t", index=False)
    print(f"✅ Results written to: {args.output}")

if __name__ == "__main__":
    main()
