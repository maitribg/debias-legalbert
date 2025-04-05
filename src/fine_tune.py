# fine_tune.py 
# Adapted from Mustafa Bozdag's original 
# Description: Fine-tuning logic for debiasing BERT-based models using either
#               - GAP-Flipped dataset (for MLM Loss)
#               - ECtHR-Gtuned dataset (for binary classification)


import time 
import torch 
from transformers import AdamW, get_linear_schedule_with_warmup 
from utils import format_time, mask_tokens 


def fine_tune(model, dataset_name, dataloader, epochs, learning_rate, epsilon, tokenizer, device):
    """
    
    Fine-tunes a BERT-based model using either MLM or sequence classification, depending on dataset. 

    Args: 
        model: The model to fine-tune (MaskedLM or SequenceClassification).
        dataset_name: Either gap-flipped or ecthr-gtuned
        epochs: number of training epochs 
        learning_rate: learning rate for the optimizer
        epsilon: Epsilon value for the AdamW
        tokenizer: Tokenizer used for tokenizing input 
        device: cuda or cpu 

    Returns:
        The fine-tuned model
    """

    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
    total_steps = len(dataloader) * epochs 

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = total_steps
    )

    print(f"\n--- Starting fine-tuning on '{dataset_name}' for {epochs} epochs ---")

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        t0 = time.time()
        total_loss = 0

        for step, batch in enumerate(dataloader):
            if step % 100 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step}/{len(dataloader)} — Elapsed: {elapsed}")

            # GAP-Flipped uses Masked Language Modeling (MLM) objective
            if dataset_name == "gap-flipped":
                b_input_ids, b_labels = mask_tokens(batch[0], tokenizer)
                b_input_ids = b_input_ids.to(device)
                b_input_mask = batch[1].to(device)
                b_labels = b_labels.to(device)

            # ECtHR-Gtuned uses binary classification (predicting gender)
            elif dataset_name == "ecthr-gtuned":
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[2].to(device)
                b_labels = batch[3].to(device)

            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # Forward pass
            model.zero_grad()
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )
            loss = outputs[0]
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        epoch_time = format_time(time.time() - t0)

        print(f"  Avg training loss: {avg_loss:.4f}")
        print(f"  Epoch duration: {epoch_time}")

    print("\n--- Fine-tuning complete ---")
    return model
