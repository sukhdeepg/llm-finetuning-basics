"""
train.py — Fine-tuning GPT-2 on a custom Q&A dataset

THE TRAINING LOOP — MENTAL MODEL:

  for each batch:
    1. FORWARD PASS  — run the model, compute predictions
    2. LOSS          — measure how wrong the predictions are
    3. BACKWARD PASS — compute gradients (how to fix each weight)
    4. OPTIMIZER STEP — nudge every weight in the right direction
    5. ZERO GRADS    — clear gradients so they don't accumulate

  Repeat until loss is low enough.

WHAT IS ACTUALLY CHANGING?
  The model starts with weights learned from ~40GB of internet text.
  Fine-tuning adjusts ALL weights slightly so the model prefers
  our specific Q&A format and content. The key insight:
    - We don't retrain from scratch (that would take weeks)
    - We continue training from where GPT-2 left off
    - A few passes over 15 examples is enough to observe adaptation

OVERFITTING IS EXPECTED AND EDUCATIONAL HERE:
  With 15 examples and ~10 epochs, the model will memorize this data.
  That's fine, we want to SEE the loss go to zero and watch behavior change.
  In production we'd use more data, regularization, and early stopping.
"""

import os
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_dataloader
from model import load_model, generate


CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def evaluate_loss(model, dataloader, device: str) -> float:
    """Compute average loss on the dataset without updating weights."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


def train(
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    save_every: int = 5,
):
    """
    Fine-tuning parameters explained:
    
    num_epochs: How many full passes over the dataset.
      - With 15 examples, even 10 epochs is fast (<2 min locally)
      - Watch loss decrease. It should roughly halve in first few epochs
    
    batch_size: Examples per gradient update.
      - Small batch (2-4) works fine here.
      - Too large = needs more memory; too small = noisy gradients
    
    learning_rate: 5e-5 is the standard starting point for GPT-2 fine-tuning.
      - Lower than pretraining LR (which was ~1e-3) to avoid forgetting
      - AdamW adapts per-parameter; this is the global scale
    
    The AdamW optimizer:
      - "W" = weight decay (L2 regularization built in)
      - Adapts the learning rate per parameter based on gradient history
      - Standard choice for transformer fine-tuning
    """
    
    print("\n" + "="*60)
    print("FINE-TUNING GPT-2")
    print("="*60)
    print(f"  Epochs:        {num_epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print("="*60 + "\n")
    
    # Load model & data
    model, tokenizer, device = load_model()
    dataloader = get_dataloader(tokenizer, batch_size=batch_size)
    
    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing: smoothly reduces LR from max → 0 over training.
    # This helps the model settle into a good minimum at the end.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(dataloader),
        eta_min=learning_rate * 0.1,
    )
    
    # Baseline: what does the model produce BEFORE fine-tuning?
    test_question = "What is fine-tuning?"
    prompt = f"Q: {test_question}\nA:"
    
    print("--- BEFORE FINE-TUNING ---")
    before = generate(model, tokenizer, device, prompt, max_new_tokens=60, temperature=0.7)
    print(f"{prompt}{before}\n")
    
    # Training loop
    model.train()
    loss_history = []
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for step, batch in enumerate(dataloader):
            # 1. Move data to the right device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 2. FORWARD PASS
            #    The model internally computes cross-entropy loss between
            #    its predictions and the ground-truth next tokens (labels).
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # 3. BACKWARD PASS — compute gradients
            loss.backward()
            
            # 4. Gradient clipping: prevent any single gradient from being
            #    too large (common training stability trick)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 5. OPTIMIZER STEP — update weights
            optimizer.step()
            scheduler.step()
            
            # 6. ZERO GRADIENTS — crucial, gradients accumulate by default
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        elapsed = time.time() - epoch_start
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch:3d}/{num_epochs}  |  "
              f"loss: {avg_loss:.4f}  |  "
              f"lr: {lr_now:.2e}  |  "
              f"time: {elapsed:.1f}s")
        
        # Save checkpoint periodically
        if epoch % save_every == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")
    
    # After training: see what changed
    model.eval()
    print("\n--- AFTER FINE-TUNING ---")
    after = generate(model, tokenizer, device, prompt, max_new_tokens=60, temperature=0.7)
    print(f"{prompt}{after}\n")
    
    # Save the final model
    final_path = f"{CHECKPOINT_DIR}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to: {final_path}/")
    
    print("\n--- LOSS SUMMARY ---")
    print(f"  Start: {loss_history[0]:.4f}")
    print(f"  End:   {loss_history[-1]:.4f}")
    print(f"  Drop:  {(1 - loss_history[-1]/loss_history[0])*100:.1f}%")
    
    return model, tokenizer, device


if __name__ == "__main__":
    train(num_epochs=10, batch_size=2, learning_rate=5e-5)
