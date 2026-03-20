"""
infer.py — Test and compare the model before and after fine-tuning

Run this AFTER train.py to see the effect of fine-tuning.
Also useful as a playground to probe model behavior.
"""

import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import load_model, generate
from data import QA_CORPUS


def load_finetuned(checkpoint_dir: str = "./checkpoints/final"):
    """Load the saved fine-tuned model."""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"No fine-tuned model found at '{checkpoint_dir}'. "
            "Run train.py first."
        )
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[infer] Loading fine-tuned model from {checkpoint_dir} on {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


def compare(question: str, base_model, base_tok, ft_model, ft_tok, device: str):
    """Side-by-side comparison: base GPT-2 vs fine-tuned GPT-2."""
    prompt = f"Q: {question}\nA:"
    
    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")
    
    base_out = generate(base_model, base_tok, device, prompt,
                         max_new_tokens=80, temperature=0.7)
    print(f"[BASE GPT-2]  {base_out}")
    
    ft_out = generate(ft_model, ft_tok, device, prompt,
                       max_new_tokens=80, temperature=0.7)
    print(f"[FINE-TUNED]  {ft_out}")


def interactive_mode(model, tokenizer, device: str):
    """Type your own questions and see what the model says."""
    print("\n" + "="*60)
    print("INTERACTIVE MODE — type a question, press Enter")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        
        prompt = f"Q: {question}\nA:"
        for temp in [0.3, 0.8]:
            result = generate(model, tokenizer, device, prompt,
                              max_new_tokens=100, temperature=temp)
            print(f"  [temp={temp}] {result}")


def probe_memorization(model, tokenizer, device: str):
    """
    Test how well the model memorized training data.
    
    WHAT THIS REVEALS:
    A fine-tuned model given just "Q: What is a token?\nA:" should
    complete it in the style of our training data. If loss went low,
    it will nearly reproduce the training answer. This is overfitting, 
    bad for production, educational for understanding what training does.
    """
    print("\n" + "="*60)
    print("MEMORIZATION PROBE (first 5 training examples)")
    print("="*60)
    
    for question, expected in QA_CORPUS[:5]:
        prompt = f"Q: {question}\nA:"
        result = generate(model, tokenizer, device, prompt,
                           max_new_tokens=80, temperature=0.3)
        print(f"\nQ: {question}")
        print(f"Expected: {expected[:80]}...")
        print(f"Got:      {result[:80]}...")


if __name__ == "__main__":
    import sys
    
    # Load base model for comparison
    print("Loading base GPT-2 (pretrained, not fine-tuned)...")
    base_model, base_tok, device = load_model()
    
    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    try:
        ft_model, ft_tok, _ = load_finetuned()
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("Run python train.py first, then come back here.")
        sys.exit(1)
    
    # Compare on a few questions
    test_questions = [
        "What is a gradient?",
        "What is fine-tuning?",
        "What is temperature in text generation?",
    ]
    
    print("\n" + "="*60)
    print("BASE vs FINE-TUNED COMPARISON")
    print("="*60)
    
    for q in test_questions:
        compare(q, base_model, base_tok, ft_model, ft_tok, device)
    
    # Memorization probe
    probe_memorization(ft_model, ft_tok, device)
    
    # Interactive session
    interactive_mode(ft_model, ft_tok, device)
