"""
model.py — GPT-2 (small) loaded from HuggingFace

WHY GPT-2 SMALL?
- 124M parameters: big enough to be real, small enough to run locally
- Well-documented architecture (identical to modern LLMs, just smaller)
- Pretrained weights freely available, we get a working model instantly
- Inference: ~200ms per token on CPU/MPS

MENTAL MODEL OF A LANGUAGE MODEL:
  Input text → Tokenizer → Token IDs → Embedding layer
  → N × Transformer blocks (Attention + MLP) → Final projection → Logits
  → Sample next token → Repeat

Every "modern" LLM (GPT-4, LLaMA, Mistral, Gemma) follows this exact pattern.
The differences are: scale, attention variant, and training data.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(device: str = "auto"):
    """
    Load GPT-2 small (~500MB download, cached after first run).
    
    GPT-2 architecture at a glance:
      - 12 transformer layers
      - 12 attention heads per layer
      - 768-dimensional hidden states
      - 50,257 token vocabulary
      - Max context window: 1024 tokens
    """
    if device == "auto":
        # As I have a M2 Mac, it has a unified GPU called MPS (Metal Performance Shaders)
        # MPS is ~3-5x faster than CPU for transformer inference
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"[model] Loading GPT-2 small on device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # GPT-2 has no pad token by default — use EOS as pad (standard practice)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {param_count:,}  (~{param_count/1e6:.0f}M)")
    print(f"[model] Layers: {model.config.n_layer}, "
          f"Heads: {model.config.n_head}, "
          f"Hidden dim: {model.config.n_embd}")
    
    return model, tokenizer, device


def generate(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """
    Generate text from a prompt.

    KEY CONCEPTS:
    - temperature: controls randomness.
        • 1.0 = sample from raw distribution
        • <1.0 = sharper (more confident, less creative)
        • >1.0 = flatter (more random, creative)
    - top_k: at each step, only consider the top-k most likely tokens.
        This prevents sampling from the long tail of unlikely tokens.
    - The model always outputs a full vocabulary distribution (logits).
      We sample from it rather than always picking argmax (greedy) to get
      varied, natural-sounding text.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def inspect_forward_pass(model, tokenizer, device: str, text: str = "Hello world"):
    """
    Run one forward pass and show the internal tensor shapes.
    
    This demystifies what happens inside the model on every token prediction.
    """
    print("\n" + "="*60)
    print("FORWARD PASS INSPECTION")
    print("="*60)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    print(f"\nInput text:    '{text}'")
    print(f"Token IDs:      {input_ids[0].tolist()}")
    print(f"Tokens:         {[tokenizer.decode([t]) for t in input_ids[0].tolist()]}")
    print(f"Input shape:    {input_ids.shape}  ← (batch_size, seq_len)")
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    
    logits = outputs.logits
    print(f"\nLogits shape:   {logits.shape}  ← (batch, seq_len, vocab_size)")
    print(f"  → For each of the {input_ids.shape[1]} input tokens,")
    print(f"    the model outputs a score for all {logits.shape[-1]:,} possible next tokens.")
    
    # Show what the model predicts after the last token
    last_logits = logits[0, -1, :]           # shape: (vocab_size,)
    probs = torch.softmax(last_logits, dim=-1)
    top5 = torch.topk(probs, 5)
    
    print(f"\nTop-5 predictions after '{text}':")
    for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        token_str = tokenizer.decode([idx])
        print(f"  {prob:.4f}  →  '{token_str}'")
    
    hidden = outputs.hidden_states
    print(f"\nHidden states:  {len(hidden)} tensors  ← (embedding + {len(hidden)-1} transformer layers)")
    print(f"  Each shape:   {hidden[0].shape}  ← (batch, seq_len, hidden_dim)")
    
    attn = outputs.attentions
    if attn and len(attn) > 0:
        print(f"\nAttention maps: {len(attn)} layers × {attn[0].shape[1]} heads")
        print(f"  Each shape:   {attn[0].shape}  ← (batch, heads, seq_len, seq_len)")
        print(f"  → This is the famous 'attention matrix' — each head learns")
        print(f"    which tokens should 'attend to' which other tokens.")
    else:
        print(f"\nAttention maps: not returned (try model.config.output_attentions = True before loading)")
    print("="*60)


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    
    inspect_forward_pass(model, tokenizer, device, "The capital of France is")
    
    print("\n--- Generation Example ---")
    prompt = "The best way to learn machine learning is"
    print(f"Prompt: {prompt}")
    result = generate(model, tokenizer, device, prompt, max_new_tokens=60)
    print(f"Output: {result}")