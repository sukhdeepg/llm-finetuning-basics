# LLM Fine-Tuning

A project for understanding how language models work
and how fine-tuning changes them. No abstraction layers, just PyTorch and
HuggingFace Transformers.

<img width="1511" height="1619" alt="image" src="https://github.com/user-attachments/assets/677431c4-33a2-4065-9697-301fff5ccaf2" />
<img width="1671" height="1083" alt="image" src="https://github.com/user-attachments/assets/22c1b00e-60f7-42cf-9852-979eb0a973e1" />
<img width="1741" height="1421" alt="image" src="https://github.com/user-attachments/assets/7090c34d-37f9-437f-8064-9cce88ec807a" />
<img width="1798" height="983" alt="image" src="https://github.com/user-attachments/assets/c19febaa-0c71-4985-a0d2-12050f95f6f1" />

---

## Learning

| Concept | File | Key idea |
|---|---|---|
| Model architecture | `model.py` | What GPT-2 looks like internally |
| Forward pass | `model.py` → `inspect_forward_pass()` | Shapes, logits, attention maps |
| Data preparation | `data.py` | How text becomes training signal |
| Training loop | `train.py` | Forward → Loss → Backward → Step |
| Behavior change | `infer.py` | Before vs after comparison |

---

## Setup

```bash
pip install -r requirements.txt
```

First run downloads GPT-2 (~500MB, cached afterward).

---

## Run in order

### Step 1: Understanding the model

```bash
python model.py
```

Prints the architecture, runs a forward pass, and shows internal tensor
shapes. Reading the output carefully is important because that is the whole model in one screen.

**What to notice:**
- `Logits shape: (1, 4, 50257)`: for 4 input tokens, 50,257 next-token scores each
- The top-5 predictions after "The capital of France is": these are just
  probabilities over the vocabulary
- Attention maps: `(1, 12, 4, 4)`: 12 heads, each producing a 4×4 matrix
  of "which tokens attend to which"

---

### Step 2: Understanding the data

```bash
python data.py
```

Shows what the model actually sees as input/output during training.

**Key insight:** Fine-tuning is just more next-token prediction. The model
sees `"Q: What is a token?\nA: A token is..."` and learns that after this
Q: prefix, this kind of A: should follow.

---

### Step 3: Fine-tune

```bash
python train.py
```

Runs 10 epochs (~1-2 minutes locally). Watch the loss column:

```
Epoch   1/10  |  loss: 3.4821  |  lr: 5.00e-05  |  time: 8.3s
Epoch   2/10  |  loss: 2.9104  |  ...
...
Epoch  10/10  |  loss: 0.8231  |  ...
```

Loss dropping = model getting better at predicting our training text.

**Before vs after is printed automatically**: we'll see raw GPT-2 ramble
vs. fine-tuned GPT-2 give a structured answer.

---

### Step 4: Compare and probe

```bash
python infer.py
```

Side-by-side comparison, memorization probe, and an interactive REPL.

---

### Why fine-tuning works

GPT-2 was pretrained to predict next tokens across ~40GB of web text.
It learned: grammar, facts, reasoning patterns, many writing styles.

When we fine-tune on Q&A pairs:
- The model doesn't forget what it learned (weights change slowly)
- It learns a new distribution: "after `Q: ... \nA:`, produce this style"
- A little data goes a long way because the foundation is already there

### The training loop (one iteration)

```python
outputs = model(input_ids=x, labels=x)  # forward pass
loss = outputs.loss                       # cross-entropy vs ground truth
loss.backward()                           # compute gradients
optimizer.step()                          # update all 124M weights
optimizer.zero_grad()                     # clear gradients for next step
```

That's it. Everything else is scaffolding.

---

## Experiment ideas

1. **Change the dataset**: Edit `QA_CORPUS` in `data.py`. Add 5 new Q&A
   pairs in a different domain. Re-run `train.py`. How quickly does it adapt?

2. **Change the learning rate**: Try `1e-4` (too high, loss bounces) and
   `1e-6` (too low, barely moves). See the effect directly.

3. **Change the temperature**: In `infer.py`, try `temperature=0.1` vs
   `temperature=1.5`. Same model, very different outputs.

4. **Freeze layers**: Add this before training to only update the last 2 layers:
   ```python
   for name, param in model.named_parameters():
       if not name.startswith("transformer.h.1"):
           param.requires_grad = False
   ```
   This is the core idea behind parameter-efficient fine-tuning (PEFT/LoRA).

5. **Use a real dataset**: Replace `QA_CORPUS` with HuggingFace datasets:
   ```python
   from datasets import load_dataset
   ds = load_dataset("tatsu-lab/alpaca", split="train[:200]")
   ```

---

## File map

```
llm-finetuning-basics/
├── requirements.txt   # dependencies
├── model.py           # load, inspect, generate — start here
├── data.py            # dataset and tokenization
├── train.py           # training loop
├── infer.py           # compare base vs fine-tuned
├── checkpoints/       # saved during training
└── README.md          # this file
```

---

## Going further

The natural next steps are:

- **LoRA / QLoRA**: Fine-tune only a tiny fraction of weights (rank decomposition).
  Same result, 10× less memory. See `peft` library.
- **LLaMA 3.2 (1B)**: A modern architecture locally. Use `mlx-lm` for
  Apple Silicon native inference.
- **Instruction tuning**: Fine-tuning on (instruction, response) pairs.
  This is how ChatGPT-style models are made.
- **RLHF**: Reinforcement learning from human feedback. Adds a reward model
  on top of fine-tuning.
