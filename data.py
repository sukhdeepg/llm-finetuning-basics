"""
data.py - How to prepare data for fine-tuning

MENTAL MODEL FOR FINE-TUNING DATA:
  Fine-tuning teaches the model a NEW distribution on top of its existing knowledge.
  The model already knows English, grammar, reasoning, we're just steering it.

  The training signal is simple:
    Given tokens [t1, t2, t3, ... tN], predict [t2, t3, t4, ... tN+1]
    i.e. at every position, predict the NEXT token.
    Loss = how wrong the predictions were (cross-entropy).

DATASET CHOICES IN THIS FILE:
  We use a tiny hand-crafted Q&A dataset. This is intentional:
  - It fits in memory
  - We can read every training example
  - Changes in behavior are traceable to specific examples
  - Perfect for building intuition before scaling up
"""

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

"""
The training corpus
# We'll fine-tune GPT-2 to answer questions in a specific style.
# Format: "Q: {question}\nA: {answer}<|endoftext|>"
# The <|endoftext|> token tells the model "response is complete here".
"""

QA_CORPUS = [
    ("What is a neural network?",
     "A neural network is a function with millions of parameters that transforms numbers into numbers. Training adjusts those parameters so the function becomes useful."),
    
    ("What is a transformer?",
     "A transformer is a neural network architecture that uses attention mechanisms to process sequences. It reads all tokens simultaneously, not one by one."),
    
    ("What is attention in machine learning?",
     "Attention lets a model decide which parts of the input to focus on when producing each output token. Each token looks at all other tokens and assigns them relevance scores."),
    
    ("What is a token?",
     "A token is the basic unit of text a language model processes. It is roughly a word fragment, about 4 characters on average. 'Hello world' is 2 tokens."),
    
    ("What is fine-tuning?",
     "Fine-tuning is continuing to train a pretrained model on a smaller, specific dataset. The model adapts its weights to the new distribution while retaining general knowledge."),
    
    ("What is a loss function?",
     "A loss function measures how wrong the model's predictions are. Training minimizes this value. For language models the loss is cross-entropy over next-token predictions."),
    
    ("What is a gradient?",
     "A gradient is the direction and magnitude of how much each parameter should change to reduce the loss. Backpropagation computes it efficiently using the chain rule."),
    
    ("What is a learning rate?",
     "The learning rate controls how large a step to take in the direction of the gradient. Too high and training diverges. Too low and training is very slow."),
    
    ("What is overfitting?",
     "Overfitting happens when a model memorizes training data instead of learning generalizable patterns. It performs well on training data but poorly on unseen data."),
    
    ("What is temperature in text generation?",
     "Temperature scales the model's output logits before sampling. Low temperature makes the model pick likely tokens. High temperature makes output more random and creative."),
    
    ("What is a hyperparameter?",
     "A hyperparameter is a setting chosen before training that is not updated by gradient descent. Learning rate, batch size, and number of layers are common hyperparameters."),
    
    ("What is backpropagation?",
     "Backpropagation computes gradients by applying the chain rule from the loss backward through every layer. It tells us exactly how each weight contributed to the error."),
    
    ("What is an embedding?",
     "An embedding maps discrete tokens to dense vectors in a continuous space. Semantically similar tokens end up with similar vectors after training."),
    
    ("What is a batch in training?",
     "A batch is a group of training examples processed together in one forward and backward pass. Larger batches give more stable gradient estimates but use more memory."),
    
    ("What is weight initialization?",
     "Weight initialization sets the starting values of model parameters. Good initialization prevents gradients from vanishing or exploding at the start of training."),
]


def format_example(question: str, answer: str) -> str:
    """
    The prompt template matters enormously in fine-tuning.
    During inference we'll use the same template (minus the answer)
    so the model knows it's in 'answer mode'.
    """
    return f"Q: {question}\nA: {answer}<|endoftext|>"


class QADataset(Dataset):
    """
    A minimal PyTorch Dataset that:
    1. Formats each QA pair into a string
    2. Tokenizes it
    3. Creates (input_ids, labels) pairs for causal LM training

    In causal LM training:
      input_ids = [t1, t2, t3, t4]
      labels    = [t2, t3, t4, t5]  ← shifted by 1

    HuggingFace's GPT2LMHeadModel does this shift internally —
    we just pass labels = input_ids and it handles it.
    """
    
    def __init__(self, tokenizer: GPT2Tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for question, answer in QA_CORPUS:
            text = format_example(question, answer)
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            
            # Labels: same as input_ids, but -100 on padding tokens
            # -100 is PyTorch's convention for "don't compute loss here"
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        
        print(f"[data] Dataset ready: {len(self.examples)} examples, "
              f"max_length={max_length}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def get_dataloader(tokenizer: GPT2Tokenizer, batch_size: int = 4) -> DataLoader:
    dataset = QADataset(tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def preview_dataset(tokenizer: GPT2Tokenizer):
    """Print what the model actually sees during training."""
    print("\n" + "="*60)
    print("DATASET PREVIEW (first 2 examples)")
    print("="*60)
    
    dataset = QADataset(tokenizer, max_length=128)
    
    for i in range(min(2, len(dataset))):
        item = dataset[i]
        ids = item["input_ids"]
        mask = item["attention_mask"]
        
        # Only show non-padded tokens
        real_ids = ids[mask == 1]
        text = tokenizer.decode(real_ids, skip_special_tokens=False)
        
        print(f"\nExample {i+1}:")
        print(f"  Text:   {text!r}")
        print(f"  Tokens: {len(real_ids)} real + {(mask==0).sum().item()} padding")
        print(f"  IDs:    {real_ids[:10].tolist()}...")
    print("="*60)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    preview_dataset(tokenizer)
