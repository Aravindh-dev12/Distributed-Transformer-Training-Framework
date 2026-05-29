import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import numpy as np

class LLMTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        learning_rate=3e-4,
        seq_len=32,
        batch_size=1,
        gradient_accumulation_steps=1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.grad_acc_steps = gradient_accumulation_steps
        
        self.model.to(device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.global_step = 0
        self.tokens_processed = 0

    def prepare_dataset(self, dataset_source: str, num_samples: int = 100):
        """
        Creates tokenized chunks from either a Hugging Face dataset name, a path to a raw txt file, 
        or an inline sample text.
        """
        print(f"Preparing dataset from: {dataset_source}...")
        
        raw_text = ""
        
        # Check if source is a local file
        if os.path.exists(dataset_source) and os.path.isfile(dataset_source):
            with open(dataset_source, "r", encoding="utf-8") as f:
                raw_text = f.read()
        elif dataset_source.startswith("hf:"):
            # Load from HF Datasets, e.g. "hf:roneneldan/TinyStories"
            hf_path = dataset_source.split("hf:")[-1]
            try:
                ds = load_dataset(hf_path, split="train", streaming=True)
                # Read a few samples
                texts = []
                for i, item in enumerate(ds):
                    if i >= num_samples:
                        break
                    texts.append(item.get("text", ""))
                raw_text = "\n\n".join(texts)
            except Exception as e:
                print(f"Error loading Hugging Face dataset: {e}. Falling back to default corpus.")
                raw_text = self._get_fallback_text()
        else:
            # Inline raw text or fallback
            if len(dataset_source.strip()) > 50:
                raw_text = dataset_source
            else:
                raw_text = self._get_fallback_text()

        # Tokenize the corpus
        print("Tokenizing corpus...")
        tokenized = self.tokenizer.encode(raw_text, add_special_tokens=True)
        
        # Chunk into sequence_length + 1
        sequence_length = self.seq_len
        chunks = []
        for i in range(0, len(tokenized) - sequence_length, sequence_length):
            chunk = tokenized[i : i + sequence_length + 1]
            if len(chunk) == sequence_length + 1:
                chunks.append(chunk)
                
        print(f"Dataset prepared! Total sequence chunks: {len(chunks)}")
        return chunks

    def train_step(self, batch_chunks):
        """
        Performs a single gradient step. Handles batching and gradient accumulation.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        accumulated_loss = 0.0
        
        # Loop through gradient accumulation steps
        for step in range(self.grad_acc_steps):
            # Select chunk slice for this micro-batch
            start_idx = step * self.batch_size
            end_idx = start_idx + self.batch_size
            
            # Pad or slice if needed
            micro_batch = batch_chunks[start_idx:end_idx]
            if not micro_batch:
                continue
                
            # Prepare tensor data
            tensor_batch = torch.tensor(micro_batch, dtype=torch.long, device=self.device)
            input_ids = tensor_batch[:, :-1].contiguous()
            target_ids = tensor_batch[:, 1:].contiguous()
            
            # Forward pass
            outputs = self.model(input_ids=input_ids) # [batch, seq, vocab]
            
            # Compute loss
            b, s = input_ids.shape
            outputs = outputs.view(b * s, -1)
            target_ids = target_ids.reshape(-1)
            
            loss = F.cross_entropy(outputs, target_ids, reduction="mean") / self.grad_acc_steps
            loss.backward()
            
            accumulated_loss += loss.item() * self.grad_acc_steps
            self.tokens_processed += b * s
            
        self.optimizer.step()
        self.global_step += 1
        
        return accumulated_loss

    def _get_fallback_text(self):
        return """
Distributed systems allow multiple computer networks to collaborate and compute large workloads together.
Transformer neural networks are highly scalable attention-based models that form the backbone of modern Generative AI.
Pre-training involves training large language models on large corpora of text datasets, teaching them syntax, logic, and base knowledge.
Fine-tuning adapts these models to specific downstream tasks, like customer support, coding assistance, or instruction following.
This framework is an advanced, intelligent tool that makes it incredibly easy to load, adapt, and serve open-source LLMs.
"""

    def fit_generator(self, dataset_source: str, max_steps: int = 50, callback=None):
        """
        An active generator that runs the fine-tuning loop and yields metrics step-by-step.
        """
        chunks = self.prepare_dataset(dataset_source)
        if not chunks:
            yield {"status": "error", "message": "Dataset preparation failed."}
            return
            
        step = 0
        total_chunks = len(chunks)
        batch_capacity = self.batch_size * self.grad_acc_steps
        
        chunk_idx = 0
        
        start_time = time.time()
        
        while step < max_steps:
            # Check if we ran out of chunks and loop them
            if chunk_idx + batch_capacity > total_chunks:
                chunk_idx = 0
                
            batch_chunks = chunks[chunk_idx : chunk_idx + batch_capacity]
            if len(batch_chunks) < batch_capacity:
                chunk_idx = 0
                continue
                
            chunk_idx += batch_capacity
            
            # Perform training step
            step_start = time.time()
            loss = self.train_step(batch_chunks)
            step_duration = time.time() - step_start
            
            step += 1
            
            # Log metrics
            tokens_per_sec = (batch_capacity * self.seq_len) / step_duration
            elapsed = time.time() - start_time
            
            metrics = {
                "step": step,
                "max_steps": max_steps,
                "loss": round(loss, 4),
                "speed": f"{tokens_per_sec:.1f} tokens/s",
                "tokens": self.tokens_processed,
                "elapsed": f"{elapsed:.1f}s",
                "memory": f"{torch.cuda.memory_reserved() / 1e9:.2f}GB" if torch.cuda.is_available() else "0.00GB"
            }
            
            if callback:
                callback(metrics)
                
            yield metrics
