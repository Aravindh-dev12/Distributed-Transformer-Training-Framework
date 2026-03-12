"""
Single-process version of train.py for Windows testing
"""
import os
import datetime
import torch
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from transformers import AutoConfig

from model import Llama
from utils import set_all_seed, print

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Training script for LLaMA model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=1)

    # Logging arguments
    parser.add_argument("--run_name", type=str, default="default_run")
    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    device = torch.device("cpu")
    dtype = torch.bfloat16

    set_all_seed(args.seed)

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    print(f"Loading model with {args.num_hidden_layers} layers, {args.num_attention_heads} heads...")
    model = Llama(config=model_config)
    model.to(dtype).to(device)            
    model.train()
    
    print(f"Model loaded. Vocab size: {model_config.vocab_size}, Hidden size: {model_config.hidden_size}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Create dummy data
    input_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_len), device=device)
    target_ids = torch.randint(0, model_config.vocab_size, (args.micro_batch_size, args.seq_len), device=device)

    print(f"Training step with batch_size={args.micro_batch_size}, seq_len={args.seq_len}")
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    # Compute loss
    target_ids = target_ids.reshape(-1)
    outputs = outputs.view(-1, model_config.vocab_size)
    loss = F.cross_entropy(outputs, target_ids)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print(f"Output shape: {outputs.shape}")
    print("Training step completed successfully!")
