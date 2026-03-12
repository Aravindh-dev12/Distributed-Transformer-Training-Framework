# Distributed-Transformer-Training-Framework

A comprehensive framework for distributed transformer training from scratch, exploring **data**, **tensor**, and **pipeline parallel** training for Transformers — inspired by minigpt and picotron.

## What is this project?

**Distributed-Transformer-Training-Framework** is an educational implementation of distributed transformer training from scratch. It demonstrates how large language models (LLMs) are trained across multiple GPUs using advanced parallelism techniques. This project is designed for learning and understanding the internals of distributed training systems.

## Key Features

- **Data Parallelism (DP)**: Replicate the model across multiple GPUs and split the training data
- **Tensor Parallelism (TP)**: Split individual model layers/tensors across multiple GPUs
- **Pipeline Parallelism (PP)**: Split model layers across multiple GPUs in a pipeline
- **Complete LLaMA-style Transformer**: Full implementation with attention, MLP, and normalization layers
- **Rotary Position Embeddings (RoPE)**: Modern positional encoding for transformers
- **Grouped Query Attention (GQA)**: Efficient attention mechanism used in current LLMs

## Project Structure

```
Distributed-Transformer-Training-Framework/
├── examples/                    # Tensor parallelism demos
│   ├── column_parallel_linear_demo.py
│   ├── row_parallel_linear_demo.py
│   └── README.md
├── 3d_parallel/                 # Step-by-step implementation
│   ├── step1_modelling/        # Base transformer model
│   ├── step2_PGM/             # Parallel Group Management
│   ├── step3_dataloader/      # Distributed data loading
│   ├── step4_TP/              # Tensor Parallelism
│   └── step5_DP/              # Data Parallelism
├── pyproject.toml             # Dependencies
└── README.md
```

## Use Cases

### Educational Purposes
- **Learn distributed training concepts**: Understand how large models like GPT-4, Claude, and Llama are trained
- **Study parallelism techniques**: Deep dive into data, tensor, and pipeline parallelism
- **Research and experimentation**: Test new parallelism strategies and optimizations

### Practical Applications
- **Custom training pipelines**: Build your own distributed training system
- **Resource optimization**: Learn to maximize GPU utilization for training
- **Model scaling**: Understand how to scale models across multiple GPUs/nodes

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Running the Project

### Option 1: Single-Process Training (Windows Compatible)

For Windows users or quick testing without distributed setup:

```bash
# Navigate to project directory
cd Distributed-Transformer-Training-Framework

# Run with default settings (32 layers, 16 heads)
uv run python 3d_parallel/step1_modelling/train_single.py

# Run with small model (faster for testing)
uv run python 3d_parallel/step1_modelling/train_single.py --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 2 --seq_len 16

# Run with medium model (balanced)
uv run python 3d_parallel/step1_modelling/train_single.py --num_hidden_layers 4 --num_attention_heads 8 --num_key_value_heads 2 --seq_len 32

# Run with custom batch size and learning rate
uv run python 3d_parallel/step1_modelling/train_single.py --num_hidden_layers 2 --micro_batch_size 2 --learning_rate 1e-4 --seq_len 32

# Run simple tensor parallelism demo
uv run python simple_demo.py
```

### Option 2: Distributed Training (Linux + GPU)

For full distributed training with multiple GPUs (Linux only):

```bash
# Column parallel linear demo (3 GPUs)
uv run torchrun --nproc_per_node=3 examples/column_parallel_linear_demo.py

# Row parallel linear demo (4 GPUs)
uv run torchrun --nproc_per_node=4 examples/row_parallel_linear_demo.py

# Full distributed training with 4 GPUs
uv run torchrun --nproc_per_node=4 3d_parallel/step1_modelling/train.py --num_hidden_layers 32 --num_attention_heads 16

# Distributed training with custom parameters
uv run torchrun --nproc_per_node=8 3d_parallel/step1_modelling/train.py --num_hidden_layers 64 --num_attention_heads 32 --num_key_value_heads 8 --seq_len 512 --micro_batch_size 4
```

## Training Parameters

- `--num_hidden_layers`: Number of transformer layers (default: 32)
- `--num_attention_heads`: Number of attention heads (default: 16)
- `--num_key_value_heads`: Number of KV heads for GQA (default: 4)
- `--seq_len`: Sequence length (default: 32)
- `--micro_batch_size`: Batch size (default: 1)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--model_name`: HuggingFace model config (default: HuggingFaceTB/SmolLM-360M-Instruct)

## Technical Details

### Model Architecture
- **LLaMA-style transformer** with rotary embeddings
- **Grouped Query Attention (GQA)** for efficient inference
- **RMSNorm** instead of LayerNorm
- **SwiGLU** activation function in MLP

### Parallelism Strategies
1. **Data Parallelism**: Each GPU has a full model copy, processes different data batches
2. **Tensor Parallelism**: Model layers are split across GPUs, communication via all_reduce/all_gather
3. **Pipeline Parallelism**: Model stages are distributed across GPUs in a pipeline

## Windows Limitations

The distributed examples require Linux with GPU support due to:
- **NCCL backend**: Only available on Linux for GPU communication
- **GLOO backend**: Limited support on Windows for CPU communication
- **TCPStore/libuv**: Compatibility issues on Windows

The `train_single.py` script provides a Windows-compatible single-process version for testing the core modeling concepts.

## Dependencies

- PyTorch >= 2.9.0
- Transformers >= 4.57.1
- Datasets >= 4.4.1
- NumPy >= 2.3.4
- Rich >= 14.2.0
- WandB >= 0.23.0 (optional, for logging)

## License

This is an educational project. Please refer to the original licenses of the dependencies used.