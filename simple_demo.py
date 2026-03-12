import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinearDemo(nn.Module):
    """
    Simplified Column Parallel Linear for single-process demo.
    In real distributed training, each rank holds a shard of the weight matrix.
    """
    def __init__(self, in_features, out_features, num_shards=2, bias=True):
        super().__init__()
        self.num_shards = num_shards
        self.in_features = in_features
        self.out_per_shard = out_features // num_shards
        
        # Simulate sharded weights (in real TP, each rank holds one shard)
        self.weight_shards = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, self.out_per_shard))
            for _ in range(num_shards)
        ])
        
        self.bias_shards = nn.ParameterList([
            nn.Parameter(torch.empty(self.out_per_shard))
            for _ in range(num_shards)
        ]) if bias else None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for w in self.weight_shards:
            nn.init.kaiming_uniform_(w, a=5**0.5)
        if self.bias_shards is not None:
            for b in self.bias_shards:
                fan_in = b.size(0)
                bound = 1.0 / (fan_in**0.5)
                nn.init.uniform_(b, -bound, bound)
    
    def forward(self, x):
        # Simulate parallel computation: each shard computes partial output
        partial_outputs = []
        for i in range(self.num_shards):
            y_local = x @ self.weight_shards[i]
            if self.bias_shards is not None:
                y_local = y_local + self.bias_shards[i]
            partial_outputs.append(y_local)
        
        # Gather: concatenate partial outputs (all_gather in distributed)
        y = torch.cat(partial_outputs, dim=-1)
        return y


def main():
    print("=== mini-trainer Simple Demo ===\n")
    print("This demonstrates Column Parallel Linear (Tensor Parallelism)\n")
    
    B, Din, Dout = 4, 8, 12
    num_shards = 2
    
    print(f"Batch size: {B}, Input dim: {Din}, Output dim: {Dout}")
    print(f"Number of shards (simulating GPUs): {num_shards}")
    print(f"Each shard handles: {Dout // num_shards} output features\n")
    
    layer = ColumnParallelLinearDemo(Din, Dout, num_shards=num_shards, bias=True)
    
    # Input
    x = torch.randn(B, Din, requires_grad=True)
    t = torch.randint(0, Dout, (B,))
    
    print(f"Input shape: {x.shape}")
    
    # Forward
    y = layer(x)
    print(f"Output shape: {y.shape}")
    
    # Loss and backward
    loss = F.cross_entropy(y, t)
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    
    print(f"\nGradients computed:")
    print(f"  Input grad: {x.grad is not None}")
    print(f"  Weight grads: {all(w.grad is not None for w in layer.weight_shards)}")
    print(f"  Bias grads: {all(b.grad is not None for b in layer.bias_shards)}")
    
    print("\n=== Demo Complete ===")
    print("\nIn real distributed training:")
    print("  - Each GPU would hold one weight shard")
    print("  - Communication happens via all_gather/all_reduce")
    print("  - This enables training larger models across multiple GPUs")


if __name__ == "__main__":
    main()
