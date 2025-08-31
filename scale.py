import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- Configuration ---
# We use a config class to keep parameters organized, similar to the original script.
class Config:
    # Model parameters are kept constant to isolate the effect of sequence length
    N_EMBED = 512
    N_HEAD = 8
    DROPOUT = 0.1
    BATCH_SIZE = 16 # A realistic batch size for the benchmark

    # GAM specific
    NUM_MEMORY_SLOTS = 512
    CONV_KERNEL_SIZE = 3

    # Benchmark parameters
    WARMUP_ITERATIONS = 5
    TIMED_ITERATIONS = 20
    # Sequence lengths to test. We go high to see the scaling effect.
    SEQUENCE_LENGTHS_TO_TEST = [256, 512, 1024, 2048, 4096, 8192]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# --- Model Implementations (Copied from your original script) ---

# A. Gated Associative Memory (GAM) Network
class GAMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Note: BLOCK_SIZE is not needed in the constructor, making it flexible.
        self.causal_conv = nn.Conv1d(
            in_channels=config.N_EMBED, out_channels=config.N_EMBED,
            kernel_size=config.CONV_KERNEL_SIZE, padding=config.CONV_KERNEL_SIZE - 1,
            groups=config.N_EMBED
        )
        self.memory_bank = nn.Parameter(torch.randn(config.NUM_MEMORY_SLOTS, config.N_EMBED))
        nn.init.xavier_uniform_(self.memory_bank)
        self.gate = nn.Linear(config.N_EMBED, 2 * config.N_EMBED)
        self.ffn = nn.Sequential(
            nn.Linear(config.N_EMBED, 4 * config.N_EMBED), nn.GELU(),
            nn.Linear(4 * config.N_EMBED, config.N_EMBED), nn.Dropout(config.DROPOUT)
        )
        self.ln1 = nn.LayerNorm(config.N_EMBED)
        self.ln2 = nn.LayerNorm(config.N_EMBED)
    def forward(self, x):
        res = x
        x = self.ln1(x)
        x_permuted = x.permute(0, 2, 1)
        local_context = self.causal_conv(x_permuted)
        # Trim the conv output to match the input sequence length
        local_context = local_context[:, :, :x.shape[1]].permute(0, 2, 1)
        scores = x @ self.memory_bank.T
        weights = F.softmax(scores, dim=-1)
        global_context = weights @ self.memory_bank
        local_g, global_g = self.gate(x).chunk(2, dim=-1)
        fused_context = torch.sigmoid(local_g) * local_context + torch.sigmoid(global_g) * global_context
        x = res + fused_context
        x = x + self.ffn(self.ln2(x))
        return x

# B. Transformer Model
class MultiHeadAttention(nn.Module):
    def __init__(self, config, block_size):
        super().__init__()
        assert config.N_EMBED % config.N_HEAD == 0
        self.n_head = config.N_HEAD
        self.head_dim = config.N_EMBED // config.N_HEAD
        self.c_attn = nn.Linear(config.N_EMBED, 3 * config.N_EMBED)
        self.c_proj = nn.Linear(config.N_EMBED, config.N_EMBED)
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.resid_dropout = nn.Dropout(config.DROPOUT)
        # The causal mask buffer depends on block_size, so we pass it in
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.N_EMBED)
        self.attn = MultiHeadAttention(config, block_size) # Pass block_size here
        self.ln_2 = nn.LayerNorm(config.N_EMBED)
        self.ffn = nn.Sequential(nn.Linear(config.N_EMBED, 4 * config.N_EMBED), nn.GELU(), nn.Linear(4 * config.N_EMBED, config.N_EMBED), nn.Dropout(config.DROPOUT))
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

# --- Benchmarking Framework ---

def run_single_benchmark(model_class, config, seq_len):
    """
    Measures the forward/backward time and peak memory for a given model class.
    Returns (avg_time_ms, peak_mem_mb).
    Returns (inf, inf) on CUDA OOM error.
    """
    # TransformerBlock requires block_size at initialization for the causal mask
    if model_class == TransformerBlock:
        model = model_class(config, block_size=seq_len).to(config.DEVICE)
    else:
        model = model_class(config).to(config.DEVICE)
    
    model.train() # Use train mode to include gradient calculations

    # Create dummy input data
    dummy_input = torch.randn(
        config.BATCH_SIZE, seq_len, config.N_EMBED,
        device=config.DEVICE, dtype=torch.float16
    )
    
    # Use AMP for more realistic performance and to fit larger models
    scaler = torch.cuda.amp.GradScaler()

    try:
        # Warm-up iterations
        for _ in range(config.WARMUP_ITERATIONS):
            with torch.cuda.amp.autocast():
                output = model(dummy_input)
                loss = output.sum() # Dummy loss
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)

        # Clear cache and reset stats before timing
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(config.DEVICE)
        
        # Timed iterations
        total_time = 0
        for _ in range(config.TIMED_ITERATIONS):
            start_time = time.time()
            
            with torch.cuda.amp.autocast():
                output = model(dummy_input)
                loss = output.sum()
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            
            torch.cuda.synchronize() # Wait for all GPU operations to finish
            end_time = time.time()
            total_time += (end_time - start_time)

        avg_time_ms = (total_time / config.TIMED_ITERATIONS) * 1000
        peak_mem_mb = torch.cuda.max_memory_allocated(config.DEVICE) / (1024 * 1024)

        # Clean up memory
        del model, dummy_input, output, loss
        torch.cuda.empty_cache()

        return avg_time_ms, peak_mem_mb

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"    - OOM Error for {model_class.__name__} at sequence length {seq_len}.")
            del model, dummy_input
            torch.cuda.empty_cache()
            return float('inf'), float('inf')
        else:
            raise e


def plot_results(df):
    """Generates and displays plots for the benchmark results."""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Time vs. Sequence Length
    sns.lineplot(data=df, x='Sequence Length', y='Time (ms)', hue='Model', marker='o', ax=ax1)
    ax1.set_title('Compute Time vs. Sequence Length')
    ax1.set_ylabel('Avg. Fwd+Bwd Time (ms)')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    ax1.legend(title='Model')

    # Plot 2: Memory vs. Sequence Length
    sns.lineplot(data=df, x='Sequence Length', y='Peak Memory (MB)', hue='Model', marker='o', ax=ax2)
    ax2.set_title('Peak GPU Memory vs. Sequence Length')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_xscale('log', base=2)
    # Memory for Transformer scales quadratically, so log-log plot should be linear
    ax2.set_yscale('log', base=10)
    ax2.legend(title='Model')

    fig.suptitle(f'GAM vs. Transformer Scaling Benchmark (Batch Size: {config.BATCH_SIZE}, Embed Dim: {config.N_EMBED})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    if config.DEVICE != 'cuda':
        print("This benchmark requires a CUDA-enabled GPU. Exiting.")
        return

    print(f"--- Starting Scaling Benchmark on {config.DEVICE} ---")
    print(f"Config: Batch={config.BATCH_SIZE}, Embed Dim={config.N_EMBED}, Heads={config.N_HEAD}")
    print(f"Sequence Lengths to test: {config.SEQUENCE_LENGTHS_TO_TEST}\n")

    results = []
    models_to_test = [("GAM", GAMBlock), ("Transformer", TransformerBlock)]

    for seq_len in tqdm(config.SEQUENCE_LENGTHS_TO_TEST, desc="Testing Sequence Lengths"):
        print(f"\n--- Testing Sequence Length: {seq_len} ---")
        for model_name, model_class in models_to_test:
            print(f"  Benchmarking {model_name}...")
            
            time_ms, mem_mb = run_single_benchmark(model_class, config, seq_len)
            
            if time_ms != float('inf'):
                print(f"    - Avg Time: {time_ms:.2f} ms")
                print(f"    - Peak Memory: {mem_mb:.2f} MB")
            
            results.append({
                'Model': model_name,
                'Sequence Length': seq_len,
                'Time (ms)': time_ms,
                'Peak Memory (MB)': mem_mb
            })

    # Create and display a DataFrame
    results_df = pd.DataFrame(results)
    print("\n\n--- Benchmark Results ---")
    # Pivot for easier comparison
    pivoted_df = results_df.pivot(index='Sequence Length', columns='Model')
    print(pivoted_df.round(2))
    
    # Plot the results
    plot_results(results_df)

if __name__ == '__main__':
    main()