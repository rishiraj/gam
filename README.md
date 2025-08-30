# Gated Associative Memory (GAM)

[![arXiv](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX) <!-- Replace with your actual arXiv ID -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for the paper: **"Gated Associative Memory: A Parallel O(N) Architecture for Efficient Sequence Modeling"**.

GAM is a novel, fully parallel architecture for sequence modeling that exhibits linear complexity (O(N)) with respect to sequence length. It serves as a highly efficient and performant alternative to the standard Transformer's quadratic self-attention mechanism.

## Key Features

- **Linear Complexity:** Computational and memory costs scale linearly, O(N), with sequence length, making it ideal for processing long contexts.
- **Fully Parallelizable:** Designed without any recurrent components, GAM fully leverages the parallel processing power of modern hardware like GPUs.
- **Strong Performance:** Outperforms both a standard Transformer and a modern Mamba baseline in perplexity and training speed on benchmark datasets.
- **Hybrid Context Modeling:** Explicitly decomposes context modeling into two parallel pathways:
    1. A **Causal Convolution** to efficiently capture local, position-dependent patterns.
    2. A **Parallel Associative Memory** to retrieve global, content-based information from a learned memory bank.
- **Dynamic Fusion:** A learnable gating mechanism dynamically combines the local and global pathways for each token, allowing the model to flexibly prioritize different types of context.

## Architecture Overview

The core of the GAM network is the `GAMBlock`, which replaces the self-attention block in a Transformer. It processes an input by branching it into parallel local and global context pathways, which are then dynamically fused by a learned gate.


*Figure 1: The GAM Block architecture.*

## Results

GAM was benchmarked against a standard Transformer and a strong Mamba baseline on the WikiText-2 and TinyStories datasets. It consistently demonstrated faster training times and achieved superior or competitive final validation perplexity.

#### WikiText-2 Benchmark

| Model       | Params | Avg. Time / Epoch | Val. Perplexity (↓) |
|-------------|--------|-------------------|---------------------|
| Transformer | 24.2 M | 131.9 s           | 918.99              |
| Mamba       | 20.5 M | 127.1 s           | 1017.54             |
| **GAM (Ours)**  | **22.6 M** | **117.2 s**           | **882.57**              |

#### Scaling Analysis

The O(N) complexity of GAM provides significant advantages in compute time and memory usage as sequence length increases, whereas the Transformer's O(N²) complexity quickly becomes a bottleneck.


*Figure 2: Compute time and peak memory usage vs. sequence length. GAM scales linearly, while the Transformer's quadratic growth leads to OOM errors.*

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rishiraj/gam.git
   cd gam
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(A `requirements.txt` file should include `torch`, `datasets`, `tokenizers`, `mamba_ssm`, `tqdm`, `wandb`, `pandas`, `matplotlib`)*

## Usage

You can easily import and use the `GAM_Model` in your own projects. Here is a minimal example:

```python
import torch
from gam_model import GAM_Model, GAMConfig

# 1. Define the model configuration
config = GAMConfig(
    vocab_size=10000,
    block_size=256,
    n_embed=512,
    n_layer=6,
    num_memory_slots=512, # GAM-specific
    conv_kernel_size=3,   # GAM-specific
)

# 2. Instantiate the model
model = GAM_Model(config)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters.")

# 3. Create some dummy data
batch_size = 8
dummy_input = torch.randint(0, config.vocab_size, (batch_size, config.block_size))

# 4. Forward pass
logits, loss = model(dummy_input, targets=dummy_input)

print("Logits shape:", logits.shape)
print("Loss:", loss.item())
```

## Running the Benchmarks

This repository includes scripts to reproduce the results from the paper.

### Main Benchmark (`benchmark.py`)

This script trains and evaluates GAM, Transformer, and Mamba models on the WikiText-2 and TinyStories datasets. Results are logged using Weights & Biases.

1.  **Log in to W&B (optional but recommended):**
    ```bash
    wandb login
    ```
2.  **Run the script:**
    ```bash
    python benchmark.py
    ```
    You can modify hyperparameters and dataset settings directly within the `Config` class in the script.

### Scaling Benchmark (`scale.py`)

This script reproduces the scaling analysis (Figure 2 from the paper), measuring the compute time and peak memory usage of a single GAM block versus a Transformer block at various sequence lengths.

```bash
python scale.py
```
This will print a results table to the console and save a plot named `scaling_benchmark.png`.

## Citation

If you find this work useful in your research, please consider citing the paper:

```bibtex
@misc{acharya2025gated,
      title={Gated Associative Memory: A Parallel O(N) Architecture for Efficient Sequence Modeling}, 
      author={Rishiraj Acharya},
      year={2025},
      eprint={24XX.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

This project is licensed under the MIT License.
