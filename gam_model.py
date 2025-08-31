import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the GAM model."""
    VOCAB_SIZE: int = 10000
    BLOCK_SIZE: int = 256
    N_EMBED: int = 512
    N_LAYER: int = 6
    DROPOUT: float = 0.1
    # GAM specific parameters
    NUM_MEMORY_SLOTS: int = 512
    CONV_KERNEL_SIZE: int = 3

class GAMBlock(nn.Module):
    """
    A single block of the Gated Associative Memory (GAM) model.
    
    This block is designed as a parallelizable, O(N) complexity replacement for the
    standard self-attention mechanism in Transformers. It processes information
    through two parallel pathways:
    
    1.  **Local Context Pathway**: A causal depthwise convolution captures local
        syntactic and sequential patterns within a fixed window.
    2.  **Global Context Pathway**: An associative memory mechanism allows each token
        to retrieve information from a shared, trainable 'memory bank'. This captures
        long-range, semantic dependencies across the entire sequence.
        
    These two pathways are then dynamically fused using learned gates, allowing the
    model to decide how much local vs. global information to prioritize for each
    token. This is followed by a standard feed-forward network (FFN).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # --- Local Context Pathway ---
        # Causal depthwise convolution to efficiently capture local patterns.
        # Padding is set to (kernel_size - 1) to ensure the output sequence length
        # matches the input length after slicing, maintaining causality.
        self.causal_conv = nn.Conv1d(
            in_channels=config.N_EMBED, 
            out_channels=config.N_EMBED,
            kernel_size=config.CONV_KERNEL_SIZE, 
            padding=config.CONV_KERNEL_SIZE - 1,
            groups=config.N_EMBED  # Depthwise convolution
        )
        
        # --- Global Context Pathway ---
        # A trainable memory bank that stores global concepts.
        self.memory_bank = nn.Parameter(torch.randn(config.NUM_MEMORY_SLOTS, config.N_EMBED))
        nn.init.xavier_uniform_(self.memory_bank)
        
        # --- Gating and Fusion Mechanism ---
        # A linear layer to produce two gates that control the fusion of local and global contexts.
        self.gate = nn.Linear(config.N_EMBED, 2 * config.N_EMBED)
        
        # --- Standard FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(config.N_EMBED, 4 * config.N_EMBED),
            nn.GELU(),
            nn.Linear(4 * config.N_EMBED, config.N_EMBED),
            nn.Dropout(config.DROPOUT)
        )
        
        # --- Layer Normalization ---
        self.ln1 = nn.LayerNorm(config.N_EMBED)
        self.ln2 = nn.LayerNorm(config.N_EMBED)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GAMBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is batch size,
                              T is sequence length, and C is embedding dimension.
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Store residual for the first skip connection
        residual = x
        
        # Pre-normalization for the context fusion part
        x = self.ln1(x)
        
        # 1. Local Context via Causal Convolution
        # Permute from (B, T, C) to (B, C, T) for Conv1d
        x_permuted = x.permute(0, 2, 1)
        local_context = self.causal_conv(x_permuted)
        # Slice to remove extra padding and maintain sequence length, ensuring causality
        local_context = local_context[:, :, :x.shape[1]]
        # Permute back to (B, T, C)
        local_context = local_context.permute(0, 2, 1)
        
        # 2. Global Context via Memory Bank Retrieval
        # Calculate similarity scores between input tokens and memory slots
        scores = x @ self.memory_bank.T  # (B, T, C) @ (C, M) -> (B, T, M)
        # Normalize scores into weights
        weights = F.softmax(scores, dim=-1) # (B, T, M)
        # Retrieve information from memory as a weighted sum
        global_context = weights @ self.memory_bank # (B, T, M) @ (M, C) -> (B, T, C)
        
        # 3. Gating and Fusion
        # Generate gates from the input
        local_g, global_g = self.gate(x).chunk(2, dim=-1)
        # Fuse contexts using sigmoid gates
        fused_context = torch.sigmoid(local_g) * local_context + torch.sigmoid(global_g) * global_context
        
        # First residual connection
        x = residual + fused_context
        
        # Second residual connection with FFN
        x = x + self.ffn(self.ln2(x))
        
        return x


class GAM_Model(nn.Module):
    """
    The complete Gated Associative Memory (GAM) language model.
    
    This model stacks multiple GAMBlocks to form a deep neural network capable
    of learning complex patterns in sequential data. It follows the standard
    architecture of a decoder-only language model.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input embeddings
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED)
        self.pos_embedding = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Stack of GAMBlocks
        self.blocks = nn.ModuleList([GAMBlock(config) for _ in range(config.N_LAYER)])
        
        # Final layer normalization and output head
        self.ln_f = nn.LayerNorm(config.N_EMBED)
        self.head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False)
        
        # Weight tying between token embeddings and the final linear layer
        self.token_embedding.weight = self.head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass for the full GAM model.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).
            targets (torch.Tensor, optional): Target token indices of shape (B, T).
                                              If provided, computes cross-entropy loss.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                - logits (torch.Tensor): Output logits of shape (B, T, VOCAB_SIZE).
                - loss (torch.Tensor | None): The cross-entropy loss if targets are provided,
                                              otherwise None.
        """
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # (1, T)
        pos_emb = self.pos_embedding(pos) # (1, T, C)
        
        # Combine embeddings and apply dropout
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through the stack of GAMBlocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x) # (B, T, VOCAB_SIZE)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

# --- Demonstration ---
if __name__ == '__main__':
    # 1. Configuration
    config = ModelConfig(
        VOCAB_SIZE=10000,
        BLOCK_SIZE=256,
        N_EMBED=512,
        N_LAYER=6,
        DROPOUT=0.1,
        NUM_MEMORY_SLOTS=512,
        CONV_KERNEL_SIZE=3
    )

    # 2. Model Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GAM_Model(config).to(device)
    print(f"Model initialized on device: {device}")
    print(model)

    # 3. Parameter Count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # 4. Dummy Data and Forward Pass
    batch_size = 8
    seq_len = config.BLOCK_SIZE

    # Create random input and target tensors
    dummy_input = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len)).to(device)
    dummy_targets = torch.randint(0, config.VOCAB_SIZE, (batch_size, seq_len)).to(device)

    print(f"\nRunning a forward pass with dummy data...")
    print(f"Input shape: {dummy_input.shape}")

    # Perform a forward pass
    logits, loss = model(dummy_input, dummy_targets)

    # 5. Verify Output
    print(f"Logits output shape: {logits.shape}")
    print(f"Calculated loss: {loss.item():.4f}")

    # Check if backpropagation works
    loss.backward()
    print("Backward pass successful.")
