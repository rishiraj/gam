import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GAMConfig:
    """Configuration for the GAM Model."""
    vocab_size: int = 10000
    block_size: int = 256
    n_embed: int = 512
    n_layer: int = 6
    dropout: float = 0.1
    # GAM specific
    num_memory_slots: int = 512
    conv_kernel_size: int = 3


class GAMBlock(nn.Module):
    """
    A single block of the Gated Associative Memory model.
    This block is designed as a parallel, O(N) replacement for self-attention.
    It combines a local context pathway (causal convolution) and a global
    context pathway (associative memory) using a learned gate.
    """
    def __init__(self, config: GAMConfig):
        super().__init__()
        
        # --- Local Context Pathway: Causal Convolution ---
        # Depthwise convolution to process each feature channel independently
        self.causal_conv = nn.Conv1d(
            in_channels=config.n_embed,
            out_channels=config.n_embed,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1, # Asymmetric padding for causality
            groups=config.n_embed
        )
        
        # --- Global Context Pathway: Associative Memory ---
        # Learnable memory bank of prototypical patterns
        self.memory_bank = nn.Parameter(torch.randn(config.num_memory_slots, config.n_embed))
        nn.init.xavier_uniform_(self.memory_bank)
        
        # --- Gating and Fusion Mechanism ---
        # Linear layer to compute gates for local and global pathways
        self.gate = nn.Linear(config.n_embed, 2 * config.n_embed)
        
        # --- Standard FFN and Layer Normalization ---
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.GELU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout)
        )
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GAM Block.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        B, T, C = x.shape
        
        # Residual connection from the original input
        residual = x
        
        x_norm = self.ln1(x)
        
        # --- 1. Local Context Pathway ---
        # Permute to (B, C, T) for Conv1d
        x_permuted = x_norm.permute(0, 2, 1)
        local_context = self.causal_conv(x_permuted)
        # Trim the padding to maintain sequence length T
        local_context = local_context[:, :, :T].permute(0, 2, 1)

        # --- 2. Global Context Pathway ---
        # (B, T, C) @ (C, M) -> (B, T, M) where M is num_memory_slots
        scores = x_norm @ self.memory_bank.T
        # Softmax over memory slots to get retrieval weights
        weights = F.softmax(scores, dim=-1)
        # (B, T, M) @ (M, C) -> (B, T, C)
        global_context = weights @ self.memory_bank

        # --- 3. Gating and Fusion ---
        # Compute gates from the normalized input
        local_gate, global_gate = self.gate(x_norm).chunk(2, dim=-1)
        
        # Modulate pathways with sigmoid gates
        fused_context = torch.sigmoid(local_gate) * local_context + \
                        torch.sigmoid(global_gate) * global_context

        # Add the fused context to the initial residual connection
        x = residual + fused_context
        
        # Apply the FFN with another residual connection
        x = x + self.ffn(self.ln2(x))
        
        return x


class GAM_Model(nn.Module):
    """
    The Gated Associative Memory (GAM) language model.
    """
    def __init__(self, config: GAMConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([GAMBlock(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding.weight = self.head.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass for the GAM Model.
        Args:
            idx (torch.Tensor): Input sequence of token indices, shape (B, T)
            targets (torch.Tensor, optional): Target sequence, shape (B, T). Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Logits and loss. Loss is None if targets are not provided.
        """
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through GAM blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
