import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import math
import time
import os
from tqdm import tqdm
import wandb
from mamba_ssm import Mamba


# --- Configuration ---
class Config:
    # --- Wandb Configuration ---
    WANDB_PROJECT = "GAM_vs_Transformer_vs_Mamba_Test"
    WANDB_ENTITY = None # Your wandb username or team name, or None

    # --- Dataset Configuration ---
    # List of datasets to run experiments on.
    # Each dict must have 'name' and 'config'. 'config' can be None.
    DATASETS_TO_RUN = [
        {'name': "wikitext", 'config': "wikitext-2-raw-v1"},
        {'name': "roneneldan/TinyStories", 'config': None}
    ]
    # Trim datasets to prevent memory issues. Set to None to use the full dataset.
    MAX_TRAIN_ROWS = 50000
    MAX_VAL_ROWS = 5000

    # --- Model & Training Configuration ---
    VOCAB_SIZE = 10000
    BLOCK_SIZE = 256

    N_EMBED = 512
    N_HEAD = 8
    N_LAYER = 6
    DROPOUT = 0.1

    # GAM specific
    NUM_MEMORY_SLOTS = 512
    CONV_KERNEL_SIZE = 3 # Kernel for local context

    # Mamba specific
    D_STATE = 16
    D_CONV = 4
    EXPAND = 3

    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    EPOCHS = 5 # Reduced for quicker testing, can be increased
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Scheduler params
    WARMUP_STEPS = 100

config = Config()
print(f"Using device: {config.DEVICE}")

# --- 1. Data Preparation ---
def prepare_data(dataset_name, dataset_config, max_rows):
    print(f"--- Preparing Data for {dataset_name} ---")

    # Load and trim the dataset first to save memory
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)

    print("Trimming dataset...")
    if max_rows['train'] and 'train' in dataset:
        num_rows = min(max_rows['train'], len(dataset['train']))
        dataset['train'] = dataset['train'].select(range(num_rows))
    if max_rows['validation'] and 'validation' in dataset:
        num_rows = min(max_rows['validation'], len(dataset['validation']))
        dataset['validation'] = dataset['validation'].select(range(num_rows))

    print(f"Train split size after trimming: {len(dataset['train'])}")
    print(f"Validation split size after trimming: {len(dataset['validation'])}")


    def get_training_corpus():
        # Iterate over all available splits for tokenizer training
        return (dataset[split]['text'] for split in dataset if dataset[split].num_rows > 0 and 'text' in dataset[split].column_names)

    tokenizer_filename = dataset_name.replace('/', '_') # Handle names like 'roneneldan/TinyStories'
    tokenizer_path = f'{tokenizer_filename}_bpe_tokenizer.json'

    if not os.path.exists(tokenizer_path):
        print("Training new tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=config.VOCAB_SIZE)
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)

    def tokenize_function(examples):
        return tokenizer.encode_batch(examples["text"])

    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda examples: {'tokens': [t.ids for t in tokenize_function(examples)]},
        batched=True,
        remove_columns=["text"]
    )

    class CustomDataset(Dataset):
        def __init__(self, token_list, block_size):
            all_tokens = [token for sublist in token_list for token in sublist]
            num_blocks = len(all_tokens) // block_size
            self.data = torch.tensor(all_tokens[:num_blocks * block_size]).view(-1, block_size)
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            chunk = self.data[idx]
            return chunk[:-1], chunk[1:]

    train_dataset = CustomDataset(tokenized_datasets['train']['tokens'], config.BLOCK_SIZE + 1)
    val_dataset = CustomDataset(tokenized_datasets['validation']['tokens'], config.BLOCK_SIZE + 1)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, pin_memory=True, num_workers=4)

    # Update global config with the actual vocab size from the trained tokenizer
    config.VOCAB_SIZE = tokenizer.get_vocab_size()

    print(f"Actual Vocab Size: {config.VOCAB_SIZE}")
    print(f"Train dataset size: {len(train_dataset)} blocks")
    return train_loader, val_loader

# --- 2. Model Implementation ---

# A. Gated Associative Memory (GAM) Network
class GAMBlock(nn.Module):
    """ A single block of the GAM model, designed to be a parallel O(N) replacement for self-attention. """
    def __init__(self, config):
        super().__init__()
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
        local_context = local_context[:, :, :x.shape[1]].permute(0, 2, 1)
        scores = x @ self.memory_bank.T
        weights = F.softmax(scores, dim=-1)
        global_context = weights @ self.memory_bank
        local_g, global_g = self.gate(x).chunk(2, dim=-1)
        fused_context = torch.sigmoid(local_g) * local_context + torch.sigmoid(global_g) * global_context
        x = res + fused_context
        x = x + self.ffn(self.ln2(x))
        return x

class GAM_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED)
        self.pos_embedding = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.blocks = nn.ModuleList([GAMBlock(config) for _ in range(config.N_LAYER)])
        self.ln_f = nn.LayerNorm(config.N_EMBED)
        self.head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False)
        self.token_embedding.weight = self.head.weight
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# B. Transformer Model
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.N_EMBED % config.N_HEAD == 0
        self.n_head = config.N_HEAD
        self.head_dim = config.N_EMBED // config.N_HEAD
        self.c_attn = nn.Linear(config.N_EMBED, 3 * config.N_EMBED)
        self.c_proj = nn.Linear(config.N_EMBED, config.N_EMBED)
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.resid_dropout = nn.Dropout(config.DROPOUT)
        self.register_buffer("bias", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
                                     .view(1, 1, config.BLOCK_SIZE, config.BLOCK_SIZE))
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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.N_EMBED)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.N_EMBED)
        self.ffn = nn.Sequential(nn.Linear(config.N_EMBED, 4 * config.N_EMBED), nn.GELU(), nn.Linear(4 * config.N_EMBED, config.N_EMBED), nn.Dropout(config.DROPOUT))
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED)
        self.pos_embedding = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.N_LAYER)])
        self.ln_f = nn.LayerNorm(config.N_EMBED)
        self.head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False)
        self.token_embedding.weight = self.head.weight
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks: x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None: loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# C. Mamba Model
class MambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.N_EMBED)
        # Mamba does not typically use positional embeddings, but we add it for a fair comparison
        self.pos_embedding = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.dropout = nn.Dropout(config.DROPOUT)

        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=config.N_EMBED,
                    d_state=config.D_STATE,
                    d_conv=config.D_CONV,
                    expand=config.EXPAND,
                )
                for _ in range(config.N_LAYER)
            ]
        )
        self.ln_f = nn.LayerNorm(config.N_EMBED)
        self.head = nn.Linear(config.N_EMBED, config.VOCAB_SIZE, bias=False)
        self.token_embedding.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# --- 3. Training and Evaluation Framework ---

def get_scheduler(optimizer, num_training_steps):
    def lr_lambda(current_step):
        if current_step < config.WARMUP_STEPS:
            return float(current_step) / float(max(1, config.WARMUP_STEPS))
        progress = float(current_step - config.WARMUP_STEPS) / float(max(1, num_training_steps - config.WARMUP_STEPS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, global_step):
    model.train()
    total_loss = 0
    start_time = time.time()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    for batch, (X, Y) in enumerate(progress_bar):
        X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

        # Log to wandb
        if wandb.run:
            wandb.log({"train/batch_loss": loss.item(), "train/lr": scheduler.get_last_lr()[0], "global_step": global_step})
        global_step += 1


    avg_loss = total_loss / len(dataloader)
    elapsed_time = time.time() - start_time
    return avg_loss, elapsed_time, global_step

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    for X, Y in dataloader:
        X, Y = X.to(config.DEVICE), Y.to(config.DEVICE)
        _, loss = model(X, Y)
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')
    return avg_loss, perplexity

def run_experiment(model, model_name, train_loader, val_loader, dataset_name):
    print(f"\n--- Starting Experiment for: {model_name} on {dataset_name} ---")

    run_name = f"{model_name}-{dataset_name.replace('/', '_')}-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=run_name,
        config={k: v for k, v in vars(config).items() if not k.startswith('__')}
    )
    wandb.watch(model, log="all", log_freq=100)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    wandb.config.update({"total_params": total_params})


    model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    num_training_steps = config.EPOCHS * len(train_loader)
    scheduler = get_scheduler(optimizer, num_training_steps)

    global_step = 0
    try:
        for epoch in range(config.EPOCHS):
            train_loss, epoch_time, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, global_step)
            val_loss, val_ppl = evaluate(model, val_loader)

            print(f"Epoch {epoch+1}/{config.EPOCHS} | Time: {epoch_time:.2f}s")
            print(f"\tTrain Loss: {train_loss:.4f}")
            print(f"\tVal Loss:   {val_loss:.4f} | Val Perplexity: {val_ppl:.4f}")

            # Log epoch-level metrics to wandb
            if wandb.run:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "val/perplexity": val_ppl,
                    "epoch_time_s": epoch_time
                })
    except KeyboardInterrupt:
        print("Experiment interrupted by user.")
    finally:
        print(f"--- Experiment for {model_name} Finished ---")
        if wandb.run:
            wandb.finish()


# --- 4. Main Execution ---

if __name__ == '__main__':
    max_rows = {'train': config.MAX_TRAIN_ROWS, 'validation': config.MAX_VAL_ROWS}

    for dataset_info in config.DATASETS_TO_RUN:
        dataset_name = dataset_info['name']
        dataset_config_name = dataset_info['config']

        print(f"\n{'='*20} RUNNING EXPERIMENTS ON: {dataset_name} {'='*20}")

        # 1. Prepare data for the current dataset
        train_loader, val_loader = prepare_data(dataset_name, dataset_config_name, max_rows)

        # 2. Re-initialize models for each dataset to ensure a fair comparison
        # This is crucial! Otherwise, the second experiment would continue training the same model.
        print("\nInitializing models...")
        # We need to pass the updated config object since vocab_size may have changed
        gam_model = GAM_Model(config)
        transformer_model = TransformerModel(config)
        mamba_model = MambaModel(config)

        # 3. Run experiments for all models on the current dataset
        run_experiment(gam_model, "GAM", train_loader, val_loader, dataset_name)
        run_experiment(transformer_model, "Transformer", train_loader, val_loader, dataset_name)
        run_experiment(mamba_model, "Mamba", train_loader, val_loader, dataset_name)
