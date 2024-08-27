import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PianistModel(nn.Module):
    """ transformer model """
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # embeddings for duration and velocity (continuous values)
        self.duration_embedding = nn.Linear(1, n_embd)
        self.velocity_embedding = nn.Linear(1, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm

        # output layers for tokens, durations, and velocities
        self.lm_head_tokens = nn.Linear(n_embd, vocab_size)
        self.lm_head_durations = nn.Linear(n_embd, 1) 
        self.lm_head_velocities = nn.Linear(n_embd, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, durations, velocities, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        duration_emb = self.duration_embedding(durations.unsqueeze(-1))  # (B,T,C)
        velocity_emb = self.velocity_embedding(velocities.unsqueeze(-1))  # (B,T,C)

        x = x + duration_emb + velocity_emb  # (B,T,C)

        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        # separate heads for different predictions
        logits_tokens = self.lm_head_tokens(x)  # (B,T,vocab_size)
        logits_durations = self.lm_head_durations(x).squeeze(-1)  # (B,T)
        logits_velocities = self.lm_head_velocities(x).squeeze(-1)  # (B,T)

        loss = None
        if targets is not None:
            targets_tokens, targets_durations, targets_velocities = targets

            B, T, C = logits_tokens.shape
            logits_tokens = logits_tokens.view(B*T, C)
            targets_tokens = targets_tokens.view(B*T)
            loss_tokens = F.cross_entropy(logits_tokens, targets_tokens)

            loss_durations = F.mse_loss(logits_durations, targets_durations)
            loss_velocities = F.mse_loss(logits_velocities, targets_velocities)

            # total loss is the sum of individual losses
            loss = loss_tokens + loss_durations + loss_velocities

        return logits_tokens, logits_durations, logits_velocities, loss


    def generate(self, idx, durations, velocities, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            durations_cond = durations[:, -block_size:]
            velocities_cond = velocities[:, -block_size:]

            logits_tokens, logits_durations, logits_velocities, _ = self(idx_cond, durations_cond, velocities_cond)

            logits = logits_tokens[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            duration_next = logits_durations[:, -1].unsqueeze(-1)  # (B, 1)
            velocity_next = logits_velocities[:, -1].unsqueeze(-1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            durations = torch.cat((durations, duration_next), dim=1)  # (B, T+1)
            velocities = torch.cat((velocities, velocity_next), dim=1)  # (B, T+1)

        return idx, durations, velocities

