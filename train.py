import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import numpy as np
from model import PianistModel
from tokenizer import Tokenizer
import sys

if len(sys.argv) < 2:
    print("Usage: python train.py <model_name>")
    sys.exit(1)
    
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 100 # typicallys set to > 5000 for full training
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.2
# ------------

tokenizer = Tokenizer('/workspace/training/')

tokens, durations, velocities = tokenizer.tokenize_multiple_files(os.listdir('/workspace/training/'))
notes = tokenizer.decode(tokens)

print(tokens[:5])
print(durations[:5])
print(velocities[:5])

# vector of encoded notes 
data = torch.tensor(tokens, dtype=torch.long)

data[:block_size+1]

vocab_size = len(tokenizer.notes)

durations_tensor = torch.tensor(durations, dtype=torch.float32) / tokenizer.max_duration
velocities_tensor = torch.tensor(velocities, dtype=torch.float32) / tokenizer.max_velocity

def get_batch(split):
    # small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    d = torch.stack([durations_tensor[i:i+block_size] for i in ix])
    v = torch.stack([velocities_tensor[i:i+block_size] for i in ix])
    # targets
    y_tokens = torch.stack([data[i+1:i+block_size+1] for i in ix])
    y_durations = torch.stack([durations_tensor[i+1:i+block_size+1] for i in ix])
    y_velocities = torch.stack([velocities_tensor[i+1:i+block_size+1] for i in ix])

    x, d, v, y_tokens, y_durations, y_velocities = x.to(device), d.to(device), v.to(device), y_tokens.to(device), y_durations.to(device), y_velocities.to(device)

    return x, d, v, (y_tokens, y_durations, y_velocities)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, D, V, Y = get_batch(split)
            targets_tokens, targets_durations, targets_velocities = Y
            logits_tokens, logits_durations, logits_velocities, loss = model(X, D, V, (targets_tokens, targets_durations, targets_velocities))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = PianistModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # evaluate the loss on train sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}")

    # sample a batch of data
    xb, db, vb, (yb_tokens, yb_durations, yb_velocities) = get_batch('train')

    # evaluate loss
    logits_tokens, logits_durations, logits_velocities, loss = model(xb, db, vb, (yb_tokens, yb_durations, yb_velocities))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model_save_path = f'/workspace/models/{sys.argv[1]}.pth'
torch.save(m.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# save model settings
model_settings = {
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'block_size': block_size,
    'n_head': n_head,
    'n_layer': n_layer,
    'dropout': dropout
}
model_settings_save_path = f'/workspace/models/{sys.argv[1]}.json'
np.save(model_settings_save_path, model_settings)
print(f"Model settings saved to {model_settings_save_path}")

# save tokenizer
tokenizer_save_path = f'/workspace/models/{sys.argv[1]}_tokenizer.pkl'
with open(tokenizer_save_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {tokenizer_save_path}")