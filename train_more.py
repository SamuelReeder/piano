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
import json

if len(sys.argv) < 3:
    print("Usage: python continue_training.py <model_name> <iters>")
    sys.exit(1)

model_name = sys.argv[1]

model_settings_path = f'/workspace/models/{model_name}.json.npy'
model_settings = np.load(model_settings_path, allow_pickle=True).item()

tokenizer_path = f'/workspace/models/{model_name}_tokenizer.pkl'
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64
block_size = model_settings['block_size']
max_iters = int(sys.argv[2])  # additional training iterations
eval_interval = 50
learning_rate = 3e-4  # lower learning rate 
eval_iters = 200

model = PianistModel(
    vocab_size=model_settings['vocab_size'],
    n_embd=model_settings['n_embd'],
    block_size=model_settings['block_size'],
    n_head=model_settings['n_head'],
    n_layer=model_settings['n_layer'],
    dropout=model_settings['dropout']
)
model.load_state_dict(torch.load(f'/workspace/models/{model_name}.pth'))
model.to(device)

print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

tokens, durations, velocities = tokenizer.tokenize_multiple_files(os.listdir('/workspace/training/'))
data = torch.tensor(tokens, dtype=torch.long)
durations_tensor = torch.tensor(durations, dtype=torch.float32) / tokenizer.max_duration
velocities_tensor = torch.tensor(velocities, dtype=torch.float32) / tokenizer.max_velocity

def get_batch(split):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    d = torch.stack([durations_tensor[i:i+block_size] for i in ix])
    v = torch.stack([velocities_tensor[i:i+block_size] for i in ix])
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}")

    xb, db, vb, (yb_tokens, yb_durations, yb_velocities) = get_batch('train')

    logits_tokens, logits_durations, logits_velocities, loss = model(xb, db, vb, (yb_tokens, yb_durations, yb_velocities))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

new_model_name = f"{model_name}_continued"
model_save_path = f'/workspace/models/{new_model_name}.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Fine-tuned model saved to {model_save_path}")

model_settings_save_path = f'/workspace/models/{new_model_name}.json'
np.save(model_settings_save_path, model_settings)
print(f"Updated model settings saved to {model_settings_save_path}")

tokenizer_save_path = f'/workspace/models/{new_model_name}_tokenizer.pkl'
with open(tokenizer_save_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer saved to {tokenizer_save_path}")
