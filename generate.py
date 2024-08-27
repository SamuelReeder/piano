from model import PianistModel
import torch
import sys
import numpy as np
import pickle

if len(sys.argv) < 3:
    print("Usage: python generate.py <model_name> <max_new_tokens>")
    sys.exit(1)

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer
model_name = sys.argv[1]
tokenizer_save_path = f'/workspace/models/{model_name}_tokenizer.pkl'
with open(tokenizer_save_path, 'rb') as f:
    tokenizer = pickle.load(f)

# model settings
model_settings_save_path = f'/workspace/models/{model_name}.json.npy'
model_settings = np.load(model_settings_save_path, allow_pickle=True).item()

vocab_size = model_settings['vocab_size']
n_embd = model_settings['n_embd']
block_size = model_settings['block_size']
n_head = model_settings['n_head']
n_layer = model_settings['n_layer']
dropout = model_settings['dropout']

# model
model = PianistModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)
m = model.to(device)
model_save_path = f'/workspace/models/{model_name}.pth'
m.load_state_dict(torch.load(model_save_path))
m.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
initial_durations = torch.zeros((1, 1), dtype=torch.float32, device=device)
initial_velocities = torch.zeros((1, 1), dtype=torch.float32, device=device)

generated_tokens, generated_durations, generated_velocities = m.generate(
    idx=context,
    durations=initial_durations,
    velocities=initial_velocities,
    max_new_tokens=int(sys.argv[2]),
    block_size=block_size
)

decoded_durations = generated_durations * tokenizer.max_duration * 1000
decoded_velocities = generated_velocities

decoded_durations_list = decoded_durations.flatten().tolist()
decoded_velocities_list = decoded_velocities.flatten().tolist()

generated_notes = tokenizer.decode(generated_tokens[0].tolist())

print("Generated Notes:", generated_notes)
print("Decoded Durations:", decoded_durations_list)
print("Decoded Velocities:", decoded_velocities_list)
assert len(generated_notes) == len(decoded_durations_list) == len(decoded_velocities_list)
