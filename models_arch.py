# print_architecture.py

import os
import torch
from torchinfo import summary  # pip install torchinfo

# Make paths independent of machine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

slow_fast_path = os.path.join(parent_dir, 'slow_fast_phoenix2014_dev_18.01_test_18.28.pt')
twostream_path = os.path.join(parent_dir, 'twostreamslr.ckpt')

# Output file
output_file = os.path.join(current_dir, 'model_architectures.txt')


def describe_state_dict(state_dict, name, file):
    print(f"\n{'='*30}\n{name.upper()} ARCHITECTURE\n{'='*30}\n", file=file)
    for k, v in state_dict.items():
        if hasattr(v, 'shape'):
            print(f"{k:60}  →  shape: {tuple(v.shape)}", file=file)


# Load state_dicts
sf_ckpt = torch.load(slow_fast_path, map_location='cpu')
ts_ckpt = torch.load(twostream_path, map_location='cpu')

sf_state = sf_ckpt['model_state_dict']
ts_state = ts_ckpt['model_state']

# Save to file (overwrite each time)
with open(output_file, "w") as f:
    describe_state_dict(sf_state, "SlowFast", f)
    describe_state_dict(ts_state, "TwoStream", f)

print("[✅] Layer-wise architecture (name + shape) saved to model_architectures.txt")
