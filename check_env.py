"""Environment check; never changes installed software."""
import sys
import platform
import importlib.metadata as meta

print('Python:', sys.version.split()[0])
print('Platform:', platform.platform())
for package in ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow', 'streamlit']:
    try:
        print(f'{package}: {meta.version(package)}')
    except meta.PackageNotFoundError:
        raise SystemExit(f'Missing package: {package}; complete installation first.')

import torch
from torchvision.models import efficientnet_b0

print('CUDA available:', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
    print('PyTorch CUDA runtime:', torch.version.cuda)
else:
    print('CPU mode works for checks and inference; training will be slower.')
torch.set_num_threads(2)
model = efficientnet_b0(weights=None).to(device).eval()
with torch.no_grad():
    out = model(torch.zeros(1, 3, 224, 224, device=device))
assert out.shape == (1, 1000)
print('EfficientNet-B0 forward check: PASS (no pretrained download needed)')
