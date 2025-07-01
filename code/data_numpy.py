import torch
import numpy as np
from pathlib import Path 
# Lade die .pt-Datei
data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"
data_path_end = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.npz"
data = torch.load(data_path)

# Angenommen, data ist ein Dict mit Tensors, z.B.:
# {'input_ids': tensor(...), 'attention_mask': tensor(...), 'labels': tensor(...)}

# Wandle jeden Tensor in ein numpy array um und speichere als .npy
for key, tensor in data.items():
    np.save(f"{key}.npy", tensor.numpy())

# Alternativ: alles zusammen als .npz speichern
np.savez(data_path_end, **{k: v.numpy() for k, v in data.items()})