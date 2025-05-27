import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

def get_agnews_dataloader(batch_size=32):
    data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"
    data = torch.load(data_path)
    dataset = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

import sys
import os

# Brevitas-Ordner zum Python-Pfad hinzufügen

main_py_dir = Path(__file__).resolve().parent.parent.parent / "brevitas" / "src" / "brevitas_examples" / "llm"
sys.path.append(str(main_py_dir))
print("Brevitas main.py path:", main_py_dir)
print(main_py_dir.exists())
for f in main_py_dir.glob("**/*"):
    print(f)
# ausgegebener pfad: /home/hanna/git/brevitas/src/brevitas_examples/llm
# richtiger pfad: /home/git/brevitas/src/brevitas_examples/llm

import main

# Argumente wie von der Kommandozeile
sys.argv = [
    "main.py",
    "--model", "prajjwal1/bert-tiny",
    "--dataset", "agnews",
    "--export-target", "onnx_qcdq",
    "--weight-quant-granularity", "per_tensor",
    "--input-bit-width", "8",
    "--input-quant-type", "sym",
    "--act-calibration",
    # "--input-quant-format", "qint",
    # "--quantize-input-zero-point", "True",
    # ...weitere Optionen...
]

main.main()