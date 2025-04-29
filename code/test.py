import torch

if torch.cuda.is_available():
    print("✅ CUDA ist verfügbar!")
    print(f"🖥️ GPU-Name: {torch.cuda.get_device_name(0)}")
    print(f"📦 Speicher: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")
else:
    print("❌ CUDA ist NICHT verfügbar. Läuft auf CPU.")
