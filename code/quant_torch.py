from transformers import AutoModelForSequenceClassification
import torch
import os
from pathlib import Path


# Modellstruktur laden
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=4)

# Trainierte Gewichte laden
model_weights = Path(__file__).resolve().parent.parent / "models" / "tinybert_model_weights.pt"
model.load_state_dict(torch.load(model_weights, map_location="cpu"))  # Pfad anpassen
model.eval()

# Dynamische Quantisierung anwenden
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Nur Linear-Layer quantisieren
    dtype=torch.qint8
)

# Dummy-Eingabe vorbereiten
dummy_input = {
    "input_ids": torch.ones(1, 128, dtype=torch.long),  # Beispielhafte Eingabe
    "attention_mask": torch.ones(1, 128, dtype=torch.long)
}

# Modell auf CPU (ONNX erwartet CPU)
quantized_model.cpu()

# Export
torch.onnx.export(
    quantized_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "quantized_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=13
)

print("ONNX-Export abgeschlossen: quantized_model.onnx")