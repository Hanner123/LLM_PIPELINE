from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from pathlib import Path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./models/tinybert_saved_model")
model = AutoModelForSequenceClassification.from_pretrained("./models/tinybert_saved_model")

dummy_input = tokenizer(
    "This is a dummy input for ONNX export.",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True,
)
dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

model.eval().to(device)




# Export nach ONNX
model_dir = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"
torch.onnx.export(
    model,                                 # Modell
    (dummy_input["input_ids"], dummy_input["attention_mask"]),  # Eingaben
    model_dir,                     # Output-Pfad
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, 
                    "attention_mask": {0: "batch_size"},
                    "logits": {0: "batch_size"}},
    opset_version=17,
)

print(f"ONNX-Modell exportiert zu: {model_dir}")  