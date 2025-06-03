from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from pathlib import Path



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_dir = Path(__file__).resolve().parent.parent / "models" / "tinybert_saved_model"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForSequenceClassification.from_pretrained(tokenizer_dir)

batch_size = 16


vocab_size = tokenizer.vocab_size  # z. B. 30522

input_ids = torch.randint(0, vocab_size, (batch_size, 128), dtype=torch.long)
attention_mask = torch.ones((batch_size, 128), dtype=torch.long)


model.eval().to("cpu")


# Export nach ONNX
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:  
    model_name = f"tinybert_{batch_size}.onnx"
    input_ids = torch.randint(0, vocab_size, (batch_size, 128), dtype=torch.long)
    attention_mask = torch.ones((batch_size, 128), dtype=torch.long)
    # Export nach ONNX
    model_dir = Path(__file__).resolve().parent.parent / "models" / model_name

    torch.onnx.export(
        model,
        (input_ids, input_ids),
        model_dir,
        export_params=True,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=None,  # kein dynamisches Shape!
        opset_version=17
    )

    print(f"ONNX-Modell exportiert zu: {model_dir}")  