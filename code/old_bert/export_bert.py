from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from pathlib import Path
import torch
from bert_dataset import BERTLM  # falls nötig
from bert_dataset import BERT, BERTLM  # sicherstellen, dass die Definition identisch ist
from transformers import BertTokenizer

tokenizer_path = Path(__file__).resolve().parent.parent / "bert-it-1"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

print("Vocab length tokenizer", len(tokenizer.vocab))


bert_model = BERT(
  vocab_size=len(tokenizer.vocab),
  d_model=768,
  n_layers=2,
  heads=12,
  dropout=0.1
)
model = BERTLM(bert_model, vocab_size=len(tokenizer.vocab))
# Gewichte laden
model_dir = Path(__file__).resolve().parent.parent / "models" / "bert.pth"
model.load_state_dict(torch.load(model_dir, map_location='cpu'))
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 1
seq_len = 64

dummy_input_ids = torch.randint(0, len(tokenizer.vocab), (batch_size, seq_len))       # Vocab-Größe anpassen
dummy_segment_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
dummy_input_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

dummy_inputs = (dummy_input_ids, dummy_segment_ids)



# Export nach ONNX
model_dir = Path(__file__).resolve().parent.parent / "models" / "bert.onnx"

torch.onnx.export(
    model,
    dummy_inputs,  # Tuple of inputs
    model_dir,
    input_names=["input_ids", "segment_ids"],
    output_names=["logits", "nsp_logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "segment_ids": {0: "batch_size", 1: "sequence"},
        "input_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
        "nsp_logits": {0: "batch_size"},
    },
    opset_version=13
)


print(f"ONNX-Modell exportiert zu: {model_dir}")  