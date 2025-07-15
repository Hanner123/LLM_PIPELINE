from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from pathlib import Path
from evaluate import load as load_metric_eval  # falls 'datasets.load_metric' deprecated ist
import onnx

print("Neu:")
# Modellverzeichnis
model_dir = Path(__file__).resolve().parent.parent / "models" / "tinybert_saved_model"

# Modell und Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# CUDA verwenden, wenn vorhanden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# AG News-Datensatz laden
dataset = load_dataset("ag_news")

# Tokenisierung
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Entferne Textspalten für Kompatibilität mit Trainer
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Accuracy-Metrik laden

accuracy_metric = load_metric_eval("accuracy")

# Compute-Funktion für die Evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Neue TrainingArguments (für Evaluation)
eval_args = TrainingArguments(
    output_dir="../eval_results",  # muss existieren, wird aber nicht wirklich benutzt
    per_device_eval_batch_size=32,
    do_train=False,
    do_eval=True,
    logging_dir="./logs",
)

# Trainer mit geladenem Modell + Tokenizer + Eval-Metrik
trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Evaluation starten
results = trainer.evaluate()
print("Evaluationsergebnis ohne Quantisierung:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")


import onnxruntime as ort
import torch

#chatGPT Lösung
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1  # oder 2 – hängt von Jetson-Modell ab
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

print("vor session")



# Lade das ONNX-Modell
model_dir = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"

onnx.checker.check_model(model_dir)
print("ONNX-Modell ist gültig")


session = ort.InferenceSession(model_dir, sess_options)




print("Model inputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

# Beispiel-Eingabe (z.B. aus deinem Testset)
data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"
data = torch.load(data_path)
for i in range(50):
    input_ids = data["input_ids"][i:i+1].numpy().astype("int32")
    attention_mask = data["attention_mask"][i:i+1].numpy().astype("int32")
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "token_type_ids" in [inp.name for inp in session.get_inputs()]:
        inputs["token_type_ids"] = (torch.zeros_like(data["input_ids"][i:i+1])).numpy().astype("int32")
    outputs = session.run(None, inputs)
