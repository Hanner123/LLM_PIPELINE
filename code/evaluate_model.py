from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from pathlib import Path
from evaluate import load as load_metric_eval  # falls 'datasets.load_metric' deprecated ist

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
    output_dir="./eval_results",  # muss existieren, wird aber nicht wirklich benutzt
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
print("Evaluationsergebnisse:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
