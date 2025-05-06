from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import time
from pathlib import Path
import yaml


def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

params = load_params()
batch_size = params["train"]["batch_size"]
learning_rate = params["train"]["lr"]
epochs = params["train"]["epochs"]
weight_decay = params["train"]["weight_decay"]

# Modell & Tokenizer laden
model_name = "prajjwal1/bert-tiny"  # TinyBERT alternative


tokenizer = AutoTokenizer.from_pretrained(model_name) # wählt dann automatisch den richtigen tokenizer für bert aus
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 Klassen bei AG-News, pre trained model wird an qualifizierungsaufgabe angepasst
# fügt wahrscheinlich zum model hinzu:
# - dense layer mit aktivierung -> Vorhersage zur Klasse
# - soft-max (wähle Klasse mit höchster Wahrscheinlichkeit)



# Datensatz laden
dataset = load_dataset("ag_news")

# Tokenisierung der Daten, alle 128 Zeichen lang, wenn zu lang abgeschnitten
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenizer wird angewendet
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Trainings-Argumente festlegen
training_args = TrainingArguments(
    output_dir="../tinybert_results",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=128,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
)

# Trainer bauen
# Trainer: führt training und eval loops durch, automatische nutzung von cuda wenn vorhanden, checkpoints und neustarts, evaluation während training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Relevante Felder extrahieren und in Tensorn speichern
test_inputs = torch.tensor(tokenized_datasets["test"]["input_ids"])
test_attention = torch.tensor(tokenized_datasets["test"]["attention_mask"])
test_labels = torch.tensor(tokenized_datasets["test"]["label"])

# Speichern als dictionary
torch.save({
    "input_ids": test_inputs,
    "attention_mask": test_attention,
    "labels": test_labels
}, "./datasets/tokenized_agnews_test.pt")

# Training starten
trainer.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


model_dir = Path(__file__).resolve().parent.parent / "models" / "tinybert_saved_model"
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print("Model saved to:", model_dir)
