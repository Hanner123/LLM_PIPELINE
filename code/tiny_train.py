from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Modell & Tokenizer laden
model_name = "prajjwal1/bert-tiny"  # TinyBERT alternative


tokenizer = AutoTokenizer.from_pretrained(model_name) # wählt dann automatisch den richtigen tokenizer für bert aus
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 Klassen bei AG-News, pre trained model wird an qualifizierungsaufgabe angepasst
# fügt wahrscheinlich zum model hinzu:
# - dense layer mit aktivierung -> Vorhersage zur Klasse
# - soft-max (wähle Klasse mit höchster Wahrscheinlichkeit)

model.save_pretrained("../models/tinybert_saved_model")
tokenizer.save_pretrained("../models/tinybert_saved_model")

dummy_input = tokenizer(
    "This is a dummy input for ONNX export.",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True,
)

# Export nach ONNX
onnx_program = torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "../models/model_before_train.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17,
    do_constant_folding=True
)



# Datensatz laden
dataset = load_dataset("ag_news")

# Tokenisierung der Daten, alle 128 Zeichen lang, wenn zu lang abgeschnitten
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Tokenizer wird angewendet
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Trainings-Argumente festlegen
training_args = TrainingArguments(
    output_dir="./tinybert_results",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
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

# Training starten
trainer.train()

model.save_pretrained("./tinybert_saved_model")
tokenizer.save_pretrained("./tinybert_saved_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

dummy_input = tokenizer(
    "This is a dummy input for ONNX export.",
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True,
)
dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

model.eval().to(device)
dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

# Export nach ONNX
torch.onnx.export(
    model,                                 # Modell
    (dummy_input["input_ids"], dummy_input["attention_mask"]),  # Eingaben
    "../models/tinybert_post_training.onnx",                     # Output-Pfad
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, 
                  "attention_mask": {0: "batch_size"},
                  "logits": {0: "batch_size"}},
    opset_version=17,
)


# dvc projekt (init, yaml file ...)
# quantisierung

