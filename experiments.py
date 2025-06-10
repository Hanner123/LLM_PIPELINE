

import subprocess
import numpy as np
import json
from pathlib import Path
import time

# 10 logarithmisch verteilte LRs zwischen 1e-5 und 1e-3
learning_rates = np.logspace(-5, -3, num=10)
#   learning_rates = [0.0001, 0.0005]

results = []
accuracy_path = Path("eval_results/accuracy_FP32.json")  

for lr in learning_rates:
    lr = float(lr)
    print(f"Starte Experiment mit lr={lr:.6f}")
    subprocess.run([
        "dvc", "exp", "run", "--set-param", f"train.lr={lr}"
    ], check=True)

    # Warte kurz, damit Datei geschrieben wird (optional, je nach Pipeline)
    time.sleep(1)

    # Accuracy auslesen
    if accuracy_path.exists():
        with open(accuracy_path) as f:
            acc_data = json.load(f)
            # Falls Liste, nimm letzten Eintrag
            if isinstance(acc_data, list):
                acc = acc_data[-1]["value"]
            else:
                acc = acc_data["value"]
        results.append({"lr": lr, "accuracy": acc})
    else:
        print(f"Warnung: {accuracy_path} nicht gefunden!")

# Vergleichsdatei speichern
comparison_path = Path("eval_results/lr_accuracy_comparison.json")
with open(comparison_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Fertig! Ergebnisse gespeichert in {comparison_path}")