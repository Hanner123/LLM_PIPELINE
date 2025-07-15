import os
import urllib.request
import onnx
import onnxruntime as ort

# 1. MNIST-Modell-URL (raw)
url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
model_path = "mnist-8.onnx"

# 2. Laden, falls nötig
if not os.path.exists(model_path):
    print("Lade MNIST-Modell herunter...")
    urllib.request.urlretrieve(url, model_path)
    print("Download abgeschlossen.")

# 3. Validierung
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX-Modell ist gültig.")

# 4. Session mit CPU-Provider erstellen
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
print("✅ InferenceSession erfolgreich erstellt.")

# 5. Eingabe-Info anzeigen
inp = session.get_inputs()[0]
print(f"Eingabetensor: {inp.name}, Shape: {inp.shape}, Typ: {inp.type}")
