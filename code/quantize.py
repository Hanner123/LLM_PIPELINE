import torch
from transformers import AutoModelForSequenceClassification
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Modell laden (z. B. TinyBERT)



# funktioniert nicht, weil asynchron (bei measure)
# quantize_dynamic(
#     model_input=onnx_model_path,
#     model_output=onnx_model_path_quantized,
#     weight_type=QuantType.QInt16  # Oder QuantType.QUInt8
# )

import tensorrt as trt
import numpy as np
import torch
import onnx
from torch.utils.data import DataLoader, Dataset




# Lade das ONNX-Modell
onnx_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"
onnx_model_path_quantized = Path(__file__).resolve().parent.parent / "models" / "tinybert_quantized.onnx"
onnx_model = onnx.load(onnx_model_path)

import tensorrt as trt

# Beispiel Kalibrator (muss angepasst werden)
class MyCalibrator(trt.IInt8Calibrator):
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.index = 0

    def get_batch_size(self):
        # Gibt die Batch-Größe zurück
        return 1  # Beispiel, anpassen je nach Bedarf

    def get_batch(self, names):
        # Liefert den nächsten Batch (in TensorRT erwartet die Methode eine Liste von Eingabename)
        batch_data = next(self.data_loader)  # Hier anpassen
        return batch_data

    def read_calibration_cache(self):
        # Hier könnte ein Cache des Kalibrators geladen werden
        return None

    def write_calibration_cache(self, cache):
        # Hier könnte der Kalibrator-Cache gespeichert werden
        pass

# TensorRT-Engine bauen und quantisieren
def build_tensorrt_engine(onnx_model_path, data_loader):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()

    # Setze Quantisierungsflags (INT8)
    config.set_flag(trt.BuilderFlag.INT8)

        
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 40)

    # Set optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        profile.set_shape(name, (1, 128), (8, 128), (1024, 128))  # min, opt, max batch size
    config.add_optimization_profile(profile)

    # Setze den Kalibrator
    calibrator = MyCalibrator(data_loader)
    config.int8_calibrator = calibrator  # Hier den Kalibrator richtig zuweisen

    # Baue die Engine mit INT8-Quantisierung
    
    serialized_engine = builder.build_serialized_network(network, config)

    # Speichere die Engine als .plan-Datei
    with open("quantized_model.plan", "wb") as f:
        f.write(serialized_engine)
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()


    return engine


class ExampleDataset(Dataset):
    def __init__(self, data):
        # Hier sollte dein tatsächlicher Input-Daten für die Kalibrierung kommen (z.B. Bilder)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

# Beispielaufruf
# Beispielhafte Dummy-Daten für die Kalibrierung
data = torch.randn(100, 128)  # Beispiel-Daten (z.B. Tensor mit 100 Samples)
dataset = ExampleDataset(data)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
engine = build_tensorrt_engine(onnx_model_path, data_loader)
if engine is None:
    print("Engine building failed")
    exit()

# Serialisiere die Engine, um sie später zu laden
serialized_engine = engine.serialize()

# Speicher die Engine
engine_path = 'quantized_model_int8.trt'
with open(engine_path, 'wb') as f:
    f.write(serialized_engine)
print(f"INT8 TensorRT Engine gespeichert unter: {engine_path}")