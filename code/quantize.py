import torch
from pathlib import Path
import tensorrt as trt
import numpy as np
import onnx
from torch.utils.data import DataLoader, Dataset
import pycuda.driver as cuda
import pycuda.autoinit

# Lade das ONNX-Modell
onnx_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"
onnx_model_path_quantized = Path(__file__).resolve().parent.parent / "models" / "tinybert_quantized.onnx"
onnx_model = onnx.load(onnx_model_path)

# Beispiel Kalibrator (muss angepasst werden)
class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file="calibration.cache"):
        super().__init__()
        self.data_loader = data_loader
        self.index = 0
        self.cache_file = cache_file
        self.device_input = None
        self.batch_size = 1  # Beispiel, anpassen je nach Bedarf

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch_data = next(self.data_loader)
            # Hier wird der Batch auf das Gerät kopiert
            self.device_input = cuda.mem_alloc(batch_data.nbytes)
            cuda.memcpy_htod(self.device_input, batch_data)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except IOError:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# TensorRT-Engine bauen und quantisieren
def build_tensorrt_engine(onnx_model_path, data_loader):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()

    # Annahme: das Modell hat die Inputs ["input_ids", "attention_mask"]
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        profile.set_shape(name, (1, 128), (8, 128), (1024, 128))  # min, opt, max batch size
    config.add_optimization_profile(profile)


    # Setze den Kalibrator
    calibrator = MyCalibrator(data_loader)
    config.int8_calibrator = calibrator

    

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
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Beispielaufruf
data = torch.randn(100, 128)  # Beispiel-Daten
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
