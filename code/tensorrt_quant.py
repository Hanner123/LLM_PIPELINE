import tensorrt as trt
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pycuda.driver as cuda
import pycuda.autoinit
import os

import yaml

# ---- Kalibrator-Klasse ----
class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, batch_size, cache_file="calibration.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        example = next(self.data_iter)
        self.device_input_ids = cuda.mem_alloc(example[0].numpy().astype(np.int32).nbytes)
        self.device_attention_mask = cuda.mem_alloc(example[1].numpy().astype(np.int32).nbytes)
        self.cache_file = cache_file
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            return None
        input_ids = batch[0].cpu().numpy().astype(np.int32)
        attention_mask = batch[1].cpu().numpy().astype(np.int32)
        cuda.memcpy_htod(self.device_input_ids, input_ids)
        cuda.memcpy_htod(self.device_attention_mask, attention_mask)
        return [int(self.device_input_ids), int(self.device_attention_mask)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ---- Hilfsfunktion für DataLoader ----
def create_calib_dataloader(data_path, batch_size):
    data = torch.load(data_path)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    # Labels werden für Kalibrierung nicht benötigt
    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ---- Hauptfunktion ----
def build_int8_engine(engine_path, onnx_model_path, calib_loader, batch_size, cache_file="calibration.cache"):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX Parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.INT8)

    # Kalibrator setzen
    calibrator = MyEntropyCalibrator(calib_loader, batch_size, cache_file)
    config.int8_calibrator = calibrator

    # Dynamische Batchgrößen
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        profile.set_shape(name, (1, 128), (batch_size, 128), (batch_size, 128))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"INT8 TensorRT-Engine gespeichert unter: {engine_path}")

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

if __name__ == "__main__":
    params = load_params()
    batch_sizes = params["measure"]["batch_sizes"]

    for batch_size in batch_sizes:
        onnx_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"
        data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"

        calib_loader = create_calib_dataloader(data_path, batch_size)
        engine_name = f"tinybert_int8_{batch_size}.engine"
        engine_path = Path(__file__).resolve().parent.parent / "models" / "engines" /engine_name

        build_int8_engine(engine_path, onnx_model_path, calib_loader, batch_size)