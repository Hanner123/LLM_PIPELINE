import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import onnx_tool
import torch
import pynvml
import onnx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
import pycuda.driver as cuda
import pycuda.autoinit
import os

import yaml


def to_device(data,device):
    if isinstance(data, (list,tuple)): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)
    


def accuracy(labels, outputs):
    correct_predictions = 0
    total_predictions = 0
    i = 0
    for label in labels:
        _, predicted = torch.max(torch.tensor(outputs[i]), dim=0)
        total_predictions = total_predictions + 1
        if predicted == label:
            correct_predictions = correct_predictions + 1
        i = i+1
    return correct_predictions, total_predictions

def save_json(log, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)

class MyEntropyCalibrator(trt.IInt8EntropyCalibrator2):
     
    def __init__(self, dataloader, batch_size, cache_file="calibration.cache"):
        super(MyEntropyCalibrator, self).__init__()
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        example_input = next(self.data_iter)[0].numpy().astype(np.int32)
        attention_mask = next(self.data_iter)[1].numpy().astype(np.int32)
        self.device_input_ids = cuda.mem_alloc(example_input.nbytes)
        self.device_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
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
        attention_mask = batch[1].numpy().astype(np.int32)

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



def measure_latency(context, test_loader, device_input, device_attention_mask, device_output, stream_ptr, torch_stream, batch_size=1):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """
    total_time = 0
    total_time_synchronize = 0
    total_time_datatransfer = 0 
    iterations = 0  
    for input_ids, attention_mask, labels in test_loader: 

        start_time_datatransfer = time.time()  # Startzeit messen
        
        device_input.copy_(input_ids.to(torch.int32))           # Eingabe auf GPU übertragen
        device_attention_mask.copy_(attention_mask.to(torch.int32)) # Eingabe auf GPU übertragen

        start_time_synchronize = time.time()  # Startzeit messen
        torch_stream.synchronize()  

        start_time_inteference = time.time()  # Startzeit messen
        with torch.cuda.stream(torch_stream):
            context.execute_async_v3(stream_ptr)  # TensorRT-Inferenz durchführen
        torch_stream.synchronize()  # GPU-Synchronisation nach Inferenz
        end_time = time.time()

        output = device_output.cpu().numpy()
        end_time_datatransfer = time.time() 

        latency = end_time - start_time_inteference  # Latenz für diesen Batch
        latency_synchronize = end_time - start_time_synchronize  # Latenz für diesen Batch
        latency_datatransfer = end_time_datatransfer - start_time_datatransfer  # Latenz für diesen Batch

        total_time += latency
        total_time_synchronize += latency_synchronize
        total_time_datatransfer += latency_datatransfer
        iterations += 1
        
        # labels auswerten - zeit messen, bar plots

    average_latency = (total_time / iterations) * 1000  # In Millisekunden
    average_latency_synchronize = (total_time_synchronize / iterations) * 1000  # In Millisekunden
    average_latency_datatransfer = (total_time_datatransfer / iterations) * 1000  # In Millisekunden


    return average_latency, average_latency_synchronize, average_latency_datatransfer

def print_latency(latency_ms, latency_synchronize, latency_datatransfer, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size):
    print("For Batch Size: ", batch_size)
    print(f"Gemessene durchschnittliche Latenz für Inteferenz : {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation : {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer : {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time-start_time:.4f} s")
    print("num_batches", num_batches)
    print(f"Throughput: {throughput_batches:.4f} Batches/Sekunde")
    print(f"Throughput: {throughput_images:.4f} Bilder/Sekunde")

def build_tensorrt_engine(onnx_model_path, test_loader, batch_size):
    """
    Erstellt und gibt die TensorRT-Engine und den Kontext zurück.
    :param onnx_model_path: Pfad zur ONNX-Modell-Datei.
    :param logger: TensorRT-Logger.
    :return: TensorRT-Engine und Execution Context.
    """

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse the ONNX model
    with open(onnx_model_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX Parsing failed")

    config = builder.create_builder_config()
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 40)


    # INT8 aktivieren
    config.set_flag(trt.BuilderFlag.INT8) 

    # Set optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()

    # Annahme: das Modell hat die Inputs ["input_ids", "attention_mask"]
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        profile.set_shape(name, (1, 128), (8, 128), (1024, 128))  # min, opt, max batch size
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    return engine, context



def create_test_dataloader(data_path, batch_size, device):
    """
    Erstellt den DataLoader für die Testdaten.
    :param data_path: Pfad zur Testdaten-Datei.
    :param batch_size: Die Batchgröße.
    :param device: Zielgerät (z. B. 'cpu' oder 'cuda').
    :return: DataLoader-Objekt für die Testdaten.
    """

    # test_dataset_hf = dataset["train"].shuffle(seed=42).select(range(1000))  # Beispiel: 1000 zufällige Beispiele
    # Wie kann ich so einen test data loader mit den ag_news daten erstellen?
    # test_data = torch.load(data_path, map_location=device, weights_only=False) 
    # Data_Loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    # test_loader = DeviceDataLoader(Data_Loader, device)  # test_loader in die cpu laden, wegen numpy im evaluate_model

    data = torch.load(data_path)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    labels = data["labels"]
    test_dataset = TensorDataset(input_ids, attention_mask, labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    return test_loader

def run_inference(batch_size=1):
    """pynvml-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param max_iterations: Maximalanzahl der Iterationen.
    :return: (Anzahl der korrekten Vorhersagen, Gesamtanzahl der Vorhersagen).
    """
    test_loader = create_test_dataloader(data_path, batch_size, "cpu")
    engine, context = build_tensorrt_engine(onnx_model_path, test_loader, batch_size)
    device_input, device_attention_mask, device_output, stream_ptr, torch_stream = test_data(context, batch_size)

    total_predictions = 0
    correct_predictions = 0

    for input_ids, attention_mask, labels in test_loader:

        device_input.copy_(input_ids.to(torch.int64))           # Eingabe auf GPU übertragen
        device_attention_mask.copy_(attention_mask.to(torch.int64)) # Eingabe auf GPU übertragen
        torch_stream.synchronize()
        
        with torch.cuda.stream(torch_stream): # nicht für mehr als 64 Bildern möglich
            context.execute_async_v3(stream_ptr)
        torch_stream.synchronize()

        output = device_output.cpu().numpy()

        correct, total = accuracy(labels, output)
        total_predictions += total
        correct_predictions += correct

    return correct_predictions, total_predictions

def calculate_latency_and_throughput(context, batch_sizes, onnx_model_path):
    """
    Berechnet die durchschnittliche Latenz und den Durchsatz (Bilder und Batches pro Sekunde) für verschiedene Batchgrößen.
    :param context: TensorRT-Execution-Context.
    :param test_loader: DataLoader mit Testdaten.
    :param device_input: Eingabebuffer auf der GPU.
    :param device_output: Ausgabebuffer auf der GPU.
    :param stream_ptr: CUDA-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param batch_sizes: Liste der Batchgrößen.
    :return: (Throughput-Log, Latenz-Log).
    """
    

    throughput_log = []
    latency_log = []
    latency_log_batch = []

    for batch_size in batch_sizes:
        test_loader = create_test_dataloader(data_path, batch_size, "cpu")
        engine, context = build_tensorrt_engine(onnx_model_path, test_loader, batch_size)
        device_input, device_attention_mask, device_output, stream_ptr, torch_stream = test_data(context, batch_size)

        
        # Schleife für durchschnitt
        latency_ms_sum = 0
        latency_synchronize_sum = 0
        lantency_datatransfer_sum = 0
        total_time_sum = 0
        num_executions = 10.0
        for i in range(int(num_executions)):
            start_time = time.time()
            latency_ms, latency_synchronize, latency_datatransfer = measure_latency(
                context=context,
                test_loader=test_loader,
                device_input=device_input,
                device_attention_mask=device_attention_mask,
                device_output=device_output,
                stream_ptr=stream_ptr,
                torch_stream=torch_stream,
                batch_size=batch_size
            )
            latency_ms_sum = latency_ms_sum + latency_ms
            latency_synchronize_sum = latency_synchronize_sum + (latency_synchronize-latency_ms)
            lantency_datatransfer_sum = lantency_datatransfer_sum + (latency_datatransfer-latency_synchronize)

            end_time = time.time()
            total_time_sum = total_time_sum + (end_time-start_time)


        latency_avg = float(latency_ms_sum/num_executions)
        latency_synchronize_avg = float(latency_synchronize_sum/num_executions)
        latency_datatransfer_avg = float(lantency_datatransfer_sum/num_executions)
        total_time_avg = float(total_time_sum/num_executions)

        num_batches = int(7600/batch_size) 
        throughput_batches = num_batches/(total_time_avg) 
        throughput_images = (num_batches*batch_size)/(total_time_avg)


        log_latency_inteference = {"batch_size": batch_size, "type":"inteference", "value": latency_avg/batch_size} # pro datensatz?
        log_latency_synchronize = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize_avg/batch_size)} # pro datensatz?
        log_latency_datatransfer = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer_avg/batch_size)} # pro datensatz?
        log_latency_inteference_batch = {"batch_size": batch_size, "type":"inteference", "value": latency_avg} #pro batch
        log_latency_synchronize_batch = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize_avg)} #pro batch
        log_latency_datatransfer_batch = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer_avg)} #pro batch 
        throughput = {"batch_size": batch_size, "throughput_images_per_s": throughput_images, "throughput_batches_per_s": throughput_batches}


        throughput_log.append(throughput)
        latency_log.extend([log_latency_inteference, log_latency_synchronize, log_latency_datatransfer])
        latency_log_batch.extend([log_latency_inteference_batch, log_latency_synchronize_batch, log_latency_datatransfer_batch])
        # print_latency(latency_avg, latency_synchronize_avg+latency_avg, latency_datatransfer_avg+latency_synchronize_avg+latency_avg, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size)

    return throughput_log, latency_log, latency_log_batch

def test_data(context, batch_size):
    input_name = "input_ids"
    input_name_2 = "attention_mask"
    output_name = "logits"
    input_shape = (batch_size, 128) # anpassen
    output_shape = (batch_size, 4)
    device_input = torch.empty(input_shape, dtype=torch.int64, device='cuda')  # Eingabe auf der GPU
    device_attention_mask = torch.empty(input_shape, dtype=torch.int64, device='cuda') #Maske auf der GPU
    device_token_type_ids = torch.zeros(input_shape, dtype=torch.int64, device='cuda')
    device_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')  # Ausgabe auf der GPU
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream
    context.set_tensor_address(input_name, device_input.data_ptr()) 
    context.set_tensor_address(input_name_2, device_attention_mask.data_ptr())  # EingabeTensor verknüpfen
    context.set_tensor_address("token_type_ids", device_token_type_ids.data_ptr())  # Token Type IDs verknüpfen
    context.set_tensor_address(output_name, device_output.data_ptr())  # AusgabeTensor verknüpfen
    context.set_input_shape("input_ids", (batch_size, 128)) # muss man bei dynamischen batch sizes machen
    context.set_input_shape("attention_mask", (batch_size, 128)) # muss man bei dynamischen batch sizes machen
    context.set_input_shape("token_type_ids", (batch_size, 128))
    return device_input, device_attention_mask, device_output, stream_ptr, torch_stream

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

if __name__ == "__main__":
    onnx_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert_int8" / "model.onnx"

    data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"

    # engine, context = build_tensorrt_engine(onnx_model_path)


    # dvc pipeline vervollständigen: parameter bei quantisierung als option, so dass ich nichts im code ändern muss - fertig
    # accuracy ag_news dataset - is 92% enough - ja, sieht gut aus https://www.researchgate.net/figure/Performance-test-accuracy-on-AG-News_fig4_360591395
    # dvc experiment tracking dvc exp - abends laufen lassen
    # calibration mit tensorrt - funktioniert, aber ist schlechter als mit brevitas... ganzes experiment (viele layer konnten nicht quantisiert werden)
    # mehrere tensorrt engines bauen, yaml file, weitere stage mit plots - fertig

    # paper schreiben
    # Die Quantisierung mit Tensorrt auf Int8 hat teilweise funktioniert, aber es ist danach auch nicht schneller. Es konnten auch viele nodes nicht quantisiert werden, ich weiß noch nicht ob das an meinem Kalibrator liegt.
    correct_predictions, total_predictions = run_inference(batch_size=1)  # Teste Inferenz mit Batch Size 1
    print(f"Accuracy : {correct_predictions / total_predictions:.2%}")
    accuracy_result = {
        "quantisation_type": "INT8",
        "value": correct_predictions / total_predictions
    }
    accuracy_path = Path(__file__).resolve().parent.parent / "eval_results" /"accuracy_INT8.json"
    save_json(accuracy_result, accuracy_path)


    params = load_params()
    batch_sizes = params["measure"]["batch_sizes"]

    context=0
    throughput_log, latency_log, latency_log_batch = calculate_latency_and_throughput(context, batch_sizes, onnx_model_path)

    throughput_results = Path(__file__).resolve().parent.parent / "throughput" / "INT8" /"throughput_results.json"
    throughput_results2 = Path(__file__).resolve().parent.parent / "throughput"/ "INT8" / "throughput_results_2.json"
    latency_results = Path(__file__).resolve().parent.parent / "throughput" / "INT8"/ "latency_results.json"
    latency_results_batch = Path(__file__).resolve().parent.parent / "throughput" / "INT8"/ "latency_results_batch.json"
    save_json(throughput_log, throughput_results)
    save_json(throughput_log, throughput_results2)
    save_json(latency_log, latency_results)
    save_json(latency_log_batch, latency_results_batch)

    batch_size = 1



    correct_predictions, total_predictions = run_inference(batch_size)


