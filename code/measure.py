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

FP16 = os.environ.get("FP16", "0") == "1"

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
    

# accuracy vom LLM berechnen

def accuracy(labels, outputs):
    # funktioniert nicht mit größerer batch size
    correct_predictions = 0
    total_predictions = 0
    i = 0
    for label in labels: 
        predicted = np.argmax(outputs, axis=1)
        total_predictions = total_predictions + 1
        if predicted == label:
            correct_predictions = correct_predictions + 1
        i = i+1
    return correct_predictions, total_predictions

def save_json(log, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ordner anlegen, falls nicht vorhanden
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

        # Nur Eingaben verwenden (nicht Labels)
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
    total_time_datatransfer = 0  # Gesamte Laufzeit aller gemessenen Batches
    iterations = 0  # Anzahl gemessener Batches
    # wie kann ich die input-sätze von dem Dataloader in den device_input buffer laden?
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


    # FP16 Quantisierung
    if FP16 == True:
        config.set_flag(trt.BuilderFlag.FP16)
        

    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        name = network.get_input(i).name
        profile.set_shape(name, (1, 128), (8, 128), (1024, 128))
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
        #print_latency(latency_avg, latency_synchronize_avg+latency_avg, latency_datatransfer_avg+latency_synchronize_avg+latency_avg, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size)

    return throughput_log, latency_log, latency_log_batch

def test_data(context, batch_size):
    input_name = "input_ids"
    input_name_2 = "attention_mask"
    output_name = "logits"
    input_shape = (batch_size, 128) # anpassen
    output_shape = (batch_size, 4)
    device_input = torch.empty(input_shape, dtype=torch.int32, device='cuda')  # Eingabe auf der GPU
    device_attention_mask = torch.empty(input_shape, dtype=torch.int32, device='cuda') #Maske auf der GPU
    device_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')  # Ausgabe auf der GPU
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream
    context.set_tensor_address(input_name, device_input.data_ptr()) 
    context.set_tensor_address(input_name_2, device_attention_mask.data_ptr())  # EingabeTensor verknüpfen
    context.set_tensor_address(output_name, device_output.data_ptr())  # AusgabeTensor verknüpfen
    context.set_input_shape("input_ids", (batch_size, 128)) # muss man bei dynamischen batch sizes machen
    context.set_input_shape("attention_mask", (batch_size, 128)) # muss man bei dynamischen batch sizes machen
    return device_input, device_attention_mask, device_output, stream_ptr, torch_stream

def append_json(new_entry, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Bestehende Daten laden, falls vorhanden
    if filepath.exists():
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except Exception:
                data = []
    else:
        data = []
    data.append(new_entry)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

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

        device_input.copy_(input_ids.to(torch.int32))

        

        device_attention_mask.copy_(attention_mask.to(torch.int32))

        torch_stream.synchronize()
        
        with torch.cuda.stream(torch_stream): # nicht für mehr als 64 Bildern möglich
            context.execute_async_v3(stream_ptr)
        torch_stream.synchronize()

        output = device_output.cpu().numpy()

        correct, total = accuracy(labels, output)
        total_predictions += total
        correct_predictions += correct
    return correct_predictions, total_predictions

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

if __name__ == "__main__":
    onnx_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"

    data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"

    params = load_params()
    batch_sizes = params["measure"]["batch_sizes"]


    context=0
    correct_predictions, total_predictions = run_inference(batch_size=1)  # Teste Inferenz mit Batch Size 1
    print(f"Accuracy : {correct_predictions / total_predictions:.2%}")
    
    accuracy_path = Path(__file__).resolve().parent.parent / "eval_results" /"accuracy_FP16.json" if FP16 else Path(__file__).resolve().parent.parent / "eval_results" /"accuracy_FP32.json"
    quantisation_type = "FP16" if FP16 else "FP32"
    accuracy_result = {
        "quantisation_type": quantisation_type,
        "value": correct_predictions / total_predictions
    }
    save_json(accuracy_result, accuracy_path)


    throughput_log, latency_log, latency_log_batch = calculate_latency_and_throughput(context, batch_sizes, onnx_model_path)
    if FP16:
        throughput_results = Path(__file__).resolve().parent.parent / "throughput" / "FP16" / "throughput_results.json"
        throughput_results2 = Path(__file__).resolve().parent.parent / "throughput" / "FP16"/ "throughput_results_2.json"
        latency_results = Path(__file__).resolve().parent.parent / "throughput" / "FP16"/ "latency_results.json"
        latency_results_batch = Path(__file__).resolve().parent.parent / "throughput" / "FP16"/ "latency_results_batch.json"
    else:
        throughput_results = Path(__file__).resolve().parent.parent / "throughput" / "FP32"/ "throughput_results.json"
        throughput_results2 = Path(__file__).resolve().parent.parent / "throughput" / "FP32"/ "throughput_results_2.json"
        latency_results = Path(__file__).resolve().parent.parent / "throughput" / "FP32"/ "latency_results.json"
        latency_results_batch = Path(__file__).resolve().parent.parent / "throughput" / "FP32"/ "latency_results_batch.json"
    save_json(throughput_log, throughput_results)
    save_json(throughput_log, throughput_results2)
    save_json(latency_log, latency_results)
    save_json(latency_log_batch, latency_results_batch)

