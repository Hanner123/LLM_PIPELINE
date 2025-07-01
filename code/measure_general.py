import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import torch
import onnx
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pycuda.driver as cuda
import pycuda.autoinit
import os
import yaml
from onnxconverter_common import float16 # zu requirements hinzufügen
import onnxruntime as ort

# tensorrt, datasets(hugging face), pycuda
FP16 = os.environ.get("FP16", "0") == "1"
if FP16:
    dtype = torch.float16
    print("FP16 enabled")
else:
    dtype = torch.float32
    print("FP32")

def to_device(data,device):
    if isinstance(data, (list,tuple)): 
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

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def accuracy(labels, outputs):
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
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)


def parse_shape(shape, batch_value):
    """Ersetzt 'batch_size' durch batch_value in der shape-Liste."""
    return tuple(batch_value if d == "batch_size" else d for d in shape)


ONNX_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(uint8)": torch.uint8,
    "tensor(int8)": torch.int8,
    "tensor(bool)": torch.bool,
    # Füge weitere Typen bei Bedarf hinzu
}


def onnx_dtype_to_torch(onnx_dtype_str):
    """
    Wandelt einen ONNX-Datentyp-String in einen torch.dtype um.
    """
    return ONNX_TO_TORCH_DTYPE.get(onnx_dtype_str, torch.float32)  # Default: float32


def get_model_io_info(model_path):
    """
    Liest Input- und Output-Infos aus einem ONNX-Modell.
    Gibt Listen von Dictionaries mit Name, Shape und Dtype zurück.
    """
    session = ort.InferenceSession(model_path)
    input_info = [
        {
            "name": inp.name,
            "shape": inp.shape,
            "dtype": inp.type
        }
        for inp in session.get_inputs()
    ]
    output_info = [
        {
            "name": out.name,
            "shape": out.shape,
            "dtype": out.type
        }
        for out in session.get_outputs()
    ]
    return input_info, output_info


def print_latency(latency_ms, latency_synchronize, latency_datatransfer, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size):
    print("For Batch Size: ", batch_size)
    print(f"Gemessene durchschnittliche Latenz für Inteferenz : {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation : {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer : {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time-start_time:.4f} s")
    print("num_batches", num_batches)
    print(f"Throughput: {throughput_batches:.4f} Batches/Sekunde")
    print(f"Throughput: {throughput_images:.4f} Bilder/Sekunde")


def create_test_dataloader(data_path, batch_size):
    """
    Erstellt den DataLoader für die Testdaten.
    :param data_path: Pfad zur Testdaten-Datei.
    :param batch_size: Die Batchgröße.
    :return: DataLoader-Objekt für die Testdaten.
    """
    data = np.load(data_path)
    input_info, output_info = get_model_io_info(onnx_model_path)
    key_list = list(data.keys())
    if len(input_info) > 1:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    else:
        input_key = key_list[0]
        attention_mask_key = None
        output_key = key_list[1]
    
    
    input_ids = torch.from_numpy(data[input_key])
    attention_mask = torch.from_numpy(data[attention_mask_key]) if attention_mask_key else None
    labels = torch.from_numpy(data[output_key])

    if len(input_info) > 1:
        test_dataset = TensorDataset(input_ids, attention_mask, labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    else:
        test_dataset = TensorDataset(input_ids, labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    return test_loader

# Spezifisch für den Datensatz und das Modell!
# numpy array als allgemeine form!! Einrichten
def create_test_dataloader_radiollm(data_path, batch_size, seq_len=32, emb_dim=64):
    import h5py
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader

    with h5py.File(data_path, "r") as f:
        X = np.array(f["X"][:10000])  # Nur die ersten 1000 Datensätze
        Y = np.array(f["Y"][:10000])

    X = X.reshape(X.shape[0], -1)           # [samples, 2048]
    X = X.reshape(-1, seq_len, emb_dim)     # [samples', 32, 64]

    # Labels ggf. anpassen (z.B. argmax, expand, ... wie im Training)
    if Y.ndim == 2 and Y.shape[1] > 1:
        Y = np.argmax(Y, axis=1)
    Y = np.tile(Y[:, None], (1, seq_len))   

    X = torch.tensor(X, dtype=dtype) 
    Y = torch.tensor(Y, dtype=torch.long)
    test_dataset = TensorDataset(X, Y)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    return test_loader

def test_data_old(context, batch_size):
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


def test_data(context, batch_size, input_info, output_info):
    device_inputs = {}
    device_outputs = {}
    device_attention_masks = {}
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream

    # Inputs vorbereiten
    inp = input_info[0]
    name = inp["name"]
    shape = parse_shape(inp["shape"], batch_size)
    dtype = onnx_dtype_to_torch(inp["dtype"])  # ONNX-Datentyp in PyTorch-Datentyp umwandeln
    tensor = torch.empty(shape, dtype=dtype, device='cuda')
    context.set_tensor_address(name, tensor.data_ptr())
    context.set_input_shape(name, shape)
    device_inputs[name] = tensor
    
    # Wenn mehrere Inputs: attention_mask
    if len(input_info) > 1:
        att_mask_name = input_info[1]["name"]
        att_mask_shape = parse_shape(input_info[1]["shape"], batch_size)
        att_mask_dtype = onnx_dtype_to_torch(input_info[1]["dtype"])
        att_mask_tensor = torch.empty(att_mask_shape, dtype=att_mask_dtype, device='cuda')
        context.set_tensor_address(att_mask_name, att_mask_tensor.data_ptr())

        device_attention_masks[att_mask_name] = att_mask_tensor

    # Outputs vorbereiten
    for out in output_info:
        name = out["name"]
        shape = parse_shape(out["shape"], batch_size)
        dtype = onnx_dtype_to_torch(out["dtype"])  # ONNX-Datentyp in PyTorch-Datentyp umwandeln
        tensor = torch.empty(shape, dtype=dtype, device='cuda')
        context.set_tensor_address(name, tensor.data_ptr())
        device_outputs[name] = tensor

    device_input = next(iter(device_inputs.values()))
    device_output = next(iter(device_outputs.values()))
    # if another input: nächste attmask adresse
    if len(input_info) > 1:
        device_attention_mask = next(iter(device_attention_masks.values()))
    else:
        device_attention_mask = None

    return device_input, device_output, device_attention_mask, stream_ptr, torch_stream


def build_tensorrt_engine(onnx_model_path, test_loader, batch_size, input_info=None, min_bs=1, opt_bs=8, max_bs=1024):
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

    if FP16 == True:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()

    for inp in input_info:
        name = inp["name"]
        shape = inp["shape"]
        min_shape = parse_shape(shape, min_bs)
        opt_shape = parse_shape(shape, opt_bs)
        max_shape = parse_shape(shape, max_bs)
        profile.set_shape(name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    return engine, context


def measure_latency(context, test_loader, device_input, device_output, device_attention_mask, stream_ptr, torch_stream, batch_size=1, input_info=None, output_info=None):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """
    total_time = 0
    total_time_synchronize = 0
    total_time_datatransfer = 0  
    iterations = 0 
    for xb, att_mask, yb in test_loader:  
        start_time_datatransfer = time.time()  # Startzeit

        # Buffer-Addresses und Shape JEDES MAL neu setzen!
        input_name = input_info[0]["name"]
        if len(input_info) > 1:
            att_mask_name = input_info[1]["name"]
        output_name = output_info[0]["name"]
        dtype = onnx_dtype_to_torch(input_info[0]["dtype"])

        device_input.copy_(xb.to(dtype))
        if len(input_info) > 1:
            device_attention_mask.copy_(att_mask.to(dtype))

        context.set_tensor_address(input_name, device_input.data_ptr())
        if len(input_info) > 1:
            context.set_tensor_address(att_mask_name, device_attention_mask.data_ptr())
        context.set_tensor_address(output_name, device_output.data_ptr())
        context.set_input_shape(input_name, device_input.shape)
        if len(input_info) > 1:
            context.set_input_shape(att_mask_name, device_attention_mask.shape)

        torch_stream.synchronize()

        start_time_synchronize = time.time()  
        torch_stream.synchronize()  

        start_time_inteference = time.time() 
        try:
            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
        except Exception as e:
            print("TensorRT Error:", e)
        torch_stream.synchronize() 
        end_time = time.time()

        output = device_output.cpu().numpy()
        end_time_datatransfer = time.time() 

        latency = end_time - start_time_inteference  
        latency_synchronize = end_time - start_time_synchronize  
        latency_datatransfer = end_time_datatransfer - start_time_datatransfer  

        total_time += latency
        total_time_synchronize += latency_synchronize
        total_time_datatransfer += latency_datatransfer
        iterations += 1
        
       
    average_latency = (total_time / iterations) * 1000  # In Millisekunden
    average_latency_synchronize = (total_time_synchronize / iterations) * 1000  # In Millisekunden
    average_latency_datatransfer = (total_time_datatransfer / iterations) * 1000  # In Millisekunden


    return average_latency, average_latency_synchronize, average_latency_datatransfer


def calculate_latency_and_throughput(context, batch_sizes, onnx_model_path, input_info, output_info):
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
        test_loader = create_test_dataloader(data_path, batch_size) 
        engine, context = build_tensorrt_engine(onnx_model_path, test_loader, batch_size, input_info)
        device_input, device_output, device_attention_mask, stream_ptr, torch_stream = test_data(context, batch_size, input_info, output_info)

        
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
                device_output=device_output,
                device_attention_mask=device_attention_mask,
                stream_ptr=stream_ptr,
                torch_stream=torch_stream,
                batch_size=batch_size,
                input_info=input_info,
                output_info=output_info
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
        print_latency(latency_avg, latency_synchronize_avg+latency_avg, latency_datatransfer_avg+latency_synchronize_avg+latency_avg, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size)

    return throughput_log, latency_log, latency_log_batch


def run_inference(batch_size=1, input_info=None, output_info=None):
    """pynvml-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param max_iterations: Maximalanzahl der Iterationen.
    :return: (Anzahl der korrekten Vorhersagen, Gesamtanzahl der Vorhersagen).
    """

    test_loader = create_test_dataloader(data_path, batch_size)
    engine, context = build_tensorrt_engine(onnx_model_path, test_loader, batch_size, input_info)
    device_input, device_output, device_attention_mask, stream_ptr, torch_stream = test_data(context, batch_size, input_info, output_info)
    
    total_predictions = 0
    correct_predictions = 0


    # ist unterschiedlich je nach modell - aber eig. sind die ähnlich aufgebaut, hier kommt noch die att_mask dazu!!
    for xb, att_mask, yb in test_loader:

        input_name = input_info[0]["name"]
        att_mask_name = input_info[1]["name"]  # Falls attention_mask 
        output_name = output_info[0]["name"]
        dtype = onnx_dtype_to_torch(input_info[0]["dtype"])

        device_input.copy_(xb.to(dtype))
        device_attention_mask.copy_(att_mask.to(dtype)) # eig dtype von der mask...


        context.set_tensor_address(input_name, device_input.data_ptr())
        context.set_tensor_address(att_mask_name, device_attention_mask.data_ptr())
        context.set_tensor_address(output_name, device_output.data_ptr())
        context.set_input_shape(input_name, device_input.shape) 
        context.set_input_shape(att_mask_name, device_attention_mask.shape) 
        torch_stream.synchronize()
        
        try:
            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
        except Exception as e:
            print("TensorRT Error:", e)
        torch_stream.synchronize()
        torch.cuda.synchronize()  # Warten auf Abschluss der Inferenz

        output = device_output.cpu().numpy()
        # print("Labels:", yb[:5])
        # print("Output:", output[:5])

        correct, total = accuracy(yb, output)
        total_predictions += total
        correct_predictions += correct
    return correct_predictions, total_predictions


if __name__ == "__main__":
    # muss in parameter datei:
    onnx_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"
    data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"
    data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.npz"


    params = load_params()
    batch_sizes = params["measure"]["batch_sizes"]


    model = onnx.load(onnx_model_path)
    # funktioniert bei radioml, hier auch
    if FP16:
        model = float16.convert_float_to_float16(model) 

    input_info, output_info = get_model_io_info(onnx_model_path)

    context=0

    correct_predictions, total_predictions = run_inference(batch_size=1, input_info=input_info, output_info=output_info) 
    print(f"Accuracy : {correct_predictions / total_predictions:.2%}")

    accuracy_path = Path(__file__).resolve().parent.parent / "eval_results" /"accuracy_FP16.json" if FP16 else Path(__file__).resolve().parent.parent / "eval_results" /"accuracy_FP32.json"
    quantisation_type = "FP16" if FP16 else "FP32"
    accuracy_result = {
        "quantisation_type": quantisation_type,
        "value": correct_predictions / total_predictions
    }
    save_json(accuracy_result, accuracy_path)
    


    throughput_log, latency_log, latency_log_batch = calculate_latency_and_throughput(context, batch_sizes, onnx_model_path, input_info=input_info, output_info=output_info)
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


# code generalisiert - fertig
# code verschönert - fertig
# mit llm pipeline testen - fertig, aber doch noch änderungen mit attention mask in inference schleifen..
# fp16 vergleich datentypen - passt
# bei llm: daten in numpy umwandeln
# test_data_loader anpassen (alle daten als numpy dateien lesen)
# christoph schreiben - er schickt mir andere models zum testen
# mit anderen modellen testen
# mit jetson testen (gleiche zugangsdaten wie pc)
