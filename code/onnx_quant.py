from onnxruntime.quantization import CalibrationDataReader, QuantFormat
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.quantize import quantize_static, quantize_dynamic
from onnxruntime.quantization import QuantType
import onnx
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Beispiel-Dummy-Datenleser
class DummyDataReader(CalibrationDataReader):
    def __init__(self, input_name, input_mask_name):
        # Dynamische Batch-Größe (z.B. 1 für Batch-Größe 1 oder beliebig)
        seq_len = 128  # Maximale Sequenzlänge
        vocab_size = 30522  # Typische Vokabgröße für BERT-Modelle

        # Erstellen der Dummy-Daten mit einer variablen Batch-Größe
        self.data = [
            {
                input_name: np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int32),  # Batch-Größe 1
                input_mask_name: np.ones((1, seq_len), dtype=np.int32)  # Alle Tokens sichtbar
            }
        ]
        self.enum_data = iter(self.data)

    def get_next(self):
        # Return data with dynamic batch size
        print("DummyDataReader get_next")
        return next(self.enum_data, None)
    


class MyTestSetDataset(Dataset):
    def __init__(self, data, labels, mask=None):
        # Daten und Labels (optional auch Masken) müssen geladen werden
        self.data = data
        self.labels = labels
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx]
        input_mask = self.mask[idx] if self.mask is not None else None
        return input_data, input_mask

class RealDataReader:
    def __init__(self, input_data, input_mask, batch_size=8):
        print("RealDataReader init")
        self.input_data = input_data[:8]
        self.input_mask = input_mask[:8]
        self.batch_size = batch_size
        self.index = 0  # Zum Tracken des aktuellen Index für den Batch
        self.iter = 0  # Zum Tracken der Iterationen
        self.max_iter = 1000

    def get_next(self):
        # Hole einen Batch von den Daten
        #print("next", self.iter)
        if self.iter >= self.max_iter:
            return None  # Keine Daten mehr, daher None zurückgeben
        batch_input = self.input_data[self.index:self.index + self.batch_size]
        batch_mask = self.input_mask[self.index:self.index + self.batch_size]

        # Konvertiere die Eingabedaten und Masken zu int32
        batch_input = batch_input.astype(np.int32)
        batch_mask = batch_mask.astype(np.int32)
        
        # Erstelle ein Dictionary mit den Eingabedaten und der Maske
        inputs = {
            'input_ids': batch_input,  # Der Name des Eingabewerts sollte mit dem Eingabe-Tensor im ONNX-Modell übereinstimmen
            'attention_mask': batch_mask  # Achte darauf, dass der Name hier mit dem Modell übereinstimmt
        }

        # Aktualisiere den Index für den nächsten Batch
        self.index += self.batch_size
        if self.index >= len(self.input_data):
            self.index = 0  # Zum Wiederholen der Daten
            self.iter += 1

        return inputs  # Gib das Dictionary zurück

# Lade ONNX-Modell
model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert.onnx"
preprocessed_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert_preprocessed.onnx"
cleaned_model_path = Path(__file__).resolve().parent.parent / "models" / "tinybert_cleaned.onnx"
data_path = Path(__file__).resolve().parent.parent / "datasets" / "tokenized_agnews_test.pt"
model = onnx.load(model_path)


# Namen der Input-Tensoren ermitteln
input_name = model.graph.input[0].name
input_mask_name = model.graph.input[1].name

# Lade die tokenisierte AG News Testdaten
data = torch.load(data_path)

input_data = data['input_ids'].numpy()  # Tokenisierte Eingabedaten
input_mask = data['attention_mask'].numpy()  # Attention-Maske
labels = data['labels'].numpy()  # Labels
print("after loading data")
# Erstelle das Dataset mit den echten Daten
dataset = MyTestSetDataset(input_data, labels=None, mask=input_mask)
print("after creating dataset")
# Erstelle den Kalibrierungsleser
calibration_data_reader = RealDataReader(input_data, input_mask)
print("after creating calibration data reader")
# Quantisiere Modell (PTQ mit QDQ-Nodes)
quantized_model_path = Path(__file__).resolve().parent.parent / "models" / "model_quantized_onnx_run.onnx"
model = quant_pre_process(
    input_model=model_path,
    output_model_path=preprocessed_model_path,
    quant_format=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Conv", "Gemm", "Add", "Sub", "Mul", "Div"],
)
print("after preprocessing model")
import onnx_graphsurgeon as gs

model = onnx.load(preprocessed_model_path)
graph = gs.import_onnx(model)

# Original Nodes sichern (Kopie!)
original_nodes = list(graph.nodes)

# Neue Nodes vorbereiten
new_nodes = []

for node in original_nodes:
    if node.op == "LayerNormalization" and "LayerNorm" in node.name:
        print(f"Ersetze Node: {node.name}")

        # Nur wenn Node tatsächlich noch im Graph ist!
        if node in graph.nodes:
            input_tensor = node.inputs[0]
            output_tensor = node.outputs[0]

            # Ersatz-Node
            identity_node = gs.Node(op="Identity", inputs=[input_tensor], outputs=[output_tensor], name=node.name + "_identity")
            new_nodes.append(identity_node)

            # Alten entfernen
            graph.nodes.remove(node)

# Neue Identity-Nodes anhängen
graph.nodes.extend(new_nodes)

# Cleanup und Toposort
graph.cleanup().toposort()


onnx.save(gs.export_onnx(graph), cleaned_model_path)



print("after surgeon")
quantize_static(
    model_input=cleaned_model_path,
    model_output=quantized_model_path,
    #calibration_data_reader=DummyDataReader(input_name, input_mask_name),
    calibration_data_reader=calibration_data_reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
    reduce_range=False,
    # nodes_to_exclude=[
    #     "/bert/embeddings/LayerNorm/LayerNormalization", 
    #     "/bert/encoder/layer.0/attention/output/LayerNorm/LayerNormalization",
    #     "/bert/encoder/layer.0/output/LayerNorm/LayerNormalization"     #ITensor::getDimensions: Error Code 3: API Usage Error (bert.encoder.layer.0.output.LayerNorm.bias_DequantizeLinear: only activation types allowed as input to this layer.)
    #     # [05/13/2025-15:47:45] [TRT] [E] ITensor::getDimensions: Error Code 3: API Usage Error (bert.encoder.layer.0.output.LayerNorm.bias_DequantizeLinear: only activation types allowed as input to this layer.)
    #     # [05/13/2025-15:47:45] [TRT] [E] In node 12 with name: bert.encoder.layer.0.output.LayerNorm.bias_DequantizeLinear and operator: DequantizeLinear (parseNode): INVALID_NODE: Invalid Node - bert.encoder.layer.0.output.LayerNorm.bias_DequantizeLinear
        
    #     "/bert/encoder/layer.1/attention/output/LayerNorm/LayerNormalization",
    #     "/bert/encoder/layer.1/output/LayerNorm/LayerNormalization"
    # ],
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "OpTypesToExcludeOutputQuantization": ["LayerNormalization"],  # <-- wichtig
        # "AddQDQPairToWeight": False  # Deaktiviert die Quantisierung von Bias-Werten
    }
)

print("Quantisiertes ONNX-Modell gespeichert unter:", quantized_model_path)

# mit den echten Kalibrierungsdaten dauert die quantisierung ewig (auf der CPU) --> anzahl daten verringert
# mit den dummy Daten geht es schnell, aber es gibt fehler (eventuell gibts die Fehler auch bei den echten Daten)
# Fehler: es gibt scheinbar einen Layer der nicht quantisiert werden kann
# https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
# zu wenige kalibrierungsdaten führen zu overflows?





# https://github.com/Xilinx/brevitas
# brevitas nutzen!