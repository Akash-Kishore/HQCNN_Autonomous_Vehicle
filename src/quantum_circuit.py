
import pennylane as qml
import torch
import torch.nn as nn

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2, device_name="lightning.gpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        try:
            self.dev = qml.device(device_name, wires=n_qubits)
        except:
            print("⚠️ GPU device not found, falling back to CPU.")
            self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), pad_with=0., normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)
