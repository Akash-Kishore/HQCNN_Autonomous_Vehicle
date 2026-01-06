
import torch
import torch.nn as nn
from src.quantum_circuit import QuantumLayer

class HQCNN(nn.Module):
    def __init__(self, n_classes=43):
        super(HQCNN, self).__init__()
        self.quantum_layer = QuantumLayer(n_qubits=8, n_layers=2)
        self.classical_layers = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 84), nn.ReLU(),
            nn.Linear(84, n_classes)
        )
    def forward(self, x):
        batch_size = x.size(0)
        # Split 32x32 image into four 16x16 patches (256 pixels each)
        patches = [
            x[:, 0, 0:16, 0:16].reshape(batch_size, 256),
            x[:, 0, 0:16, 16:32].reshape(batch_size, 256),
            x[:, 0, 16:32, 0:16].reshape(batch_size, 256),
            x[:, 0, 16:32, 16:32].reshape(batch_size, 256)
        ]
        q_results = [self.quantum_layer(p) for p in patches]
        combined = torch.cat(q_results, dim=1)
        return self.classical_layers(combined)
