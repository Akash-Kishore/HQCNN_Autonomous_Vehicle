import torch
import torch.nn as nn
# We use a relative import here to ensure stability within the src package
from .quantum_circuit import QuantumLayer 

class HQCNN(nn.Module):
    def __init__(self, n_classes=43):
        super(HQCNN, self).__init__()
        
        # QUANTUM LAYER
        # Processes 256 inputs (16x16 patch) -> Output 8 quantum features
        self.quantum_layer = QuantumLayer(n_qubits=8, n_layers=2)
        
        # CLASSICAL HEAD
        # Input: 4 patches * 8 quantum features = 32 Total Features
        self.classical_layers = nn.Sequential(
            nn.Linear(32, 128), 
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(128, 84), 
            nn.ReLU(),
            nn.Linear(84, n_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # PATCHING LOGIC
        # Split 32x32 image into four 16x16 quadrants (256 pixels each)
        # x shape is [Batch, 1, 32, 32]
        patches = [
            x[:, 0, 0:16, 0:16].reshape(batch_size, 256),   # Top-Left
            x[:, 0, 0:16, 16:32].reshape(batch_size, 256),  # Top-Right
            x[:, 0, 16:32, 0:16].reshape(batch_size, 256),  # Bottom-Left
            x[:, 0, 16:32, 16:32].reshape(batch_size, 256)  # Bottom-Right
        ]
        
        # QUANTUM PROCESSING
        # Run the circuit 4 times per image (once per patch)
        q_results = [self.quantum_layer(p) for p in patches]
        
        # FUSION
        # Concatenate the 4 outputs: [Batch, 8] x 4 -> [Batch, 32]
        combined = torch.cat(q_results, dim=1)
        
        # CLASSIFICATION
        return self.classical_layers(combined)