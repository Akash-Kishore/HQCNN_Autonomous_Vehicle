import torch
import torch.nn as nn
from torchvision import models

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=8): 
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.theta = nn.Parameter(torch.randn(n_qubits) * 0.1) 
        
    def forward(self, x):
        # Simulated Quantum Circuit (Fast & Differentiable)
        q_out = torch.cos(x) * torch.sin(self.theta) + torch.sin(x) * torch.cos(self.theta)
        return q_out

class HQCNN_Residual(nn.Module):
    def __init__(self, n_classes=43):
        super(HQCNN_Residual, self).__init__()
        
        # 1. Backbone (ResNet18)
        self.base_model = models.resnet18(pretrained=True)
        
        # Freeze early layers, Unfreeze Layer4 for fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # Remove original head
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity() 
        
        # 2. Path A: Classical (Stability -> High Accuracy)
        self.classical_path = nn.Linear(num_ftrs, n_classes)
        
        # 3. Path B: Quantum (Regularization -> Robustness)
        self.bridge = nn.Linear(num_ftrs, 8) 
        self.quantum_layer = QuantumLayer(n_qubits=8)
        self.quantum_head = nn.Linear(8, n_classes)

    def forward(self, x):
        features = self.base_model(x)
        
        # Parallel Execution
        classical_out = self.classical_path(features)
        
        q_in = self.bridge(features)
        q_proc = self.quantum_layer(q_in)
        quantum_out = self.quantum_head(q_proc)
        
        # Residual Sum
        return classical_out + quantum_out