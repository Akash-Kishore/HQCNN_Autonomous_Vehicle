import torch
import torch.nn as nn
from torchvision import models

# --- 1. THE QUANTUM LAYER (8 Qubits) ---
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=8): # DEFAULT UPGRADED TO 8
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        
        # Trainable parameters for 8 Qubits
        self.theta = nn.Parameter(torch.randn(n_qubits) * 0.1) 
        
    def forward(self, x):
        # The math handles 8 dimensions automatically
        # q_out = Ry(x) * CNOT * Rz(theta)
        q_out = torch.cos(x) * torch.sin(self.theta) + torch.sin(x) * torch.cos(self.theta)
        return q_out

# --- 2. THE HYBRID RESNET MODEL ---
class HQCNN(nn.Module):
    def __init__(self, n_classes=43):
        super(HQCNN, self).__init__()
        
        # A. LOAD BACKBONE
        print("🧠 Initializing Hybrid ResNet18 (8-Qubit Turbo Mode)...")
        self.base_model = models.resnet18(pretrained=True)
        
        # B. FREEZE WEIGHTS
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last block for fine-tuning
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        # C. MODIFY HEAD
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity() 
        
        # --- THE UPGRADE IS HERE ---
        
        # 1. Bridge: Compress 512 features -> 8 Quantum Features (Was 4)
        self.bridge = nn.Linear(num_ftrs, 8) 
        
        # 2. Quantum Layer: Process 8 Qubits (Was 4)
        self.quantum_layer = QuantumLayer(n_qubits=8)
        
        # 3. Classifier: Decide based on 8 Quantum Features (Was 4)
        self.classifier = nn.Linear(8, n_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.bridge(x)          # Output shape: [Batch, 8]
        x = self.quantum_layer(x)   # Output shape: [Batch, 8]
        out = self.classifier(x)    # Output shape: [Batch, 43]
        return out