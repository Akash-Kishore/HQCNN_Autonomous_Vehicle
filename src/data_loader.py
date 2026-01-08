import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import numpy as np

# --- 1. DATASET CLASS ---
class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read path from CSV
        img_rel_path = self.df.iloc[idx]['Path']
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        # Open as PIL RGB (Critical for ResNet)
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            # Fallback for corrupt images
            return self.__getitem__((idx + 1) % len(self.df))

        label = int(self.df.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 2. LOADER FUNCTION (Scientifically Rigorous) ---
def get_data_loaders(data_dir, batch_size=32, percent=1.0):
    """
    Args:
        percent (float): Set to 0.20 for the Paper Experiment (20% data).
    """
    train_csv = os.path.join(data_dir, 'Train.csv')
    
    # --- A. DEFINE TRANSFORMS ---
    # ResNet18 REQUIRES specific Normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Training: Add Noise/Rotation (Robustness)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),       # ResNet Input Size
        transforms.RandomRotation(15),       # Paper Novelty
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Validation: Clean (No Rotation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- B. LOAD INDICES ---
    # We load the CSV once just to get the length and indices
    df = pd.read_csv(train_csv)
    n_samples = len(df)
    indices = list(range(n_samples))

    # --- C. EFFICIENCY SLICING (The Paper Logic) ---
    if percent < 1.0:
        # We reduce the TOTAL dataset to 20% before splitting
        n_samples = int(n_samples * percent)
        np.random.seed(42) # Reproducible Science
        np.random.shuffle(indices)
        indices = indices[:n_samples]
        print(f"✂️ SLICING DATA: Reduced dataset to {n_samples} images ({percent*100}%)")

    # --- D. SPLIT TRAIN/VAL INDICES ---
    # 85% Train, 15% Val
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    # --- E. CREATE DATASETS ---
    # CRITICAL FIX: We create TWO dataset objects.
    # One with train_transform, one with val_transform.
    # We then use Subset to assign the indices.
    
    train_set_full = GTSRBDataset(data_dir, train_csv, transform=train_transform)
    val_set_full = GTSRBDataset(data_dir, train_csv, transform=val_transform)

    train_dataset = Subset(train_set_full, train_idx)
    val_dataset = Subset(val_set_full, val_idx)

    # --- F. LOADERS ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"✅ DATA READY: {len(train_dataset)} Train | {len(val_dataset)} Val")
    return train_loader, val_loader