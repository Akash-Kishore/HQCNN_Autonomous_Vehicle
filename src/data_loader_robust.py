import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torchvision import transforms
import numpy as np

# --- 1. ROBUSTNESS TOOLKIT ---
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return torch.clamp(tensor + noise, 0., 1.)
        return tensor

# --- 2. GTSRB DATASET (CSV-BASED) ---
class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # GTSRB paths in CSV usually start with "Train/" or "Test/"
        img_rel_path = self.df.iloc[idx]['Path']
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            return self.__getitem__((idx + 1) % len(self.df))

        label = int(self.df.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 3. CTSD DATASET (EXCEL-BASED + FLAT FOLDER) ---
# UPDATED for your screenshot structure
class CTSDDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, mapping=None):
        """
        root_dir: Path to the 'images' folder
        annotation_file: Path to 'annotations.xlsx'
        mapping: Dict mapping CTSD Class ID -> GTSB Class ID
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mapping = mapping
        
        # Load Excel
        # NOTE: Adjust 'usecols' or column names if your Excel header is different
        # I am assuming columns are roughly [FileName, ClassId]
        raw_df = pd.read_excel(annotation_file)
        
        self.samples = []
        print(f"🔍 CTSD Loader: Scanning {len(raw_df)} rows in Excel...")

        # Filter and Map
        # We iterate through the Excel file and only keep classes we can map to GTSRB
        for idx, row in raw_df.iterrows():
            # Adjust these keys based on your actual Excel headers!
            # Common names: 'Path', 'Filename', 'Name' / 'ClassId', 'label', 'Roi.ClassId'
            filename = str(row.iloc[0]) # Assuming filename is 1st column
            class_id = int(row.iloc[-1]) # Assuming class ID is last column
            
            if self.mapping and class_id in self.mapping:
                gtsb_id = self.mapping[class_id]
                self.samples.append((filename, gtsb_id))
        
        print(f"✅ CTSD Loaded: {len(self.samples)} images matched with GTSRB classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            # Fallback
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 4. LOADER FUNCTIONS ---

def get_robust_loaders(data_dir, batch_size=32, noise_level=0.1):
    """ Loads GTSRB for Training """
    train_csv = os.path.join(data_dir, 'Train.csv')
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Robust Training: Augmentation + Noise
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=noise_level, p=0.3), 
        transforms.Normalize(mean, std)
    ])

    # Clean Validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load All and Split
    full_dataset = GTSRBDataset(data_dir, train_csv, transform=train_transform)
    indices = list(range(len(full_dataset)))
    
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.85 * len(indices))
    
    # We use Subset, but we need to ensure Val set gets Val transforms
    # Trick: Load dataset twice with diff transforms
    train_base = GTSRBDataset(data_dir, train_csv, transform=train_transform)
    val_base = GTSRBDataset(data_dir, train_csv, transform=val_transform)
    
    train_set = Subset(train_base, indices[:split])
    val_set = Subset(val_base, indices[split:])

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    )

def get_ctsd_loader(ctsd_root, mapping, batch_size=32):
    """ Loads CTSD for Testing """
    # Based on your screenshot:
    # Images are in: .../CTSD/images
    # Excel is in: .../CTSD/annotations.xlsx
    img_dir = os.path.join(ctsd_root, 'images')
    xlsx_file = os.path.join(ctsd_root, 'annotations.xlsx')
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = CTSDDataset(img_dir, xlsx_file, transform=test_transform, mapping=mapping)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)