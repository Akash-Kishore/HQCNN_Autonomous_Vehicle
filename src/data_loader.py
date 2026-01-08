import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from src.transforms import get_train_transforms, get_val_transforms

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read file path from CSV (e.g., 'Train/0/00005_00029.png')
        img_rel_path = self.df.iloc[idx]['Path']
        
        # Construct full path
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        # Read image
        image = imread(img_path)
        
        # Error handling
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")

        # Convert BGR (OpenCV default) to RGB
        image = cvtColor(image, COLOR_BGR2RGB)
        
        # Get label
        label = int(self.df.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=50):
    """
    Creates Training and Validation DataLoaders.
    Uses Dual-Instance strategy to ensure Train gets Augmentation 
    while Validation remains Clean.
    """
    train_csv = os.path.join(data_dir, 'Train.csv')
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Could not find Train.csv at {train_csv}")

    # 1. Create TWO separate dataset instances
    # One with Augmentation (Train)
    train_full = GTSRBDataset(
        root_dir=data_dir, 
        csv_file=train_csv, 
        transform=get_train_transforms()
    )
    
    # One without Augmentation (Validation)
    val_full = GTSRBDataset(
        root_dir=data_dir, 
        csv_file=train_csv, 
        transform=get_val_transforms()
    )

    # 2. Calculate Split Indices (85/15)
    total_size = len(train_full)
    train_size = int(0.85 * total_size)
    
    # Generate random indices
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 3. Create Subsets using the indices
    # We grab the training indices from the Augmented dataset
    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    # We grab the validation indices from the Clean dataset
    val_dataset = torch.utils.data.Subset(val_full, val_indices)

    # 4. Create Loaders (Num workers=0 is safest for WSL to avoid shared memory errors)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset Loaded: {len(train_dataset)} Train | {len(val_dataset)} Val")
    return train_loader, val_loader