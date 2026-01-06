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

        # Convert BGR to RGB
        image = cvtColor(image, COLOR_BGR2RGB)
        
        # Get label
        label = int(self.df.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=50):
    """
    Creates Training and Validation DataLoaders.
    """
    train_csv = os.path.join(data_dir, 'Train.csv')
    
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Could not find Train.csv at {train_csv}")

    # Initialize Dataset
    full_dataset = GTSRBDataset(
        root_dir=data_dir, 
        csv_file=train_csv, 
        transform=get_train_transforms()
    )

    # Split: 85% Train, 15% Validation
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Override validation transform
    val_dataset.dataset.transform = get_val_transforms()

    # Create Loaders (Num workers=0 is safest for WSL)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
