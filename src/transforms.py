import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class PaperGrayscale(object):
    """
    Converts image to grayscale using the specific formula from the research paper:
    I_gray = 0.2989*R + 0.5870*G + 0.1140*B
    """
    def __call__(self, img):
        # Convert PIL image to numpy array (RGB)
        img_np = np.array(img, dtype=np.float32)
        
        # Check if image has 3 channels, otherwise it's already gray
        if len(img_np.shape) == 3:
            r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = img_np
        
        # Convert back to uint8 for histogram equalization compatibility
        gray = gray.astype(np.uint8)
        return gray

class HistogramEqualization(object):
    """
    Applies Histogram Equalization to enhance contrast.
    """
    def __call__(self, img):
        # cv2.equalizeHist requires a grayscale image (uint8)
        return cv2.equalizeHist(img)

class QuantumResize(object):
    """
    Resizes image to 32x32 to fit the HQCNN input requirement.
    """
    def __call__(self, img):
        # Resize using INTER_CUBIC for better quality
        return cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)

def get_train_transforms():
    """
    Combines preprocessing with Data Augmentation for the training set.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        # Augmentation: Rotation (+/-10%), Zoom (+/-20%), Shift (+/-10%)
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        PaperGrayscale(),
        HistogramEqualization(),
        QuantumResize(),
        transforms.ToTensor(), # Automatically scales 0-255 to 0.0-1.0
    ])

def get_val_transforms():
    """
    Validation transforms (Preprocessing ONLY, no augmentation).
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        PaperGrayscale(),
        HistogramEqualization(),
        QuantumResize(),
        transforms.ToTensor(),
    ])