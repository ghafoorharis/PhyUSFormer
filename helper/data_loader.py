import os
import numpy as np
import pandas as pd
import random
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

class BUSIDataset(Dataset):
    """
    Dataset class for loading and preprocessing Breast Ultrasound Images (BUSI).
    
    Args:
        image_paths (list): List of paths to ultrasound images
        mask_paths (list): List of paths to mask images
        labels (list): List of labels for each image
        transform (callable, optional): Optional transform to be applied to samples
        image_processor (callable, optional): Image processor for segmentation model
    """
    def __init__(self, image_paths, mask_paths, labels, transform=None, image_processor=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image paths
        x_img_path = self.image_paths[idx]
        x_mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        
        # Load and convert images
        image = Image.open(x_img_path).convert("RGB")
        image = np.array(image)
        mask = np.array(Image.open(x_mask_path), dtype=np.uint8)
        
        # Resize to standard dimensions
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image / image.max()
        
        # Normalize mask
        if mask.max() > 0:
            mask = mask / mask.max()

        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # Process with Segformer Image Processor
        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        # Add metadata
        metadata = {
            "image_path": x_img_path,
            "mask_path": x_mask_path,
            "label": label,
        }
        encoded_inputs["metadata"] = metadata
        return encoded_inputs
    
class UDIATDataset(Dataset):
    """
    Dataset class for loading and preprocessing UDIAT Ultrasound Images.
    
    Args:
        data (list): List of tuples containing (image_path, mask_path, label)
        transform (callable, optional): Optional transform to be applied to samples
        image_processor (callable, optional): Image processor for segmentation model
    """
    def __init__(self, data, transform=None, image_processor=None):
        self.data = data
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image data
        image_path, mask_path, label = self.data[idx]
        
        # Load and convert images
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        
        # Resize to standard dimensions
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image / image.max()
        
        # Normalize mask
        if mask.max() > 0:
            mask = mask / mask.max()

        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # Process with Segformer Image Processor
        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        # Add metadata
        metadata = {
            "image_path": image_path,
            "mask_path": mask_path,
            "label": label,
        }
        encoded_inputs["metadata"] = metadata
        return encoded_inputs
    
class UltrasoundDataset(Dataset):
    """
    Dataset class for loading and preprocessing generic ultrasound scans and labels.
    
    Args:
        scans (numpy.ndarray): Array of ultrasound scans
        labels (numpy.ndarray): Array of ground truth labels
        transform (callable, optional): Optional transform to be applied to samples
    """
    def __init__(self, scans, labels, transform=None):
        self.scans = scans
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        noisy_image = self.scans[idx]
        clean_image = self.labels[idx]
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        else:
            noisy_image = torch.from_numpy(noisy_image).float()
            clean_image = torch.from_numpy(clean_image).float()
            
        return noisy_image, clean_image


def load_data(data_dir, img_size=(256, 256), test_split=0.1):
    """
    Load data from disk, preprocess, and split into training and testing sets.
    
    Args:
        data_dir (str): Path to data directory
        img_size (tuple): Target size for images
        test_split (float): Proportion of data to use for testing
    
    Returns:
        tuple: (train_scans, train_labels, test_scans, test_labels)
    """
    scans_folder = os.path.join(data_dir, "scans")
    labels_folder = os.path.join(data_dir, "labels")
    scan_files = sorted(os.listdir(scans_folder))
    label_files = sorted(os.listdir(labels_folder))

    scans = np.zeros((len(scan_files), *img_size))
    labels = np.zeros((len(label_files), *img_size))

    for idx, (scan_file, label_file) in enumerate(zip(scan_files, label_files)):
        if scan_file.split("_")[1] == label_file.split("_")[1]:
            scans[idx] = pd.read_pickle(os.path.join(scans_folder, scan_file))["noisy_us_scan_b_mode"]
            labels[idx] = pd.read_pickle(os.path.join(labels_folder, label_file))["clean_phantom_sound_speed"]

    scans = preprocess_images(scans, img_size)
    labels = preprocess_images(labels, img_size)

    # Split into training and testing datasets
    random.seed(0)  # Set seed for reproducibility
    indices = list(range(len(scans)))
    test_size = int(len(scans) * test_split)
    test_indices = random.sample(indices, test_size)
    train_indices = list(set(indices) - set(test_indices))

    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of testing samples: {len(test_indices)}")
    
    return (
        scans[train_indices], labels[train_indices],
        scans[test_indices], labels[test_indices]
    )


def preprocess_images(images, img_size):
    """
    Preprocess images by resizing and normalizing.
    
    Args:
        images (numpy.ndarray): Array of images to preprocess
        img_size (tuple): Target size for images
    
    Returns:
        numpy.ndarray: Preprocessed images
    """
    processed = np.zeros((len(images), *img_size))
    for idx, img in enumerate(images):
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        processed[idx] = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
    return np.expand_dims(processed, axis=1)

def read_data_numpy(PATH):
    """
    Read data from a NPZ file.
    
    Args:
        PATH (str): Path to the NPZ file (without .npz extension)
    
    Returns:
        dict: Dictionary containing the data from the NPZ file
    """
    data = np.load(f"{PATH}.npz")
    return data
