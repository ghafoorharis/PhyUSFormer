import os
import numpy as np
import pandas as pd
import random
import cv2
import torch
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths,labels, transform=None, image_processor=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        x_img_path = self.image_paths[idx] # Image path
        x_mask_path = self.mask_paths[idx][0] # Mask path
        label = self.labels[idx]# Label
        image = Image.open(x_img_path).convert("RGB")  # Ensure 3-channel RGB
        image = np.array(image)

        mask = np.array(Image.open(x_mask_path),dtype=np.uint8)
        # print(f"Mask after loading (unique values): {np.unique(mask)}")
        # Resize
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        # Ensure mask is uint8 before resizing
        mask = np.array(mask, dtype=np.uint8)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # print(f"Mask after resizing (unique values): {np.unique(mask)}")

        # Normalize image
        image = image / image.max()
        # Normalize mask safely
        if mask.max() > 0:
            mask = mask / mask.max()


        # Ensure mask remains binary (0 or 1)
        mask = torch.tensor(mask, dtype=torch.float32)
        # print(f"Mask after transformation (unique values): {np.unique(mask.numpy())}")
        # Process with Segformer Image Processor
        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        # print(f"Encoded Mask unique values: {np.unique(encoded_inputs['labels'].numpy())}")
        # Add metadata
        metadata = {
            "image_path": x_img_path,
            "mask_path": x_mask_path,
            "label": label,
        }
        encoded_inputs["metadata"] = metadata
        return encoded_inputs
    
class UDIATDataset(Dataset):
    def __init__(self, data, transform=None, image_processor=None):
        self.data = data
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        image_path,mask_path,label = self.data[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB
        image = np.array(image)
        mask = np.array(Image.open(mask_path), dtype=np.uint8)
        # print(f"Mask after loading (unique values): {np.unique(mask)}")
        # Resize
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        # Ensure mask is uint8 before resizing
        mask = np.array(mask, dtype=np.uint8)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # print(f"Mask after resizing (unique values): {np.unique(mask)}")

        # Normalize image
        image = image / image.max()
        # Normalize mask safely
        if mask.max() > 0:
            mask = mask / mask.max()

        # Ensure mask remains binary (0 or 1)
        mask = torch.tensor(mask, dtype=torch.float32)
        # print(f"Mask after transformation (unique values): {np.unique(mask.numpy())}")
        # Process with Segformer Image Processor
        encoded_inputs = self.image_processor(image, mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        # print(f"Encoded Mask unique values: {np.unique(encoded_inputs['labels'].numpy())}")

        
        metadata = {
            "image_path": image_path,
            "mask_path": mask_path,
            "label": label,
        }

        encoded_inputs["metadata"] = metadata
        return encoded_inputs
    
class UltrasoundDataset(Dataset):
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
    random.seed(0)
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
    processed = np.zeros((len(images), *img_size))
    for idx, img in enumerate(images):
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        processed[idx] = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min())
    return np.expand_dims(processed, axis=1)

def read_data_numpy(PATH):
    """
    Read the data from the NPZ file.

    Args:
        None

    Returns:
        np.array: Scans
        np.array: Labels
    """
    data = np.load(f"{PATH}.npz")
    return data
