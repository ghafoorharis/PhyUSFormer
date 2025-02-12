import os
import numpy as np
import pandas as pd
import random
import cv2
import torch
from torch.utils.data import Dataset


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
