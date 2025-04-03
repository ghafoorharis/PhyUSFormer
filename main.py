# External Imports
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import wandb
import argparse
# Internal Imports
from helper.trainer import Trainer
from config import (EPOCHS, BATCH_SIZE,
                    LEARNING_RATE, DATA_DIR,
                    LOG_DIR, SAVE_DIR,ALPHA)
from helper.data_loader import load_data, UltrasoundDataset,read_data_numpy
from helper.models import UNet
# from helper.model_ae import Autoencoder
from helper.loss import DenoisingLoss
from helper.utils import (log_hyperparameters,
                        log_inference,
                        )

def parse_args():
    """
    Parse command-line arguments for model training and evaluation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train and evaluate ultrasound denoising model')
    parser.add_argument('--fold_id', type=int, default=0, help='Fold ID for cross-validation')
    parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='Directory for TensorBoard logs')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Directory to save model checkpoints')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    return parser.parse_args()

def main(FOLD_ID, LOG_DIR, SAVE_DIR, use_wandb=False):
    """
    Main function for training and evaluating the ultrasound denoising model.
    
    Args:
        FOLD_ID (int): Fold ID for cross-validation
        LOG_DIR (str): Directory for TensorBoard logs
        SAVE_DIR (str): Directory to save model checkpoints
        use_wandb (bool): Whether to use Weights & Biases for logging
    """
    # Create necessary directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(
            project="ultrasound-denoising",
            config={
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "alpha": ALPHA,
                "fold_id": FOLD_ID
            }
        )
    
    # 1. Load and Preprocess Data
    print("Loading and preprocessing data...")
    try:
        X_train_noisy, X_train, X_test_noisy, X_test = load_data(DATA_DIR)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split validation set from training set
    val_split = int(0.1 * len(X_train))
    X_val_noisy = X_train_noisy[:val_split]
    X_val = X_train[:val_split]
    X_train_noisy = X_train_noisy[val_split:]
    X_train = X_train[val_split:]
    
    # 2. Create PyTorch Datasets and DataLoaders
    print("Creating datasets and dataloaders...")
    train_dataset = UltrasoundDataset(X_train_noisy, X_train)
    val_dataset = UltrasoundDataset(X_val_noisy, X_val)
    test_dataset = UltrasoundDataset(X_test_noisy, X_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model, Optimizer, and Loss Function
    print("Initializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1).to(device)
    # model = Autoencoder(n_channels=1, n_classes=1).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = DenoisingLoss(alpha=ALPHA)

    # 4. Initialize TensorBoard Writer
    writer = SummaryWriter(LOG_DIR)

    # 5. Train the Model
    print("Starting training...")
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, EPOCHS, writer, device, save_dir=SAVE_DIR)
    trainer.fit()

    # 6. Save Final Model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    print(f"Training complete. Model saved at {SAVE_DIR}/final_model.pth")

    # 7. Log Hyperparameters
    DATA_PARAMS = {
        "EPOCHS": EPOCHS, 
        "BATCH_SIZE": BATCH_SIZE, 
        "DATA_DIR": DATA_DIR,
        "FOLD_ID": FOLD_ID
    }
    
    TRAINING_PARAMS = {
        "LEARNING_RATE": LEARNING_RATE,
        "ALPHA": ALPHA,
    }
    
    LOGGING_PARAMS = {
        "LOG_DIR": LOG_DIR, 
        "SAVE_DIR": SAVE_DIR
    }
    
    # Log to tensorboard
    log_hyperparameters(writer, {
        "DATA_PARAMS": DATA_PARAMS, 
        "TRAINING_PARAMS": TRAINING_PARAMS, 
        "LOGGING_PARAMS": LOGGING_PARAMS
    })

    # 8. Evaluate the model on the test set
    print("Evaluating model on test set...")
    model.eval()
    # model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
    checkpoint = torch.load(os.path.join(SAVE_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=trainer.device))
    # Perform inference on the test set
    inference_metrics = trainer.test(model, test_loader = test_loader)
    # Log inference metrics
    log_inference(writer, inference_metrics)
    
    # Log test metrics to wandb if enabled
    if use_wandb:
        wandb.log(inference_metrics)
        wandb.finish()
    
    # Close the TensorBoard writer
    writer.close()
    
    return inference_metrics

if __name__ == "__main__":
    args = parse_args()
    main(
        FOLD_ID=args.fold_id,
        LOG_DIR=args.log_dir,
        SAVE_DIR=args.save_dir,
        use_wandb=args.use_wandb
    )

