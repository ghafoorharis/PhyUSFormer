# External Imports
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import os
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

def main(FOLD_ID,LOG_DIR,SAVE_DIR):
    """
    Main function for training the UNet model.

    Args:
        EPOCHS (int): Number of epochs to train the model.
        BATCH_SIZE (int): Number of samples per batch.
        LEARNING_RATE (float): Learning rate for the optimizer.
        DATA_DIR (str): Path to the directory containing the dataset.
        LOG_DIR (str): Path to the directory to save TensorBoard logs.
        SAVE_DIR (str): Path to the directory to save the trained model.
    
    Returns:
        None
    """
   

    # 2. Load Data
    print("Loading data...")
    # fold_id = FOLD_ID # FOLD ID : 0,1,2,3,4
    DATA_DIR_TEST = "/home/user/haris/data/combine_datasets/data_npz/train_test_data"
    DATA_DIR_TRAIN_VAL = f"/home/user/haris/data/combine_datasets/data_npz/train_val_data_fold_{str(FOLD_ID)}"
    train_test_data = read_data_numpy(DATA_DIR_TEST)
    X_test_noisy = train_test_data['X_test']
    X_test = train_test_data['y_test']
    train_val_data = read_data_numpy(DATA_DIR_TRAIN_VAL)
    X_train_noisy = train_val_data['X_train']
    X_train = train_val_data['y_train']
    X_val_noisy = train_val_data['X_val']
    X_val = train_val_data['y_val']
    # X_train_noisy, X_train, X_test_noisy, X_test = load_data(DATA_DIR)
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
    loss_fn = DenoisingLoss(alpha = ALPHA)

    # 4. Initialize TensorBoard Writer
    writer = SummaryWriter(LOG_DIR)

    # 5. Train the Model
    print("Starting training...")
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, EPOCHS, writer, device,save_dir=SAVE_DIR)
    trainer.fit()

    # 6. Save Final Model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    print(f"Training complete. Model saved at {SAVE_DIR}/final_model.pth")

    # 7. Log Hyperparameters
    DATA_PARAMS = {"EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "DATA_DIR": DATA_DIR,
                   "FOLD_ID": FOLD_ID}
    
    TRAINING_PARAMS = {"LEARNING_RATE": LEARNING_RATE,
                    "ALPHA": ALPHA,
                    }
    LOGGING_PARAMS = {"LOG_DIR": LOG_DIR, "SAVE_DIR": SAVE_DIR}
    log_hyperparameters(writer, {"DATA_PARAMS": DATA_PARAMS, "TRAINING_PARAMS": TRAINING_PARAMS, "LOGGING_PARAMS": LOGGING_PARAMS})

    # 8 . Evaluate the model on the test set
    model.eval()
    # model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
    checkpoint = torch.load(os.path.join(SAVE_DIR, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=trainer.device))
    # Perform inference on the test set
    inference_metrics = trainer.test(model, test_loader = test_loader)
    # Log inference metrics
    log_inference(writer, inference_metrics)
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    for fold_id in range(5):
        print(f"Training fold {fold_id}...")
        log_dir = LOG_DIR + f"_fold_{str(fold_id)}"
        save_dir = SAVE_DIR + f"_fold_{str(fold_id)}"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        main(fold_id,log_dir,save_dir)
        # Clear CUDA cache after each fold
        torch.cuda.empty_cache()
        # Clear out evrything related to CUDA
        # torch.cuda.reset_max_memory_allocated()
        print("CUDA cache cleared.")

