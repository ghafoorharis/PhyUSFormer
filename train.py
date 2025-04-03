import os
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset as BaseDataset
from torch.optim import lr_scheduler
import pytorch_lightning as pl

# Internal Imports
from config import EPOCHS, BATCH_SIZE, DATA_DIR, LOG_DIR, SAVE_DIR
from helper.data_loader import read_data_numpy
from helper.utils import calculate_metrics, calculate_psnr_ssim
from transformers import SegformerImageProcessor
import evaluate

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import SegformerForSemanticSegmentation
import pandas as pd
from PIL import Image

# =============================================================================
# Hybrid Loss: Combines Dice loss and Binary Cross-Entropy (BCE) loss.
# =============================================================================
class HybridLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: Weight for Dice loss. (1 - alpha) is used for BCE loss.
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # Compute Binary Cross-Entropy loss.
        bce = self.bce_loss(preds, targets)
        # Compute Dice loss.
        smooth = 1e-7
        preds = torch.sigmoid(preds)  # bring predictions into [0,1]
        intersection = (preds * targets).sum(dim=(1,2,3))
        union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice_loss = 1 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()
        # Return the weighted sum.
        return self.alpha * dice_loss + (1 - self.alpha) * bce

# =============================================================================
# Dataset Classes
# =============================================================================
class synthetic_data(BaseDataset):
    def __init__(self, scans, labels, image_processor, transform=None):
        self.scans = scans  # Expected shape: [Total_images, 1, 256, 256]
        self.labels = labels  # Expected shape: [Total_images, 1, 256, 256] or [Total_images, 256,256]
        self.transform = transform
        self.image_processor = image_processor
        self.id2label = {0: "background", 1: "object"}

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        # Load raw data and convert to float32
        noisy_image = self.scans[idx].astype(np.float32)
        clean_image = self.labels[idx].astype(np.float32)
        # Convert from CHW to HWC for Albumentations if needed.
        if noisy_image.shape[0] == 1:
            noisy_image = np.transpose(noisy_image, (1, 2, 0))
        if len(clean_image.shape) == 3 and clean_image.shape[0] == 1:
            clean_image = np.transpose(clean_image, (1, 2, 0))
        # Apply Albumentations transform if provided.
        if self.transform:
            augmented = self.transform(image=noisy_image, mask=clean_image)
            noisy_image = augmented["image"]
            clean_image = augmented["mask"]
        # Convert back to torch tensors (from HWC to CHW).
        noisy_image = torch.from_numpy(noisy_image).permute(2, 0, 1).float()
        if len(clean_image.shape) == 3:
            clean_image = torch.from_numpy(clean_image).permute(2, 0, 1).float().squeeze(0)
        else:
            clean_image = torch.from_numpy(clean_image).float()
        # Process with image_processor.
        encoded_inputs = self.image_processor(noisy_image, clean_image, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        # Ensure pixel_values have 3 channels.
        if encoded_inputs["pixel_values"].shape[1] == 1:
            encoded_inputs["pixel_values"] = encoded_inputs["pixel_values"].repeat(1, 3, 1, 1)
        return encoded_inputs

# =============================================================================
# Segformer Finetuner with Dynamic Learning Rate Scheduler
# =============================================================================
class SegformerFinetuner(pl.LightningModule):
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.validation_step_loss = []
        self.test_step_loss = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        # Load a pretrained SegFormer model with pretrained MiT-B5 weights.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            return_dict=False,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return outputs

    def training_step(self, batch, batch_nb):
        images, masks = batch["pixel_values"], batch["labels"]
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        # Upsample logits to match mask dimensions.
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
        )
        if batch_nb % self.metrics_interval == 0:
            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes, ignore_index=255, reduce_labels=False
            )
            metrics = {"loss": loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            for k, v in metrics.items():
                self.log(k, v)
            return metrics
        else:
            return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        images, masks = batch["pixel_values"], batch["labels"]
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
        )
        self.validation_step_loss.append(outputs[0])
        self.validation_step_outputs.append(outputs[1])
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes, ignore_index=255, reduce_labels=False
        )
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        metrics = {"val_loss": 1, "val_mean_iou": val_mean_iou, "val_mean_accuracy": val_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)
        return metrics

    def test_step(self, batch, batch_nb):
        images, masks = batch["pixel_values"], batch["labels"]
        outputs = self(images, masks)
        loss, logits = outputs[0], outputs[1]
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
        )
        self.test_step_outputs.append(outputs)
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes, ignore_index=255, reduce_labels=False
        )
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]
        metrics = {"test_loss": 1, "test_mean_iou": test_mean_iou, "test_mean_accuracy": test_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)
        return metrics

    def configure_optimizers(self):
        # Use AdamW with a dynamic learning rate scheduler based on validation loss.
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=2e-5, eps=1e-8)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        # Return a dictionary that PyTorch Lightning can use for scheduling.
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

# =============================================================================
# Main Function: Data Loading, Model Training and Evaluation
# =============================================================================
def main(FOLD_ID=0, TRAIN=True):
    MODEL_NAME = f"segformer-mit-b5_dataset_combined_Fold_{FOLD_ID}_v3"
    print("Loading data...")
    # Use generic paths instead of user-specific names.
    DATA_DIR_TEST = "/path/to/data/train_test_data"
    DATA_DIR_TRAIN_VAL = f"/path/to/data/train_val_data_fold_{FOLD_ID}"
    train_test_data = read_data_numpy(DATA_DIR_TEST)
    X_test_noisy = train_test_data['X_test']
    X_test = train_test_data['y_test']
    train_val_data = read_data_numpy(DATA_DIR_TRAIN_VAL)
    X_train_noisy = train_val_data['X_train']
    X_train = train_val_data['y_train']
    X_val_noisy = train_val_data['X_val']
    X_val = train_val_data['y_val']

    # Initialize the image processor for SegFormer.
    image_processor = SegformerImageProcessor(
        reduce_labels=False,
        do_normalize=False,
        do_rescale=False,
        size={"height": 256, "width": 256},
    )
    # Data augmentation using Albumentations.
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.3),
    ])
    train_dataset = synthetic_data(X_train_noisy, X_train, transform=transforms, image_processor=image_processor)
    val_dataset = synthetic_data(X_val_noisy, X_val, transform=transforms, image_processor=image_processor)
    test_dataset = synthetic_data(X_test_noisy, X_test, transform=transforms, image_processor=image_processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing model...")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    segformer_finetuner = SegformerFinetuner(
        train_dataset.id2label,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        metrics_interval=10,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_mean_iou",
        min_delta=0.01,
        patience=20,
        verbose=True,
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_mean_iou",
        mode="max",
        dirpath=f"weights/{MODEL_NAME}",
        filename=f"{MODEL_NAME}_checkpoint",
        verbose=True,
    )
    logger = TensorBoardLogger(save_dir="lightning_logs", name=MODEL_NAME)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=500,
        min_epochs=100,
        val_check_interval=len(train_loader),
        check_val_every_n_epoch=None,
        logger=logger,
    )
    if TRAIN:
        trainer.fit(segformer_finetuner)
        smp_model = segformer_finetuner.model
        # Save the model.
        smp_model.save_pretrained(
            save_directory=f"weights/{MODEL_NAME}",
            push_to_hub=False,
            dataset="Cubic Bezier From Scratch",
        )
    # Load model for evaluation.
    loaded_model = SegformerForSemanticSegmentation.from_pretrained(
        f"weights/{MODEL_NAME}",
        return_dict=False,
        num_labels=segformer_finetuner.num_classes,
        id2label=segformer_finetuner.id2label,
        label2id=segformer_finetuner.label2id,
        ignore_mismatched_sizes=True,
    )
    valid_metrics = trainer.validate(segformer_finetuner, dataloaders=val_loader, verbose=True)
    print(valid_metrics)
    test_metrics = trainer.test(segformer_finetuner, dataloaders=test_loader, verbose=True)
    print(test_metrics)

    # Inference on test data.
    model = loaded_model.to(device)
    list_of_test_metrics_per_batch = []
    for batch in test_loader:
        with torch.no_grad():
            model.eval()
            image = batch["pixel_values"].to(device)
            logits = model(image)[0]
        pr_masks = torch.sigmoid(logits)
        mask = torch.nn.functional.interpolate(pr_masks, size=(256, 256), mode="bilinear", align_corners=False)
        mask = mask.argmax(1)
        for idx, (image, gt_mask, pr_mask) in enumerate(zip(batch["pixel_values"], batch["labels"], mask)):
            if idx <= 4:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(image.numpy().transpose(1, 2, 0))
                plt.title("Image")
                plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask.numpy().squeeze(), cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(pr_mask.cpu().numpy(), cmap="gray")
                plt.title("Prediction")
                plt.axis("off")
                os.makedirs("results", exist_ok=True)
                plt.savefig(f"results/{MODEL_NAME}_{idx}.png")
                plt.close()
        batch_metrics = calculate_metrics(outputs=mask.unsqueeze(1).to(device),
                                           targets=batch["labels"].to(device))
        batch_metrics.update(calculate_psnr_ssim(outputs=mask.unsqueeze(1).to(device),
                                                 target=batch["labels"].to(device)))
        list_of_test_metrics_per_batch.append(batch_metrics)
    df = pd.DataFrame(list_of_test_metrics_per_batch).mean()
    time_now = time.strftime("%Y%m%d-%H%M%S")
    df.to_csv(f"results/{MODEL_NAME}_{time_now}.csv")
    print("Inference done!")

if __name__ == "__main__":
    import time
    for fold_id in [4]:
        print(f"Training fold {fold_id}...")
        main(FOLD_ID=fold_id, TRAIN=True)
        torch.cuda.empty_cache()
