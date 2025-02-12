import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
from helper.utils import (
    log_images_to_tensorboard,
    calculate_metrics,
    calculate_psnr_ssim,
)


# %% Trainer Class
class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs,
        writer,
        device,
        save_dir,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = epochs
        self.writer = writer
        self.device = device
        self.save_dir = save_dir

    def fit(self):
        model = self.model
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            list_of_dice_scores = []
            list_of_iou_scores = []
            list_of_precision_scores = []
            list_of_recall_scores = []
            list_of_accuracy_scores = []
            list_of_psnr_scores = []
            list_of_ssim_scores = []
            for batch_idx, (noisy_imgs, clean_imgs) in enumerate(self.train_loader):
                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(
                    self.device
                )
                # Convert the target to a binary image
                clean_imgs = (clean_imgs > 0.5).float()
                # Forward pass
                outputs = model(noisy_imgs)
                loss = loss_fn(outputs, clean_imgs)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Log training loss
                metrics = calculate_metrics(outputs, clean_imgs)
                psnr_ssim_metrics = calculate_psnr_ssim(outputs, clean_imgs)
                for metric_name, metric_value in psnr_ssim_metrics.items():
                    self.writer.add_scalar(
                        f"Train/{metric_name}", metric_value, global_step
                    )

                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(
                        f"Train/{metric_name}", metric_value, global_step
                    )
                self.writer.add_scalar("Train/Loss", loss.item(), global_step)

                global_step += 1

                # Append metrics to list
                list_of_dice_scores.append(metrics["dice"])
                list_of_iou_scores.append(metrics["iou"])
                list_of_precision_scores.append(metrics["precision"])
                list_of_recall_scores.append(metrics["recall"])
                list_of_accuracy_scores.append(metrics["accuracy"])
                list_of_psnr_scores.append(psnr_ssim_metrics["psnr"])
                list_of_ssim_scores.append(psnr_ssim_metrics["ssim"])

            avg_loss = running_loss / len(self.train_loader)
            elapsed_time = time.time() - start_time
            # Calculate average metrics
            metrics_epoch = {
                "dice": np.mean(list_of_dice_scores),
                "iou": np.mean(list_of_iou_scores),
                "precision": np.mean(list_of_precision_scores),
                "recall": np.mean(list_of_recall_scores),
                "accuracy": np.mean(list_of_accuracy_scores),
            }
            print(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s"
            )
            print(
                f"Train Metrics - Dice: {metrics_epoch['dice']:.4f}, IoU: {metrics_epoch['iou']:.4f}, "
                f"Precision: {metrics_epoch['precision']:.4f}, Recall: {metrics_epoch['recall']:.4f}, "
                f"Accuracy: {metrics_epoch['accuracy']:.4f}, PSNR: {np.mean(list_of_psnr_scores):.4f}, SSIM: {np.mean(list_of_ssim_scores):.4f}"
            )
            # Validation
            val_loss = self.validate(epoch, global_step)

            # Save model checkpoint
            best_val_loss = self.save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                save_dir=self.save_dir,
            )

    def validate(self, epoch, global_step):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (noisy_imgs, clean_imgs) in enumerate(self.val_loader):
                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(
                    self.device
                )
                # Convert the target to a binary image
                clean_imgs = (clean_imgs > 0.5).float()
                outputs = self.model(noisy_imgs)
                outputs = (outputs > 0.5).float()
                loss_val = self.loss_fn(outputs, clean_imgs)
                val_loss += loss_val.item()
                # Log validation loss
                metrics = calculate_metrics(outputs, clean_imgs)
                psnr_ssim_metrics = calculate_psnr_ssim(outputs, clean_imgs)
                for metric_name, metric_value in psnr_ssim_metrics.items():
                    self.writer.add_scalar(
                        f"Validation/{metric_name}", metric_value, global_step
                    )
                for metric_name, metric_value in metrics.items():
                    self.writer.add_scalar(
                        f"Validation/{metric_name}", metric_value, global_step
                    )
                self.writer.add_scalar("Validation/Loss", loss_val.item(), global_step)

                # Log images every few batches
                if batch_idx % 10 == 0:
                    log_images_to_tensorboard(
                        self.writer,
                        noisy_imgs,
                        clean_imgs,
                        outputs,
                        epoch=epoch,
                        num_images=10,
                    )
                

        avg_val_loss = val_loss / len(self.val_loader)
        print(
            f"Validation Loss after Epoch [{epoch + 1}/{self.num_epochs}]: {avg_val_loss:.4f}"
        )
        # Print all metrics values after each epoch
        print(
            f"Validation Metrics - Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, PSNR: {psnr_ssim_metrics['psnr']:.4f}, SSIM: {psnr_ssim_metrics['ssim']:.4f}"
        )
        return avg_val_loss

    def test(self, model, test_loader,threshold=0.5):
        # Initialize lists to store the metrics
        list_of_dice_scores = []
        list_of_iou_scores = []
        list_of_precision_scores = []
        list_of_recall_scores = []
        list_of_accuracy_scores = []
        list_of_psnr_scores = []
        list_of_ssim_scores = []
        # Iterate over the test dataset
        with torch.no_grad():
            for batch_idx, (noisy_imgs, clean_imgs) in enumerate(test_loader):
                noisy_imgs, clean_imgs = noisy_imgs.to(self.device), clean_imgs.to(
                    self.device
                )
                # Forward pass
                outputs = model(noisy_imgs)
                outputs_bin = (outputs > 0.5).float()
                # Calculate metrics
                metrics = calculate_metrics(
                                        outputs_bin,
                                        clean_imgs,
                                        threshold = threshold
                                        )
                psnr_ssim_metrics = calculate_psnr_ssim(outputs, clean_imgs)
                # Append metrics to list
                list_of_dice_scores.append(metrics["dice"])
                list_of_iou_scores.append(metrics["iou"])
                list_of_precision_scores.append(metrics["precision"])
                list_of_recall_scores.append(metrics["recall"])
                list_of_accuracy_scores.append(metrics["accuracy"])
                list_of_psnr_scores.append(psnr_ssim_metrics["psnr"])
                list_of_ssim_scores.append(psnr_ssim_metrics["ssim"])

        # Calculate average metrics
        metrics_test = {
            "dice": np.mean(list_of_dice_scores),
            "iou": np.mean(list_of_iou_scores),
            "precision": np.mean(list_of_precision_scores),
            "recall": np.mean(list_of_recall_scores),
            "accuracy": np.mean(list_of_accuracy_scores),
            "psnr": np.mean(list_of_psnr_scores),
            "ssim": np.mean(list_of_ssim_scores),
        }
        print(
            f"Test Metrics - Dice: {metrics_test['dice']:.4f}, IoU: {metrics_test['iou']:.4f}, "
            f"Precision: {metrics_test['precision']:.4f}, Recall: {metrics_test['recall']:.4f}, "
            f"Accuracy: {metrics_test['accuracy']:.4f}, PSNR: {metrics_test['psnr']:.4f}, SSIM: {metrics_test['ssim']:.4f}"
        )
        return metrics_test

    def save_model_checkpoint(
        self, model, optimizer, epoch, val_loss, best_val_loss, save_dir="./weights"
    ):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, f"best_model.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(
                f"Best Model saved at {checkpoint_path} with validation loss {val_loss:.4f}"
            )
        return best_val_loss
