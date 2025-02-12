import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

def log_hyperparameters(writer, config):
    """
    Log hyperparameters as a Markdown-style table in TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        config (dict): Dictionary of hyperparameters.
    """
    table_lines = ["| Parameter | Value |", "|-----------|-------|"]

    for section, params in config.items():
        if isinstance(params, dict):  # Handle nested dictionaries
            for key, value in params.items():
                table_lines.append(f"| {key} | {value} |")
        else:
            table_lines.append(f"| {section} | {params} |")
    
    # Join the table as a Markdown string
    table_markdown = "\n".join(table_lines)
    
    # Log the table to TensorBoard
    writer.add_text("Hyperparameters", table_markdown)

def log_inference(writer, scores):
    """
    Log hyperparameters as a Markdown-style table in TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        scores (dict): Dictionary of metrics values.
    """
    table_lines = ["| Metrics | Score |", "|-----------|-------|"]
    for key, value in scores.items():
        table_lines.append(f"| {key} | {value} |")
        
    # Join the table as a Markdown string
    table_markdown = "\n".join(table_lines)
    
    # Log the table to TensorBoard
    writer.add_text("Evaluation", table_markdown)

def log_images_to_tensorboard(
        writer, noisy_imgs, clean_imgs, output_imgs, epoch, num_images=10
    ):
        """
        Log random images to TensorBoard by creating a grid of subplots using Matplotlib.

        Args:
            writer (SummaryWriter): TensorBoard writer object.
            noisy_imgs (Tensor): Batch of noisy images (B, C, H, W).
            clean_imgs (Tensor): Batch of clean images (B, C, H, W).
            output_imgs (Tensor): Batch of denoised images from the model (B, C, H, W).
            epoch (int): Current epoch number.
            num_images (int): Number of random images to plot and log.
        """
        # Choose random indices from the batch
        num_images = min(num_images, noisy_imgs.size(0))
        indices = random.sample(range(noisy_imgs.size(0)), num_images)

        # Set up the figure
        fig, axes = plt.subplots(3, num_images, figsize=(15, 15))
        for i, idx in enumerate(indices):
            # Plot Noisy Images
            if noisy_imgs.shape[1] == 1:  # If the image has 1 channel (grayscale)
                axes[0, i].imshow(noisy_imgs[idx].cpu().squeeze(), cmap="gray")
            else:  # If the image has 3 channels (RGB)
                axes[0, i].imshow(noisy_imgs[idx].cpu().permute(1, 2, 0))
            axes[0, i].set_title("Noisy")
            axes[0, i].axis("off")

            # Plot Clean Images
            if clean_imgs.shape[1] == 1:  # If the image has 1 channel (grayscale)
                axes[1, i].imshow(clean_imgs[idx].cpu().squeeze(), cmap="gray")
            else:  # If the image has 3 channels (RGB)
                axes[1, i].imshow(clean_imgs[idx].cpu().permute(1, 2, 0))
            axes[1, i].set_title("Target")
            axes[1, i].axis("off")

            # Plot Output Images
            if output_imgs.shape[1] == 1:  # If the image has 1 channel (grayscale)
                axes[2, i].imshow(output_imgs[idx].cpu().squeeze(), cmap="gray")
            else:  # If the image has 3 channels (RGB)
                axes[2, i].imshow(output_imgs[idx].cpu().permute(1, 2, 0))
            axes[2, i].set_title("Output")
            axes[2, i].axis("off")

        plt.tight_layout()
        # Log the plot to TensorBoard
        writer.add_figure(
            f"Visualization of Random Samples from Validation Set",
            fig,
            global_step=epoch,
        )
        plt.close(fig)



# Function to calculate PSNR
def calculate_psnr(original, denoised, max_pixel_value=255.0):
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float("inf")
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# Function to calculate SSIM
def calculate_ssim(original, denoised):
    return ssim(original, denoised, data_range=original.max() - original.min())


def calculate_metrics(outputs, targets, threshold=0.5):
    """
    Calculates evaluation metrics for binary segmentation per image and averages over the batch.
    Args:
        outputs (torch.Tensor): Model predictions, expected to be probabilities in [0,1], shape [batch_size, 1, H, W]
        targets (torch.Tensor): Ground truth labels, shape [batch_size, 1, H, W]
    Returns:
        metrics_dict (dict): Dictionary containing averaged 'dice', 'iou', 'precision', 'recall', 'accuracy'
    """
    epsilon = 1e-7  # Small constant to avoid division by zero
    # Threshold outputs to obtain binary predictions
    outputs_bin = (outputs > threshold).float()
    targets_bin = (targets > threshold).float()

    batch_size = outputs.size(0)
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    for i in range(batch_size):
        output_i = outputs_bin[i].view(-1)
        target_i = targets_bin[i].view(-1)

        # True Positives, False Positives, True Negatives, False Negatives
        TP = (output_i * target_i).sum()
        FP = (output_i * (1 - target_i)).sum()
        TN = ((1 - output_i) * (1 - target_i)).sum()
        FN = ((1 - output_i) * target_i).sum()

        # Dice Coefficient
        dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
        dice_scores.append(dice.item())

        # Intersection over Union (IoU)
        iou = (TP + epsilon) / (TP + FP + FN + epsilon)
        iou_scores.append(iou.item())

        # Precision
        precision = (TP + epsilon) / (TP + FP + epsilon)
        precision_scores.append(precision.item())

        # Recall
        recall = (TP + epsilon) / (TP + FN + epsilon)
        recall_scores.append(recall.item())

        # Accuracy
        accuracy = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)
        accuracy_scores.append(accuracy.item())

    # Average over batch
    metrics_dict = {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "accuracy": np.mean(accuracy_scores),

    }

    return metrics_dict


def calculate_psnr_ssim(outputs: torch.tensor, target: torch.tensor) -> dict:
    """This function calculates the PSNR and SSIM for a batch of images

    Args:
        outputs (torch.tensor): The output of the model (batch)
        target (torch.tensor): The target image (batch)

    Returns:
        tuple: The average PSNR and SSIM for the batch
    """
    batch_psnr = []
    batch_ssim = []
    outputs = outputs.cpu().detach().numpy().squeeze()
    target = target.cpu().detach().numpy().squeeze()
    for i in range(outputs.shape[0]):
        target_image = target[i]
        output_image = outputs[i]
        batch_psnr.append(calculate_psnr(target_image, output_image))
        batch_ssim.append(calculate_ssim(target_image, output_image))

    psnr = np.mean(batch_psnr)
    ssim = np.mean(batch_ssim)
    # Average over batch
    metrics_dict = {"psnr": psnr, "ssim": ssim}
    return metrics_dict

def post_process_target_tensor(targets):
    """
    Convert the target tensor to a binary tensor with values 0 and 1.
    Args:
        targets (torch.Tensor): Target tensor with values 0 and 255.

    Returns:
        torch.Tensor: Binary tensor with values 0 and 1.
    """
    list_of_tensors = []
    for x_target in targets:
        # find the indices where x_target is equal to 255
        bg_pixels_idx = torch.where(x_target == 255)
        # find the indices where x_target is equal to 0 
        fg_pixels_idx = torch.where(x_target == 0)
        # create a tensor of zeros with the same shape as x_target
        new_target = torch.zeros_like(x_target)
        # set the background pixels to 0
        new_target[bg_pixels_idx] = 0
        # set the foreground pixels to 1
        new_target[fg_pixels_idx] = 1
        list_of_tensors.append(new_target)

    targets_new = torch.stack(list_of_tensors, dim=0)

    # invert the binary tensor using some logical operations
    targets_new = torch.logical_not(targets_new).float()
    
    assert targets_new.shape == targets.shape
    return targets_new