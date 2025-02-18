import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
from matplotlib.patches import Patch
import cv2 # type: ignore
import os
INITIAL_SEED = 42
np.random.seed(INITIAL_SEED)
random.seed(INITIAL_SEED)
torch.manual_seed(INITIAL_SEED)
torch.cuda.manual_seed(INITIAL_SEED)
torch.cuda.manual_seed_all(INITIAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_seed(seed):
    """
    Set seed for reproducibility.
    Args:
        seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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




import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

def compute_hd95(pred_mask, gt_mask):
    """
    Compute the 95th percentile of the Hausdorff Distance (HD95) between two binary masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted binary mask, shape (H, W) or (1, H, W)
        gt_mask (torch.Tensor): Ground truth binary mask, shape (H, W) or (1, H, W)
    
    Returns:
        hd95 (float): The 95th percentile of the Hausdorff Distance
    """
    assert pred_mask.shape == gt_mask.shape, "Masks must have the same shape"

    # Convert to numpy and binarize
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)  # Shape (H, W)
    gt_mask = gt_mask.squeeze().cpu().numpy().astype(np.uint8)      # Shape (H, W)

    # Get coordinates of foreground pixels (x, y)
    pred_coords = np.column_stack(np.where(pred_mask > 0))
    gt_coords = np.column_stack(np.where(gt_mask > 0))

    # If either mask is empty, assign a penalty value
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        print("Empty mask detected")
        return 50  # Reasonable high penalty for empty masks

    # Compute directed Hausdorff distances
    hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd2 = directed_hausdorff(gt_coords, pred_coords)[0]

    # Compute HD95 (95th percentile of all Hausdorff distances)
    hd95 = np.percentile([hd1, hd2], 95)

    return hd95


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
    hd95_scores = []
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
        # Compute HD95
        hd95 = compute_hd95(outputs_bin[i].squeeze(0), targets_bin[i].squeeze(0))
        hd95_scores.append(hd95)

    # Average over batch
    metrics_dict = {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "precision": np.mean(precision_scores),
        "recall": np.mean(recall_scores),
        "accuracy": np.mean(accuracy_scores),
        "hd95": np.mean(hd95_scores),

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


def filter_multi_lesions(mask):
    """Removes small lesions and keeps only the largest connected component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    if num_labels > 1:
        max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Find largest region
        return (labels == max_label).astype(np.uint8)  # Keep only the largest region

    return mask



def calculate_metrics_ind(outputs, targets, threshold=0.5):
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
    hd95_scores = []

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

        # Compute HD95
        hd95 = compute_hd95(outputs_bin[i].squeeze(0), targets_bin[i].squeeze(0))
        hd95_scores.append(hd95)
    
    # Average over batch
    metrics_dict = {
        "dice": dice_scores,
        "iou": iou_scores,
        "precision": precision_scores,
        "recall": recall_scores,
        "accuracy": accuracy_scores,
        "hd95": hd95_scores,
    }

    return metrics_dict

def visualize_batch(batch, mask, num_imgs=1, preview=False,
                    MODEL_NAME="UNet", threshold=0.5,
                    save = False,
                    save_dir = None,
                    batch_metrics: dict = None):
    for idx, (image, gt_mask, pr_mask,x_path,x_label,x_dice,x_hd) in enumerate(
        zip(batch["pixel_values"], batch["labels"], mask,batch["metadata"]['image_path'],batch["metadata"]['label'],batch_metrics["dice"],
            batch_metrics["hd95"])
    ):
        if idx <= num_imgs:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.cpu().numpy().squeeze(), cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.cpu().numpy().squeeze(), cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

            # Add metrics to title
            plt.suptitle(
                f"Model: {MODEL_NAME}\nDice: {x_dice:.2f}, HD95: {x_hd:.2f}"
            )
            # tumor_class = x_label
            # tumor_class = "benign" if tumor_class == True else "malignant"
            if save:
                sample_idx = x_path.split("/")[-1].split(".")[0]
                plt.savefig(f"{save_dir}/sample_idx_{sample_idx}.png")
            if preview:
                plt.show()
            else:
                plt.close()
    # print("Output Shape: ", mask.shape)  # Should be (B, 1, 256, 256)



def visualize_batch_overlay(batch, mask, num_imgs=1, preview=False,
                            MODEL_NAME="UNet", threshold=0.5,
                            save=False, save_dir=None,
                            batch_metrics: dict = None,
                            contour_thickness: int = 2):
    for idx, (image, gt_mask, pr_mask, x_path, x_label, x_dice, x_hd) in enumerate(
        zip(batch["pixel_values"], batch["labels"], mask, 
            batch["metadata"]['image_path'], batch["metadata"]['label'], 
            batch_metrics["dice"], batch_metrics["hd95"])
    ):
        if idx > num_imgs:
            break

        # Convert image to NumPy and normalize for visualization
        image_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
        image_np = (image_np * 255).astype(np.uint8)  # Convert to uint8 (0-255)

        gt_mask_np = gt_mask.cpu().numpy().squeeze().astype(np.uint8)  # (H, W)
        pr_mask_np = pr_mask.cpu().numpy().squeeze().astype(np.uint8)  # (H, W)

        # Define Colors for Masks
        gt_color = (0, 255, 0)  # Green for Ground Truth
        pr_color = (255, 0, 0)  # Red for Prediction

        # Step 1: Create Overlay
        overlay = image_np.copy()

        # Fill Ground Truth Mask
        gt_overlay = np.zeros_like(image_np, dtype=np.uint8)
        gt_overlay[gt_mask_np == 1] = gt_color  # Fill ground truth in green

        # Fill Prediction Mask
        pr_overlay = np.zeros_like(image_np, dtype=np.uint8)
        pr_overlay[pr_mask_np == 1] = pr_color  # Fill prediction in red

        # Blend with original image using alpha transparency
        alpha = 0.4  # Transparency level
        overlay = cv2.addWeighted(overlay, 1, gt_overlay, alpha, 0)  # Overlay GT
        overlay = cv2.addWeighted(overlay, 1, pr_overlay, alpha, 0)  # Overlay Prediction

        # Step 2: Draw Contours (Same Colors as Overlays)
        contours_gt, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pr, _ = cv2.findContours(pr_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(overlay, contours_gt, -1, gt_color, thickness=contour_thickness)  # GT Contour
        cv2.drawContours(overlay, contours_pr, -1, pr_color, thickness=contour_thickness)  # Prediction Contour

        # Step 3: Plot the Figure
        plt.figure(figsize=(10, 10), dpi=200)
        plt.imshow(overlay)
        plt.axis("off")

        # Add Legend
        legend_patches = [
            Patch(color=np.array(gt_color) / 255, label="Ground Truth"),
            Patch(color=np.array(pr_color) / 255, label="Prediction"),
        ]
        plt.legend(handles=legend_patches, loc="upper right", fontsize=10)

        # Add title with model name and metrics
        plt.title(f"Model: {MODEL_NAME}\nDice: {x_dice:.2f}, HD95: {x_hd:.2f}", fontsize=12)

        # Save or display the figure
        if save:
            os.makedirs(save_dir, exist_ok=True)
            sample_idx = x_path.split("/")[-1].split(".")[0]
            plt.savefig(f"{save_dir}/sample_idx_{sample_idx}.png", bbox_inches="tight")

        if preview:
            plt.show()
        else:
            plt.close()
