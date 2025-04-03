import torch
import torch.nn as nn


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (predict * target).sum(1)
        dice = (2.0 * intersection + self.smooth) / (predict.sum(1) + target.sum(1) + self.smooth)
        return 1 - dice.mean()

class DenoisingLoss(nn.Module):
    """
    DenoisingLoss: Combines Dice loss and MSE loss to evaluate denoising performance.
    
    This loss function balances segmentation performance (Dice loss) with pixel-wise 
    reconstruction accuracy (MSE loss) using a weighting parameter alpha.
    
    Args:
        alpha (float): Weight factor to balance Dice loss and MSE loss. 
                      Higher alpha gives more weight to Dice loss.
    """
    def __init__(self, alpha=0.8):
        super(DenoisingLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def dice_loss(self, preds, targets):
        """
        Calculate Dice loss between predictions and targets.
        
        Args:
            preds (torch.Tensor): Predicted segmentation masks
            targets (torch.Tensor): Ground truth segmentation masks
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Add small epsilon to avoid division by zero
        smooth = 1e-5
        
        # Calculate intersection and union
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        
        # Calculate Dice coefficient and convert to loss
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss
    
    def forward(self, preds, targets):
        """
        Calculate combined loss of Dice loss and MSE loss.
        
        Args:
            preds (torch.Tensor): Predicted output from the model
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Weighted combination of Dice loss and MSE loss
        """
        # Calculate individual loss components
        d_loss = self.dice_loss(preds, targets)
        mse_loss = self.mse_loss(preds, targets)
        
        # Return weighted combination
        return self.alpha * d_loss + (1 - self.alpha) * mse_loss
