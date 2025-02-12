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
    def __init__(self, alpha=0.5):
        super(DenoisingLoss, self).__init__()
        self.alpha = alpha
        # Define loss functions
        # Define RMSE loss
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.dice_loss = BinaryDiceLoss()

    def forward(self, outputs, targets):
        bce = self.bce_loss(outputs, targets)
        mse = self.mse_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        return self.alpha * dice + (1 - self.alpha) * mse
