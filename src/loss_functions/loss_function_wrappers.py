import torch
import torch.nn as nn
import torch.nn.functional as F

class EffectSumLosswrapper(nn.Module):
    def __init__(self, loss_function=F.mse_loss):
        super(EffectSumLosswrapper, self).__init__()
        self.loss_function = loss_function
        
    def forward(self, pred, target, mask):
        mask = mask.bool()
        valid_pred = ~torch.isnan(pred)
        valid_target = ~torch.isnan(target)
        valid_elements = valid_pred & valid_target & ~mask
        filtered_pred = pred[valid_elements]
        filtered_target = target[valid_elements]
        loss = self.loss_function(filtered_pred, filtered_target)
        if loss.numel() > 1:
            loss = loss.sum()
        return loss
    
class EffectMeanLossWrapper(nn.Module):
    def __init__(self, loss_function=F.mse_loss):
        super(EffectMeanLossWrapper, self).__init__()
        self.loss_function = loss_function
        
    def forward(self, pred, target, mask):
        mask = mask.bool()
        valid_pred = ~torch.isnan(pred)
        valid_target = ~torch.isnan(target)
        valid_elements = valid_pred & valid_target & ~mask
        filtered_pred = pred[valid_elements]
        filtered_target = target[valid_elements]
        loss = self.loss_function(filtered_pred, filtered_target)
        if loss.numel() > 1:
            loss = loss.mean()
        return loss
    
class ConvergedLossWrapper(nn.Module):
    def __init__(self, loss_function=F.cross_entropy):
        super(ConvergedLossWrapper, self).__init__()
        self.loss_function = loss_function
        
    def forward(self, pred, target, mask):
        mask = mask.bool()
        valid_pred = ~torch.isnan(pred)
        valid_target = ~torch.isnan(target)
        valid_elements = valid_pred & valid_target & ~mask
        filtered_pred = pred[valid_elements]
        filtered_target = target[valid_elements]
        loss = self.loss_function(filtered_pred, filtered_target)
        if loss.numel() > 1:
            loss = loss.mean()
        return loss