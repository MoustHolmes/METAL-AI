import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFuncMaskWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        """
        Args:
            loss_fn (nn.Module): A PyTorch loss function like CrossEntropyLoss, MSELoss, etc.
        """
        super(LossFuncMaskWrapper, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target, mask= None):
        """
        Args:
            input (Tensor): Predicted values of size (N, *) where * means any number of additional dimensions.
            mask (Tensor): Mask of size (N) to filter out the padded values.
            target (Tensor): True values of size (N, *).

        Returns:
            Tensor: Computed loss after applying the mask.
        """
        # Apply the mask to input and target
        if mask is None:
            return self.loss_fn(input, target)
        else:
            return self.loss_fn(input[mask], target[mask])
        
        # masked_input = input[mask].float()
        # masked_target = target[mask].float()

        # # Compute the loss using the provided loss function
        # loss = self.loss_fn(masked_input, masked_target)
        
        # return loss

class LossFuncWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        """
        Args:
            loss_fn (nn.Module): A PyTorch loss function like CrossEntropyLoss, MSELoss, etc.
        """
        super(LossFuncWrapper, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, input, target, mask= None):
        
        return self.loss_fn(input, target)

class GaussianNLLLossWrapper(nn.Module):
    def __init__(self, loss_fn: nn.Module):
        """
        Args:
            loss_fn (nn.Module): A PyTorch loss function like GaussianNLLLoss which takes two inputs and returns the loss.
        """
        super(GaussianNLLLossWrapper, self).__init__()

        self.loss_fn = loss_fn

    def forward(self, input, target,  mask = None):
        """
        Args:
            input (Tensor): Predicted means of size (N, 2) first column is the mean and the second column is the variance.
            mask (Tensor): Mask of size (N) to filter out the padded values.
            target (Tensor): True values of size (N)

        Returns:
            Tensor: Computed Gaussian negative log likelihood loss.
        """
        # Ensure variances are non-negative by adding eps (if not handled elsewhere)
        mean = input[:,:,0]#[mask]
        var = input[:,:,1]#[mask]#.clamp(min=self.eps)
        target = target.float() #[mask]
        
        # Compute the loss using GaussianNLLLoss
        loss = self.loss_fn(mean, target, var)
        
        return loss

class DiscretizedNLLLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, num_bins: int, min_value: float, max_value: float):
        """
        :param num_bins: Number of bins to discretize the continuous range
        :param min_value: Minimum value of the range to discretize
        :param max_value: Maximum value of the range to discretize
        """
        super(DiscretizedNLLLoss, self).__init__()
        self.loss_fn = loss_fn
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        
        # Calculate the width of each bin
        self.bin_width = (max_value - min_value) / num_bins
        
        # Bins are represented by their center values
        self.bin_centers = torch.linspace(min_value + self.bin_width / 2,
                                          max_value - self.bin_width / 2, num_bins)

    def forward(self, logits, targets):
        """
        :param predictions: The continuous predictions from the model [batch_size, 1]
        :param targets: The continuous target values [batch_size, 1]
        """

        # Convert continuous target values into bin indices
        target_bin_indices = ((targets - self.min_value) / self.bin_width).long().clamp(0, self.num_bins - 1)
        
        # Apply CrossEntropyLoss (which includes log-softmax)
        loss = self.loss_fn(logits, target_bin_indices.squeeze(-1))

        return loss


