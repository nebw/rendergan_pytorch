import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


# Adapted from tetratrio: 
# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, sigma_learnable=False, max_sigma=1):
        super(GaussianSmoothing, self).__init__()
        
        self.kernel_size = kernel_size
        self.dim = dim
        self.max_sigma = max_sigma
        
        self.sigma = torch.FloatTensor([np.log(sigma / (1 - sigma))])
        if sigma_learnable:
            self.sigma = torch.nn.Parameter(self.sigma)
                    
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
            
    def get_kernel(self, device):
        sigma = torch.sigmoid(self.sigma.to(device)) * self.max_sigma
        kernel_size = self.kernel_size
        dim = self.dim
        channels = self.groups
        
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        #if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim
            
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32, device=device)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        
        return kernel

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        x = torch.nn.functional.pad(input, pad=[self.kernel_size // 2] * 4, mode='reflect')
        return self.conv(x, weight=self.get_kernel(input.device), groups=self.groups, padding=0)
    

class GaussianHighpass(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianHighpass, self).__init__()
        
        self.smoothing = GaussianSmoothing(channels, kernel_size, 1. - 1e-15, dim, sigma_learnable=False, max_sigma=sigma)
        
    def forward(self, x):
        return x - self.smoothing(x)