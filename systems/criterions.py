import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLoss(nn.Module):
    @property
    def func(self):
        raise NotImplementedError

    def forward(self, inputs, targets, weight=None, reduction='mean'):
        assert reduction in ['none', 'sum', 'mean', 'valid_mean']
        loss = self.func(inputs, targets, reduction='none')
        if weight is not None:
            while weight.ndim < inputs.ndim:
                weight = weight[..., None]
            loss *= weight.float()
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'valid_mean':
            return loss.sum() / weight.float().sum()


class MSELoss(WeightedLoss):
    @property
    def func(self):
        return F.mse_loss


class L1Loss(WeightedLoss):
    @property
    def func(self):
        return F.l1_loss


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian
        
        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        
        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale)**2
        self.c2 = (k2 * data_scale)**2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = F.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = F.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0) : (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        elif reduction == 'sum':
            return _ssim.sum()
        elif reduction == 'mean':
            return _ssim.mean()


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.prediction_depth = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        
        depth_error = (self.prediction_depth - target) * mask
        depth_error = depth_error.reshape(1, -1).permute(1, 0)
        depth_loss = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / (mask.sum() + 1e-5)

        return depth_loss

class TVLoss(nn.Module):
    def __init__(self, mask=None):
        super(TVLoss,self).__init__()
        self.TVLoss_mask = mask

    def forward(self,x):
        # x: [C, H, W]
        # mask: [H, W]
        h_x = x.size()[1]
        w_x = x.size()[2]
        count_h = self._tensor_size(x[:,1:,:])
        count_w = self._tensor_size(x[:,:,1:])
        h_tv = torch.pow((x[:,1:,:]-x[:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,1:]-x[:,:,:w_x-1]),2).sum()
        TV_Loss = 2*(h_tv/count_h+w_tv/count_w)
        if self.TVLoss_mask:
           TV_Loss = TV_Loss * self.TVLoss_mask
        return TV_Loss

    def _tensor_size(self,t):
        return t.size()[0]*t.size()[1]*t.size()[2]