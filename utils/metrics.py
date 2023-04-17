import torch
import torch.nn.functional as F


def calc_rmse(input_: torch.Tensor, target: torch.Tensor):
    """Calculate Root Mean Squre Error
    params:
        input_: torch.Tensor
        target: torch.Tensor
    """
    assert input_.shape == target.shape
    return torch.sqrt(F.mse_loss(input_, target)).item()

def calc_mre(input_: torch.Tensor, target: torch.Tensor):
    """Calculate Mean Relative Error
    parmas:
        input_: torch.Tensor
        target: torch.Tensor
    """
    assert input_.shape == target.shape
    return torch.sum((input_ - target) / target).item() / target.numel()

def calc_mad(input_: torch.Tensor, target: torch.Tensor):
    """Calculate Mean Absolute Deviation
    parmas:
        input_: torch.Tensor
        target: torch.Tensor
    """
    assert input_.shape == target.shape
    mad = torch.abs(input_ - target).sum().item() / input_.numel()
    return mad

def calc_pcc(input_: torch.Tensor, target: torch.Tensor):
    """Calculate Pearson Correlation Coefficient across rows
    parmas:
        input_: torch.Tensor
        target: torch.Tensor
    """
    assert input_.shape == target.shape
    dim = 0
    centered_input  = input_ - input_.mean(dim=dim, keepdim=True)
    centered_target = target - target.mean(dim=dim, keepdim=True)
    covariance = (centered_input * centered_target).sum(dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (input_.shape[dim] - 1)
    input_std = input_.std(dim=dim, keepdim=True)
    target_std = target.std(dim=dim, keepdim=True)
    pcc = bessel_corrected_covariance / (input_std * target_std)
    return pcc


if __name__ == '__main__':
    a = torch.rand((10, 1))
    b = torch.rand((10, 1))
    print('rmse:', calc_rmse(a, b))
    print('mre:', calc_mre(a, b))
    print('mad:', calc_mad(a, b))
    # print('pcc:', calc_pcc(a, b))