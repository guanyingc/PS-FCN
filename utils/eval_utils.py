import torch
import math
   
def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid   = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean  = ang_valid.sum() / valid
    value = {'n_err_mean': n_err_mean.item()}
    return value
