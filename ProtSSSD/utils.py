""" 
"""
import torch
import numpy as np
from torchmetrics import MatthewsCorrCoef, Accuracy, Recall, Metric
from torchmetrics.regression import SpearmanCorrCoef
from torch import Tensor, tensor

def _mean_absolute_error_update(preds: Tensor, target: Tensor):
    """Update and returns variables required to compute Mean Absolute Error.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor

    """
    # _check_same_shape(preds, target)
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    err = torch.abs(preds - target)
    sum_abs_error = torch.sum(torch.fmin(err, 360-err))
    return sum_abs_error, target.numel()

def arctan_dihedral(sin: torch.tensor, cos: torch.tensor) -> torch.tensor:
    """ Converts sin and cos back to diheral angles
    Args:
        sin: tensor with sin values 
        cos: tensor with cos values
    """
    result = torch.where(cos >= 0, torch.arctan(sin / cos),
                         torch.arctan(sin / cos) + np.pi)
    result = torch.where((sin <= 0) & (cos <= 0), result - np.pi * 2, result)

    return result * 180 / np.pi

class AngleMeanAbsoluteError(Metric):
    r"""`Compute Mean Absolute Error`_ (MAE).

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_error, num_obs = _mean_absolute_error_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return self.sum_abs_error/self.total

    def plot(
        self, val= None, ax= None
    ):
        """Plot a single or multiple values from the metric.
        """
        return self._plot(val, ax)
    
class ProtMetrics(object):
    def __init__(self,ss3=3, ss8=8, device='cuda', disorder_infer=True) -> None:
        # pass
        # self.metrics()
        self.ss3 = ss3 
        self.ss8 = ss8
        self.disorder_infer = disorder_infer

        ss3_acc = Accuracy(task="multiclass", num_classes=self.ss3).to(device)
        ss8_acc = Accuracy(task="multiclass", num_classes=self.ss8).to(device)

        phi_mae = AngleMeanAbsoluteError().to(device)
        psi_mae = AngleMeanAbsoluteError().to(device)
        rsa_pcc = SpearmanCorrCoef().to(device)

        self.metrics_fn = {"ss3":ss3_acc, "ss8":ss8_acc, "phi_mae":phi_mae,"psi_mae":psi_mae, "rsa_pcc":rsa_pcc}
        
        if self.disorder_infer:
            disorder_mcc = MatthewsCorrCoef(task="binary").to(device)
            disorder_recall = Recall(task="binary").to(device)
            self.metrics_fn["disorder_mcc"] = disorder_mcc
            self.metrics_fn["disorder_recall"] = disorder_recall

    
    def update_step(self, predict, target):
        mask = target["valid_mask"] == 1
        if self.disorder_infer:
            disorder_mask = target["disorder"] == 1
            mask_w_dis = mask & disorder_mask # [b, l]
        else:
            mask_w_dis = mask
        # print(predict["ss3"].device, target["ss3"].device)
        ss3_acc_step = self.metrics_fn["ss3"](predict["ss3"][mask], target["ss3"][mask])
        ss8_acc_step = self.metrics_fn["ss8"](predict["ss8"][mask], target["ss8"][mask])


        rsa_pcc_step = self.metrics_fn["rsa_pcc"](predict["rsa"][mask_w_dis], target["rsa"][mask_w_dis])

        predict_phi_norm = torch.norm(predict["phi"], dim=-1, keepdim=True)
        predict["phi"] = predict["phi"] / predict_phi_norm

        phi_norm = torch.norm(target["phi"], dim=-1, keepdim=True)
        target["phi"]= target["phi"]/ phi_norm # [b, l, 2]
        predict_phi_dihedral = arctan_dihedral(predict["phi"][:,:,0], predict["phi"][:,:,1])
        phi_dihedral = arctan_dihedral(target["phi"][:,:,0], target["phi"][:,:,1])

        phi_mae_step = self.metrics_fn["phi_mae"](predict_phi_dihedral[mask_w_dis], phi_dihedral[mask_w_dis])
        
        predict_psi_norm = torch.norm(predict["psi"], dim=-1, keepdim=True)
        predict["psi"] = predict["psi"] / predict_psi_norm

        phi_norm = torch.norm(target["psi"], dim=-1, keepdim=True)
        target["psi"]= target["psi"]/ phi_norm
        predict_psi_dihedral = arctan_dihedral(predict["psi"][:,:,0], predict["psi"][:,:,1])
        psi_dihedral = arctan_dihedral(target["psi"][:,:,0], target["psi"][:,:,1])

        psi_mae_step = self.metrics_fn["psi_mae"](predict_psi_dihedral[mask_w_dis], psi_dihedral[mask_w_dis])

        metrics_step = {"ss3_acc_step":ss3_acc_step, "ss8_acc_step":ss8_acc_step,"rsa_pcc_step":rsa_pcc_step, 
                        "phi_mae_step":phi_mae_step, "psi_mae_step":psi_mae_step}
        
        if self.disorder_infer:
            disorder_mcc_step = self.metrics_fn["disorder_mcc"](predict["disorder"][mask], 
                                                                target["disorder"][mask])
            disorder_recall_step = self.metrics_fn["disorder_recall"](predict["disorder"][mask], 
                                                                    target["disorder"][mask])
            
            metrics_step['disorder_mcc_step'] = disorder_mcc_step
            metrics_step["disorder_recall_step"] = disorder_recall_step

        return metrics_step
    

    def compute(self):
        return {k: v.compute() for k, v in self.metrics_fn.items() if hasattr(v, "compute") and callable(v.compute)}
    
    def reset(self):
        return {k: v.reset() for k, v in self.metrics_fn.items() if hasattr(v, "reset") and callable(v.reset)}

if __name__ == "__main__":
    a = torch.randn((2, 10))
    sin, cos = torch.sin(a), torch.cos(a)
    c = arctan_dihedral(sin, cos)
    print(c.shape)



