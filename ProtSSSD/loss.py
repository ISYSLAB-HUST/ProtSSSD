""" 
"""
import torch
from einops import rearrange

class LossTotal(object):
    def __init__(self, loss_weight=None):
        self.mse_loss_fn = torch.nn.MSELoss(reduction='none')
        self.crossentropyLoss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bin_crossentropyLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.weight = loss_weight

    def l2_loss(self, input, target):
        norm = torch.norm(input, dim=-1, keepdim=True)
        input = input/norm
        diff_norm_gt = torch.norm(input - target, dim=-1) # [batch, N, 2] -> [batch, N]
        l_torsion = diff_norm_gt + torch.abs(norm.squeeze(-1) - 1) * 0.02
        return l_torsion
  
    def loss_fn(self, input, target):
        mask = target["valid_mask"] == 1
        disorder_mask = target["disorder"] == 1
        # ss8 -> [b, l]
        ss8_loss = self.crossentropyLoss(rearrange(input["ss8"], 'b l c -> b c l'), target["ss8"])*self.weight["ss8"]
        ss3_loss = self.crossentropyLoss(rearrange(input["ss3"], 'b l c -> b c l'), target["ss3"])*self.weight["ss3"]
        disorder_loss = self.bin_crossentropyLoss(input["disorder"], target["disorder"].float())*self.weight["disorder"]

        # phi_diff_norm_gt = torch.norm(input["phi"] - target["phi"], dim=-1)
        rsa_loss = self.mse_loss_fn(input["rsa"], target["rsa"])* self.weight["rsa"] # [b, l]
        phi_loss = self.l2_loss(input["phi"], target["phi"])* self.weight["phi"] # [b, l]
        psi_loss = self.l2_loss(input["psi"], target["psi"])* self.weight["psi"] # [b, l]

        # print(mask.shape, ss8_loss.shape)

        loss = ss8_loss[mask].mean() + ss3_loss[mask].mean() + disorder_loss[mask].mean() + \
                rsa_loss[disorder_mask & mask].mean() + phi_loss[disorder_mask & mask].mean()  + psi_loss[disorder_mask & mask].mean()
        return loss
    
    def dynamic_loss_fn(self, input, target):
        mask = target["valid_mask"] == 1
        disorder_mask = target["disorder"] == 1
        # ss8 -> [b, l]
        eps = 1e-7
        ss8_loss = self.crossentropyLoss(rearrange(input["ss8"], 'b l c -> b c l'), target["ss8"])[mask].mean()
        ss3_loss = self.crossentropyLoss(rearrange(input["ss3"], 'b l c -> b c l'), target["ss3"])[mask].mean()
        disorder_loss = self.bin_crossentropyLoss(input["disorder"], target["disorder"].float())[mask].mean()
        rsa_loss = self.mse_loss_fn(input["rsa"], target["rsa"])[disorder_mask & mask].mean() # [b, l]

        phi_loss = self.l2_loss(input["phi"], target["phi"])[disorder_mask & mask].mean() # [b, l]
        psi_loss = self.l2_loss(input["psi"], target["psi"])[disorder_mask & mask].mean() # [b, l]

        # print(mask.shape, ss8_loss.shape)

        loss = ss8_loss/(ss8_loss.detach()+eps) + ss3_loss/(ss3_loss.detach()+eps) + disorder_loss/(disorder_loss.detach()+eps)+ \
                rsa_loss/(rsa_loss.detach()+eps) + phi_loss/(phi_loss.detach()+eps) + psi_loss/(psi_loss.detach()+eps)
        return loss
    
    
if __name__ == "__main__":
    b, l  = 2, 10
    input_tensor = {"ss8": torch.randn((b, l, 8)), "ss3":torch.randn((b, l, 3)), 
                    "disorder":torch.randn((b, l)), "rsa":torch.randn((b, l)), 
                    "phi":torch.randn((b, l, 2)), "psi":torch.randn((b, l, 2))}
    
    target_tensor = {"ss8": torch.ones((b, l), dtype=torch.long), "ss3": torch.ones((b, l),dtype=torch.long), 
                    "disorder":torch.ones((b, l), dtype=torch.int), "rsa":torch.randn((b, l)), 
                    "phi":torch.randn((b, l, 2)), "psi":torch.randn((b, l, 2)),
                    "valid_mask":torch.ones((b, l), dtype=torch.int)}
    loss_weight = {"ss8":1,"ss3":5,"disorder":5,"rsa":100,"phi":5,"psi":5}
    loss_metrics = LossTotal(loss_weight)
    loss = loss_metrics.loss_fn(input_tensor, target_tensor)
    print(loss)


