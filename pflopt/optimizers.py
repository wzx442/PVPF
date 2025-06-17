import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class MaskLocalAltSGD(optim.Optimizer):
    def __init__(self, params, mask: OrderedDict = None, lr=0.01):
       
        self.mask: list[torch.Tensor] = [value.long() for key, value in mask.items()]
        self.names: list[torch.Tensor] = [key for key, value in mask.items()]
        self.named_mask: OrderedDict = mask
        self._toggle = True
        defaults = dict(lr=lr, _toggle=True)

        if mask is None:
            raise ValueError("MaskLocalAltSGD requires a mask")
        super(MaskLocalAltSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskLocalAltSGD, self).__setstate__(state)

    def toggle(self):
        self._toggle = not self._toggle

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
       
        for group in self.param_groups:
            step = 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
              
                if torch.isnan(p).any():
                    print(f"Warning: NaN detected in parameter at step {step}")
                    p.data.zero_()
                    p.grad.data.zero_()
                    continue
                    
                if torch.isnan(p.grad).any():
                    print(f"Warning: NaN detected in gradient at step {step}")
                    p.grad.data.zero_()
                    continue
                
                
                if step < len(self.mask):
                    mask = self.mask[step]
                    param_name = self.names[step] if step < len(self.names) else f"unknown_{step}"
                    
                  
                    if mask.shape != p.shape:
                        print(f"Warning: Mask shape {mask.shape} doesn't match parameter shape {p.shape} for {param_name}")
                        
                       
                        if 'batch_norm' in param_name or '.bn' in param_name or '.running_' in param_name:
                           
                            if self._toggle:
                               
                                uniform_mask_value = 0
                            else:
                                
                                uniform_mask_value = 1
                            mask = torch.ones_like(p.data, dtype=torch.long) * uniform_mask_value
                        else:
                            
                            try:
                                
                                if mask.numel() == p.numel():
                                    mask = mask.reshape(p.shape)
                                else:
                                    
                                    if len(p.shape) == 1:  
                                        if len(mask.shape) > 1:  
                                            
                                            if mask.shape[0] == p.shape[0]:
                                                mask = mask[:, 0, 0, 0] if len(mask.shape) == 4 else mask[:, 0]
                                            else:
                                                raise ValueError(f"Cannot match dimensions: mask {mask.shape}, param {p.shape}")
                                    else:
                                        raise ValueError(f"Incompatible shapes: mask {mask.shape}, param {p.shape}")
                            except Exception as e:
                                print(f"Error reshaping mask: {e}")
                                
                                if self._toggle:
                                    
                                    mask = torch.ones_like(p.data, dtype=torch.long)
                                else:
                                    
                                    mask = torch.zeros_like(p.data, dtype=torch.long)
                    
                    
                    if self._toggle:
                        
                        grad = p.grad.data * mask
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                        p.data.add_(grad, alpha=-group["lr"])
                    else:
                        
                        
                        grad = p.grad.data * (1 - mask)
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                        p.data.add_(grad, alpha=-group["lr"])
                else:
                    print(f"Warning: No mask found for parameter at step {step}")
                    grad = p.grad.data
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                    p.data.add_(grad, alpha=-group["lr"])
                step += 1
        return loss


def local_alt(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    clip_grad_norm=False,
    max_grad_norm=3.50,
):
    assert isinstance(optimizer, MaskLocalAltSGD), "optimizer must be MaskLocalAltSGD" # 断言优化器必须是MaskLocalAltSGD
    avg_loss_1 = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_loss_1 += loss.item()
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    avg_loss_1 /= len(data_loader)
    optimizer.toggle()

    avg_loss_2 = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        avg_loss_2 += loss.item()
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    avg_loss_2 /= len(data_loader)
    optimizer.toggle()

    train_loss = (avg_loss_1 + avg_loss_2) / 2
    return train_loss


if __name__ == "__main__":
    pass
