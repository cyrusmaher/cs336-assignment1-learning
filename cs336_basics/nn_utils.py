import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    largest_logit = torch.max(logits, dim=-1, keepdim=True).values
    correct_class_logit = logits[torch.arange(logits.shape[0]), labels]
    loss = (
        largest_logit 
        + torch.log(torch.sum(torch.exp(logits - largest_logit), dim=-1, keepdim=True)) 
        - correct_class_logit
    )
    return loss.mean()



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
                return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        beta1, beta2 = betas
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta values: beta1={beta1}, beta2={beta2}")
        
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get gradient
                grad = p.grad.data # Get the gradient of loss with respect to p.

                # Get states
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.

                momentum = state.get("momentum", 0) # Get the momentum parameter from the state, or 0.
                velocity = state.get("velocity", 0) # Get the velocity parameter from the state, or 0.
                
                # apply weight decay
                p.data = p.data - group["lr"] * self.weight_decay * p.data
          
                # update states
                t = t + 1

                alpha_t = group["lr"] * ((1 - group["beta2"]**t) ** 0.5) / (1 - group["beta1"]**t)
                momentum = group["beta1"] * momentum + (1 - group["beta1"]) * grad
                velocity = group["beta2"] * velocity + (1 - group["beta2"]) * grad**2
                
                # apply momentum-adjusted weight updates
                p.data = p.data - alpha_t * momentum / (torch.sqrt(velocity) + group["eps"])
                
                # Write back to state
                state["t"] = t
                state['momentum'] = momentum
                state['velocity'] = velocity

        return loss