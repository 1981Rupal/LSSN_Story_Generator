import copy
import torch


class EMA:
    """
    Exponential moving average of model weights. Standard practice across
    DDPM (Ho et al. 2020), Stable Diffusion, and most modern diffusion
    training code -- the raw (non-EMA) weights are noisy from
    step-to-step SGD variance; sampling from the EMA weights is
    consistently reported to give more stable, higher-quality generations
    than sampling from the live training weights.
    """

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)
