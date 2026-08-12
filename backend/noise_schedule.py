import torch


class NoiseSchedule:
    """
    Variance-preserving (VP) forward diffusion schedule, per Ho et al. 2020 (DDPM).

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    This was previously absent from the codebase: train_lssn.py fed x_0 (the
    clean latent) directly into the UNet with no noise added, and the earlier
    research-repo variant mixed noise with a fixed scalar (alpha=0.5) instead
    of a timestep-dependent schedule. In both cases the sampled timestep t was
    passed to the network as a conditioning signal but never controlled the
    actual corruption level of its input, so t-conditioning was vestigial and
    the objective did not correspond to a valid diffusion training loss.
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Draws x_t ~ q(x_t | x_0) for a batch of timesteps t, shape (B,).
        x_0, noise: (B, C, H, W). Returns x_t of the same shape.
        """
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x_0 + sqrt_1m_ac * noise
