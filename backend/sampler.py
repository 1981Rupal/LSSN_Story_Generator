import torch


class DDIMSampler:
    """
    Deterministic DDIM reverse sampling (Song, Meng & Ermon 2021, "Denoising
    Diffusion Implicit Models"), with classifier-free guidance (Ho &
    Salimans 2022, "Classifier-Free Diffusion Guidance").

    This did not exist anywhere in the codebase before: noise_schedule.py
    only implements q_sample, the *forward* corruption process used during
    training. Nothing turned a trained model's noise predictions back into
    an image. DDIM is used here instead of ancestral DDPM sampling because
    it needs far fewer steps (~50 vs. 1000) to reach a comparable result,
    which matters when each step is a full LSSN_UNet forward pass.

    Guidance is applied over the *fused* dual-modality prediction: the UNet
    is run twice per step (conditional wih real c_text/c_image, unconditional
    with null embeddings from CLIPConditioner.null_text/null_image), and the
    prediction is extrapolated away from the unconditional one:
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    """

    def __init__(self, schedule, num_inference_steps=50):
        self.schedule = schedule
        self.num_inference_steps = num_inference_steps

        step_ratio = schedule.timesteps // num_inference_steps
        # descending timesteps, e.g. [980, 960, ..., 20, 0] for 50 steps / 1000 total
        self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = torch.flip(self.timesteps, dims=[0]).to(schedule.alphas_cumprod.device)

    @torch.no_grad()
    def sample(self, model, shape, c_text, c_image, uncond_c_text, uncond_c_image,
               guidance_scale=7.5, generator=None, device="cuda"):
        """
        shape: (B, C, H, W) latent shape to generate.
        c_text/c_image: real conditioning embeddings, (B, 77, D) / (B, 1, D).
        uncond_c_text/uncond_c_image: null embeddings of the same shape, for CFG.
        Returns the final denoised latent x_0, (B, C, H, W).
        """
        x_t = torch.randn(shape, device=device, generator=generator)
        batch_size = shape[0]

        for i, t in enumerate(self.timesteps):
            t_batch = t.expand(batch_size)

            eps_cond = model(x_t, t_batch, c_text, c_image, mode="fused")
            if guidance_scale != 1.0:
                eps_uncond = model(x_t, t_batch, uncond_c_text, uncond_c_image, mode="fused")
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            alpha_bar_t = self.schedule.alphas_cumprod[t]
            t_prev = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else torch.tensor(-1, device=x_t.device)
            alpha_bar_prev = self.schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(alpha_bar_t)

            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-4.0, 4.0)  # latents are roughly unit-scaled; guards against divergence early in training

            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev) * eps

        return x_t
