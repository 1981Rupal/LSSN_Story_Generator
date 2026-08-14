import torch
from diffusers import AutoencoderKL


class LatentVAE:
    """
    Frozen pretrained VAE, stabilityai/sd-vae-ft-mse -- an MSE-finetuned
    improvement over the original SD1.x VAE (Rombach et al. 2022, "High-
    Resolution Image Synthesis with Latent Diffusion Models").

    LSSN_UNet already operates on 4-channel latents (in_channels=4,
    out_channels=4) at whatever resolution the dummy loop's 32x32 example
    implies -- that is exactly this VAE's 8x-downsampled latent space
    (256x256 image -> 32x32x4 latent). Diffusion happens in this compressed
    latent space rather than raw pixels, which is the core efficiency idea
    of LDM and the reason a consumer GPU can train this at all.

    Frozen: the VAE is not part of LSSN's training objective, only a fixed
    image<->latent codec.
    """

    MODEL_ID = "stabilityai/sd-vae-ft-mse"
    IMAGE_SIZE = 256  # -> 32x32x4 latent, matching the model's existing defaults

    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(self.MODEL_ID, torch_dtype=dtype).to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor  # 0.18215

    @torch.no_grad()
    def encode(self, images):
        """images: (B, 3, H, W) float tensor in [-1, 1] -> (B, 4, H/8, W/8) latents"""
        images = images.to(self.device, self.dtype)
        latent_dist = self.vae.encode(images).latent_dist
        return latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents):
        """(B, 4, H/8, W/8) latents -> (B, 3, H, W) float tensor in [-1, 1]"""
        latents = latents.to(self.device, self.dtype) / self.scaling_factor
        images = self.vae.decode(latents).sample
        return images.clamp(-1, 1)

    @staticmethod
    def to_pil(images):
        """(B, 3, H, W) tensor in [-1, 1] -> list[PIL.Image]"""
        from PIL import Image
        import numpy as np

        images = ((images.float().cpu() + 1.0) * 127.5).clamp(0, 255).byte()
        images = images.permute(0, 2, 3, 1).numpy()
        return [Image.fromarray(img) for img in images]
