"""
Standalone text(+optional reference image)-conditioned inference for a
trained LSSN checkpoint. This is the piece that was completely missing
from the repo before this pass: train_lssn.py's dummy loop and
train_real.py both only ever run the *forward* process (add noise, predict
it); nothing turned a trained model's predictions back into an image until
sampler.py's DDIMSampler. This script wires that sampler up to a real
checkpoint + the frozen CLIP/VAE codecs for end-to-end text -> image.

Usage:
    python sample_lssn.py "a castle on a floating island" --out out.png
    python sample_lssn.py "a castle on a floating island" --ref_image path.png --out out.png
"""
import argparse
import os

import torch
from PIL import Image

from lssn_model import LSSN_UNet
from noise_schedule import NoiseSchedule
from encoders import CLIPConditioner
from vae_utils import LatentVAE
from sampler import DDIMSampler

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LSSN_UNet(
        model_channels=ckpt["model_channels"],
        context_dim=ckpt["context_dim"],
    ).to(device)
    # Sample from the EMA weights (Ho et al. 2020 / SD convention) -- lower
    # variance than the raw training weights, see ema.py.
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--ref_image", type=str, default=None,
                         help="Optional reference image for the image-conditioning branch.")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(CKPT_DIR, "lssn_latest.pt"))
    parser.add_argument("--out", type=str, default="lssn_sample.png")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"No checkpoint at {args.checkpoint}. Run train_real.py first -- "
            f"there is no pretrained LSSN checkpoint bundled with this repo."
        )

    conditioner = CLIPConditioner(device=device, dtype=torch.float16)
    vae = LatentVAE(device=device, dtype=torch.float16)
    schedule = NoiseSchedule(timesteps=1000, device=device)
    model = load_model(args.checkpoint, device)

    c_text = conditioner.encode_text([args.prompt]).float()
    if args.ref_image:
        ref = Image.open(args.ref_image).convert("RGB")
        c_image = conditioner.encode_image([ref]).float()
    else:
        c_image = conditioner.null_image(1).float()

    uncond_text = conditioner.null_text(1).float()
    uncond_image = conditioner.null_image(1).float()

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    sampler = DDIMSampler(schedule, num_inference_steps=args.steps)
    latent_shape = (1, 4, LatentVAE.IMAGE_SIZE // 8, LatentVAE.IMAGE_SIZE // 8)

    x0 = sampler.sample(
        model, latent_shape, c_text, c_image, uncond_text, uncond_image,
        guidance_scale=args.guidance_scale, generator=generator, device=device,
    )
    image = vae.to_pil(vae.decode(x0))[0]
    image.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
