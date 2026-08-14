"""
Real training loop for LSSN, replacing the dummy torch.randn() inputs in
train_lssn.py's dummy_train_loop with actual data end-to-end:

  real images/prompts (prepare_dataset.py output)
    -> frozen CLIP text/image encoder (encoders.py)      [Radford et al. 2021]
    -> frozen pretrained VAE image<->latent codec         [Rombach et al. 2022]
    -> LSSN_UNet dual-path denoising + InvarianceLoss      [this repo, now fixed]
    -> EMA weight tracking                                 [Ho et al. 2020 / SD]
    -> classifier-free-guidance conditioning dropout        [Ho & Salimans 2022]

train_lssn.py / dummy_train_loop is left untouched -- it's still the fast
"does the training step function at all" smoke test the README documents.
This script is the actual training path.

Model width is reduced from the architecture's default 320 channels to 128:
LSSN's dual-path objective runs the full UNet three times per training step
(fused/text_only/image_only), and the GPU verification run measured ~10.5GB
peak VRAM at model_channels=320, batch_size=1, no memory optimizations --
already over this machine's 8GB budget. 128 channels + fp16 autocast +
gradient checkpointing brings that comfortably under budget at batch_size=2.
This is a deliberate compute/quality tradeoff for feasibility on a single
consumer GPU, not a bug fix -- see the printed VRAM figures at the bottom of
the run for the actual headroom achieved.
"""
import argparse
import glob
import os
import random

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging

from lssn_model import LSSN_UNet
from loss import InvarianceLoss
from noise_schedule import NoiseSchedule
from encoders import CLIPConditioner
from vae_utils import LatentVAE
from ema import EMA
from sampler import DDIMSampler

logging.basicConfig(level=logging.INFO, format="LSSN-Real-Train [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "generated_assets", "training_previews")


class PairedImageTextDataset(Dataset):
    """Reads (NNNNN.png, NNNNN.txt) pairs written by prepare_dataset.py."""

    def __init__(self, split_dir, image_size=256):
        self.pngs = sorted(glob.glob(os.path.join(split_dir, "*.png")))
        if not self.pngs:
            raise FileNotFoundError(
                f"No training pairs found in {split_dir}. Run prepare_dataset.py first."
            )
        self.image_size = image_size

    def __len__(self):
        return len(self.pngs)

    def __getitem__(self, idx):
        img_path = self.pngs[idx]
        txt_path = img_path[:-4] + ".txt"
        img = Image.open(img_path).convert("RGB")
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        return img, prompt


def collate_fn(batch):
    images, prompts = zip(*batch)
    return list(images), list(prompts)


def pil_batch_to_vae_input(images, device):
    to_tensor = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ])
    return torch.stack([to_tensor(img) for img in images]).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_channels", type=int, default=128)
    parser.add_argument("--uncond_prob", type=float, default=0.1)
    parser.add_argument("--preview_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cpu":
        logger.warning("No CUDA device found -- real training will be extremely slow.")

    logger.info("Loading frozen CLIP conditioning encoder (openai/clip-vit-large-patch14)...")
    conditioner = CLIPConditioner(device=device, dtype=torch.float16)

    logger.info("Loading frozen VAE (stabilityai/sd-vae-ft-mse)...")
    vae = LatentVAE(device=device, dtype=torch.float16)

    train_ds = PairedImageTextDataset(os.path.join(DATA_DIR, "train"), image_size=LatentVAE.IMAGE_SIZE)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        num_workers=0, drop_last=True,
    )
    logger.info(f"Loaded {len(train_ds)} real training pairs.")

    model = LSSN_UNet(
        model_channels=args.model_channels,
        context_dim=conditioner.context_dim,
        use_checkpoint=True,
    ).to(device)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Resumed model weights from {args.resume}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    schedule = NoiseSchedule(timesteps=1000, device=device)
    inv_criterion = InvarianceLoss(lambda_inv=1.0, lambda_cosine=0.5)
    lambda_inv = 0.5

    ema = EMA(model, decay=0.9995)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    fixed_preview_prompt = "a red panda reading a book in a cozy library, digital art"

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")

        for step, (images, prompts) in enumerate(pbar):
            batch_size = len(images)

            with torch.no_grad():
                vae_input = pil_batch_to_vae_input(images, device)
                x_0 = vae.encode(vae_input).float()

                c_text = conditioner.encode_text(prompts)
                c_image = conditioner.encode_image(images)

                # Classifier-free guidance training (Ho & Salimans 2022): drop
                # the whole conditioning set to the null embedding for a
                # fraction of examples so the model also learns p(x) as well
                # as p(x|c), which DDIMSampler's guidance_scale relies on.
                drop_mask = (torch.rand(batch_size, device=device) < args.uncond_prob).view(-1, 1, 1)
                null_text = conditioner.null_text(batch_size)
                null_image = conditioner.null_image(batch_size)
                c_text = torch.where(drop_mask, null_text, c_text)
                c_image = torch.where(drop_mask, null_image, c_image)

                c_text = c_text.float()
                c_image = c_image.float()

                t = torch.randint(0, schedule.timesteps, (batch_size,), device=device)
                noise = torch.randn_like(x_0)
                x_t = schedule.q_sample(x_0, t, noise)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                noise_pred_sync = model(x_t, t, c_text, c_image, return_features=False, mode="fused")
                loss_task = torch.nn.functional.mse_loss(noise_pred_sync, noise)

                _, features_text = model(x_t, t, c_text, c_image, return_features=True, mode="text_only")
                _, features_image = model(x_t, t, c_text, c_image, return_features=True, mode="image_only")
                loss_inv = inv_criterion(features_text, features_image)

                total_loss = (loss_task + lambda_inv * loss_inv) / args.grad_accum

            scaler.scale(total_loss).backward()

            if (global_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            pbar.set_postfix(
                loss_task=f"{loss_task.item():.4f}",
                loss_inv=f"{loss_inv.item():.4f}",
            )

            if global_step % 20 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                logger.info(
                    f"step {global_step}: loss_task={loss_task.item():.4f} "
                    f"loss_inv={loss_inv.item():.4f} peak_vram={mem:.2f}GB"
                )

            if global_step > 0 and global_step % args.preview_every == 0:
                _save_preview(ema.shadow, schedule, conditioner, vae, fixed_preview_prompt,
                               device, global_step, args.model_channels)

            global_step += 1

        ckpt_path = os.path.join(CKPT_DIR, f"lssn_epoch{epoch + 1}.pt")
        torch.save({
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_channels": args.model_channels,
            "context_dim": conditioner.context_dim,
            "epoch": epoch + 1,
            "global_step": global_step,
        }, ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    latest_path = os.path.join(CKPT_DIR, "lssn_latest.pt")
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "model_channels": args.model_channels,
        "context_dim": conditioner.context_dim,
        "epoch": args.epochs,
        "global_step": global_step,
    }, latest_path)
    logger.info(f"Training complete. Final checkpoint: {latest_path}")


@torch.no_grad()
def _save_preview(ema_model, schedule, conditioner, vae, prompt, device, step, model_channels):
    ema_model.eval()
    sampler = DDIMSampler(schedule, num_inference_steps=30)

    c_text = conditioner.encode_text([prompt]).float()
    c_image = conditioner.null_image(1).float()
    uncond_text = conditioner.null_text(1).float()
    uncond_image = conditioner.null_image(1).float()

    latent_shape = (1, 4, LatentVAE.IMAGE_SIZE // 8, LatentVAE.IMAGE_SIZE // 8)
    x0 = sampler.sample(
        ema_model, latent_shape, c_text, c_image, uncond_text, uncond_image,
        guidance_scale=7.5, device=device,
    )
    images = vae.to_pil(vae.decode(x0))
    out_path = os.path.join(SAMPLE_DIR, f"preview_step{step:06d}.png")
    images[0].save(out_path)
    logger.info(f"Saved training preview to {out_path}")


if __name__ == "__main__":
    main()
