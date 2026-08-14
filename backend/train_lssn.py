import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging

from lssn_model import LSSN_UNet
from loss import InvarianceLoss
from noise_schedule import NoiseSchedule

logging.basicConfig(level=logging.INFO, format='LSSN-Train [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class LSSNTrainer:
    """
    Core Training Framework for the Latent-Space Synchronization Network (LSSN).
    Implements a Dual-Path Latent Diffusion Model (LDM) to resolve
    Latent Trajectory Divergence (LTD) by eliminating Modality Bias.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Initializing LSSN Dual-Path LDM on {self.device}")

        self.model = LSSN_UNet().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

        # BUG FIX 1: no forward diffusion process existed anywhere in this file --
        # x_0 (clean latent) was fed straight into the UNet with no noise added,
        # so timestep conditioning was never actually tied to a noise level.
        # NoiseSchedule implements the standard DDPM variance-preserving process.
        self.schedule = NoiseSchedule(timesteps=1000, device=self.device)

        # BUG FIX 2: this trainer never imported InvarianceLoss from loss.py and
        # reimplemented a bare MSE-only version below, silently dropping the
        # cosine-alignment term the loss is documented to need. Restored here.
        self.inv_criterion = InvarianceLoss(lambda_inv=1.0, lambda_cosine=0.5)

        self.lambda_inv = 0.5

    def train_step(self, x_0, t, c_text, c_image):
        """
        x_0: Clean latent image, (B, C, H, W)
        t: Timestep, (B,) long tensor in [0, schedule.timesteps)
        c_text: Embedded text description
        c_image: Embedded visual reference
        """
        self.optimizer.zero_grad()

        # 1. Forward diffusion: sample x_t and the noise that produced it.
        #    (Previously: x_0 was used directly as model input -- no noise, no t-dependence.)
        noise = torch.randn_like(x_0)
        x_t = self.schedule.q_sample(x_0, t, noise)

        # 2. Standard Denoising Path (Synchronized Modal Fusion)
        noise_pred_sync = self.model(x_t, t, c_text, c_image, return_features=False, mode="fused")
        loss_task = nn.functional.mse_loss(noise_pred_sync, noise)

        # 3. Parallel Path Text (Branch A) -- mode="text_only" bypasses the
        #    learnable gate entirely instead of zeroing c_image and relying on
        #    the fused path to collapse to alpha*out_text (see lssn_modules.py).
        _, features_text = self.model(x_t, t, c_text, c_image, return_features=True, mode="text_only")

        # 4. Parallel Path Image (Branch B)
        _, features_image = self.model(x_t, t, c_text, c_image, return_features=True, mode="image_only")

        # 5. Invariance Loss: hybrid L2 + cosine alignment, on ungated features.
        loss_inv = self.inv_criterion(features_text, features_image)

        total_loss = loss_task + (self.lambda_inv * loss_inv)

        total_loss.backward()
        self.optimizer.step()

        return loss_task.item(), loss_inv.item(), total_loss.item()

    def dummy_train_loop(self, epochs=1):
        logger.info("Starting LSSN Training Optimization...")

        batch_size = 1
        channels, h, w = 4, 32, 32
        context_dim = 768

        for epoch in range(epochs):
            logger.info(f"--- Epoch {epoch+1}/{epochs} ---")

            for step in tqdm(range(2), desc="Simulated Batches"):
                x_0 = torch.randn(batch_size, channels, h, w).to(self.device)
                t = torch.randint(0, self.schedule.timesteps, (batch_size,)).to(self.device)

                c_text = torch.randn(batch_size, 77, context_dim).to(self.device)
                c_image = torch.randn(batch_size, 77, context_dim).to(self.device)

                l_task, l_inv, l_tot = self.train_step(x_0, t, c_text, c_image)

            logger.info(f"Epoch Results -> Total Loss: {l_tot:.4f} | Denoising Loss: {l_task:.4f} | Invariance Loss (L_inv): {l_inv:.4f}")

if __name__ == "__main__":
    trainer = LSSNTrainer()
    trainer.dummy_train_loop()
    print("\nDual-Path Latent Diffusion Optimization Complete.")
