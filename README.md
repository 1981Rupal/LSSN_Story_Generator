# LSSN Story Generator

**LSSN** (Latent-Space Synchronization Network) is a dual-path latent diffusion
architecture for text+image-conditioned generation, wrapped in a full-stack
application that turns a text prompt into an illustrated, narrated short story.

This repository contains two things: (1) the LSSN research architecture —
a conditional latent-diffusion UNet with a novel cross-modal synchronization
mechanism and an invariance-regularized dual-path training objective — and
(2) a FastAPI + React application that uses it to generate multi-page
illustrated stories.

## Research contribution

Standard conditional latent diffusion models condition on a single modality
(usually text). LSSN conditions on **two** modalities simultaneously — a text
description and a reference image embedding — and asks a specific question:
*how do you fuse two conditioning signals without either one systematically
dominating the generation trajectory (modality bias)?*

**Synchronization Module (SM).** Each transformer block's cross-attention is
replaced with `SynchronizationModule` (`backend/lssn_modules.py`): the latent
attends to text and image conditioning independently, and the two attention
outputs are combined with a learnable gate `α = σ(g)`:

```
out = α · Attn(x, c_text) + (1 − α) · Attn(x, c_image)
```

**Invariance Regularization Loss (L_inv).** During training, the UNet is run
three times per step: once with both modalities (the actual denoising
prediction), once with only text, and once with only image. `InvarianceLoss`
(`backend/loss.py`) penalizes the L2 and cosine distance between the
intermediate SpatialTransformer features of the text-only and image-only
passes, pushing both modalities toward a shared latent trajectory rather than
letting one modality's path diverge from the other's:

```
L_total = L_denoise(x_t, t, c_text, c_image) + λ · L_inv(f_text, f_image)
```

## Architecture corrections (this revision)

An earlier revision of this code had three issues that would have silently
produced an invalid training signal. Documenting them here because the fixes
are as informative as the design itself:

1. **Missing forward diffusion process.** The trainer previously fed the
   clean latent `x_0` directly into the UNet with no noise added — `t` was
   sampled and passed as conditioning but never used to control the actual
   corruption level of the input. Added `backend/noise_schedule.py`, a
   standard DDPM variance-preserving schedule (`x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`,
   Ho et al. 2020), and wired it into `train_lssn.py`.
2. **Invariance loss regression.** `train_lssn.py` had drifted from
   `loss.py`'s documented hybrid L2 + cosine objective to a bare MSE-only
   reimplementation, silently dropping the directional-alignment term.
   Restored the import and use of `InvarianceLoss`.
3. **Gated fusion biased the invariance loss.** The dual-path branches were
   computed by zeroing one modality and running the *fused* (gated) forward
   pass. Since the zeroed modality's attention output is exactly zero
   (bias-free projections), this made the "text-only" branch always equal
   `α·out_text` and the "image-only" branch always `(1−α)·out_image` — a
   magnitude gap manufactured by the shared gate, not by genuine
   representational divergence, which corrupted the very loss meant to
   measure that divergence. `SynchronizationModule` now supports
   `mode="text_only"`/`"image_only"` that bypass the gate for these branches.

## Repository structure

```
backend/
  lssn_model.py        LSSN_UNet: ResNet + SpatialTransformer UNet backbone
  lssn_modules.py       SynchronizationModule, dual-modality cross-attention
  loss.py               InvarianceLoss (L2 + cosine)
  noise_schedule.py     DDPM forward-diffusion schedule
  train_lssn.py         LSSNTrainer: dual-path training loop
  story_engine.py        Story/scene generation via local LLM (Ollama)
  lssn_service.py        Image + video generation service
  main.py                FastAPI application
frontend/                React + Vite UI
```

## Status

The UNet, synchronization module, invariance loss, and diffusion schedule are
implemented and GPU-verified end-to-end. Beyond that, a full real (non-dummy)
training + inference pipeline now exists:

- `encoders.py` — frozen CLIP ViT-L/14 (openai/clip-vit-large-patch14) text
  and image conditioning, matching SD1.x's exact 768-dim/77-token convention.
- `vae_utils.py` — frozen pretrained VAE (stabilityai/sd-vae-ft-mse) for
  image<->latent encode/decode, per Latent Diffusion Models (Rombach et al. 2022).
- `sampler.py` — DDIM reverse sampling (Song et al. 2021) with
  classifier-free guidance (Ho & Salimans 2022) — previously nothing turned a
  trained model's predictions back into an image.
- `prepare_dataset.py` — materializes a real (prompt, image) training set
  from DiffusionDB (Wang et al. 2023), not synthetic placeholders.
- `train_real.py` — the actual training loop over that real data, with EMA
  weight averaging and CFG conditioning dropout. `train_lssn.py`'s dummy loop
  is unchanged and still serves as the fast architecture smoke test.
- `sample_lssn.py` — standalone text(+optional reference image) inference
  from a trained checkpoint.

A checkpoint has been trained on a small proof-of-concept run (540 real
image/prompt pairs, 3 epochs, `model_channels=128` instead of the
architecture's 320-channel default — a deliberate size/VRAM tradeoff for a
single consumer GPU). `/generate/visualize` now uses that checkpoint via
`lssn_service.py` when one exists at `backend/checkpoints/lssn_latest.pt`,
falling back to the external Pollinations API otherwise.

**On output quality**: at this data/compute scale (540 images, ~800
gradient steps, no dataloader augmentation), samples are unstructured
color/texture fields, not coherent images — this is expected, not a bug.
Production-grade text-to-image quality (comparable to Stable Diffusion/SDXL)
requires orders of magnitude more paired data and compute than a single
consumer GPU session provides; see Roadmap below for what scaling this up
would take.

## Running locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Train (dummy loop, validates the training step end-to-end)**
```bash
cd backend
python train_lssn.py
```

**Train (real data, real encoders/VAE, produces a usable checkpoint)**
```bash
cd backend
python prepare_dataset.py   # one-time: materializes backend/data/{train,val}
python train_real.py        # writes backend/checkpoints/lssn_latest.pt
python sample_lssn.py "a castle on a floating island" --out sample.png
```

`start_app.bat` launches both backend and frontend on Windows.

## Roadmap

- Scale up training data: thousands-to-millions of real paired examples
  instead of the current 540-image proof of concept (e.g. a larger
  DiffusionDB slice, LAION-derived subsets, or a licensed dataset).
- Scale up model capacity back toward `model_channels=320` (or beyond) once
  training infrastructure isn't a single 8GB GPU — likely via gradient
  checkpointing + multi-GPU or cloud training, or parameter-efficient
  fine-tuning (LoRA) on top of a larger pretrained base.
- More training steps with a proper LR schedule/warmup, data augmentation,
  and a validation-loss-based early-stopping signal (data/prepare_dataset.py
  already reserves a val split that train_real.py doesn't yet use).
- Ablate λ (invariance loss weight) and report FID/CLIP-score trade-offs
  vs. a single-modality-conditioned baseline.
- Explore per-modality CFG dropout (drop text/image independently rather
  than jointly) for finer guidance control at inference.

## License

MIT — see `LICENSE`.
