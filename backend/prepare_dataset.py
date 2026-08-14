"""
Materializes a small real (prompt, image) training set to backend/data/{train,val}.

Source: svjack/diffusiondb_random_10k, a parquet re-upload of a random sample
of poloclub/diffusiondb (Wang et al. 2023, arXiv:2210.14896) -- real user
prompts paired with real Stable Diffusion outputs, CC0-1.0-licensed at the
source. The parquet mirror is used (instead of loading poloclub/diffusiondb
directly) because that dataset ships as a loading *script*, which the
installed `datasets` 4.x no longer executes; the mirror is data-only parquet.

Streaming mode pulls only as many shards as needed for N_EXAMPLES rows,
so this does not download the full 5.9GB dataset -- with N_EXAMPLES=600
it needs exactly one of the 13 parquet shards (~450MB).
"""
import os
import io
import logging
from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="prepare-data [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_EXAMPLES = 600
VAL_FRACTION = 0.1
IMAGE_SIZE = 256
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    train_dir = os.path.join(OUT_DIR, "train")
    val_dir = os.path.join(OUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    logger.info("Streaming svjack/diffusiondb_random_10k (only as many shards as needed)...")
    ds = load_dataset("svjack/diffusiondb_random_10k", split="train", streaming=True)

    n_val = int(N_EXAMPLES * VAL_FRACTION)
    n_train = N_EXAMPLES - n_val
    count = 0
    skipped = 0

    for example in ds:
        if count >= N_EXAMPLES:
            break

        img = example.get("image")
        prompt = example.get("prompt") or example.get("Prompt") or example.get("text")

        if img is None or not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            skipped += 1
            continue

        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else None
            if img is None:
                skipped += 1
                continue

        img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        split_dir = train_dir if count < n_train else val_dir
        idx = count if count < n_train else count - n_train

        img.save(os.path.join(split_dir, f"{idx:05d}.png"))
        with open(os.path.join(split_dir, f"{idx:05d}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt.strip())

        count += 1
        if count % 100 == 0:
            logger.info(f"{count}/{N_EXAMPLES} pairs written (skipped {skipped} malformed rows)")

    logger.info(f"Done. {n_train} train pairs in {train_dir}, {n_val} val pairs in {val_dir}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
