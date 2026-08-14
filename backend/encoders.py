import torch
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor


class CLIPConditioner:
    """
    Frozen CLIP ViT-L/14 (openai/clip-vit-large-patch14) conditioning encoder.

    This is the exact text tower Stable Diffusion 1.x uses (768-dim hidden
    states, 77-token sequence length), which is why LSSN_UNet's default
    context_dim=768 and the (B, 77, context_dim) shape already assumed
    throughout lssn_model.py / lssn_modules.py / train_lssn.py line up with
    it directly -- no architecture changes needed to consume real embeddings.

    Text conditioning: last_hidden_state, (B, 77, 768) -- per-token features,
    the same representation SD's cross-attention conditions on.

    Image conditioning: the projected pooled image embedding via
    get_image_features(), (B, 768) -- CLIP's joint text/image embedding
    space for ViT-L/14 is 768-dim, matching lssn_service.py's existing
    `dummy_c_image = torch.randn(1, 1, 768)` convention (a single conditioning
    token per reference image). This mirrors how IP-Adapter (Ye et al. 2023)
    conditions diffusion UNets on a CLIP image embedding of a reference image
    alongside text, which is the closest published analog to LSSN's
    dual text/image SynchronizationModule.

    Both towers are frozen (no gradient, eval mode) -- LSSN trains only the
    UNet, consistent with Latent Diffusion Models (Rombach et al. 2022),
    which also keep the conditioning encoder frozen.
    """

    MODEL_ID = "openai/clip-vit-large-patch14"

    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = CLIPTokenizer.from_pretrained(self.MODEL_ID)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID, dtype=dtype).to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.context_dim = self.model.config.projection_dim  # 768 for ViT-L/14

    @torch.no_grad()
    def encode_text(self, texts):
        """texts: list[str] -> (B, 77, context_dim)"""
        tokens = self.tokenizer(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.text_model(**tokens)
        return out.last_hidden_state.to(self.dtype)

    @torch.no_grad()
    def encode_image(self, images):
        """images: list[PIL.Image] -> (B, 1, context_dim)"""
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device, self.dtype)
        embeds = self.model.get_image_features(pixel_values=pixel_values)
        return embeds.unsqueeze(1).to(self.dtype)

    @torch.no_grad()
    def null_text(self, batch_size):
        """Unconditional text embedding (empty-string prompt), for CFG."""
        return self.encode_text([""] * batch_size)

    @torch.no_grad()
    def null_image(self, batch_size):
        """Unconditional image embedding (zero vector), for CFG."""
        return torch.zeros(batch_size, 1, self.context_dim, device=self.device, dtype=self.dtype)
