import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SynchronizationModule(nn.Module):
    """
    Synchronization Module (SM) for LSSN.
    Integrates information from both text and image modalities into the latent features.
    It performs symmetric cross-attention where the latent features attend to both
    text and image conditioning inputs.
    """
    def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0, context_dim=768):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_text = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_text = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_image = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_image = nn.Linear(context_dim, inner_dim, bias=False)

        # Learnable gating parameter for Gated Fusion
        self.gate = nn.Parameter(torch.tensor([0.0]))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def _attend(self, q, k, v, h):
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        return attn @ v

    def forward(self, x, c_text, c_image, mode="fused"):
        """
        x: Latent features (Batch, Sequence/Pixels, Dim)
        c_text: Text conditioning (Batch, Seq_len, Context_Dim)
        c_image: Image conditioning (Batch, Seq_len, Context_Dim)
        mode:
            "fused" (default)   -> gated combination of both branches, alpha*text + (1-alpha)*image.
                                    Used for the primary denoising forward pass.
            "text_only"         -> raw (un-gated) attention over c_text only.
            "image_only"        -> raw (un-gated) attention over c_image only.

        BUG FIX: previously the dual-path trainer computed the "text-only" and
        "image-only" branches by calling this module in "fused" mode with the
        other modality zeroed out. Because to_k_*/to_v_* are bias-free Linear
        layers, a zeroed modality produces an exact-zero attention output for
        that branch, so the result was always alpha*out_text or
        (1-alpha)*out_image -- scaled by the *same shared* learnable gate.
        Whenever alpha != 0.5 this manufactures a systematic magnitude gap
        between the two branches that has nothing to do with genuine
        text/image representational divergence, and the Invariance Loss (an
        L2 + cosine distance between these branches) was penalizing that
        artifact instead of real modality bias. "text_only"/"image_only"
        modes below bypass the gate entirely so L_inv operates on the raw,
        comparable representations.
        """
        h = self.num_heads
        q = self.to_q(x)
        q = q.view(q.shape[0], -1, h, q.shape[-1] // h).transpose(1, 2)

        def split(t):
            return t.view(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2)

        if mode in ("fused", "text_only"):
            k_text, v_text = split(self.to_k_text(c_text)), split(self.to_v_text(c_text))
            out_text = self._attend(q, k_text, v_text, h)

        if mode in ("fused", "image_only"):
            k_image, v_image = split(self.to_k_image(c_image)), split(self.to_v_image(c_image))
            out_image = self._attend(q, k_image, v_image, h)

        if mode == "fused":
            alpha = torch.sigmoid(self.gate)
            out = alpha * out_text + (1 - alpha) * out_image
        elif mode == "text_only":
            out = out_text
        elif mode == "image_only":
            out = out_image
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * h)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, dropout=0.0, context_dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SynchronizationModule(dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * 4, dropout=dropout)

    def forward(self, x, c_text, c_image, mode="fused"):
        x = x + self.attn(self.norm1(x), c_text, c_image, mode=mode)
        x = x + self.ff(self.norm2(x))
        return x
