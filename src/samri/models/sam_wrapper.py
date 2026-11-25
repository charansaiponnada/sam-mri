import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from samri.common import REGIONS

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def build_sam(kind, ckpt=None):
    from segment_anything import sam_model_registry
    from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

    if kind == "test":  # tiny random SAM for smoke tests only
        enc = ImageEncoderViT(depth=2, embed_dim=64, img_size=1024, mlp_ratio=4,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=2, patch_size=16,
                              qkv_bias=True, use_rel_pos=True, global_attn_indexes=[1], window_size=14,
                              out_chans=256)
        return Sam(
            image_encoder=enc,
            prompt_encoder=PromptEncoder(embed_dim=256, image_embedding_size=(64, 64),
                                         input_image_size=(1024, 1024), mask_in_chans=16),
            mask_decoder=MaskDecoder(num_multimask_outputs=3,
                                     transformer=TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
                                     transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256),
            pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])
    if ckpt is not None and not Path(ckpt).exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {ckpt}\n"
                                f"Download e.g. {SAM_URLS.get(kind, SAM_URLS['vit_b'])} "
                                f"(MedSAM weights: see the MedSAM GitHub README)")
    return sam_model_registry[kind](checkpoint=ckpt)


# ---------------------------------------------------------------- LoRA
class LoRAQKV(nn.Module):
    """LoRA on the q and v projections of a fused qkv Linear (as in SAMed)."""

    def __init__(self, qkv, rank, alpha):
        super().__init__()
        self.qkv, d, self.scale = qkv, qkv.in_features, alpha / rank
        self.aq, self.bq = nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)
        self.av, self.bv = nn.Linear(d, rank, bias=False), nn.Linear(rank, d, bias=False)
        for a in (self.aq, self.av):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (self.bq, self.bv):
            nn.init.zeros_(b.weight)

    def forward(self, x):
        out, d = self.qkv(x), self.qkv.in_features
        dq = self.bq(self.aq(x)) * self.scale
        dv = self.bv(self.av(x)) * self.scale
        return torch.cat([out[..., :d] + dq, out[..., d:2 * d], out[..., 2 * d:] + dv], dim=-1)


def add_lora(image_encoder, rank, alpha):
    for blk in image_encoder.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, rank, alpha)


def enable_grad_checkpoint(image_encoder):
    for blk in image_encoder.blocks:
        orig = blk.forward
        blk.forward = lambda x, _o=orig: checkpoint(_o, x, use_reentrant=False) if torch.is_grad_enabled() else _o(x)


# ---------------------------------------------------------------- multi-sequence input handling
class ChannelStem(nn.Module):
    """C_in -> 3 channels. Starts as a pass-through of `init_idx` channels (= the naive RGB baseline),
    plus a zero-initialised 3x3 residual, so training can only improve on the naive mapping."""

    def __init__(self, c_in, init_idx):
        super().__init__()
        self.pw = nn.Conv2d(c_in, 3, 1)
        self.res = nn.Conv2d(c_in, 3, 3, padding=1)
        with torch.no_grad():
            self.pw.weight.zero_()
            self.pw.bias.zero_()
            self.res.weight.zero_()
            self.res.bias.zero_()
            for o, i in enumerate(init_idx):
                self.pw.weight[o, i, 0, 0] = 1.0

    def forward(self, x):
        return self.pw(x) + self.res(x)


class FusionHead(nn.Module):
    """Fuse per-sequence image embeddings (B, S, 256, h, w) -> (B, 256, h, w). Starts as the mean."""

    def __init__(self, n_groups, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(n_groups * dim, dim, 1)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for s in range(n_groups):
                for i in range(dim):
                    self.proj.weight[i, s * dim + i, 0, 0] = 1.0 / n_groups

    def forward(self, e):
        b, s, c, h, w = e.shape
        return self.proj(e.reshape(b, s * c, h, w))


def decode_one_to_one(decoder, emb, image_pe, sparse, dense):
    """segment-anything's MaskDecoder repeats the image embedding once per prompt, which assumes ONE image
    with many prompts. For batches where image i is paired with prompt i (training, eval) that yields B*B
    items and breaks, so this re-implements MaskDecoder.predict_masks (single-mask output) one-to-one."""
    out_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
    out_tokens = out_tokens.unsqueeze(0).expand(sparse.size(0), -1, -1)
    tokens = torch.cat((out_tokens, sparse), dim=1)
    src = emb + dense
    pos = image_pe.expand(src.shape[0], -1, -1, -1)
    b, c, h, w = src.shape
    hs, src = decoder.transformer(src, pos, tokens)
    mask_tokens = hs[:, 1:1 + decoder.num_mask_tokens, :]
    src = src.transpose(1, 2).view(b, c, h, w)
    up = decoder.output_upscaling(src)
    hyper = torch.stack([decoder.output_hypernetworks_mlps[i](mask_tokens[:, i, :])
                         for i in range(decoder.num_mask_tokens)], dim=1)
    b, c, h, w = up.shape
    masks = (hyper @ up.view(b, c, h * w)).view(b, -1, h, w)
    return masks[:, :1]


def to_sam_input(x_uint8):
    """(B, C, S, S) uint8 -> float [0,1] at 1024 x 1024."""
    x = x_uint8.float() / 255.0
    if x.shape[-1] != 1024:
        x = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
    return x


class SamMRI(nn.Module):
    def __init__(self, sam, adapter_kind, n_groups, group_size, input_norm, stem_init, region_cond=False):
        super().__init__()
        self.sam, self.kind, self.input_norm = sam, adapter_kind, input_norm
        self.n_groups, self.k = n_groups, group_size
        self.region_cond = region_cond
        c_in = n_groups * group_size
        self.adapter = self.fusion = self.region_emb = None
        if region_cond:
            # Zero-initialized so decoding starts out identical to the unconditioned model;
            # training can only add region-specific information.
            self.region_emb = nn.Embedding(len(REGIONS), 256)
            nn.init.zeros_(self.region_emb.weight)
        if adapter_kind == "identity":
            if c_in != 3:
                raise ValueError(f"adapter=identity needs exactly 3 input channels, got {c_in}")
        elif adapter_kind == "stem":
            self.adapter = ChannelStem(c_in, stem_init)
        elif adapter_kind == "fusion":
            if group_size not in (1, 3):
                raise ValueError("fusion needs context 0 (k=1) or 1 (k=3)")
            self.fusion = FusionHead(n_groups)
        else:
            raise ValueError(adapter_kind)

    def _norm(self, x):
        if self.input_norm == "sam":
            return (x * 255.0 - self.sam.pixel_mean) / self.sam.pixel_std
        return x  # 'unit': MedSAM feeds [0,1] images directly

    def encode(self, x):
        """x: (B, C, 1024, 1024) float in [0,1] -> (B, 256, 64, 64)."""
        if self.kind == "fusion":
            b, c, h, w = x.shape
            x = x.reshape(b * self.n_groups, self.k, h, w)
            if self.k == 1:
                x = x.expand(-1, 3, -1, -1)
            emb = self.sam.image_encoder(self._norm(x))
            return self.fusion(emb.reshape(b, self.n_groups, *emb.shape[1:]))
        if self.adapter is not None:
            x = self.adapter(x)
        return self.sam.image_encoder(self._norm(x))

    def decode(self, emb, box=None, point=None, region=None):
        """box: (B,4) xyxy in 1024-frame px; point: (B,2) xy. emb[i] is paired with prompt i.
        region: (B,) long region index (WT/TC/ET) used to condition the prompt tokens.
        Returns low-res logits (B,1,256,256)."""
        pe = self.sam.prompt_encoder
        if box is not None:
            sparse, dense = pe(points=None, boxes=box, masks=None)
        else:
            labels = torch.ones(point.shape[0], 1, dtype=torch.int, device=point.device)
            sparse, dense = pe(points=(point[:, None, :], labels), boxes=None, masks=None)
        if region is not None and self.region_cond:
            sparse = sparse + self.region_emb(region.to(sparse.device))[:, None, :]
        return decode_one_to_one(self.sam.mask_decoder, emb, pe.get_dense_pe(), sparse, dense)

    def forward(self, x, box=None, point=None, region=None):
        return self.decode(self.encode(x), box, point, region)


def build_model(cfg):
    m, d = cfg["model"], cfg["data"]
    sam = build_sam(m["sam_type"], m.get("sam_checkpoint"))
    for p in sam.parameters():
        p.requires_grad = False
    n_groups, k = len(d["sequences"]), 2 * d.get("context", 0) + 1
    names = m.get("stem_init") or (["t1ce", "t2", "flair"] if all(s in d["sequences"] for s in ["t1ce", "t2", "flair"])
                                    else d["sequences"][:3])
    stem_init = [d["sequences"].index(n) * k + k // 2 for n in names]
    model = SamMRI(sam, m.get("adapter", "identity"), n_groups, k, m.get("input_norm", "sam"), stem_init,
                   m.get("region_cond", False))
    if m.get("lora_rank", 0) > 0:
        add_lora(sam.image_encoder, m["lora_rank"], m.get("lora_alpha", m["lora_rank"]))
    if m.get("train_decoder", True):
        for p in sam.mask_decoder.parameters():
            p.requires_grad = True
    if m.get("grad_checkpoint", False):
        enable_grad_checkpoint(sam.image_encoder)
    return model


def trainable_state_dict(model):
    return {k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"checkpoint has keys the model lacks (config mismatch?): {res.unexpected_keys[:5]}")
    model_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    if model_keys - set(state):
        raise RuntimeError(f"checkpoint is missing trainable keys: {sorted(model_keys - set(state))[:5]}")
