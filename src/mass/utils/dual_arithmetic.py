"""
dual_arithmetic.py – Duality-map merging for mass-language-evaluation.

Provides the Modula atomic modules, mass schedules, architecture graphs
(ViT variants + FLAN-T5-base), key filtering utilities, topological
ordering helpers, and the main build_duality_map() entry point.
"""

import re
from math import sqrt

import torch
from modula.abstract import *
from modula.bond import *


# ─────────────────────────────────────────────────────────────────────────────
# SVD orthogonalisation helper
# ─────────────────────────────────────────────────────────────────────────────

def svd_orthogonalize(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


# ─────────────────────────────────────────────────────────────────────────────
# Atomic modules
# ─────────────────────────────────────────────────────────────────────────────

class LinearSVD(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        return x @ w[0].T

    def initialize(self, key=None):
        return None

    def project(self, w):
        weight = svd_orthogonalize(w[0]) * sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        expected = (self.fanout, self.fanin)
        if grad.shape != expected:
            raise ValueError(
                f"dualize shape mismatch: expected {expected}, got {grad.shape}"
            )
        scalar = sqrt(self.fanout / self.fanin) * target_norm
        return [svd_orthogonalize(grad) * scalar]


class Conv2DSVD(Atom):
    def __init__(self, fanout, fanin, kernel_size):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.kernel_size = kernel_size
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w[0], padding="same")

    def initialize(self, key=None):
        weight = torch.randn(self.fanout, self.fanin, self.kernel_size, self.kernel_size)
        scale = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin)
        return [self._ortho_spatial(weight) * scale]

    def project(self, w):
        scale = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin)
        return [self._ortho_spatial(w[0]) * scale]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        expected = (self.fanout, self.fanin, self.kernel_size, self.kernel_size)
        if grad.shape != expected:
            raise ValueError(
                f"dualize shape mismatch: expected {expected}, got {grad.shape}"
            )
        scalar = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin) * target_norm
        return [self._ortho_spatial(grad) * scalar]

    def _ortho_spatial(self, weight):
        k = self.kernel_size
        result = torch.zeros_like(weight)
        for i in range(k):
            for j in range(k):
                result[:, :, i, j] = svd_orthogonalize(weight[:, :, i, j])
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Mass schedules
# ─────────────────────────────────────────────────────────────────────────────

def uniform_mass_schedule(current_l, tot_layer):
    return 0.5


def linear_mass_schedule(current_l, tot_layer):
    return 0.01 + current_l / tot_layer * (0.5 - 0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Architecture graphs – Vision Transformers
# ─────────────────────────────────────────────────────────────────────────────

def ViT_B_16(num_classes=512, num_blocks=12, d_embed=768, patch_size=16,
              input_channels=3, mass_schedule="uniform"):
    mlp_width = 4 * d_embed
    tot_layers = 4 * num_blocks + 3
    ms = uniform_mass_schedule if mass_schedule == "uniform" else linear_mass_schedule

    conv1 = Conv2DSVD(fanin=input_channels, fanout=d_embed, kernel_size=patch_size)
    conv1.mass = ms(0, tot_layers)

    visual_pos_embed = LinearSVD(197, d_embed)
    visual_pos_embed.mass = ms(1, tot_layers)

    transformer = None
    for b in range(num_blocks):
        a1 = LinearSVD(3 * d_embed, d_embed);  a1.mass = ms(b * 4 + 2, tot_layers)
        a2 = LinearSVD(d_embed, d_embed);       a2.mass = ms(b * 4 + 3, tot_layers)
        m1 = LinearSVD(mlp_width, d_embed);     m1.mass = ms(b * 4 + 4, tot_layers)
        m2 = LinearSVD(d_embed, mlp_width);     m2.mass = ms(b * 4 + 5, tot_layers)
        att = a2 @ a1
        mlp = m2 @ m1
        transformer = (mlp @ att) @ transformer if transformer else (mlp @ att)

    proj = LinearSVD(d_embed, num_classes)
    proj.mass = ms(tot_layers, tot_layers)
    return proj @ transformer @ visual_pos_embed @ conv1


def ViT_B_32(num_classes=512, num_blocks=12, d_embed=768, patch_size=32,
              input_channels=3, mass_schedule="uniform"):
    mlp_width = 4 * d_embed
    tot_layers = 4 * num_blocks + 3
    ms = uniform_mass_schedule if mass_schedule == "uniform" else linear_mass_schedule

    conv1 = Conv2DSVD(fanin=input_channels, fanout=d_embed, kernel_size=patch_size)
    conv1.mass = ms(0, tot_layers)

    visual_pos_embed = LinearSVD(50, d_embed)
    visual_pos_embed.mass = ms(1, tot_layers)

    transformer = None
    for b in range(num_blocks):
        a1 = LinearSVD(3 * d_embed, d_embed);  a1.mass = ms(b * 4 + 2, tot_layers)
        a2 = LinearSVD(d_embed, d_embed);       a2.mass = ms(b * 4 + 3, tot_layers)
        m1 = LinearSVD(mlp_width, d_embed);     m1.mass = ms(b * 4 + 4, tot_layers)
        m2 = LinearSVD(d_embed, mlp_width);     m2.mass = ms(b * 4 + 5, tot_layers)
        att = a2 @ a1
        mlp = m2 @ m1
        transformer = (mlp @ att) @ transformer if transformer else (mlp @ att)

    proj = LinearSVD(d_embed, num_classes)
    proj.mass = ms(tot_layers, tot_layers)
    return proj @ transformer @ visual_pos_embed @ conv1


def ViT_L_14(num_classes=768, num_blocks=24, d_embed=1024, patch_size=14,
              input_channels=3, mass_schedule="uniform"):
    mlp_width = 4 * d_embed
    tot_layers = 4 * num_blocks + 3
    ms = uniform_mass_schedule if mass_schedule == "uniform" else linear_mass_schedule

    conv1 = Conv2DSVD(fanin=input_channels, fanout=d_embed, kernel_size=patch_size)
    conv1.mass = ms(0, tot_layers)

    visual_pos_embed = LinearSVD(257, d_embed)
    visual_pos_embed.mass = ms(1, tot_layers)

    transformer = None
    for b in range(num_blocks):
        a1 = LinearSVD(3 * d_embed, d_embed);  a1.mass = ms(b * 4 + 2, tot_layers)
        a2 = LinearSVD(d_embed, d_embed);       a2.mass = ms(b * 4 + 3, tot_layers)
        m1 = LinearSVD(mlp_width, d_embed);     m1.mass = ms(b * 4 + 4, tot_layers)
        m2 = LinearSVD(d_embed, mlp_width);     m2.mass = ms(b * 4 + 5, tot_layers)
        att = a2 @ a1
        mlp = m2 @ m1
        transformer = (mlp @ att) @ transformer if transformer else (mlp @ att)

    proj = LinearSVD(d_embed, num_classes)
    proj.mass = ms(tot_layers, tot_layers)
    return proj @ transformer @ visual_pos_embed @ conv1


# ─────────────────────────────────────────────────────────────────────────────
# Architecture graph – FLAN-T5-base
# ─────────────────────────────────────────────────────────────────────────────

def FlanT5Base(
    d_model=768,
    d_ff=2048,
    inner_dim=768,         # num_heads * d_kv = 12 * 64
    num_encoder_layers=12,
    num_decoder_layers=12,
    mass_schedule="uniform",
):
    """
    Modula composition graph for FLAN-T5-base.

    Execution order (matches get_t5_topological_order):
      encoder blocks 0-11  : q → k → v → o → wi_0 → wi_1 → wo   (7 atoms each)
      decoder blocks 0-11  : sq → sk → sv → so                    (self-att)
                             → cq → ck → cv → co                  (cross-att)
                             → wi_0 → wi_1 → wo                   (FFN)
                                                                   (11 atoms each)
    Total atoms: 12×7 + 12×11 = 216
    """
    ms = uniform_mass_schedule if mass_schedule == "uniform" else linear_mass_schedule
    tot_layers = num_encoder_layers * 7 + num_decoder_layers * 11
    layer_idx = 0

    # ── Encoder ──────────────────────────────────────────────────────────────
    encoder = None
    for _ in range(num_encoder_layers):
        q    = LinearSVD(inner_dim, d_model);  q.mass    = ms(layer_idx, tot_layers); layer_idx += 1
        k    = LinearSVD(inner_dim, d_model);  k.mass    = ms(layer_idx, tot_layers); layer_idx += 1
        v    = LinearSVD(inner_dim, d_model);  v.mass    = ms(layer_idx, tot_layers); layer_idx += 1
        o    = LinearSVD(d_model, inner_dim);  o.mass    = ms(layer_idx, tot_layers); layer_idx += 1
        wi_0 = LinearSVD(d_ff, d_model);       wi_0.mass = ms(layer_idx, tot_layers); layer_idx += 1
        wi_1 = LinearSVD(d_ff, d_model);       wi_1.mass = ms(layer_idx, tot_layers); layer_idx += 1
        wo   = LinearSVD(d_model, d_ff);       wo.mass   = ms(layer_idx, tot_layers); layer_idx += 1

        att   = o @ v @ k @ q
        ffn   = wo @ wi_1 @ wi_0
        block = ffn @ att

        encoder = block @ encoder if encoder is not None else block

    # ── Decoder ──────────────────────────────────────────────────────────────
    decoder = None
    for _ in range(num_decoder_layers):
        sq   = LinearSVD(inner_dim, d_model);  sq.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        sk   = LinearSVD(inner_dim, d_model);  sk.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        sv   = LinearSVD(inner_dim, d_model);  sv.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        so   = LinearSVD(d_model, inner_dim);  so.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        cq   = LinearSVD(inner_dim, d_model);  cq.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        ck   = LinearSVD(inner_dim, d_model);  ck.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        cv   = LinearSVD(inner_dim, d_model);  cv.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        co   = LinearSVD(d_model, inner_dim);  co.mass   = ms(layer_idx, tot_layers); layer_idx += 1
        wi_0 = LinearSVD(d_ff, d_model);       wi_0.mass = ms(layer_idx, tot_layers); layer_idx += 1
        wi_1 = LinearSVD(d_ff, d_model);       wi_1.mass = ms(layer_idx, tot_layers); layer_idx += 1
        wo   = LinearSVD(d_model, d_ff);       wo.mass   = ms(layer_idx, tot_layers); layer_idx += 1

        self_att  = so @ sv @ sk @ sq
        cross_att = co @ cv @ ck @ cq
        ffn       = wo @ wi_1 @ wi_0
        block     = ffn @ cross_att @ self_att

        decoder = block @ decoder if decoder is not None else block

    return decoder @ encoder


# ─────────────────────────────────────────────────────────────────────────────
# T5 key utilities
# ─────────────────────────────────────────────────────────────────────────────

def _is_t5_matrix_key(name: str) -> bool:
    """
    Returns True for the 216 weight matrices to dualize in FLAN-T5-base.
    Skips biases, layer norms, embeddings, relative attention biases, lm_head.
    """
    _SKIP = ("bias", "layer_norm", "relative_attention_bias",
             "embed_tokens", "lm_head", "shared")
    if any(s in name for s in _SKIP):
        return False
    if not name.endswith(".weight"):
        return False

    if re.search(r"encoder\.block\.\d+\.layer\.0\.SelfAttention\.[qkvo]\.weight", name):
        return True
    if re.search(r"encoder\.block\.\d+\.layer\.1\.DenseReluDense\.(wi_0|wi_1|wo)\.weight", name):
        return True
    if re.search(r"decoder\.block\.\d+\.layer\.0\.SelfAttention\.[qkvo]\.weight", name):
        return True
    if re.search(r"decoder\.block\.\d+\.layer\.1\.EncDecAttention\.[qkvo]\.weight", name):
        return True
    if re.search(r"decoder\.block\.\d+\.layer\.2\.DenseReluDense\.(wi_0|wi_1|wo)\.weight", name):
        return True

    return False


def get_t5_topological_order(keys):
    """
    Sorts FLAN-T5 state-dict keys in execution order:
      encoder blocks 0-11  (q, k, v, o, wi_0, wi_1, wo per block)
      decoder blocks 0-11  (self q/k/v/o, cross q/k/v/o, wi_0, wi_1, wo per block)
    """
    _ENC_ATT  = {"q": 0, "k": 1, "v": 2, "o": 3}
    _ENC_FFN  = {"wi_0": 4, "wi_1": 5, "wo": 6}
    _DEC_SELF  = {"q": 0, "k": 1, "v": 2, "o": 3}
    _DEC_CROSS = {"q": 4, "k": 5, "v": 6, "o": 7}
    _DEC_FFN   = {"wi_0": 8, "wi_1": 9, "wo": 10}

    def sort_key(name):
        m = re.search(r"encoder\.block\.(\d+)", name)
        if m:
            block = int(m.group(1))
            for param, sub in _ENC_ATT.items():
                if f"SelfAttention.{param}.weight" in name:
                    return (0, block, sub)
            for param, sub in _ENC_FFN.items():
                if f"DenseReluDense.{param}.weight" in name:
                    return (0, block, sub)
            return (0, block, 99)

        m = re.search(r"decoder\.block\.(\d+)", name)
        if m:
            block = int(m.group(1))
            if "layer.0.SelfAttention" in name:
                for param, sub in _DEC_SELF.items():
                    if f"SelfAttention.{param}.weight" in name:
                        return (1, block, sub)
            if "EncDecAttention" in name:
                for param, sub in _DEC_CROSS.items():
                    if f"EncDecAttention.{param}.weight" in name:
                        return (1, block, sub)
            for param, sub in _DEC_FFN.items():
                if f"DenseReluDense.{param}.weight" in name:
                    return (1, block, sub)
            return (1, block, 99)

        return (2, 0, 0)

    return sorted(keys, key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# ViT key utilities (kept for completeness)
# ─────────────────────────────────────────────────────────────────────────────

def get_vit_topological_order(keys):
    """Sorts CLIP ViT keys in topological order."""
    def sort_key(k):
        if "visual.conv1" in k:
            return (0, 0, 0)
        if "positional_embedding" in k and "vision" in k:
            return (1, 0, 0)
        if "class_embedding" in k:
            return (1, 1, 0)
        if "resblocks" in k:
            match = re.search(r"resblocks\.(\d+)", k)
            block_idx = int(match.group(1)) if match else 999
            sub = 0
            if "attn.in_proj" in k:   sub = 1
            elif "attn.out_proj" in k: sub = 2
            elif "ln_1" in k:          sub = 3
            elif "mlp.c_fc" in k:      sub = 4
            elif "mlp.c_proj" in k:    sub = 5
            elif "ln_2" in k:          sub = 6
            return (2, block_idx, sub)
        if "ln_post" in k:
            return (3, 0, 0)
        if "visual.proj" in k:
            return (4, 0, 0)
        return (5, 0, 0)

    return sorted(keys, key=sort_key)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_duality_map(layer_names, grads, device, mass_schedule, model_name):
    """
    Build the modular duality map and apply it to the task-vector gradients.

    Args:
        layer_names: Keys in topological order (already sorted by the caller).
        grads:       Dict {layer_name -> tensor} of averaged/TSV task vectors.
        device:      torch.device to move tensors to during dualisation.
        mass_schedule: "uniform" or "linear".
        model_name:  Used to pick the right architecture graph.

    Returns:
        Dict {layer_name -> dualized tensor} for all dualised keys.
    """
    # ── Select architecture graph ─────────────────────────────────────────────
    m = None
    if "t5" in model_name.lower():
        m = FlanT5Base(mass_schedule=mass_schedule)
    elif "B-16" in model_name:
        m = ViT_B_16(mass_schedule=mass_schedule)
    elif "B-32" in model_name:
        m = ViT_B_32(mass_schedule=mass_schedule)
    elif "L-14" in model_name:
        m = ViT_L_14(mass_schedule=mass_schedule)
    else:
        raise ValueError(f"No matching duality map for model_name='{model_name}'")

    # ── Collect keys and gradients to dualise ────────────────────────────────
    to_consider_name = []
    to_consider_grad = []

    if "t5" in model_name.lower():
        for name in layer_names:
            if _is_t5_matrix_key(name):
                to_consider_name.append(name)
                to_consider_grad.append(grads[name].to(device))
            else:
                print(f"  skipped: {name}")
    else:
        _VIT_SKIP = {"bias", "ln_", "class_embedding", "logit_scale"}
        for name in layer_names:
            if any(s in name for s in _VIT_SKIP):
                continue
            if (
                "visual.conv1.weight" in name
                or ("visual.proj" in name and "out_proj" not in name)
                or "visual.positional_embedding" in name
                or (
                    "visual.transformer.resblocks" in name
                    and "weight" in name
                    and any(
                        p in name
                        for p in (
                            "attn.in_proj_weight",
                            "attn.out_proj.weight",
                            "mlp.c_fc.weight",
                            "mlp.c_proj.weight",
                        )
                    )
                )
            ):
                to_consider_name.append(name)
                to_consider_grad.append(grads[name].to(device))
            else:
                print(f"  skipped: {name}")

    print(
        f"build_duality_map: atoms={m.atoms}, mass={m.mass}, "
        f"to_dualize={len(to_consider_grad)}"
    )

    dualized = m.dualize(to_consider_grad)
    return dict(zip(to_consider_name, dualized))
