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
import math

# ─────────────────────────────────────────────────────────────────────────────
# SVD orthogonalisation helper
# ─────────────────────────────────────────────────────────────────────────────

def svd_orthogonalize(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


# ─────────────────────────────────────────────────────────────────────────────
# Atomic modules
# ─────────────────────────────────────────────────────────────────────────────

class EmbedSVD(Atom):
    def __init__(self, d_embed, num_embed):
        super().__init__()
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [num_embed, d_embed]
        # PyTorch handles list/tensor indexing identically here
        return weights[x]

    def initialize(self, key=None):
        # Note: PyTorch manages randomness globally. 
        # If you want reproducibility like JAX's 'key', you can pass a torch.Generator to the 'generator' kwarg.
        # Otherwise, standard torch.randn is perfectly fine.
        weight = torch.randn(self.num_embed, self.d_embed, generator=key)
        
        # torch.linalg.vector_norm is the PyTorch equivalent of jnp.linalg.norm for specific axes
        norm = torch.linalg.vector_norm(weight, ord=2, dim=1, keepdim=True)
        weight = (weight / norm) * math.sqrt(self.d_embed)
        
        return [weight]

    def project(self, w):
        weight = w[0]
        
        norm = torch.linalg.vector_norm(weight, ord=2, dim=1, keepdim=True)
        weight = (weight / norm) * math.sqrt(self.d_embed)
        
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        
        norm = torch.linalg.vector_norm(grad, ord=2, dim=1, keepdim=True)
        d_weight = (grad / norm) * math.sqrt(self.d_embed) * target_norm
        
        # PyTorch has an identical nan_to_num function to handle division by zero
        d_weight = torch.nan_to_num(d_weight, nan=0.0)
        
        return [d_weight]
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

class CausalMaskTorch(Bond):
    """Masks the upper triangular part of the attention scores."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
    
    def forward(self, x, w):
        scores = x
        
        # Crucial: create the mask on the SAME device (CPU/GPU) as the input 'x'
        mask = torch.tril(torch.ones(scores.shape[-2:], dtype=torch.bool, device=x.device))
        
        # Apply the mask
        return torch.where(mask, scores, float('-inf'))

class SplitIntoHeadsTorch(Bond):
    """Reshapes an input to have heads."""
    def __init__(self, num_heads):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
        self.num_heads = num_heads
    
    def forward(self, x, w):
        B, T, D = x.shape
        
        # PyTorch uses .permute() to swap multiple dimensions
        return x.reshape(B, T, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

class MergeHeadsTorch(Bond):
    """Inverse of SplitIntoHeads."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1
    
    def forward(self, x, w):
        B, num_heads, T, head_dim = x.shape
        
        # .permute() to swap back, then reshape to flatten
        return x.permute(0, 2, 1, 3).reshape(B, T, num_heads * head_dim)
                                             
class AttentionQKTorch(Bond):
    """Computes the query and key matrix multiplication in attention."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # Still doing nothing here!
    
    def forward(self, x, w):
        q, k = x  # both shape [batch, n_heads, seq_len, d_query]
        
        # Note: Standard attention usually uses 1 / sqrt(d_query)
        # I left it as 1 / d_query to match your original code
        scale = 1 / q.shape[-1] 
        
        # PyTorch: .transpose() swaps exactly two dimensions. 
        # We swap the last two: seq_len (dim -2) and d_query (dim -1)
        scores = q @ k.transpose(-2, -1) * scale
        
        return scores  # shape [batch, n_heads, seq_len, seq_len]
class RopeTorch(Bond):
    """Rotates queries and keys by relative context window distance."""
    def __init__(self, d_head, base=10000):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # Still orthogonal, still doing nothing here!

        self.rope_dim = d_head // 2
        
        # PyTorch arange needs an explicit float conversion before division
        indices = torch.arange(self.rope_dim, dtype=torch.float32) / self.rope_dim
        self.inverse_frequencies = 1.0 / (base ** indices)
        
        self.seq_len_cached = None
        self.sin_cached = None
        self.cos_cached = None
    
    def get_cached(self, seq_len, device=None):
        # We also check if the device changed, so we don't accidentally cache CPU tensors 
        # on step 1 and try to use them on the GPU on step 2.
        if self.seq_len_cached != seq_len or (self.sin_cached is not None and self.sin_cached.device != device):
            self.seq_len_cached = seq_len
            
            # Ensure inverse_frequencies is on the right device
            if device is not None:
                self.inverse_frequencies = self.inverse_frequencies.to(device)
                
            distance = torch.arange(seq_len, dtype=torch.float32, device=self.inverse_frequencies.device)
            
            # torch.outer perfectly mimics jnp.outer
            freqs = torch.outer(distance, self.inverse_frequencies)  # shape [seq_len, rope_dim]
            
            # PyTorch doesn't have a multi-axis expand_dims. 
            # We use .unsqueeze(0) twice to add the batch and head dimensions.
            # Alternately, you could use None indexing: torch.cos(freqs)[None, None, :, :]
            self.cos_cached = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # shape [1, 1, seq_len, rope_dim]
            self.sin_cached = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # shape [1, 1, seq_len, rope_dim]
            
        return self.sin_cached, self.cos_cached
class ApplyAttentionScoresTorch(Bond):
    """Computes attention values from the scores."""
    def __init__(self):
        super().__init__()
        self.smooth = True
        self.sensitivity = 1  # Still boilerplate!
    
    def forward(self, x, w):
        v, scores = x
        
        # The @ operator perfectly maps to torch.matmul() under the hood
        return scores @ v
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
def Attention(num_heads, d_embed, d_query, d_value, softmax_scale, causal):
    """Multi-head attention"""
    Q = LinearSVD(num_heads * d_query, d_embed)
    K = LinearSVD(num_heads * d_query, d_embed)
    V = LinearSVD(num_heads * d_value, d_embed)
    W = LinearSVD(d_embed, num_heads * d_value) @ MergeHeadsTorch()
    vq_block = V+ Q
    att = vq_block+K
    att.sensitivity=1.0
    return W @ att
    
def FlanT5Base(
    d_model=768,
    d_ff=2048,
    inner_dim=768,         # num_heads * d_kv = 12 * 64
    num_encoder_layers=12,
    num_decoder_layers=12,
    mass_schedule="uniform",
):
    """
    Modula composition graphs for FLAN-T5-base encoder and decoder.
    
    Returns encoder and decoder as separate graphs to avoid mass-ratio 
    contamination during dualization.
    
    Encoder: 12 blocks × (4 attn + 3 ffn) + 1 rel_att_bias = 85 atoms
    Decoder: 12 blocks × (4 self + 4 cross + 3 ffn) + 1 rel_att_bias = 133 atoms
    
    Returns:
        (encoder, decoder): Tuple of separate Modula graphs
    """
    ms = uniform_mass_schedule if mass_schedule == "uniform" else linear_mass_schedule
    
    # ── Encoder ──────────────────────────────────────────────────────────────
    encoder = None
    enc_tot_layers = num_encoder_layers * 7 + 1  # 85 atoms
    enc_layer_idx = 0
    
    for i in range(num_encoder_layers):
        att = Attention(12, d_model, 64, 64, 1.0, causal=False)
        enc_layer_idx += 4
        
        wi_0 = LinearSVD(d_ff, d_model);       wi_0.mass = ms(enc_layer_idx, enc_tot_layers); enc_layer_idx += 1
        wi_1 = LinearSVD(d_ff, d_model);       wi_1.mass = ms(enc_layer_idx, enc_tot_layers); enc_layer_idx += 1
        wo   = LinearSVD(d_model, d_ff);       wo.mass   = ms(enc_layer_idx, enc_tot_layers); enc_layer_idx += 1
        
        ffn = wo @ wi_1 @ wi_0
        
        if i == 0:
            rel_att_bias = EmbedSVD(12, 32);    rel_att_bias.mass = ms(enc_layer_idx, enc_tot_layers); enc_layer_idx += 1
            att = rel_att_bias @ att
            
        block = ffn @ att
        encoder = block @ encoder if encoder is not None else block
    
    # ── Decoder ──────────────────────────────────────────────────────────────
    decoder = None
    dec_tot_layers = num_decoder_layers * 11 + 1  # 133 atoms
    dec_layer_idx = 0
    
    for i in range(num_decoder_layers):
        # Self-attention is CAUSAL (decoder tokens only)
        self_att = Attention(12, d_model, 64, 64, 1.0, causal=True)
        dec_layer_idx += 4
        
        # Cross-attention is NOT CAUSAL (attends to full encoder output)
        cross_att = Attention(12, d_model, 64, 64, 1.0, causal=False)
        dec_layer_idx += 4
        
        wi_0 = LinearSVD(d_ff, d_model);       wi_0.mass = ms(dec_layer_idx, dec_tot_layers); dec_layer_idx += 1
        wi_1 = LinearSVD(d_ff, d_model);       wi_1.mass = ms(dec_layer_idx, dec_tot_layers); dec_layer_idx += 1
        wo   = LinearSVD(d_model, d_ff);       wo.mass   = ms(dec_layer_idx, dec_tot_layers); dec_layer_idx += 1
        
        ffn = wo @ wi_1 @ wi_0
        
        if i == 0:
            rel_att_bias = EmbedSVD(12, 32);    rel_att_bias.mass = ms(dec_layer_idx, dec_tot_layers); dec_layer_idx += 1
            self_att = rel_att_bias @ self_att
            
        block = ffn @ cross_att @ self_att
        decoder = block @ decoder if decoder is not None else block
    
    return encoder, decoder

# ─────────────────────────────────────────────────────────────────────────────
# T5 key utilities
# ─────────────────────────────────────────────────────────────────────────────

def _is_t5_matrix_key(name: str) -> bool:
    """
    Returns True for the 216 weight matrices to dualize in FLAN-T5-base.
    Skips biases, layer norms, embeddings, relative attention biases, lm_head.
    """
    print("listing all layers")
    if re.search(r"relative_attention_bias",name):
        return True
    print(name)
    
    _SKIP = ("bias", "layer_norm",
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
    _ENC_ATT  = {"v": 0, "q": 1, "k": 2, "o": 3, "relative_attention_bias": 4}
    _ENC_FFN  = {"wi_0": 5, "wi_1": 6, "wo": 7}
    _DEC_SELF  = {"v": 0, "q": 1, "k": 2, "o": 3, "relative_attention_bias": 4}
    _DEC_CROSS = {"v": 5, "q": 6, "k": 7, "o": 8}
    _DEC_FFN   = {"wi_0": 9, "wi_1": 10, "wo": 11}

    def sort_key(name):
        m = re.search(r"encoder\.block\.(\d+)", name)
        if m:
            block = int(m.group(1))
            print(name)
            for param, sub in _ENC_ATT.items():
                print(f"SelfAttention.{param}.weight" in name)
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
    
    if "t5" in model_name.lower():
        # ── T5: Separate encoder/decoder handling ────────────────────────────
        encoder, decoder = FlanT5Base(mass_schedule=mass_schedule)
        
        # Collect encoder keys and gradients
        enc_names = []
        enc_grads = []
        for name in layer_names:
            if "encoder" in name and _is_t5_matrix_key(name):
                enc_names.append(name)
                enc_grads.append(grads[name].to(device))
        
        # Collect decoder keys and gradients
        dec_names = []
        dec_grads = []
        for name in layer_names:
            if "decoder" in name and _is_t5_matrix_key(name):
                dec_names.append(name)
                dec_grads.append(grads[name].to(device))
        
        print(f"build_duality_map (T5):")
        print(f"  encoder: atoms={len(encoder.atoms)}, mass={encoder.mass}, to_dualize={len(enc_grads)}")
        print(f"  decoder: atoms={len(decoder.atoms)}, mass={decoder.mass}, to_dualize={len(dec_grads)}")
        
        # Dualize separately to avoid mass-ratio contamination
        enc_dualized = encoder.dualize(enc_grads)
        dec_dualized = decoder.dualize(dec_grads)
        
        # Merge results
        dualized_dict = {}
        dualized_dict.update(zip(enc_names, enc_dualized))
        dualized_dict.update(zip(dec_names, dec_dualized))
        
        return dualized_dict
        
    else:
        # ── ViT: Single graph handling ───────────────────────────────────────
        if "B-16" in model_name:
            m = ViT_B_16(mass_schedule=mass_schedule)
        elif "B-32" in model_name:
            m = ViT_B_32(mass_schedule=mass_schedule)
        elif "L-14" in model_name:
            m = ViT_L_14(mass_schedule=mass_schedule)
        else:
            raise ValueError(f"No matching duality map for model_name='{model_name}'")
        
        # Collect keys and gradients to dualize
        to_consider_name = []
        to_consider_grad = []
        
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
        
        print(f"build_duality_map (ViT): atoms={len(m.atoms)}, mass={m.mass}, to_dualize={len(to_consider_grad)}")
        
        dualized = m.dualize(to_consider_grad)
        return dict(zip(to_consider_name, dualized))
