"""
se3_transformer.py
SE(3)-equivariant attention model for 3D point clouds in AIPlasma.

Lives in: models/equivariant/se3_transformer.py

Features are restricted to degree 0 (scalars) and degree 1 (3-vectors) and
implemented in plain PyTorch, without e3nn.

Symmetry: rotating and translating the spatial coordinates leaves the scalar
output unchanged. Only RELATIVE geometry is used, so the model cannot see
absolute positions. Absolute information must come in through input fields.

Inputs (all read from the FeatureBatch):
    spatial_dims   - the 3 coordinate columns that are positions
    invariant_dims - coordinate columns that are rotation-invariant scalars
                     (typically time)
    scalar_fields  - field columns fed in as scalars
    vector_fields  - groups of 3 field columns fed in as vectors

Do NOT list target columns in scalar_fields/vector_fields when `fields` holds
the training targets, or the model sees the answer.

Neighbours: k nearest neighbours by spatial distance. The neighbour selection
is not differentiable; relative positions are.

Limitation: no compute_gradients(). Output at a node depends on neighbouring
positions, so d(sum u)/dx is not the pointwise field derivative that PDE
residuals assume. Use with w_pde = 0 until a query/context variant exists.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base_model import PhysicsModel, ModelOutput
from data.preprocessing.feature_pipeline import FeatureBatch


# ════════════════════════════════════════════════════════════════════════════
# Building blocks
# ════════════════════════════════════════════════════════════════════════════

class RadialBasis(nn.Module):
    """Gaussian radial basis expansion of edge distances on [0, cutoff]."""

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        spacing = cutoff / max(n_rbf - 1, 1)
        self.gamma = 0.5 / spacing ** 2

    def forward(self, dist: Tensor) -> Tensor:
        # (N, k) -> (N, k, n_rbf)
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class VectorLinear(nn.Module):
    """
    Channel mixing for vector features, no bias.
    A bias would break equivariance, mixing channels does not.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels) / math.sqrt(in_channels)
        )

    def forward(self, v: Tensor) -> Tensor:
        # (..., C_in, 3) -> (..., C_out, 3)
        return torch.einsum("oc,...cx->...ox", self.weight, v)


def _radial_mlp(n_rbf: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, out))


def _kernel(
    s: Tensor, v: Tensor, u: Tensor, w: Tensor, compute_vectors: bool = True
) -> tuple:
    """
    Equivariant edge kernel for degrees 0 and 1.

    Args:
        s: neighbour scalars,          (N, k, C)
        v: neighbour vectors,          (N, k, C, 3)
        u: unit edge directions,       (N, k, 3)
        w: radial weights per path,    (N, k, 5, C)

    Paths (input degree x edge harmonic -> output degree):
        0: s          (0 x Y0 -> 0)
        1: v . u      (1 x Y1 -> 0)
        2: s * u      (0 x Y1 -> 1)
        3: v          (1 x Y0 -> 1)
        4: v cross u  (1 x Y1 -> 1)

    Returns:
        (out_s (N, k, C), out_v (N, k, C, 3) or None)
    """
    u_ = u.unsqueeze(2)                                  # (N, k, 1, 3)
    v_dot_u = (v * u_).sum(-1)                           # (N, k, C)
    out_s = w[..., 0, :] * s + w[..., 1, :] * v_dot_u

    if not compute_vectors:
        return out_s, None

    v_cross_u = torch.cross(v, u_.expand_as(v), dim=-1)  # (N, k, C, 3)
    out_v = (
        (w[..., 2, :] * s).unsqueeze(-1) * u_
        + w[..., 3, :].unsqueeze(-1) * v
        + w[..., 4, :].unsqueeze(-1) * v_cross_u
    )
    return out_s, out_v


# ════════════════════════════════════════════════════════════════════════════
# One attention layer
# ════════════════════════════════════════════════════════════════════════════

class SE3AttentionLayer(nn.Module):
    """
    Equivariant multi-head attention over the k nearest neighbours.

    Queries come from the receiving node, keys and values from edge kernels
    applied to neighbour features. The attention logit is the invariant inner
    product over scalar and vector channels, so the softmax is invariant.

    update_vectors=False skips the vector update. The last layer uses this,
    because the scalar head never reads the vector features it would produce.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int,
        n_rbf: int,
        radial_hidden: int,
        update_vectors: bool = True,
    ):
        super().__init__()
        self.channels       = channels
        self.n_heads        = n_heads
        self.update_vectors = update_vectors

        self.query_s  = nn.Linear(channels, channels)
        self.query_v  = VectorLinear(channels, channels)
        self.source_s = nn.Linear(channels, channels)
        self.source_v = VectorLinear(channels, channels)

        self.radial_key   = _radial_mlp(n_rbf, radial_hidden, 5 * channels)
        self.radial_value = _radial_mlp(n_rbf, radial_hidden, 5 * channels)

        self.norm_s = nn.LayerNorm(channels)
        self.out_s  = nn.Linear(channels, channels)
        if update_vectors:
            self.out_v = VectorLinear(channels, channels)
            self.gate  = nn.Linear(channels, channels)

    def forward(self, s: Tensor, v: Tensor, idx: Tensor, u: Tensor, rbf: Tensor):
        N, k = idx.shape
        C, H = self.channels, self.n_heads
        hd   = C // H

        q_s, q_v = self.query_s(s), self.query_v(v)       # (N, C), (N, C, 3)
        nb_s = self.source_s(s)[idx]                      # (N, k, C)
        nb_v = self.source_v(v)[idx]                      # (N, k, C, 3)

        w_key = self.radial_key(rbf).reshape(N, k, 5, C)
        w_val = self.radial_value(rbf).reshape(N, k, 5, C)
        k_s, k_v     = _kernel(nb_s, nb_v, u, w_key)
        val_s, val_v = _kernel(nb_s, nb_v, u, w_val, compute_vectors=self.update_vectors)

        # Invariant attention logits, per head
        qs = q_s.reshape(N, 1, H, hd)
        ks = k_s.reshape(N, k, H, hd)
        qv = q_v.reshape(N, 1, H, hd, 3)
        kv = k_v.reshape(N, k, H, hd, 3)
        logits = (qs * ks).sum(-1) + (qv * kv).sum((-1, -2))      # (N, k, H)
        attn   = torch.softmax(logits / math.sqrt(4 * hd), dim=1)  # over neighbours

        agg_s = (attn.unsqueeze(-1) * val_s.reshape(N, k, H, hd)).sum(1).reshape(N, C)
        s_new = self.out_s(agg_s)
        s     = s + F.silu(self.norm_s(s_new))

        if self.update_vectors:
            agg_v = (
                attn.unsqueeze(-1).unsqueeze(-1) * val_v.reshape(N, k, H, hd, 3)
            ).sum(1).reshape(N, C, 3)
            # The gate is built from scalars only, so it is invariant
            gate = torch.sigmoid(self.gate(s_new)).unsqueeze(-1)   # (N, C, 1)
            v    = v + self.out_v(agg_v) * gate

        return s, v


# ════════════════════════════════════════════════════════════════════════════
# The model
# ════════════════════════════════════════════════════════════════════════════

class SE3Transformer(PhysicsModel):
    """
    SE(3)-equivariant transformer (degrees 0 and 1) with a scalar output.

    Args:
        output_dim:     Number of output channels per point. Default: 1.
        channels:       Width of scalar and vector features. Default: 32.
        n_layers:       Number of attention layers. Default: 3.
        n_heads:        Attention heads; must divide channels. Default: 4.
        k_neighbors:    Neighbours per point (clamped to N - 1). Default: 16.
        spatial_dims:   The 3 coordinate columns that are positions.
        invariant_dims: Coordinate columns used as invariant scalars (e.g. time).
        scalar_fields:  Field columns used as scalar inputs.
        vector_fields:  Groups of 3 field columns used as vector inputs.
        n_rbf:          Radial basis functions for edge distances. Default: 16.
        rbf_cutoff:     Distance range of the radial basis. Should match the
                        typical neighbour distance in your (normalised)
                        coordinates. Default: 1.0.
        radial_hidden:  Hidden width of the radial MLPs. Default: 32.
        eps:            Smoothing for edge length, avoids a singular gradient
                        for coincident points. Default: 1e-6.

    Example (3D position + time, one scalar and one vector input field):
        model = SE3Transformer(
            output_dim=1, invariant_dims=(3,),
            scalar_fields=(0,), vector_fields=((1, 2, 3),),
        )
        pred = model.forward(batch).pred        # (N, 1), invariant
    """

    def __init__(
        self,
        output_dim:     int = 1,
        channels:       int = 32,
        n_layers:       int = 3,
        n_heads:        int = 4,
        k_neighbors:    int = 16,
        spatial_dims:   Sequence[int] = (0, 1, 2),
        invariant_dims: Sequence[int] = (),
        scalar_fields:  Sequence[int] = (),
        vector_fields:  Sequence[Sequence[int]] = (),
        n_rbf:          int = 16,
        rbf_cutoff:     float = 1.0,
        radial_hidden:  int = 32,
        eps:            float = 1e-6,
    ):
        spatial_dims   = tuple(spatial_dims)
        invariant_dims = tuple(invariant_dims)
        scalar_fields  = tuple(scalar_fields)
        vector_fields  = tuple(tuple(g) for g in vector_fields)

        if len(spatial_dims) != 3:
            raise ValueError(
                f"SE(3) equivariance needs exactly 3 spatial columns, got {len(spatial_dims)}"
            )
        if set(spatial_dims) & set(invariant_dims):
            raise ValueError("spatial_dims and invariant_dims must not overlap")
        if any(len(g) != 3 for g in vector_fields):
            raise ValueError("each entry of vector_fields must list exactly 3 columns")
        if channels % n_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by n_heads ({n_heads})")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        super().__init__()
        self.output_dim     = output_dim
        self.channels       = channels
        self.n_heads        = n_heads
        self.k_neighbors    = k_neighbors
        self.spatial_dims   = spatial_dims
        self.invariant_dims = invariant_dims
        self.scalar_fields  = scalar_fields
        self.vector_fields  = vector_fields
        self.eps            = eps

        # With no scalar input at all, a constant 1 is used as the node feature
        n_scalar_in = max(len(scalar_fields) + len(invariant_dims), 1)
        self.embed_s = nn.Linear(n_scalar_in, channels)
        self.embed_v = VectorLinear(len(vector_fields), channels) if vector_fields else None

        self.rbf = RadialBasis(n_rbf, rbf_cutoff)
        self.layers = nn.ModuleList([
            SE3AttentionLayer(
                channels, n_heads, n_rbf, radial_hidden,
                update_vectors=(i < n_layers - 1),
            )
            for i in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, output_dim)
        )

    def forward(self, batch: FeatureBatch) -> ModelOutput:
        coords, fields = batch.coords, batch.fields
        N = coords.shape[0]
        if N < 2:
            raise ValueError("SE3Transformer needs at least 2 points")

        pos = coords[:, list(self.spatial_dims)]            # (N, 3)
        idx, u, dist = self._neighbours(pos)
        rbf = self.rbf(dist)                                # (N, k, n_rbf)

        # ── Initial node features ────────────────────────────────────────────
        parts = []
        if self.scalar_fields:
            parts.append(fields[:, list(self.scalar_fields)])
        if self.invariant_dims:
            parts.append(coords[:, list(self.invariant_dims)])
        scalar_in = (
            torch.cat(parts, dim=1)
            if parts else torch.ones(N, 1, dtype=coords.dtype, device=coords.device)
        )
        s = self.embed_s(scalar_in)                         # (N, C)

        if self.embed_v is not None:
            vec = torch.stack([fields[:, list(g)] for g in self.vector_fields], dim=1)
            v = self.embed_v(vec)                           # (N, C, 3)
        else:
            v = torch.zeros(N, self.channels, 3, dtype=coords.dtype, device=coords.device)

        # ── Equivariant attention layers ─────────────────────────────────────
        for layer in self.layers:
            s, v = layer(s, v, idx, u, rbf)

        return ModelOutput(pred=self.head(s), uncertainty=None)

    def summary(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"SE3Transformer(channels={self.channels}, layers={len(self.layers)}, "
            f"heads={self.n_heads}, k={self.k_neighbors}, degrees=0,1) "
            f"- {n_params} parameters"
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _neighbours(self, pos: Tensor) -> tuple:
        """
        kNN graph from positions.

        Returns:
            idx:  neighbour indices,                  (N, k)
            u:    unit direction node -> neighbour,   (N, k, 3)
            dist: smoothed edge lengths,              (N, k)
        """
        N = pos.shape[0]
        k = min(self.k_neighbors, N - 1)

        with torch.no_grad():
            d_all = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
            d_all.fill_diagonal_(float("inf"))               # no self-edges
            idx = d_all.topk(k, dim=1, largest=False).indices

        rel  = pos[idx] - pos.unsqueeze(1)                   # (N, k, 3), differentiable
        dist = torch.sqrt((rel ** 2).sum(-1) + self.eps ** 2)
        u    = rel / dist.unsqueeze(-1)
        return idx, u, dist