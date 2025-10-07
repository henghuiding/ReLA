"""Reusable language-query modulation building blocks.

This module contains small building blocks that are shared across LQM
experiments.  They are intentionally light-weight so that decoder supervision
hooks can easily tap into the intermediate representations without modifying
network code.  Every public callable documents the tensor shapes that flow
through it and returns auxiliary tensors to facilitate debugging/visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LQCrossAttentionCache:
    """Caches the tensors produced by :class:`LQCrossAttention`.

    Attributes
    ----------
    query_proj:
        Linearly projected queries with shape ``(batch, query_tokens, hidden)``.
    key_proj:
        Linearly projected keys with shape ``(batch, context_tokens, hidden)``.
    value_proj:
        Linearly projected values with shape ``(batch, context_tokens, hidden)``.
    attention_weights:
        Attention weights from ``nn.MultiheadAttention`` with shape
        ``(batch, heads, query_tokens, context_tokens)`` when ``need_weights`` is
        ``True`` in the forward pass.
    """

    query_proj: torch.Tensor
    key_proj: torch.Tensor
    value_proj: torch.Tensor
    attention_weights: Optional[torch.Tensor]


@dataclass(frozen=True)
class TopPSelection:
    """Records the result of :func:`top_p_select`.

    Attributes
    ----------
    selection_mask:
        Boolean mask with the same shape as the input scores where ``True`` marks
        the selected entries.
    sorted_scores:
        Input scores sorted in descending order along the selection dimension.
    sorted_indices:
        Indices that map from the sorted representation back to the original
        ordering along the selection dimension.
    keep_counts:
        Number of elements kept per slice along the selection dimension.  The
        tensor matches the input shape except that the selection dimension is
        collapsed to ``1``.
    """

    selection_mask: torch.Tensor
    sorted_scores: torch.Tensor
    sorted_indices: torch.Tensor
    keep_counts: torch.Tensor


class LQCrossAttention(nn.Module):
    """Cross-attention layer specialised for language-query alignment.

    Parameters
    ----------
    query_dim:
        Feature dimensionality of the incoming query embeddings.
    context_dim:
        Feature dimensionality of the context (language) embeddings.
    hidden_dim:
        Dimensionality of the internal attention representation.
    num_heads:
        Number of attention heads to use inside ``nn.MultiheadAttention``.
    dropout:
        Dropout applied by the multi-head attention module.
    bias:
        Whether linear projections include a bias term.

    Notes
    -----
    The module does **not** cast or move any tensors.  Whatever device and
    dtype the caller provides will be preserved throughout the computation.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.key_proj = nn.Linear(context_dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(context_dim, hidden_dim, bias=bias)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, query_dim, bias=bias)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, LQCrossAttentionCache]:
        """Attend query tokens over a contextual sequence.

        Parameters
        ----------
        queries:
            Query tensor with shape ``(batch, query_tokens, query_dim)``.
        context:
            Context tensor with shape ``(batch, context_tokens, context_dim)``.
        context_mask:
            Optional padding mask of shape ``(batch, context_tokens)``.  The mask
            follows the ``nn.MultiheadAttention`` convention where ``True`` marks
            tokens that should be ignored.
        need_weights:
            Whether to store the attention weights in the returned cache.  The
            forward pass never alters the dtype/device of any tensor.

        Returns
        -------
        attended_queries:
            Tensor with shape ``(batch, query_tokens, query_dim)`` containing the
            attended query embeddings.
        cache:
            :class:`LQCrossAttentionCache` instance containing the projected
            inputs and (optionally) the attention weights.  This is meant to be
            consumed by later decoder supervision hooks.
        """

        projected_queries = self.query_proj(queries)
        projected_keys = self.key_proj(context)
        projected_values = self.value_proj(context)

        key_padding_mask: Optional[torch.Tensor]
        if context_mask is None:
            key_padding_mask = None
        else:
            key_padding_mask = context_mask.bool()

        attn_output, attn_weights = self.attn(
            projected_queries,
            projected_keys,
            projected_values,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        output = self.output_proj(attn_output)

        cache = LQCrossAttentionCache(
            query_proj=projected_queries,
            key_proj=projected_keys,
            value_proj=projected_values,
            attention_weights=attn_weights if need_weights else None,
        )
        return output, cache


class QueryScorer(nn.Module):
    """Language-query compatibility scorer.

    The scorer is a light-weight MLP that emits a scalar score per query token.
    Intermediate activations are cached so that downstream supervision modules
    can reuse them.

    Parameters
    ----------
    query_dim:
        Dimensionality of the input query embeddings.
    hidden_dim:
        Size of the hidden layer used by the scorer MLP.
    activation:
        Callable activation applied after the first linear projection.
    dropout:
        Optional dropout probability applied to the hidden representation.
    """

    def __init__(
        self,
        query_dim: int,
        hidden_dim: int,
        *,
        activation: nn.Module = nn.ReLU(inplace=False),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(query_dim, hidden_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, queries: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict scalar scores for a batch of queries.

        Parameters
        ----------
        queries:
            Tensor with shape ``(batch, query_tokens, query_dim)``.

        Returns
        -------
        scores:
            Tensor of shape ``(batch, query_tokens)`` containing unnormalised
            scores for each query token.
        cache:
            Tuple ``(hidden, activated)`` where ``hidden`` is the pre-activation
            linear output and ``activated`` the representation after the
            activation (and dropout).  Both share the input dtype/device and are
            returned for diagnostics.
        """

        hidden = self.fc1(queries)
        activated = self.dropout(self.activation(hidden))
        scores = self.fc2(activated).squeeze(-1)
        return scores, (hidden, activated)


def top_p_select(
    scores: torch.Tensor,
    top_p: float,
    *,
    min_keep: int = 1,
    dim: int = -1,
) -> TopPSelection:
    """Deterministically select the smallest prefix that covers ``top_p`` mass.

    Parameters
    ----------
    scores:
        Tensor containing probabilities (or normalised scores) along ``dim``.
    top_p:
        Threshold for nucleus/top-p selection.  Values ``>=1`` keep the entire
        distribution, whereas non-positive values fall back to ``min_keep``.
    min_keep:
        Minimum number of items to keep regardless of ``top_p``.
    dim:
        Axis along which to apply the selection.

    Returns
    -------
    selection:
        :class:`TopPSelection` dataclass containing the boolean mask of selected
        elements, the sorted scores/indices, and the number of items retained per
        slice.  All tensors preserve the device and dtype of ``scores`` with the
        exception of indices/counts which use integer types.

    Notes
    -----
    The implementation avoids randomness.  Ties are resolved by favouring lower
    indices once the scores are sorted in descending order.
    """

    if scores.ndim == 0:
        raise ValueError("top_p_select expects a tensor with at least one dimension")

    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")

    dim = dim if dim >= 0 else scores.dim() + dim
    if dim < 0 or dim >= scores.dim():
        raise ValueError("dim is out of range for scores tensor")

    min_keep = max(int(min_keep), 0)

    moved_scores = scores.movedim(dim, -1)
    original_shape = moved_scores.shape
    last_dim = original_shape[-1]
    flat_scores = moved_scores.reshape(-1, last_dim)

    sorted_scores, sorted_indices = torch.sort(flat_scores, dim=-1, descending=True, stable=True)

    if last_dim == 0:
        empty_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
        selection = TopPSelection(
            selection_mask=empty_mask.reshape(original_shape).movedim(-1, dim),
            sorted_scores=sorted_scores.reshape(original_shape).movedim(-1, dim),
            sorted_indices=sorted_indices.reshape(original_shape).movedim(-1, dim),
            keep_counts=torch.zeros(flat_scores.shape[0], 1, device=scores.device, dtype=torch.long).reshape(
                original_shape[:-1] + (1,)
            ).movedim(-1, dim),
        )
        return selection

    cumulative = sorted_scores.cumsum(-1)
    keep_counts: torch.Tensor
    min_keep_eff = min(last_dim, max(min_keep, 1 if last_dim > 0 else 0))
    if top_p >= 1.0:
        keep_counts = torch.full(
            (flat_scores.size(0), 1), last_dim, dtype=torch.long, device=scores.device
        )
    elif top_p <= 0.0:
        keep_counts = torch.full(
            (flat_scores.size(0), 1), min_keep_eff, dtype=torch.long, device=scores.device
        )
    else:
        meets = cumulative >= top_p
        meets_any = meets.any(-1, keepdim=True)
        first_idx = meets.float().argmax(-1, keepdim=True)
        fallback = torch.full_like(first_idx, last_dim - 1)
        first_idx = torch.where(meets_any, first_idx, fallback)
        keep_counts = (first_idx + 1).clamp(min=min_keep_eff, max=last_dim)

    keep_counts = keep_counts.clamp(min=min_keep_eff, max=last_dim)

    range_ = torch.arange(last_dim, device=scores.device)
    selection_mask_sorted = range_.unsqueeze(0) < keep_counts
    selection_mask_flat = torch.zeros_like(flat_scores, dtype=torch.bool)
    selection_mask_flat.scatter_(1, sorted_indices, selection_mask_sorted)

    selection_mask = selection_mask_flat.reshape(original_shape).movedim(-1, dim)
    sorted_scores = sorted_scores.reshape(original_shape).movedim(-1, dim)
    sorted_indices = sorted_indices.reshape(original_shape).movedim(-1, dim)
    keep_counts = keep_counts.reshape(original_shape[:-1] + (1,)).movedim(-1, dim)

    return TopPSelection(
        selection_mask=selection_mask,
        sorted_scores=sorted_scores,
        sorted_indices=sorted_indices,
        keep_counts=keep_counts,
    )


class QMDiversityLoss(nn.Module):
    """Penalises highly similar selected queries.

    The loss encourages the selected query embeddings to span a diverse subspace
    by discouraging large pairwise cosine similarities.  It returns both the
    scalar loss and the pairwise similarity matrix for external inspection.

    Parameters
    ----------
    eps:
        Numerical stability term added to the norm when normalising embeddings.
    reduction:
        Reduction to apply across the batch, one of ``"mean"`` (default),
        ``"sum"`` or ``"none"``.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        query_embeddings: torch.Tensor,
        selection_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the diversity regulariser.

        Parameters
        ----------
        query_embeddings:
            Tensor of shape ``(batch, num_queries, dim)`` containing the query
            descriptors.
        selection_mask:
            Boolean or integral tensor of shape ``(batch, num_queries)`` where
            ``True``/``1`` marks the queries to include in the loss.

        Returns
        -------
        loss:
            Tensor containing the reduced loss.  The dtype/device matches
            ``query_embeddings``.  When ``reduction`` is ``"none"`` a tensor of
            shape ``(batch,)`` is returned.
        pairwise_similarity:
            Tensor of shape ``(batch, num_queries, num_queries)`` containing the
            cosine similarities after applying the selection mask.  Entries
            corresponding to unselected queries are set to zero.
        """

        if selection_mask.dtype != torch.bool:
            selection_mask = selection_mask.to(dtype=torch.bool)

        normalized = F.normalize(query_embeddings, p=2, dim=-1, eps=self.eps)
        pairwise = torch.matmul(normalized, normalized.transpose(-1, -2))

        batch, num_queries, _ = pairwise.shape
        if num_queries == 0:
            zero = query_embeddings.new_zeros((batch,) if self.reduction == "none" else ())
            return zero, pairwise

        mask_pairs = selection_mask.unsqueeze(-1) & selection_mask.unsqueeze(-2)
        diag_mask = torch.eye(num_queries, dtype=torch.bool, device=mask_pairs.device)
        mask_pairs = mask_pairs & ~diag_mask

        tri_mask = torch.triu(
            torch.ones(num_queries, num_queries, dtype=torch.bool, device=mask_pairs.device), diagonal=1
        ).unsqueeze(0)
        unique_mask = mask_pairs & tri_mask

        pairwise_masked = pairwise * mask_pairs.to(dtype=pairwise.dtype)

        pair_counts = unique_mask.sum(dim=(-1, -2))
        summed = (pairwise.pow(2) * unique_mask.to(dtype=pairwise.dtype)).sum(dim=(-1, -2))

        per_sample_loss = query_embeddings.new_zeros(batch)
        non_zero = pair_counts > 0
        if non_zero.any():
            per_sample_loss = per_sample_loss.clone()
            per_sample_loss[non_zero] = summed[non_zero] / pair_counts[non_zero].to(summed.dtype)

        if self.reduction == "mean":
            loss = per_sample_loss.mean()
        elif self.reduction == "sum":
            loss = per_sample_loss.sum()
        else:
            loss = per_sample_loss

        return loss, pairwise_masked
