"""Language-aware query modulation meta-architecture."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode as CN
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from gres_model.GRES import GRES

from .lqm_blocks import LQCrossAttention, QMDiversityLoss, QueryScorer, top_p_select


@META_ARCH_REGISTRY.register()
class LQMFormer(GRES):
    """GRES architecture augmented with language-aware query modulation."""

    @configurable
    def __init__(self, *, lqm_cfg: Optional[CN] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(lqm_cfg, CN):
            self.lqm_cfg = lqm_cfg.clone()
        elif lqm_cfg:
            self.lqm_cfg = CN(lqm_cfg)
        else:
            self.lqm_cfg = CN()

        dqm_cfg_raw = self.lqm_cfg.get("DQM", CN()) if hasattr(self.lqm_cfg, "get") else CN()
        if isinstance(dqm_cfg_raw, CN):
            self.dqm_cfg = dqm_cfg_raw.clone()
        elif dqm_cfg_raw:
            self.dqm_cfg = CN(dqm_cfg_raw)
        else:
            self.dqm_cfg = CN()

        self.dqm_enabled: bool = bool(
            self.dqm_cfg.get("ENABLE", self.dqm_cfg.get("ENABLED", False))
        )
        self.dqm_top_p: float = float(self.dqm_cfg.get("TOP_P", 1.0))
        self.dqm_min_keep: int = int(self.dqm_cfg.get("MIN_KEEP", 1))
        self.dqm_qm_loss_weight: float = float(self.dqm_cfg.get("QM_LOSS_WEIGHT", 0.0))
        self.dqm_attn_cache: bool = bool(self.dqm_cfg.get("CACHE_ATTENTION", True))

        score_activation = str(self.dqm_cfg.get("SCORE_ACTIVATION", "sigmoid")).lower()
        self.score_activation: str = score_activation
        self.score_hidden_activation: str = str(
            self.dqm_cfg.get("SCORE_HIDDEN_ACTIVATION", "relu")
        ).lower()
        self.score_dropout: float = float(self.dqm_cfg.get("SCORE_DROPOUT", 0.0))

        self.lqca: Optional[LQCrossAttention] = None
        self.q_scorer: Optional[QueryScorer] = None
        self.qm_loss_fn: Optional[QMDiversityLoss] = None

        self.dqm_scores: Optional[torch.Tensor] = None
        self.dqm_keep_mask: Optional[torch.Tensor] = None
        self.dqm_idx_topk: Optional[torch.Tensor] = None
        self.dqm_attn_logits: Optional[torch.Tensor] = None
        self.dqm_q_upd: Optional[torch.Tensor] = None
        self.dqm_pairwise: Optional[torch.Tensor] = None

        self._init_lqm(self.lqm_cfg)

    def _init_lqm(self, cfg: CN) -> None:
        """Initialize DQM-related flags and log the configuration state."""

        if isinstance(cfg, CN):
            dqm_cfg = cfg.get("DQM", CN())
        elif isinstance(cfg, dict):
            dqm_cfg = cfg.get("DQM", {})
        else:
            dqm_cfg = CN()

        if isinstance(dqm_cfg, dict):
            dqm_cfg = CN(dqm_cfg)

        enable = bool(dqm_cfg.get("ENABLE", dqm_cfg.get("ENABLED", self.dqm_enabled)))
        self.dqm_enable = enable
        self.dqm_enabled = enable

        top_p = dqm_cfg.get("TOP_P", self.dqm_top_p)
        nq = dqm_cfg.get("NQ", dqm_cfg.get("NUM_QUERIES", "unknown"))

        if enable:
            print(f"[LQMFormer] Initializing DQM with TOP_P={top_p}, NQ={nq}")
        else:
            print("[LQMFormer] DQM disabled — running in GRES mode.")

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        config_dict = super().from_config(cfg)
        lqm_cfg = getattr(cfg, "LQM", None)
        if isinstance(lqm_cfg, CN):
            config_dict["lqm_cfg"] = lqm_cfg.clone()
        elif lqm_cfg:
            config_dict["lqm_cfg"] = CN(lqm_cfg)
        else:
            config_dict["lqm_cfg"] = CN()
        return config_dict

    def _build_hidden_activation(self) -> nn.Module:
        activation = self.score_hidden_activation
        if activation == "relu":
            return nn.ReLU(inplace=False)
        if activation == "gelu":
            return nn.GELU()
        if activation in {"silu", "swish"}:
            return nn.SiLU()
        if activation in {"identity", "none", "linear"}:
            return nn.Identity()
        return nn.ReLU(inplace=False)

    def _apply_score_activation(self, scores: torch.Tensor) -> torch.Tensor:
        activation = self.score_activation
        if activation == "sigmoid":
            return torch.sigmoid(scores)
        if activation == "softmax":
            return torch.softmax(scores, dim=-1)
        if activation == "relu":
            return F.relu(scores)
        return scores

    def _ensure_lqm_modules(self, query_dim: int, lang_dim: int) -> None:
        if self.lqca is None:
            hidden_dim = int(self.dqm_cfg.get("HIDDEN_DIM", query_dim))
            num_heads = int(self.dqm_cfg.get("NUM_HEADS", 8))
            dropout = float(self.dqm_cfg.get("DROPOUT", 0.0))
            bias = bool(self.dqm_cfg.get("BIAS", True))
            self.lqca = LQCrossAttention(
                query_dim=query_dim,
                context_dim=lang_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
            ).to(self.device)

        if self.q_scorer is None:
            score_hidden_dim = int(self.dqm_cfg.get("SCORE_HIDDEN_DIM", query_dim))
            activation = self._build_hidden_activation()
            self.q_scorer = QueryScorer(
                query_dim=query_dim,
                hidden_dim=score_hidden_dim,
                activation=activation,
                dropout=self.score_dropout,
            ).to(self.device)

        if self.qm_loss_fn is None:
            eps = float(self.dqm_cfg.get("QM_LOSS_EPS", 1e-6))
            reduction = str(self.dqm_cfg.get("QM_LOSS_REDUCTION", "mean"))
            self.qm_loss_fn = QMDiversityLoss(eps=eps, reduction=reduction)

    def forward(self, batched_inputs):  # type: ignore[override]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        lang_tokens = [x["lang_tokens"].to(self.device) for x in batched_inputs]
        lang_tokens = torch.cat(lang_tokens, dim=0)

        lang_attention_mask = [x["lang_mask"].to(self.device) for x in batched_inputs]
        lang_attention_mask = torch.cat(lang_attention_mask, dim=0)

        lang_sequence = self.text_encoder(lang_tokens, attention_mask=lang_attention_mask)[0]
        lang_feat = lang_sequence.permute(0, 2, 1)
        lang_mask_expanded = lang_attention_mask.unsqueeze(dim=-1)

        self.dqm_scores = None
        self.dqm_keep_mask = None
        self.dqm_idx_topk = None
        self.dqm_attn_logits = None
        self.dqm_q_upd = None
        self.dqm_pairwise = None

        dqm_state: Optional[Dict[str, torch.Tensor]] = None
        if self.dqm_enabled:
            query_embed = self.sem_seg_head.predictor.query_feat.weight
            batch_size, query_dim = lang_sequence.shape[0], query_embed.shape[1]
            self._ensure_lqm_modules(query_dim=query_dim, lang_dim=lang_sequence.shape[-1])

            assert self.lqca is not None and self.q_scorer is not None

            queries = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            language_padding_mask = lang_attention_mask == 0
            q_upd, attn_cache = self.lqca(
                queries,
                lang_sequence,
                context_mask=language_padding_mask,
                need_weights=self.dqm_attn_cache,
            )
            scores_raw, _ = self.q_scorer(q_upd)
            scores = self._apply_score_activation(scores_raw)
            selection = top_p_select(scores, self.dqm_top_p, min_keep=self.dqm_min_keep, dim=-1)
            keep_mask = selection.selection_mask
            q_gated = q_upd * keep_mask.to(dtype=q_upd.dtype).unsqueeze(-1)

            dqm_state = {
                "scores": scores,
                "keep_mask": keep_mask,
                "idx_topk": selection.sorted_indices,
                "attn_logits": attn_cache.attention_weights
                if attn_cache.attention_weights is not None
                else None,
                "q_upd": q_upd,
                "q_gated": q_gated,
            }

            self.dqm_scores = scores
            self.dqm_keep_mask = keep_mask
            self.dqm_idx_topk = selection.sorted_indices
            self.dqm_attn_logits = attn_cache.attention_weights
            self.dqm_q_upd = q_upd

            try:
                storage = get_event_storage()
            except AssertionError:
                storage = None
            if storage is not None:
                keep_float = keep_mask.float()
                keep_counts = keep_float.sum(dim=-1)
                total_queries = keep_mask.shape[-1]
                if total_queries > 0:
                    ratio = keep_counts / float(total_queries)
                else:
                    ratio = keep_counts.new_zeros(keep_counts.shape)
                kept_scores = scores * keep_float
                denom = keep_counts.clamp_min(1.0)
                mean_score = (kept_scores.sum(-1) / denom).mean().item()
                storage.put_scalar("lqm/keep_ratio", ratio.mean().item())
                storage.put_scalar("lqm/keep_score", mean_score)

        features = self.backbone(images.tensor, lang_feat, lang_mask_expanded)

        predictor = self.sem_seg_head.predictor
        previous_override = getattr(predictor, "_lqm_override", None)
        if dqm_state is not None:
            predictor._lqm_override = dqm_state["q_gated"]

        try:
            outputs = self.sem_seg_head(features, lang_feat, lang_mask_expanded)
        finally:
            if dqm_state is not None:
                if previous_override is None:
                    if hasattr(predictor, "_lqm_override"):
                        delattr(predictor, "_lqm_override")
                else:
                    predictor._lqm_override = previous_override

        if dqm_state is not None:
            outputs = dict(outputs)
            outputs["query_scores"] = dqm_state["scores"]
            self.dqm_pairwise = None
            # TODO: Supervise decoder cross-attention once decoder hooks are available.

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            losses = self.criterion(outputs, targets)

            qm_loss_scaled: Optional[torch.Tensor] = None
            if dqm_state is not None and self.qm_loss_fn is not None and self.dqm_qm_loss_weight > 0.0:
                keep_mask = dqm_state["keep_mask"]
                if keep_mask.numel() > 0 and keep_mask.any():
                    loss_qm, pairwise = self.qm_loss_fn(dqm_state["q_upd"], keep_mask)
                    self.dqm_pairwise = pairwise
                else:
                    loss_qm = dqm_state["q_upd"].new_zeros(())
                    self.dqm_pairwise = None
                qm_loss_scaled = loss_qm * self.dqm_qm_loss_weight

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            if qm_loss_scaled is not None:
                losses["loss_qm"] = qm_loss_scaled
            return losses

        mask_pred_results = outputs["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        nt_pred_results = outputs["nt_label"]
        query_scores = outputs.get("query_scores")

        del outputs

        processed_results = []
        for idx, (mask_pred_result, nt_pred_result, _, _) in enumerate(
            zip(mask_pred_results, nt_pred_results, batched_inputs, images.image_sizes)
        ):
            processed_results.append({})
            r, nt = retry_if_cuda_oom(self.refer_inference)(mask_pred_result, nt_pred_result)
            processed_results[-1]["ref_seg"] = r
            processed_results[-1]["nt_label"] = nt
            if query_scores is not None:
                processed_results[-1]["query_scores"] = query_scores[idx].detach()

        return processed_results
