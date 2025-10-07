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
from detectron2.utils.memory import retry_if_cuda_oom

from gres_model.GRES import GRES


class LanguageQueryCrossAttention(nn.Module):
    """Cross-attention layer that aligns queries with language tokens."""

    def __init__(
        self,
        query_dim: int,
        lang_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(hidden_dim, query_dim)

    def forward(
        self,
        query_features: torch.Tensor,
        language_features: torch.Tensor,
        language_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        projected_queries = self.query_proj(query_features)
        projected_language = self.lang_proj(language_features)
        attn_output, _ = self.attn(
            projected_queries,
            projected_language,
            projected_language,
            key_padding_mask=language_padding_mask,
        )
        return self.out_proj(attn_output)


@META_ARCH_REGISTRY.register()
class LQMFormer(GRES):
    """GRES architecture augmented with language-aware query modulation."""

    @configurable
    def __init__(self, *, dqm_config: Optional[CN] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(dqm_config, CN):
            self.dqm_config = dqm_config.clone()
        elif dqm_config:
            self.dqm_config = CN(dqm_config)
        else:
            self.dqm_config = CN()
        self.dqm_enabled: bool = bool(self.dqm_config.get("ENABLED", False))
        self.dqm_apply_to_masks: bool = bool(self.dqm_config.get("APPLY_TO_MASKS", False))
        self.dqm_apply_to_logits: bool = bool(self.dqm_config.get("APPLY_TO_LOGITS", False))
        self.score_activation: str = str(self.dqm_config.get("SCORE_ACTIVATION", "sigmoid")).lower()
        self._lqca_initialized: bool = False
        self.lqca: Optional[LanguageQueryCrossAttention] = None
        self.score_mlp: Optional[nn.Module] = None

    @classmethod
    def from_config(cls, cfg) -> Dict[str, Any]:
        config_dict = super().from_config(cfg)
        if hasattr(cfg.MODEL.MASK_FORMER, "DQM"):
            config_dict["dqm_config"] = cfg.MODEL.MASK_FORMER.DQM.clone()
        else:
            config_dict["dqm_config"] = CN()
        return config_dict

    def _init_lqca_modules(self, lang_feat: torch.Tensor) -> None:
        if not self.dqm_enabled or self._lqca_initialized:
            return

        query_dim = int(self.sem_seg_head.predictor.query_feat.embedding_dim)
        lang_dim = int(lang_feat.shape[-1])
        hidden_dim = int(self.dqm_config.get("HIDDEN_DIM", query_dim))
        num_heads = int(self.dqm_config.get("NUM_HEADS", 8))
        dropout = float(self.dqm_config.get("DROPOUT", 0.0))
        score_hidden_dim = int(self.dqm_config.get("SCORE_HIDDEN_DIM", query_dim))

        self.lqca = LanguageQueryCrossAttention(
            query_dim=query_dim,
            lang_dim=lang_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        ).to(self.device)

        mlp_layers = [nn.Linear(query_dim, score_hidden_dim), nn.ReLU(inplace=True), nn.Linear(score_hidden_dim, 1)]
        self.score_mlp = nn.Sequential(*mlp_layers).to(self.device)
        self._lqca_initialized = True

    def _apply_score_activation(self, scores: torch.Tensor) -> torch.Tensor:
        activation = self.score_activation
        if activation == "sigmoid":
            return torch.sigmoid(scores)
        if activation == "softmax":
            return torch.softmax(scores, dim=-1)
        if activation == "relu":
            return F.relu(scores)
        return scores

    def _apply_lqca(
        self,
        outputs: Dict[str, torch.Tensor],
        lang_feat: torch.Tensor,
        lang_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.dqm_enabled:
            return outputs

        self._init_lqca_modules(lang_feat)
        assert self.lqca is not None and self.score_mlp is not None

        batch_size = lang_feat.shape[0]
        query_embed = self.sem_seg_head.predictor.query_feat.weight
        query_features = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        language_padding_mask = lang_attention_mask == 0

        refined_queries = self.lqca(query_features, lang_feat, language_padding_mask=language_padding_mask)
        query_scores = self.score_mlp(refined_queries).squeeze(-1)
        query_scores = self._apply_score_activation(query_scores)

        outputs = dict(outputs)
        outputs["query_scores"] = query_scores

        if self.dqm_apply_to_logits and "pred_logits" in outputs:
            outputs["pred_logits"] = outputs["pred_logits"] * query_scores.unsqueeze(-1)
        if self.dqm_apply_to_masks:
            mask_scale = query_scores.view(batch_size, -1, 1, 1)
            if "pred_masks" in outputs:
                outputs["pred_masks"] = outputs["pred_masks"] * mask_scale
            if "all_masks" in outputs:
                outputs["all_masks"] = outputs["all_masks"] * mask_scale

        return outputs

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

        features = self.backbone(images.tensor, lang_feat, lang_mask_expanded)
        outputs = self.sem_seg_head(features, lang_feat, lang_mask_expanded)
        outputs = self._apply_lqca(outputs, lang_sequence, lang_attention_mask)

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
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
