import torch
from torch import nn


class RelationQueryRouter(nn.Module):
    """Per-head self-attention bias B_ij = -p_ij * S_ij + (1 - p_ij) * D_ij from pairwise query features."""

    num_pairwise_features = 8

    def __init__(
        self,
        num_heads: int = 8,
        rel_hidden: int = 32,
        max_bias: float = 10.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_bias = max_bias
        self.rel_mlp = nn.Sequential(
            nn.Linear(self.num_pairwise_features, rel_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(rel_hidden, rel_hidden),
            nn.ReLU(inplace=True),
        )
        self.gate_head = nn.Linear(rel_hidden, num_heads)
        self.sup_head = nn.Linear(rel_hidden, num_heads)
        self.del_head = nn.Linear(rel_hidden, num_heads)
        self.router_tau = 1.0
        self._last_stats = {}
        self.init_weights()

    def init_weights(self):
        for m in self.rel_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for head in (self.gate_head, self.sup_head, self.del_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        query: torch.Tensor,
        prev_class_logits: torch.Tensor = None,
        reference_boxes: torch.Tensor = None,
        return_gates: bool = False,
    ):
        B, N, _ = query.shape
        H = self.num_heads
        feats = compute_pairwise_features(reference_boxes, prev_class_logits, B, N, query)

        hidden = self.rel_mlp(feats)
        p_sup = torch.sigmoid(self.gate_head(hidden) / float(self.router_tau))
        s = torch.nn.functional.softplus(self.sup_head(hidden))
        d = torch.nn.functional.softplus(self.del_head(hidden))
        bias = -p_sup * s + (1.0 - p_sup) * d
        bias = bias.clamp(min=-self.max_bias, max=self.max_bias)

        eye = torch.eye(N, device=bias.device, dtype=torch.bool).view(1, N, N, 1)
        bias = bias.masked_fill(eye, 0.0)
        attn_bias = bias.permute(0, 3, 1, 2).reshape(B * H, N, N)

        with torch.no_grad():
            self._last_stats = {
                "p_sup_mean": float(p_sup.mean()),
                "bias_abs_mean": float(bias.abs().mean()),
                "bias_min": float(bias.min()),
                "bias_max": float(bias.max()),
            }
        if return_gates:
            return attn_bias, {"p_sup": p_sup, "S": s, "D": d}
        return attn_bias

    def get_stats(self) -> dict:
        stats = dict(self._last_stats)
        stats["tau"] = float(self.router_tau)
        return stats


def build_query_router(router_cfg: dict = None, **default_kwargs):
    cfg = dict(router_cfg or {})
    router_type = cfg.pop("type", "relation")
    if router_type != "relation":
        raise ValueError(f"Unknown router type: {router_type}")
    return RelationQueryRouter(**{**default_kwargs, **cfg})


def compute_pairwise_features(
    reference_boxes: torch.Tensor,
    prev_class_logits: torch.Tensor,
    batch_size: int,
    num_queries: int,
    like: torch.Tensor,
):
    """[B, N, N, 8]: IoU, dx, dy, log(w_j/w_i), log(h_j/h_i), class agreement, c_j - c_i, c_j."""
    B, N = batch_size, num_queries
    with torch.no_grad():
        feats = []
        if reference_boxes is not None and reference_boxes.shape[-1] >= 4:
            boxes = reference_boxes.detach().float()
            cx, cy = boxes[..., 0], boxes[..., 1]
            w, h = boxes[..., 2].clamp(min=1e-4), boxes[..., 3].clamp(min=1e-4)
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            iw = (torch.min(x1[:, :, None], x1[:, None, :]) - torch.max(x0[:, :, None], x0[:, None, :])).clamp(min=0)
            ih = (torch.min(y1[:, :, None], y1[:, None, :]) - torch.max(y0[:, :, None], y0[:, None, :])).clamp(min=0)
            inter = iw * ih
            area = w * h
            iou = inter / (area[:, :, None] + area[:, None, :] - inter).clamp(min=1e-8)
            dx = (cx[:, None, :] - cx[:, :, None]) / w[:, :, None]
            dy = (cy[:, None, :] - cy[:, :, None]) / h[:, :, None]
            dx = torch.sign(dx) * torch.log1p(dx.abs())
            dy = torch.sign(dy) * torch.log1p(dy.abs())
            dw = torch.log(w[:, None, :] / w[:, :, None])
            dh = torch.log(h[:, None, :] / h[:, :, None])
            feats += [iou, dx, dy, dw, dh]
        else:
            feats += [like.new_zeros(B, N, N, dtype=torch.float)] * 5

        if prev_class_logits is not None:
            probs = prev_class_logits.detach().float().sigmoid()
            conf = probs.max(dim=-1).values
            pn = torch.nn.functional.normalize(probs, dim=-1)
            cls_agree = torch.einsum("bnc,bmc->bnm", pn, pn)
            feats += [cls_agree, conf[:, None, :] - conf[:, :, None], conf[:, None, :].expand(B, N, N)]
        else:
            feats += [like.new_zeros(B, N, N, dtype=torch.float)] * 3

        return torch.stack(feats, dim=-1).to(like.dtype)
