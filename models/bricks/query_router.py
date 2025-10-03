import torch
from torch import nn


class QueryRouter(nn.Module):
    """
    Pairwise Query Router with Low-Rank Biases and Competition-Aware Gating.
    
    This module can be integrated into any self-attention mechanism to provide
    query-specific attention bias based on similarity, confidence, and geometric features.
    
    Args:
        embed_dim: Dimension of input query embeddings
        router_rank: Rank for low-rank factorization of pairwise biases (r in paper)
        gate_rank: Rank for pairwise gating mechanism (r_g in paper)
        router_hidden: Hidden dimension for query representation network
        num_heads: Number of attention heads (for output shape)
    
    Example:
        >>> router = QueryRouter(embed_dim=256, router_rank=16, gate_rank=32, num_heads=8)
        >>> query = torch.randn(2, 100, 256)  # [B, N, D]
        >>> attn_bias = router(query, similarity=sim, confidence=conf, geometry=geom)
        >>> # attn_bias shape: [B*H, N, N] ready to add to attention logits
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        router_rank: int = 16,
        gate_rank: int = 32,
        router_hidden: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.router_rank = router_rank
        self.gate_rank = gate_rank
        self.num_heads = num_heads
        
        # Query representation network: φ(q) -> z ∈ R^{router_rank}
        self.router_phi = nn.Sequential(
            nn.Linear(embed_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, router_rank),
        )
        
        # Low-rank bias matrices for two routes
        # Suppressor route: negative bias to reduce attention
        self.router_U_sup = nn.Linear(router_rank, router_rank, bias=False)
        self.router_V_sup = nn.Linear(router_rank, router_rank, bias=False)
        # Delegator route: positive bias to encourage exploration
        self.router_U_del = nn.Linear(router_rank, router_rank, bias=False)
        self.router_V_del = nn.Linear(router_rank, router_rank, bias=False)
        
        # Pairwise gating: bilinear interaction a_i^T b_j
        # Input: [similarity, confidence, geometry] -> 3 dimensions
        self.selector_a = nn.Linear(3, gate_rank, bias=False)  # W_a
        self.selector_b = nn.Linear(3, gate_rank, bias=False)  # W_b
        
        # Route scales (learnable, with softplus reparameterization)
        # sup is always negative, del is always positive
        self.router_gamma_sup = nn.Parameter(torch.tensor(1.0))
        self.router_gamma_del = nn.Parameter(torch.tensor(1.0))
        
        # Temperature for gating (can be annealed externally)
        self.router_tau = 1.0
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize all router parameters."""
        # Initialize query representation network
        for m in self.router_phi:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        # Initialize low-rank matrices
        for lin in [self.router_U_sup, self.router_V_sup, 
                    self.router_U_del, self.router_V_del]:
            nn.init.xavier_uniform_(lin.weight)
        
        # Initialize pairwise gating
        nn.init.xavier_uniform_(self.selector_a.weight)
        nn.init.xavier_uniform_(self.selector_b.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        similarity: torch.Tensor = None,
        confidence: torch.Tensor = None,
        geometry: torch.Tensor = None,
        return_gates: bool = False,
    ):
        """
        Compute pairwise attention bias for queries.
        
        Args:
            query: Query embeddings [B, N, D]
            similarity: Per-query similarity features [B, N] (optional, will compute if None)
            confidence: Per-query confidence scores [B, N] (optional, defaults to 0)
            geometry: Per-query geometric features [B, N] (optional, defaults to 0)
            return_gates: If True, return gating probabilities for analysis
        
        Returns:
            attn_bias: Attention bias [B*H, N, N] to add to attention logits
            gates (optional): Dict with p_sup and p_del if return_gates=True
        """
        B, N, D = query.shape
        rank = self.router_rank
        scale = (rank ** -0.5)
        
        # Compute similarity if not provided
        if similarity is None:
            with torch.no_grad():
                qn = torch.nn.functional.normalize(query, dim=-1)
                sim = torch.einsum("bne,bme->bnm", qn, qn)
                sim_pos = torch.relu(sim)
                # Remove diagonal self-similarity
                sim_pos = sim_pos - torch.diag_embed(torch.diagonal(sim_pos, dim1=1, dim2=2))
                similarity = sim_pos.mean(dim=-1)  # [B, N]
        
        # Default to zeros if not provided
        if confidence is None:
            confidence = torch.zeros_like(similarity)
        if geometry is None:
            geometry = torch.zeros_like(similarity)
        
        # Pairwise gating: x_i = [s_i, c_i, g_i]
        # a_i = x_i W_a, b_j = x_j W_b ∈ R^{gate_rank}
        # p_sup(i,j) = sigmoid(a_i^T b_j)
        x_i = torch.stack([similarity, confidence, geometry], dim=-1)  # [B, N, 3]
        a_i = self.selector_a(x_i)  # [B, N, r_g]
        b_j = self.selector_b(x_i)  # [B, N, r_g]
        
        # Bilinear interaction: [B, N, r_g] @ [B, r_g, N] -> [B, N, N]
        pairwise_logits = torch.einsum("bnr,bmr->bnm", a_i, b_j)
        p_sup_pairwise = torch.sigmoid(pairwise_logits)  # [B, N, N]
        p_del_pairwise = 1.0 - p_sup_pairwise  # [B, N, N]
        
        # Low-rank pairwise biases per route
        z_q = self.router_phi(query)  # [B, N, R]
        z_k = z_q  # Self-attention: K == Q
        
        u_sup = self.router_U_sup(z_q)  # [B, N, R]
        v_sup = self.router_V_sup(z_k)  # [B, N, R]
        u_del = self.router_U_del(z_q)  # [B, N, R]
        v_del = self.router_V_del(z_k)  # [B, N, R]
        
        # Signed reparameterization: sup is always negative, del is always positive
        gamma_sup_eff = -torch.nn.functional.softplus(self.router_gamma_sup)
        gamma_del_eff = torch.nn.functional.softplus(self.router_gamma_del)
        
        # Δ^sup = γ_sup * U^sup @ (V^sup)^T, Δ^del = γ_del * U^del @ (V^del)^T
        delta_sup = gamma_sup_eff * scale * torch.einsum("bnr,bmr->bnm", u_sup, v_sup)
        delta_del = gamma_del_eff * scale * torch.einsum("bnr,bmr->bnm", u_del, v_del)
        
        # Pairwise-routed bias: B = p_sup(i,j) ⊙ (γ_sup Δ^sup) + p_del(i,j) ⊙ (γ_del Δ^del)
        bias = p_sup_pairwise * delta_sup + p_del_pairwise * delta_del  # [B, N, N]
        bias = bias.clamp(min=-5.0, max=5.0)
        
        # Expand to multi-head format: [B*H, N, N]
        H = self.num_heads
        attn_bias = bias.unsqueeze(0).expand(H, -1, -1, -1)  # [H, B, N, N]
        attn_bias = attn_bias.permute(1, 0, 2, 3).reshape(B * H, N, N)  # [B*H, N, N]
        
        if return_gates:
            return attn_bias, {
                'p_sup': p_sup_pairwise,
                'p_del': p_del_pairwise,
                'gamma_sup': gamma_sup_eff,
                'gamma_del': gamma_del_eff,
            }
        
        return attn_bias
    
    def get_stats(self) -> dict:
        """Return routing statistics for logging."""
        stats = {}
        gamma_sup_eff = -torch.nn.functional.softplus(self.router_gamma_sup).item()
        gamma_del_eff = torch.nn.functional.softplus(self.router_gamma_del).item()
        stats['gamma_sup_eff'] = float(gamma_sup_eff)
        stats['gamma_del_eff'] = float(gamma_del_eff)
        stats['tau'] = float(self.router_tau)
        return stats


def compute_query_features(
    query: torch.Tensor,
    prev_class_logits: torch.Tensor = None,
    reference_boxes: torch.Tensor = None,
):
    """
    Helper function to compute per-query features for routing.
    
    Args:
        query: Query embeddings [B, N, D]
        prev_class_logits: Previous layer classification logits [B, N, num_classes]
        reference_boxes: Bounding boxes [B, N, 4] in format [cx, cy, w, h]
    
    Returns:
        similarity: Per-query similarity scores [B, N]
        confidence: Per-query confidence scores [B, N]
        geometry: Per-query geometric features [B, N]
    """
    B, N, D = query.shape
    
    # Compute similarity
    with torch.no_grad():
        qn = torch.nn.functional.normalize(query, dim=-1)
        sim = torch.einsum("bne,bme->bnm", qn, qn)
        sim_pos = torch.relu(sim)
        # Remove diagonal self-similarity
        sim_pos = sim_pos - torch.diag_embed(torch.diagonal(sim_pos, dim1=1, dim2=2))
        similarity = sim_pos.mean(dim=-1)  # [B, N]
    
    # Compute confidence
    if prev_class_logits is not None:
        with torch.no_grad():
            probs = torch.softmax(prev_class_logits.detach(), dim=-1)
            confidence = probs.max(dim=-1).values  # [B, N]
    else:
        confidence = torch.zeros_like(similarity)
    
    # Compute geometry (log-scale area)
    if reference_boxes is not None and reference_boxes.shape[-1] >= 4:
        with torch.no_grad():
            w = reference_boxes[..., 2]  # [B, N]
            h = reference_boxes[..., 3]  # [B, N]
            area = w * h  # [B, N]
            geometry = torch.log(area.clamp(min=1e-6))  # [B, N]
    else:
        geometry = torch.zeros_like(similarity)
    
    return similarity, confidence, geometry

