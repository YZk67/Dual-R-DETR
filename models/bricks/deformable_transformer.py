import copy
import math

import torch
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.position_encoding import get_sine_pos_embed
from util.misc import inverse_sigmoid
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.query_router import QueryRouter, compute_query_features


class DeformableTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 300,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.pos_trans = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)

        # initialize pos_trans
        nn.init.xavier_uniform_(self.pos_trans.weight)

    def forward_encoder(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds):
        """
        Run encoder and prepare decoder inputs.
        
        Returns:
            encoder_outputs: dict containing all encoder outputs and decoder inputs
        """
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            query_key_padding_mask=mask_flatten,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # select topk
        topk = self.two_stage_num_proposals
        topk_index = torch.topk(enc_outputs_class[:, :, 0], topk, dim=1)[1].unsqueeze(-1)
        topk_enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get query(target) and reference points
        # NOTE: original implementation calculates query and query_pos together.
        # To keep the interface the same with Dab, DN and DINO, we split the
        # calculation of query_pos into the DeformableDecoder
        reference_points = topk_enc_outputs_coord.detach()
        # nn.Linear can not perceive the arrangement order of elements
        # so exchange_xy=True/False does not matter results
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        target = self.pos_trans_norm(self.pos_trans(query_sine_embed))

        return {
            "memory": memory,
            "mask_flatten": mask_flatten,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "target": target,
            "reference_points": reference_points,
            "enc_outputs_class": enc_outputs_class,
            "enc_outputs_coord": enc_outputs_coord,
        }

    def forward_decoder(self, encoder_outputs):
        """
        Run decoder using encoder outputs.
        
        Args:
            encoder_outputs: dict from forward_encoder()
            
        Returns:
            outputs_classes, outputs_coords: decoder outputs
        """
        outputs_classes, outputs_coords = self.decoder(
            query=encoder_outputs["target"],
            value=encoder_outputs["memory"],
            key_padding_mask=encoder_outputs["mask_flatten"],
            reference_points=encoder_outputs["reference_points"],
            spatial_shapes=encoder_outputs["spatial_shapes"],
            level_start_index=encoder_outputs["level_start_index"],
            valid_ratios=encoder_outputs["valid_ratios"],
        )
        
        return outputs_classes, outputs_coords

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
    ):
        """
        Complete forward pass (encoder + decoder).
        For efficiency, use forward_encoder + forward_decoder separately
        when running multiple decoder passes with the same encoder output.
        """
        # Run encoder
        encoder_outputs = self.forward_encoder(
            multi_level_feats, multi_level_masks, multi_level_pos_embeds
        )
        
        # Run decoder
        outputs_classes, outputs_coords = self.forward_decoder(encoder_outputs)

        return (
            outputs_classes,
            outputs_coords,
            encoder_outputs["enc_outputs_class"],
            encoder_outputs["enc_outputs_coord"],
        )


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        # NOTE: the ref_point_head of Deformable is split from pos_trans and pos_norm,
        # which is different from DINO
        self.ref_point_head = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim), nn.LayerNorm(self.embed_dim)
        )

        # iterative bounding box refinement heads (per decoder layer)
        class_head = nn.Linear(self.embed_dim, self.num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])

        # gating control (can be modified via set_gating)
        self.gating_lambda = 0.0
        self.gating_last_n_layers = 1

        # initialize decoder heads and related layers
        self.init_weights()

    @torch.no_grad()
    def get_router_stats(self) -> dict:
        """Return per-layer routing statistics for logging."""
        stats = {}
        # gating schedule controls
        stats["router/gating_lambda"] = float(self.gating_lambda)
        stats["router/gating_last_n_layers"] = float(self.gating_last_n_layers)

        for layer_idx, layer in enumerate(self.layers):
            # Get stats from QueryRouter if available
            if hasattr(layer, "query_router"):
                router_stats = layer.query_router.get_stats()
                for key, value in router_stats.items():
                    stats[f"router/{key}_l{layer_idx}"] = value

        return stats

    def set_router_tau(self, tau: float):
        """Set temperature for router selector across all layers."""
        tau_val = float(tau)
        for layer in self.layers:
            if hasattr(layer, "query_router"):
                layer.query_router.router_tau = tau_val

    def set_gating(self, lambda_value: float = 0.0, last_n_layers: int = 1):
        """Set runtime gating strength and how many last layers to apply it to."""
        self.gating_lambda = float(lambda_value) if lambda_value is not None else 0.0
        self.gating_last_n_layers = int(max(1, last_n_layers))

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

        # initialize ref_point_head
        nn.init.xavier_uniform_(self.ref_point_head[0].weight)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
        gating_lambda: float = None,
        gating_last_n_layers: int = None,
    ):
        # NOTE: the difference between DeformableDecoder and DabDecoder is that
        # Deformable does not introduce reference refinement for query pos
        query_sine_embed = get_sine_pos_embed(
            reference_points, self.embed_dim // 2, exchange_xy=False
        )
        query_pos = self.ref_point_head(query_sine_embed)

        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale

            # decide gating strength for this layer (only last n layers)
            # pick provided or fallback to attributes
            glambda = self.gating_lambda if gating_lambda is None else gating_lambda
            last_n = self.gating_last_n_layers if gating_last_n_layers is None else gating_last_n_layers
            use_gating_lambda = 0.0
            if glambda and glambda > 0.0 and (self.num_layers - 1 - layer_idx) < max(1, last_n):
                use_gating_lambda = glambda

            # previous layer class logits for target-aware gating (only if exists)
            prev_class_logits = outputs_classes[layer_idx - 1] if layer_idx > 0 else None

            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
                gating_lambda=use_gating_lambda,
                prev_class_logits=prev_class_logits,
                reference_boxes=reference_points,  # [B,N,4] for geometric features
            )

            # get output
            output_class = self.class_head[layer_idx](query)
            output_coord = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            reference_points = output_coord.detach()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
    ):
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )

        return query



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Query router: independent module for pairwise attention bias
        self.query_router = QueryRouter(
            embed_dim=embed_dim,
            router_rank=16,      # r in paper
            gate_rank=32,        # r_g in paper
            router_hidden=64,
            num_heads=n_heads,
        )

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize FFN layers
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
        gating_lambda: float = 0.0,
        prev_class_logits: torch.Tensor = None,
        reference_boxes: torch.Tensor = None,
    ):
        # Self-attention with optional query routing
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)

        attn_mask_aug = self_attn_mask
        if gating_lambda and gating_lambda > 0.0:
            # Compute query features and get routing bias
            similarity, confidence, geometry = compute_query_features(
                query=query_with_pos,
                prev_class_logits=prev_class_logits,
                reference_boxes=reference_boxes,
            )
            
            routing_bias = self.query_router(
                query=query_with_pos,
                similarity=similarity,
                confidence=confidence,
                geometry=geometry,
            )
            routing_bias = gating_lambda * routing_bias
            
            # Combine with base mask if exists
            B, N = query.shape[:2]
            if self_attn_mask is None:
                attn_mask_aug = routing_bias
            else:
                if self_attn_mask.dim() == 2:
                    base = self_attn_mask.unsqueeze(0).to(routing_bias.device, routing_bias.dtype)
                    base = base.expand(B * self.num_heads, -1, -1)
                    attn_mask_aug = base + routing_bias
                else:
                    attn_mask_aug = self_attn_mask.to(routing_bias.device, routing_bias.dtype) + routing_bias

        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=attn_mask_aug,
            need_weights=False,
        )[0]
        
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query
