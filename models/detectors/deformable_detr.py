from typing import Dict, List

from torch import Tensor, nn

from models.detectors.base_detector import DETRDetector


class DeformableDETR(DETRDetector):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        num_classes: int,
        min_size: int = None,
        max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # NOTE: we only suppoert DeformableDETR with two-stage and box refinement

        # define model parameters
        self.num_classes = num_classes

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        original_image_sizes = self.query_original_sizes(images)
        images, targets, mask = self.preprocess(images, targets)

        # get multi-level features, masks, and pos_embeds
        multi_levels = self.get_multi_levels(images, mask)
        multi_level_feats, multi_level_masks, multi_level_pos_embeds = multi_levels

        # Read current decoder gating schedule (set by training loop)
        decoder = getattr(self.transformer, "decoder", None)
        gating_lambda_sched = 0.0
        gating_last_n_layers = 1
        if decoder is not None and hasattr(decoder, "gating_lambda"):
            gating_lambda_sched = float(decoder.gating_lambda)
            gating_last_n_layers = int(getattr(decoder, "gating_last_n_layers", 1))

        if self.training:
            # Dual-branch training: run encoder once, decoder twice with different gating
            # Step 1: Run encoder once (shared by both branches)
            encoder_outputs = self.transformer.forward_encoder(
                multi_level_feats, multi_level_masks, multi_level_pos_embeds
            )
            enc_class = encoder_outputs["enc_outputs_class"]
            enc_coord = encoder_outputs["enc_outputs_coord"]

            # Step 2: Run decoder with MAIN branch (gating OFF)
            if decoder is not None and hasattr(decoder, "set_gating"):
                decoder.set_gating(lambda_value=0.0, last_n_layers=gating_last_n_layers)
            outputs_class_main, outputs_coord_main = self.transformer.forward_decoder(encoder_outputs)
            output_main = {
                "pred_logits": outputs_class_main[-1],
                "pred_boxes": outputs_coord_main[-1],
                "aux_outputs": self._set_aux_loss(outputs_class_main, outputs_coord_main),
                "enc_outputs": {"pred_logits": enc_class, "pred_boxes": enc_coord},
            }

            # Step 3: Run decoder with AUX branch (gating ON)
            if decoder is not None and hasattr(decoder, "set_gating"):
                decoder.set_gating(lambda_value=gating_lambda_sched, last_n_layers=gating_last_n_layers)
            outputs_class_aux, outputs_coord_aux = self.transformer.forward_decoder(encoder_outputs)
            output_aux = {
                "pred_logits": outputs_class_aux[-1],
                "pred_boxes": outputs_coord_aux[-1],
                "aux_outputs": self._set_aux_loss(outputs_class_aux, outputs_coord_aux),
            }  # note: no enc_outputs on aux (encoder shared with main)

            # compute individual losses (unweighted)
            loss_main = self.criterion(output_main, targets)
            loss_aux = self.criterion(output_aux, targets)

            # combine with alpha (read from transformer if provided, else fallback)
            alpha_aux = 0.5
            if hasattr(self.transformer, "aux_alpha"):
                try:
                    alpha_aux = float(self.transformer.aux_alpha)
                except Exception:
                    pass
            combined = {}
            keys = set(loss_main.keys()) | set(loss_aux.keys())
            for k in keys:
                v_main = loss_main.get(k)
                v_aux = loss_aux.get(k)
                if v_main is not None and v_aux is not None:
                    combined[k] = v_main + alpha_aux * v_aux
                elif v_main is not None:
                    combined[k] = v_main
                else:
                    combined[k] = alpha_aux * v_aux

            # apply weight_dict reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = {k: combined[k] * weight_dict[k] for k in combined.keys() if k in weight_dict}

            # restore decoder gating to scheduled value for downstream consistency
            if decoder is not None and hasattr(decoder, "set_gating"):
                decoder.set_gating(lambda_value=gating_lambda_sched, last_n_layers=gating_last_n_layers)
            return loss_dict

        # Inference path (single branch, gating as-is)
        outputs_class, outputs_coord, enc_class, enc_coord = self.transformer(
            multi_level_feats, multi_level_masks, multi_level_pos_embeds
        )
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        detections = self.postprocessor(output, original_image_sizes)
        return detections
