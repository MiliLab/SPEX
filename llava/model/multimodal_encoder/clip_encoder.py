import torch
import torch.nn as nn
from llava.utils import rank0_print
from transformers import CLIPImageProcessor, CLIPVisionConfig
from .modeling_clip import CLIPVisionModel
from transformers import SamConfig
from transformers.models.sam.modeling_sam import SamMaskDecoder,SamPositionalEmbedding,SamPromptEncoder 
from .neck import LLAMANECK
import torch.nn.functional as F
import math


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        rank0_print("#######################################Load-Vision-InternImage-model##################################")
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.sam_path = "./pertrained_weights/segment_anything"
        self.lan_neck = LLAMANECK()
        self.sam_config = SamConfig.from_pretrained(self.sam_path).mask_decoder_config
        self.vision_config = SamConfig.from_pretrained(self.sam_path).vision_config
        self.sampromptencoder = SamPromptEncoder(SamConfig.from_pretrained(self.sam_path).prompt_encoder_config,SamPositionalEmbedding(self.vision_config))
        #model
        self.mask_decoder = SamMaskDecoder(self.sam_config)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=-1)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.cat([image_forward_outs.hidden_states[i] for i in select_layers], dim=-1)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if select_feature_type == "patch":
            image_features = image_features[:, 1:]
        elif select_feature_type == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            llm_feat, Sam_Feat = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # image_features = self.feature_select(image_forward_outs).to(images.dtype)
            image_features = llm_feat.to(images.dtype)
        return image_features,Sam_Feat
    
    def get_image_wide_positional_embeddings(self, size):
        target_device = self.sampromptencoder.shared_embedding.positional_embedding.device
        target_dtype = self.sampromptencoder.shared_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.sampromptencoder.shared_embedding(torch.stack([x_embed, y_embed], dim=-1))

        return positional_embedding.permute(2, 0, 1).unsqueeze(0)
    #######################################################dice bce loss#########################################
    def dice_loss(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                avg_factor:int,
                eps: float = 1e-3) -> float:
        # import pdb;pdb.set_trace()
        pred = pred.sigmoid()
        inputs = pred.flatten(1)
        target = target.flatten(1).float()
        a = torch.sum(inputs * target, 1)
        b = torch.sum(inputs * inputs, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)
        loss = 1 - d
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)

        return loss
    
    def focal_loss(self,
                    inputs,
                    targets,
                    num_objects,
                    alpha:float=0.25,
                    gamma:float=2,
                    loss_on_multimask=False,
                    ):
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs,targets)
        p_t = prob * targets + (1-prob) * (1-targets)
        loss = ce_loss * ((1 - p_t) **gamma)

        if alpha>0:
        
            alpha_t = alpha * targets + (1-alpha) * (1-targets)
            loss = alpha_t * loss
        
        return loss.mean()
       
    def forward_mask_decoder(self,
                            multi_layers_features,
                            decoder_lan_hidden,
                            gt_semantic_seg):
        self.dice_loss_weight = 1.0
        self.bce_loss_weight = 2.0
        img_bs,C,H,W = multi_layers_features[0].shape
        image_feature = multi_layers_features[1]
        sparse_prompt_embeddings = (self.lan_neck(decoder_lan_hidden).unsqueeze(dim=1))
        # breakpoint()
        #######################################sam model process loss########################################
        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_feature.shape[-1])
        image_positional_embeddings = image_positional_embeddings.repeat(img_bs, 1, 1, 1)
        dense_embeddings = self.sampromptencoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs, -1, image_feature.shape[2], image_feature.shape[3])
        sam_mask_pred = self.mask_decoder(
                        image_embeddings = image_feature,
                        image_positional_embeddings = image_positional_embeddings,
                        sparse_prompt_embeddings = sparse_prompt_embeddings,
                        dense_prompt_embeddings = dense_embeddings,
                        multimask_output = False,
                        output_attentions = False,
                        attention_similarity = False,
                        target_embedding = None , 
            )
        sam_mask_pred = sam_mask_pred[0].squeeze(dim=1)
        sam_masks = F.interpolate(sam_mask_pred.float(),size=(H,W),mode="bilinear",align_corners=False)
        sam_masks = sam_masks[..., : H, : W]
        sam_masks = F.interpolate(sam_masks, (H,W), mode="bilinear", align_corners=False)
        mask_ce_loss = self.bce_loss_weight * F.binary_cross_entropy_with_logits(sam_masks.squeeze(dim=1),gt_semantic_seg.float())
        mask_dice_loss = self.dice_loss_weight * self.dice_loss(sam_masks.squeeze(dim=1), gt_semantic_seg.float(), avg_factor=sam_mask_pred.shape[0])
 
        return {"mask_ce_loss":mask_ce_loss,"mask_dice_loss":mask_dice_loss}
        
    def forward_inference(self,
                        multi_layers_features,
                        decoder_lan_hidden,
                        mask_threshold=0.5
                        ):
        output = {}
        img_bs,C,H,W = multi_layers_features[0].shape
        image_feature = multi_layers_features[1]
        sparse_prompt_embeddings = (self.lan_neck(decoder_lan_hidden.squeeze(dim=0)).unsqueeze(dim=1))
        print(sparse_prompt_embeddings.shape)
        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_feature.shape[-1])
        image_positional_embeddings = image_positional_embeddings.repeat(img_bs, 1, 1, 1)
        dense_embeddings = self.sampromptencoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(img_bs, -1, image_feature.shape[2], image_feature.shape[3])
        sam_mask_pred = self.mask_decoder(
                        image_embeddings = image_feature,
                        image_positional_embeddings = image_positional_embeddings,
                        sparse_prompt_embeddings = sparse_prompt_embeddings,
                        dense_prompt_embeddings = dense_embeddings,
                        multimask_output = False,
                        output_attentions = False,
                        attention_similarity = False,
                        target_embedding = None , 
            )
        sam_mask_pred = (sam_mask_pred[0]>0).squeeze(dim=1)
        sam_masks = F.interpolate((sam_mask_pred).float(),size=(H,W),mode="bilinear",align_corners=False)
        sam_masks = sam_masks[..., : H, : W]
        sam_masks = F.interpolate(sam_masks, (H,W), mode="bilinear", align_corners=False)
        output["sam_pred"] = (sam_masks).squeeze(dim=0)
        return output

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        if "slicefour" in self.select_feature:
            _hidden_size *= 4
        if "slice_m25811_f6" in self.select_feature:
            _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size
