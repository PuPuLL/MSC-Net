import math
import time
import torch
import torch.nn as nn
import numpy as np
from functools import partial
import sys

from models.modeling_finetune import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from models.modeling_msc_helper import *
from models.modeling_DCM import age_predictor
from tools.calculate import correlation_calculation, Symmetric_loss

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class MultiSampleComparison(nn.Module):
    def __init__(self, img_size=(264, 256), patch_size=(1, 256), in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_dim=2048, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None, init_std=0.02, 
                 decoder_embed_dim=512, predictor_depth=2, decoder_num_classes=264, decoder_num_heads=12,
                 decoder_layer_scale_init_value=0.1, decoder_depth=2, fix_init_weight=False, **kwargs):
        super().__init__()

        if kwargs['args'].predictor_depth != predictor_depth: predictor_depth = kwargs['args'].predictor_depth
        if kwargs['args'].decoder_embed_dim != decoder_embed_dim: decoder_embed_dim = kwargs['args'].decoder_embed_dim
        if kwargs['args'].decoder_depth != decoder_depth: decoder_depth = kwargs['args'].decoder_depth
        print("predictor_depth: ", predictor_depth)
        print("decoder_embed_dim: ", decoder_embed_dim)
        print("decoder_depth: ", decoder_depth)

        self.batch_size = kwargs['args'].batch_size
        self.length = kwargs['args'].length

        self.encoder = SiameseEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                 embed_dim=embed_dim, depth=depth,num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)

        self.alignment_encoder = SiameseEncoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim, init_std=init_std)

        self.init_std = init_std
        self.num_patches = self.encoder.patch_embed.num_patches

        # from encoder to regresser projection, borrowed from mae.
        if decoder_embed_dim != embed_dim:
            self.encoder_to_predicotr = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.encoder_to_predicotr_norm = norm_layer(decoder_embed_dim)
        else:
            self.encoder_to_predicotr = None

        # generate position embeddings for regresser and deocder (rd) .
        self.rd_pos_embed = self.encoder.build_2d_sincos_position_embedding(decoder_embed_dim, use_cls_token=True)
        
        # predictor 
        self.predictor = LatentPredictor(embed_dim=decoder_embed_dim, predictor_depth=predictor_depth, num_heads=decoder_num_heads,
                                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                        drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std)


        # predictor is cross attention, mask tokens are querries.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=self.init_std)

        self.NetworkReconstructor = NetworkReconstructor(num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
                            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                            norm_layer=norm_layer, init_values=decoder_layer_scale_init_value, init_std=init_std)


        # age predictor with gradient
        self.encode_age_latent = age_predictor(embed_dim, self.batch_size)
        # age predictor without gradient
        self.encode_age_target = age_predictor(embed_dim, self.batch_size)

        self.symmetirc_loss = Symmetric_loss(self.batch_size, self.length)

        ### whether to use 'rescale' to init the weight, borrowed from beit.
        if not fix_init_weight:
            self.apply(self._init_weights)

        self._init_alignment_encoder()
        self._init_age_encoder()
        
    def _init_alignment_encoder(self):
        # init the weights of alignment_encoder with those of backbone
        for param_encoder, param_alignment_encoder in zip(self.encoder.parameters(), self.alignment_encoder.parameters()):
            param_alignment_encoder.detach()
            param_alignment_encoder.data.copy_(param_encoder.data)
            param_alignment_encoder.requires_grad = False

    def alignment_parameter_update(self):
        """parameter update of the alignment_encoder network."""
        for param_encoder, param_alignment_encoder in zip(self.encoder.parameters(),
                                                          self.alignment_encoder.parameters()):
            param_alignment_encoder.data = param_encoder.data # completely copy


    def _init_age_encoder(self):
        # init the weights of alignment_encoder with those of backbone
        for param_encoder_latent, param_encoder_target in zip(self.encode_age_latent.parameters(),
                                                              self.encode_age_target.parameters()):
            param_encoder_target.detach()
            param_encoder_target.data.copy_(param_encoder_latent.data)
            param_encoder_target.requires_grad = False

    def age_parameter_update(self):
        """parameter update of the alignment_encoder network."""
        for param_encoder_latent, param_encoder_target in zip(self.encode_age_latent.parameters(),
                                                              self.encode_age_target.parameters()):
            param_encoder_target.data = param_encoder_latent.data # completely copy

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    '''
    Input shape:
        x: [bs, 1, ROI_num, ts_length]
        bool_masked_pos: [bs, num_patch * num_patch]
    '''

    def forward_ts(self, x, h_label, bool_masked_pos):

        batch_size, C, seq_len, n_vars = x.size()

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''
        x_unmasked = x * ~bool_masked_pos.unsqueeze(1)

        x_unmasked = self.encoder(x_unmasked)

        # encoder to regresser projection
        if self.encoder_to_predicotr is not None:
            x_unmasked = self.encoder_to_predicotr(x_unmasked)
            x_unmasked = self.encoder_to_predicotr_norm(x_unmasked)

        '''
        Alignment branch
        '''
        with torch.no_grad():
            latent_target = self.alignment_encoder(x * bool_masked_pos.unsqueeze(1))
            latent_target = latent_target[:, 1:, :]  # remove class token
            if self.encoder_to_predicotr is not None:
                latent_target = self.encoder_to_predictor_norm(self.encoder_to_predicotr(latent_target.detach()))

            self.alignment_parameter_update()

        '''
        Latent predictor
        1. prepare masked, unmasked pos embed, and masked embedding
        '''
        _, num_visible_plus1, dim = x_unmasked.shape

        x_unmasked = x_unmasked[:, 1:, :]  # remove class token

        pos_embed = self.rd_pos_embed.expand(batch_size, self.num_patches + 1, dim).cuda(x_unmasked.device)
        pos_embed_masked = pos_embed[:, 1:, :].expand(batch_size, self.num_patches, dim).cuda(
            x_unmasked.device)

        x_masked = self.mask_token.expand(batch_size, self.num_patches, -1)
        '''
        2. predictor masked latent via predictor
        '''
        x_masked_predict = self.predictor(x_masked, x_unmasked, pos_embed_masked)

        # preserve for alignment
        latent_predict = x_masked_predict

        '''
        Label Embedding with DCM
        '''
        latent_age_predicts = self.encode_age_latent(latent_predict)
        latent_items_predicts = [latent_predict, latent_age_predicts]

        with torch.no_grad():
            latent_age_targets = self.encode_age_latent(latent_target)
            latent_items_targets = [latent_target, latent_age_targets]

            self.age_parameter_update()

        '''
        NetworkReconstructor for reconstruction
        '''
        logits = self.NetworkReconstructor(latent_predict, pos_embed_masked)
        logits = logits.view(-1, logits.shape[2])  # flatten

        fc_ori = correlation_calculation(x * bool_masked_pos.unsqueeze(1))
        loss = self.symmetirc_loss(h_label, latent_items_predicts, latent_items_targets, logits, fc_ori)
        return logits, loss

    def forward(self, x, h_label, bool_masked_pos):
        logits, loss = self.forward_ts(x, h_label, bool_masked_pos)
        return logits, loss



@register_model
def msc_base_patch_256(pretrained=False, **kwargs):
    model = MultiSampleComparison(
        img_size=(264, 256), patch_size=(1, 256), embed_dim=768, depth=2, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def msc_base_patch_145(pretrained=False, **kwargs):
    model = MultiSampleComparison(
        img_size=(264, 145), patch_size=(1, 145), embed_dim=768, depth=2, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def msc_base_patch_315(pretrained=False, **kwargs):
    model = MultiSampleComparison(
        img_size=(264, 315), patch_size=(1, 315), embed_dim=768, depth=2, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model