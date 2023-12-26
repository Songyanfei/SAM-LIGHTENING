# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
from collections import OrderedDict
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_vit_b1(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        rel = False,
    )
    
def build_sam_vit_b_dilated(checkpoint=None):
    return _build_dilated_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        rel = False,
    )


def build_sam_vit_tiny_dilated(checkpoint=None):
    return _build_dilated_sam(
        encoder_embed_dim=768,
        encoder_depth=6,
        encoder_num_heads=6,
        encoder_global_attn_indexes=[2, 5],
        checkpoint=checkpoint,
        rel = False,
    )

def build_sam_vit_tiny2_dilated(checkpoint=None):
    return _build_dilated_sam(
        encoder_embed_dim=512,
        encoder_depth=6,
        encoder_num_heads=6,
        encoder_global_attn_indexes=[2, 5],
        checkpoint=checkpoint,
        rel = False,
    )    

def build_DilatedSAM_t_np(checkpoint=None):
    return _build_dilatedSAM(
        encoder_embed_dim=384,
        encoder_depth=9,
        encoder_num_heads=8,
        use_flash= True,
        # encoder_global_attn_indexes=[0,1,2,3,4,5],
        encoder_global_attn_indexes=[range(9)],
        checkpoint=checkpoint,
        rel = False,
    )      
    
def build_DilatedSAM_t_np2(checkpoint=None):
    return _build_dilatedSAM(
        encoder_embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,
        use_flash= True,
        encoder_global_attn_indexes=[range(6)],
        checkpoint=checkpoint,
        rel = False,
    )       
def build_DilatedSAM_t(checkpoint=None, use_flash= True):
    return _build_dilatedSAM(
        encoder_embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,
        use_flash= use_flash,
        encoder_global_attn_indexes=[2, 5],
        checkpoint=checkpoint,
        rel = False,
    )        


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_b_dilated": build_sam_vit_b_dilated,
    "vit_t_dilated": build_sam_vit_tiny_dilated,
    "vit_t_dilated2": build_sam_vit_tiny_dilated,
    "DilatedSAM_vit_t_np": build_DilatedSAM_t_np,
    "DilatedSAM_vit_t_np2": build_DilatedSAM_t_np2,
    "DilatedSAM_vit_t": build_DilatedSAM_t,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    rel = True,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=rel,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=16,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


def _build_dilated_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    use_flash,
    encoder_global_attn_indexes,
    checkpoint=None,
    rel = True,
):
    prompt_embed_dim = 256
    image_size = 4096
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=rel,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            use_dilated=True,
            use_flash = use_flash
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        # 过滤掉不匹配的键
        model_dict = sam.state_dict()
        # filtered_state_dict = {k: v for k, v in state_dict.items() if k in sam.state_dict()}
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

        # 加载过滤后的状态字典
        # sam.load_state_dict(filtered_state_dict, strict=False)
        
        # 加载过滤后的状态字典
        model_dict.update(filtered_state_dict)
        sam.load_state_dict(model_dict, strict=False)

        # sam.load_state_dict(state_dict)
    return sam



def _build_dilatedSAM(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    use_flash,
    encoder_global_attn_indexes,
    checkpoint=None,
    rel = True,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=rel,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
            use_dilated=True,
            use_flash = use_flash
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        print(f"check point:{checkpoint}")
        if isinstance(checkpoint, OrderedDict):
            # 如果checkpoint是OrderedDict，直接使用它加载模型状态
            sam.load_state_dict(checkpoint)
        else:
            # 否则，认为checkpoint是文件路径，从文件中加载模型状态
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f)
                sam.load_state_dict(state_dict)
        return sam
        
        
    # if checkpoint is not None:
    #     with open(checkpoint, "rb") as f:
    #         state_dict = torch.load(f)

    #     # 过滤掉不匹配的键
    #     model_dict = sam.state_dict()
    #     # filtered_state_dict = {k: v for k, v in state_dict.items() if k in sam.state_dict()}
    #     filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

    #     # 加载过滤后的状态字典
    #     # sam.load_state_dict(filtered_state_dict, strict=False)
        
    #     # 加载过滤后的状态字典
    #     model_dict.update(filtered_state_dict)
    #     sam.load_state_dict(model_dict, strict=False)

    #     # sam.load_state_dict(state_dict)
    # return sam
