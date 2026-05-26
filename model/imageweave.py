import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.train_config import CONFIG

from model.vision_encoder import VisionEncoder
from model.qformer import QFormer
from model.cross_attention import CrossImageAttention
from model.story_pooling import StoryPooling
from model.story_head import StoryHead
from model.text_encoder import TextEncoder


class ImageWeave(nn.Module):
    """
    ImageWeave: Sequential Multimodal Retrieval Model.

    Ablation flags (read from config dict):
        use_qformer           : If False, CLIP patch tokens are mean-pooled directly.
        use_cross_attention   : If False, CrossImageAttention is skipped (identity).
        use_memory            : If False, memory embedding is zeroed (query-only).
        use_attention_pooling : If False, StoryHead (mean pool) replaces StoryPooling.
        freeze_clip           : If True, freeze CLIP vision + text backbones fully.
    """

    def __init__(self, config=None):

        super().__init__()

        # Support passing config explicitly (for evaluate.py checkpoint loading).
        cfg = config if config is not None else CONFIG

        self.use_qformer            = cfg.get("use_qformer",           True)
        self.use_cross_attention    = cfg.get("use_cross_attention",    True)
        self.use_memory             = cfg.get("use_memory",             True)
        self.use_attention_pooling  = cfg.get("use_attention_pooling",  True)
        freeze_clip                 = cfg.get("freeze_clip",            True)

        print(f"\n[ImageWeave] Config:")
        print(f"  use_qformer           = {self.use_qformer}")
        print(f"  use_cross_attention   = {self.use_cross_attention}")
        print(f"  use_memory            = {self.use_memory}")
        print(f"  use_attention_pooling = {self.use_attention_pooling}")
        print(f"  freeze_clip           = {freeze_clip}")
        print()

        # =================================================
        # VISION ENCODER (CLIP ViT-B/32)
        # =================================================

        self.vision_encoder = VisionEncoder()

        # Freeze entire CLIP vision backbone.
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        if not freeze_clip:
            # Optionally unfreeze only the final CLIP transformer block.
            try:
                last_layer = (
                    self.vision_encoder
                    .encoder
                    .vision_model
                    .encoder
                    .layers[-1]
                )
                for param in last_layer.parameters():
                    param.requires_grad = True
                print("[ImageWeave] Unfroze CLIP final block.")
            except Exception as e:
                print(f"[ImageWeave] WARNING: Could not unfreeze CLIP final block: {e}")

        # =================================================
        # QFORMER (or simple projection when disabled)
        # =================================================

        if self.use_qformer:
            self.qformer = QFormer()
        else:
            # Ablation: replace QFormer with simple mean-pool projection.
            # Maps [B, T, N_patches, 768] → [B, T, 1, 768] via mean over patches.
            # No learnable temporal interaction.
            self.qformer = None
            self.patch_projection = nn.Linear(768, 768)

        # =================================================
        # CROSS-IMAGE ATTENTION
        # =================================================

        if self.use_cross_attention:
            self.cross_attention = CrossImageAttention(
                hidden_dim=768,
                num_heads=8,
                num_layers=2,
                dropout=0.3
            )
        else:
            self.cross_attention = None

        # =================================================
        # MEMORY PROJECTION
        # =================================================

        self.memory_projection = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, 768)
        )

        self.memory_dropout = nn.Dropout(0.3)

        # =================================================
        # STORY POOLING (attention) OR STORY HEAD (mean)
        # =================================================

        if self.use_attention_pooling:
            self.story_pooling = StoryPooling()
        else:
            self.story_pooling = StoryHead()

        # =================================================
        # TEXT ENCODER (frozen CLIP text)
        # =================================================

        self.text_encoder = TextEncoder()

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        query_pixel_values,
        memory_pixel_values,
        texts
    ):

        batch_size = query_pixel_values.shape[0]

        # =================================================
        # QUERY ENCODING
        # =================================================

        query_tokens = self.vision_encoder(query_pixel_values)
        # query_tokens: [B, T, N_patches, 768]

        if self.use_qformer:
            query_tokens = self.qformer(query_tokens)
            # query_tokens: [B, T, num_queries=8, 768]

            if self.use_cross_attention:
                query_tokens = self.cross_attention(query_tokens)
                # query_tokens: [B, T, num_queries=8, 768]
        else:
            # Ablation (no QFormer): mean-pool over patches, project.
            # [B, T, N_patches, 768] → [B, T, 1, 768]
            query_tokens = query_tokens.mean(dim=2, keepdim=True)
            query_tokens = self.patch_projection(query_tokens)
            # Cross-attention requires QFormer queries, skip.

        # =================================================
        # MEMORY ENCODING
        # =================================================

        # memory_pixel_values: [B, 4, 3, 224, 224]
        memory_pixel_values = memory_pixel_values.view(
            batch_size * 4,
            1,
            3,
            224,
            224
        )

        # [B*4, 1, N_patches, 768]
        memory_tokens = self.vision_encoder(memory_pixel_values)

        if self.use_qformer:
            # [B*4, 1, num_queries, 768]
            memory_tokens = self.qformer(memory_tokens)
            # Squeeze T=1 dim → [B*4, num_queries, 768]
            memory_tokens = memory_tokens.squeeze(1)
            # Mean over queries → [B*4, 768]
            memory_tokens = memory_tokens.mean(dim=1)
        else:
            # [B*4, 1, N_patches, 768] → squeeze → [B*4, N_patches, 768] → mean → [B*4, 768]
            memory_tokens = memory_tokens.squeeze(1).mean(dim=1)

        # Project memory → [B*4, 768]
        memory_tokens = self.memory_projection(memory_tokens)

        # [B*4, 768] → [B, 4, 768]
        memory_tokens = memory_tokens.view(batch_size, 4, 768)

        # [B, 4, 768] → [B, 768]
        memory_embedding = memory_tokens.mean(dim=1)
        memory_embedding = self.memory_dropout(memory_embedding)

        # =================================================
        # MEMORY FUSION
        # =================================================

        if self.use_memory:
            # [B, 768] → [B, 1, 1, 768] for broadcast over [B, T, Q, 768]
            memory_embedding_expanded = memory_embedding.unsqueeze(1).unsqueeze(2)
            query_tokens = query_tokens + memory_embedding_expanded
        # else: query_tokens unchanged — memory is zeroed implicitly (not added)

        # =================================================
        # STORY POOLING
        # =================================================

        image_embeddings = self.story_pooling(query_tokens)
        # [B, 512]

        # =================================================
        # TEXT ENCODER
        # =================================================

        text_embeddings = self.text_encoder(texts)
        # [B, 512]

        return image_embeddings, text_embeddings