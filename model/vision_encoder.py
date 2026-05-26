import torch
import torch.nn as nn

from transformers import CLIPVisionModel


# =====================================================
# VISION ENCODER
# =====================================================

class VisionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # =================================================
        # LOAD CLIP
        # =================================================

        self.encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=True
        )

        # =================================================
        # FREEZE EVERYTHING
        # =================================================

        for param in self.encoder.parameters():

            param.requires_grad = False

        # =================================================
        # DROPOUT
        # =================================================

        self.feature_dropout = nn.Dropout(0.3)

        # =================================================
        # PARAM REPORT
        # =================================================

        total_params = sum(
            p.numel()
            for p in self.encoder.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in self.encoder.parameters()
            if p.requires_grad
        )

        print("\n========== VISION ENCODER ==========")

        print(f"Total Params: {total_params:,}")

        print(f"Trainable Params: {trainable_params:,}")

        print("====================================\n")

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, pixel_values):

        """
        pixel_values shape:

        [B, T, C, H, W]
        """

        B, T, C, H, W = pixel_values.shape

        # =================================================
        # FLATTEN TEMPORAL DIMENSION
        # =================================================

        pixel_values = pixel_values.view(

            B * T,

            C,

            H,

            W
        )

        # =================================================
        # CLIP FEATURES
        # =================================================

        outputs = self.encoder(

            pixel_values=pixel_values
        )

        # =================================================
        # PATCH TOKENS
        # =================================================

        features = outputs.last_hidden_state

        # =================================================
        # DROPOUT
        # =================================================

        features = self.feature_dropout(
            features
        )

        # =================================================
        # RESTORE TEMPORAL STRUCTURE
        # =================================================

        features = features.view(

            B,

            T,

            features.shape[1],

            features.shape[2]
        )

        return features