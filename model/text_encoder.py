import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTokenizer
from transformers import CLIPTextModel


class TextEncoder(nn.Module):

    def __init__(self, output_dim=512):

        super().__init__()

        # MUST match the vision encoder variant: openai/clip-vit-base-patch32
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=True
        )

        self.encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            local_files_only=True
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        self.projection = nn.Linear(512, output_dim)

    def forward(self, texts):

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(next(self.parameters()).device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.encoder(**inputs)

        features = outputs.pooler_output

        features = self.projection(features)

        features = F.normalize(features, dim=-1)

        return features
