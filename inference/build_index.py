import os
import sys
import json

import torch

from tqdm import tqdm
from PIL import Image

from transformers import CLIPProcessor

# =====================================================
# FIX PYTHON PATH
# =====================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# IMPORT MODEL
# =====================================================

from model.imageweave import ImageWeave

# =====================================================
# CONFIG
# =====================================================

DEVICE = "cuda"

CHECKPOINT_PATH = (
    "checkpoints/best_model.pth"
)

METADATA_PATH = (
    "dataset/processed/metadata/train_triplets.json"
)

OUTPUT_INDEX = (
    "inference/retrieval_index.pt"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = ImageWeave().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# =====================================================
# LOAD DATA
# =====================================================

with open(METADATA_PATH) as f:

    data = json.load(f)

# =====================================================
# STORAGE
# =====================================================

all_embeddings = []

all_texts = []

# =====================================================
# BUILD INDEX
# =====================================================

with torch.no_grad():

    for sample in tqdm(data):

        # =================================================
        # QUERY IMAGES
        # =================================================

        query_images = []

        for path in sample["query_images"]:

            image = Image.open(path).convert("RGB")

            query_images.append(image)

        query_inputs = processor(

            images=query_images,

            return_tensors="pt"
        )

        query_pixel_values = query_inputs[
            "pixel_values"
        ].unsqueeze(0).to(DEVICE)

        # =================================================
        # MEMORY IMAGES
        # =================================================

        memory_images = []

        for path in sample["memory_images"][:4]:

            image = Image.open(path).convert("RGB")

            memory_images.append(image)

        while len(memory_images) < 4:

            memory_images.append(
                Image.new("RGB", (224, 224))
            )

        memory_inputs = processor(

            images=memory_images,

            return_tensors="pt"
        )

        memory_pixel_values = memory_inputs[
            "pixel_values"
        ].unsqueeze(0).to(DEVICE)

        # =================================================
        # TEXT
        # =================================================

        text = [sample["text"]]

        # =================================================
        # EMBEDDING
        # =================================================

        image_embeddings, _ = model(

            query_pixel_values,

            memory_pixel_values,

            text
        )

        embedding = image_embeddings[0].cpu()

        all_embeddings.append(embedding)

        all_texts.append(sample["text"])

# =====================================================
# SAVE INDEX
# =====================================================

index = {

    "embeddings": torch.stack(all_embeddings),

    "texts": all_texts
}

torch.save(index, OUTPUT_INDEX)

print("\nRetrieval index saved successfully.")
