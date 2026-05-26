import os
import sys

import torch

from PIL import Image

from transformers import CLIPProcessor

# =====================================================
# FIX IMPORTS
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
    "checkpoints/baseline/best_model.pth"
)

INDEX_PATH = (
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
# LOAD INDEX
# =====================================================

index = torch.load(INDEX_PATH)

database_embeddings = index["embeddings"]

database_texts = index["texts"]

# =====================================================
# QUERY IMAGES
# =====================================================

query_paths = [

    "demo/query1.jpg",

    "demo/query2.jpg",

    "demo/query3.jpg"
]

images = []

for path in query_paths:

    image = Image.open(path).convert("RGB")

    images.append(image)

query_inputs = processor(

    images=images,

    return_tensors="pt"
)

query_pixel_values = query_inputs[
    "pixel_values"
].unsqueeze(0).to(DEVICE)

# =====================================================
# MEMORY PLACEHOLDER
# =====================================================

memory_images = [

    Image.new("RGB", (224, 224))

    for _ in range(4)
]

memory_inputs = processor(

    images=memory_images,

    return_tensors="pt"
)

memory_pixel_values = memory_inputs[
    "pixel_values"
].unsqueeze(0).to(DEVICE)

# =====================================================
# EMBEDDING
# =====================================================

with torch.no_grad():

    query_embedding, _ = model(

        query_pixel_values,

        memory_pixel_values,

        [""]
    )

query_embedding = query_embedding.cpu()

# =====================================================
# SIMILARITY
# =====================================================

similarities = torch.matmul(

    query_embedding,

    database_embeddings.T
)

values, indices = torch.topk(

    similarities,

    k=5
)

# =====================================================
# OUTPUT
# =====================================================

print("\n==============================")
print("TOP RETRIEVALS")
print("==============================\n")

for rank, idx in enumerate(indices[0]):

    print(f"Rank {rank+1}")

    print(database_texts[idx])

    print(
        f"\nScore: "
        f"{values[0][rank]:.4f}"
    )

    print("\n" + "-" * 60 + "\n")
