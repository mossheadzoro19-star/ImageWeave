import os
import json

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.story_dataset import StoryTripletDataset

from model.imageweave import ImageWeave

from configs.train_config import CONFIG

# =====================================================
# DEVICE
# =====================================================

device = CONFIG["device"]

# =====================================================
# DATASET
# =====================================================

dataset = StoryTripletDataset(
    "dataset/processed/metadata/val_split.json"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

# =====================================================
# LOAD MODEL
# =====================================================

model = ImageWeave().to(device)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# =====================================================
# STORAGE
# =====================================================

all_image_embeddings = []

all_text_embeddings = []

all_texts = []

# =====================================================
# EXTRACTION
# =====================================================

with torch.no_grad():

    for batch in tqdm(loader):

        query_pixel_values = batch[
            "query_pixel_values"
        ].to(device)

        memory_pixel_values = batch[
            "memory_pixel_values"
        ].to(device)

        texts = batch["text"]

        image_embeddings, text_embeddings = model(

            query_pixel_values,

            memory_pixel_values,

            texts
        )

        all_image_embeddings.append(
            image_embeddings.cpu()
        )

        all_text_embeddings.append(
            text_embeddings.cpu()
        )

        all_texts.extend(texts)

# =====================================================
# CONCATENATE
# =====================================================

image_embeddings = torch.cat(
    all_image_embeddings,
    dim=0
)

text_embeddings = torch.cat(
    all_text_embeddings,
    dim=0
)

# =====================================================
# SAVE
# =====================================================

os.makedirs(
    "retrieval_outputs",
    exist_ok=True
)

torch.save(

    image_embeddings,

    "retrieval_outputs/image_embeddings.pt"
)

torch.save(

    text_embeddings,

    "retrieval_outputs/text_embeddings.pt"
)

with open(

    "retrieval_outputs/texts.json",

    "w"
) as f:

    json.dump(
        all_texts,
        f,
        indent=4
    )

print("\nSaved embeddings successfully.")