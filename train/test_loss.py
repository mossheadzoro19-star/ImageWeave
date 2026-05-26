import torch

from torch.utils.data import DataLoader

from dataset.story_dataset import StoryTripletDataset
from model.imageweave import ImageWeave
from train.losses import ContrastiveLoss

# ---------------------------------------------------

dataset = StoryTripletDataset(
    "dataset/processed/metadata/train_triplets.json"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

batch = next(iter(loader))

pixel_values = batch["pixel_values"]
texts = batch["text"]

print("Pixel values:")
print(pixel_values.shape)

# ---------------------------------------------------

model = ImageWeave()

image_embeddings, text_embeddings = model(
    pixel_values,
    texts
)

print("\nImage embeddings:")
print(image_embeddings.shape)

print("\nText embeddings:")
print(text_embeddings.shape)

# ---------------------------------------------------

criterion = ContrastiveLoss()

loss = criterion(
    image_embeddings,
    text_embeddings
)

print("\nContrastive loss:")
print(loss.item())