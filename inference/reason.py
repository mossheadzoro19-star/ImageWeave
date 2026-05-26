import os
import sys

import torch

from PIL import Image

from transformers import CLIPProcessor

import google.generativeai as genai

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set.")

LLM_NAME = (
    "gemini-1.5-pro"
)

# =====================================================
# LOAD IMAGEWEAVE
# =====================================================

print("\nLoading ImageWeave...\n")

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

print("Loading retrieval index...\n")

index = torch.load(INDEX_PATH)

database_embeddings = index["embeddings"]

database_texts = index["texts"]

print("Configuring Gemini...\n")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

llm = genai.GenerativeModel(LLM_NAME)

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
# GENERATE EMBEDDING
# =====================================================

print("Generating semantic embedding...\n")

with torch.no_grad():

    query_embedding, _ = model(
        query_pixel_values,
        memory_pixel_values,
        [""]
    )

query_embedding = query_embedding.cpu()

# =====================================================
# RETRIEVAL
# =====================================================

print("Retrieving semantic context...\n")

similarities = torch.matmul(
    query_embedding,
    database_embeddings.T
)

values, indices = torch.topk(
    similarities,
    k=10
)

# =====================================================
# DEDUPLICATION
# =====================================================

contexts = []

seen = set()

for idx in indices[0]:

    text = database_texts[idx]

    key = text[:80]

    if key not in seen:

        seen.add(key)

        contexts.append(text)

    if len(contexts) >= 3:

        break

# =====================================================
# COMPRESSED CONTEXT
# =====================================================

summary = " ".join(contexts)

summary = summary[:500]

print("\n==============================")
print("RETRIEVED CONTEXT")
print("==============================\n")

print(summary)

# =====================================================
# CHAT LOOP
# =====================================================

print("\n====================================")
print("IMAGEWEAVE MULTI-IMAGE REASONING")
print("Type 'exit' to quit")
print("====================================")

while True:

    question = input("\nQuestion:\n> ")

    if question.lower() == "exit":

        break

    prompt = f"""
You are ImageWeave, a retrieval-grounded multimodal AI.
You have analyzed a sequence of images and retrieved the following semantic context from your memory database.

Context:
{summary}

User Question:
{question}

Provide a natural, concise, and narrative explanation. Base your reasoning STRICTLY on the provided context.
"""

    try:
        response = llm.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error communicating with Gemini: {e}"

    print("\n==============================")
    print("IMAGEWEAVE ANSWER")
    print("==============================\n")

    print(answer)

print("\nExiting ImageWeave.")