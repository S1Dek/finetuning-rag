from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed(texts):
    return model.encode(texts, convert_to_numpy=True)