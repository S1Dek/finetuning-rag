import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, vectors, texts):
        self.index.add(np.array(vectors).astype("float32"))
        self.texts.extend(texts)

    def search(self, vector, k=5):
        D, I = self.index.search(np.array([vector]).astype("float32"), k)
        return [self.texts[i] for i in I[0]]