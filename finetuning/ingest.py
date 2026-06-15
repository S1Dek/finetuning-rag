import os
from pdf_loader import extract_pdf
from embed import chunk_text, embed
from vector_db import VectorDB

db = VectorDB()

def ingest_folder(path="./data/pdfs"):
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            text, images = extract_pdf(os.path.join(path, file))

            chunks = chunk_text(text)
            vectors = embed(chunks)

            db.add(vectors, chunks)

    return db