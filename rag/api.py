from fastapi import FastAPI
from embed import embed
from vector_db import VectorDB
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

db = VectorDB()

model_name = "google/gemma-3-12b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

@app.post("/ask")
def ask(query: str):

    q_vec = embed([query])[0]
    docs = db.search(q_vec)

    context = "\n".join(docs)

    prompt = f"""
Context:
{context}

Question:
{query}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    out = model.generate(**inputs, max_new_tokens=300)

    return {
        "answer": tokenizer.decode(out[0], skip_special_tokens=True),
        "context": docs
    }