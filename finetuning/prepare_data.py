import json

DATA = [
    {
        "instruction": "Wyjaśnij czym jest RAG",
        "input": "",
        "output": "RAG (Retrieval-Augmented Generation) to metoda..."
    },
    {
        "instruction": "Co to jest LoRA?",
        "input": "",
        "output": "LoRA to metoda fine-tuningu..."
    }
]

with open("dataset.jsonl", "w") as f:
    for item in DATA:
        json.dump(item, f)
        f.write("\n")

print("Dataset zapisany.")