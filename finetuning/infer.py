from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_PATH = "./models/gemma-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

while True:
    prompt = input("You: ")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))