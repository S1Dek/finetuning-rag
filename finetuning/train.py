import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

MODEL_NAME = "google/gemma-4:31b"

dataset = load_dataset("json", data_files="dataset.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def format_example(example):
    return f"""
### Instrukcja:
{example['instruction']}

### Odpowiedź:
{example['output']}
"""

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    formatting_func=format_example,
    args=TrainingArguments(
        output_dir="./models/gemma-lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        report_to="none"
    )
)

trainer.train()

model.save_pretrained("./models/gemma-lora")
tokenizer.save_pretrained("./models/gemma-lora")
