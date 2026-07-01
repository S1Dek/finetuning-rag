#!/usr/bin/env python3
"""
=============================================================================
  PAZPIK - Fine-tuning (extraction-gguf)
=============================================================================
  Skrypt do QLoRA fine-tuningu modeli językowych na dokumentach PDF
  (lotnictwo cywilne) z ekstrakcją wag bezpośrednio z plików GGUF Ollamy.

  Przepływ:
    1. Skanuje data/finetune/ → wyciąga tekst + opisuje schematy (vision model)
    2. Pyta użytkownika który model Ollama trenować
    3. Znajduje plik GGUF w katalogu Ollamy
    4. Konwertuje GGUF → safetensors (HuggingFace format)
    5. QLoRA fine-tuning (continued pre-training)
    6. Scalenie adaptera → konwersja GGUF → ollama create
=============================================================================
"""

import os
import sys
import json
import yaml
import time
import math
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import numpy as np

# Wczytaj konfigurację
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

sys.path.insert(0, str(SCRIPT_DIR))
from pdf_utils import process_pdfs_to_corpus, estimate_tokens
from ollama_utils import OllamaClient, list_ollama_models


# ============================================================
# KONFIGURACJA
# ============================================================

PATHS = CONFIG.get("paths", {})
TRAINING = CONFIG.get("training", {})
QUANT = CONFIG.get("quantization", {})
VISION = CONFIG.get("vision", {})
OLLAMA_CFG = CONFIG.get("ollama", {})
GGUF_CFG = CONFIG.get("gguf", {})


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def print_header(text: str):
    print(f"\n{'#'*70}")
    print(f"  {text}")
    print(f"{'#'*70}")


def print_step(step: int, total: int, text: str):
    print(f"\n{'='*70}")
    print(f"  KROK {step}/{total}: {text}")
    print(f"{'='*70}")


def check_cuda() -> bool:
    """Sprawdza dostępność CUDA."""
    print_header("SPRAWDZANIE SPRZĘTU")

    if not torch.cuda.is_available():
        print("\n[!] CUDA NIE JEST DOSTĘPNA!")
        print("    Fine-tuning wymaga kart NVIDIA z CUDA.")
        print("    Sprawdź instalację:")
        print("      nvidia-smi")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"\n  Wykryto {gpu_count} karty(y) GPU:")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        print(f"    GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

    print(f"\n  PyTorch version: {torch.__version__}")
    print(f"  CUDA version:    {torch.version.cuda}")
    return True


# ============================================================
# EKSTRAKCJA MODELU Z OLLAMY (GGUF -> safetensors)
# ============================================================

def extract_model_from_ollama(ollama_model_name: str,
                               converted_dir: str) -> Optional[Path]:
    """
    Znajduje plik GGUF w Ollamie i konwertuje go do formatu HuggingFace.
    """
    # Importuj converter
    sys.path.insert(0, str(SCRIPT_DIR))
    from gguf_converter import find_gguf_blob, gguf_to_safetensors

    print(f"\n  Szukam pliku GGUF dla modelu: {ollama_model_name}")

    # Określ katalog Ollamy
    ollama_dir = GGUF_CFG.get("ollama_models_dir", "~/.ollama/models")
    ollama_dir = Path(ollama_dir).expanduser()

    # Znajdź blob GGUF
    gguf_path = find_gguf_blob(ollama_model_name, str(ollama_dir))

    if gguf_path is None:
        # Alternatywna metoda: szukaj we wszystkich blobs
        print(f"\n  Próbuję alternatywnej metody wyszukiwania...")
        blobs_dir = ollama_dir / "blobs"
        if blobs_dir.exists():
            # Szukaj największego pliku (model GGUF)
            gguf_candidates = sorted(
                [f for f in blobs_dir.iterdir() if f.is_file()],
                key=lambda f: f.stat().st_size,
                reverse=True,
            )
            # Weź największy plik > 1GB
            for f in gguf_candidates[:5]:
                size_gb = f.stat().st_size / (1024**3)
                if size_gb > 1.0:
                    print(f"  Znaleziono potencjalny plik modelu: {f.name} ({size_gb:.1f} GB)")
                    gguf_path = f
                    break

    if gguf_path is None:
        print(f"\n  [!] NIE ZNALEZIONO pliku GGUF dla modelu {ollama_model_name}")
        print(f"  Szukano w: {ollama_dir}")
        print(f"\n  Możliwe rozwiązania:")
        print(f"  1. Upewnij się, że model jest pobrany: ollama pull {ollama_model_name}")
        print(f"  2. Sprawdź katalog Ollamy: ls -la {ollama_dir / 'blobs'}")
        print(f"  3. Użyj podejścia automatic-huggingface (bezpośrednio z HF)")
        return None

    size_gb = gguf_path.stat().st_size / (1024**3)
    print(f"  Znaleziono plik GGUF:")
    print(f"    Ścieżka: {gguf_path}")
    print(f"    Rozmiar: {size_gb:.2f} GB")

    # Konwertuj GGUF -> safetensors
    output_dir = Path(converted_dir) / ollama_model_name.replace(":", "_")
    print(f"\n  Konwersja do formatu HuggingFace...")
    print(f"  To może potrwać (dekompresja i dekwantyzacja ~{size_gb:.0f} GB)...")

    result = gguf_to_safetensors(gguf_path, output_dir)
    return result


# ============================================================
# PRZYGOTOWANIE DATASETU
# ============================================================

def prepare_dataset(corpus_path: str, tokenizer, block_size: int = 2048):
    """Przygotowuje dataset do continued pre-training."""
    from datasets import Dataset

    print(f"\n  Tokenizacja korpusu...")
    start_time = time.time()

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"    Długość tekstu: {len(text):,} znaków")

    tokenized = tokenizer(
        text,
        return_tensors=None,
        truncation=False,
        add_special_tokens=True,
    )

    input_ids = tokenized["input_ids"]
    print(f"    Liczba tokenów: {len(input_ids):,}")

    total_length = (len(input_ids) // block_size) * block_size
    if total_length == 0:
        print(f"  [BŁĄD] Korpus za krótki. Minimum {block_size} tokenów.")
        return None

    input_ids = input_ids[:total_length]
    labels = input_ids.copy()

    input_ids = np.array(input_ids).reshape(-1, block_size)
    labels = np.array(labels).reshape(-1, block_size)

    dataset = Dataset.from_dict({
        "input_ids": input_ids.tolist(),
        "attention_mask": [[1] * block_size for _ in range(len(input_ids))],
        "labels": labels.tolist(),
    })

    elapsed = time.time() - start_time
    print(f"    Utworzono {len(dataset)} bloków po {block_size} tokenów")
    print(f"    Czas: {elapsed:.1f}s")

    return dataset


# ============================================================
# QLoRA - KONFIGURACJA
# ============================================================

def setup_qlora_model(model_dir: Path):
    """
    Ładuje model z przekonwertowanego katalogu w 4-bit (QLoRA).
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    model_name = str(model_dir)
    print(f"\n  Ładowanie modelu z: {model_name}")

    # Konfiguracja kwantyzacji
    compute_dtype = torch.bfloat16 if QUANT.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANT.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=QUANT.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=QUANT.get("bnb_4bit_use_double_quant", True),
    )

    print("    Ładowanie tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("    Ładowanie modelu (to może potrwać)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    model = prepare_model_for_kbit_training(model)

    # Konfiguracja LoRA
    target_modules = TRAINING.get("target_modules", "all-linear")
    print(f"\n  Konfiguracja LoRA: r={TRAINING.get('lora_r', 16)}, alpha={TRAINING.get('lora_alpha', 32)}")

    lora_config = LoraConfig(
        r=TRAINING.get("lora_r", 16),
        lora_alpha=TRAINING.get("lora_alpha", 32),
        lora_dropout=TRAINING.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Trenowalne parametry: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, tokenizer


# ============================================================
# TRENING
# ============================================================

def run_training(model, tokenizer, dataset, output_dir: str):
    """Uruchamia pętlę treningową."""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = TRAINING.get("batch_size_per_device", 1)
    grad_accum = TRAINING.get("gradient_accumulation_steps", 8)
    effective_batch = batch_size * grad_accum * max(1, torch.cuda.device_count())
    learning_rate = TRAINING.get("learning_rate", 2.0e-4)
    num_epochs = TRAINING.get("num_epochs", 3)
    max_steps = TRAINING.get("max_steps", -1)

    steps_per_epoch = max(1, len(dataset) // effective_batch) if len(dataset) > 0 else 1
    total_steps = steps_per_epoch * num_epochs
    estimated_hours = (total_steps * 8) / 3600

    print(f"\n  Parametry treningu:")
    print(f"    Effective batch size:     {effective_batch}")
    print(f"    Learning rate:            {learning_rate}")
    print(f"    Liczba epok:              {num_epochs}")
    print(f"    Sequence length:          {TRAINING.get('sequence_length', 2048)}")
    print(f"    Szacowany czas:           {estimated_hours:.1f}h")

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=TRAINING.get("warmup_steps", 10),
        max_steps=max_steps if max_steps > 0 else -1,
        num_train_epochs=num_epochs if max_steps <= 0 else 0,
        learning_rate=learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=TRAINING.get("logging_steps", 10),
        save_steps=TRAINING.get("save_steps", 100),
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print(f"\n{'='*70}")
    print(f"  ROZPOCZYNAM TRENING!")
    print(f"  Czas startu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print(f"\n  [OK] Trening zakończony!")
    print(f"  Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    adapter_path = output_dir / "lora_adapter"
    print(f"\n  Zapisuję adapter do: {adapter_path}")
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    return trainer.model, adapter_path


# ============================================================
# SCALENIE + KONWERSJA
# ============================================================

def merge_and_convert_gguf(model, tokenizer, model_dir: Path,
                           adapter_path: Path, ollama_name: str,
                           temp_dir: str) -> Optional[Path]:
    """
    Scala adapter LoRA i konwertuje do GGUF.
    """
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"  SCALANIE ADAPTERA Z BAZOWYM MODELEM")
    print(f"{'='*70}")

    # Wczytaj bazowy model w pełnej precyzji
    print(f"\n  Wczytywanie bazowego modelu (16-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Scal
    print(f"  Scalam adapter LoRA...")
    merged_model = model.merge_and_unload()
    del model
    torch.cuda.empty_cache()

    # Zapisz scalony model
    merged_path = Path(temp_dir) / "merged_model"
    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"  Zapisuję scalony model do: {merged_path}")
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

    del merged_model
    torch.cuda.empty_cache()

    # Konwersja do GGUF
    print(f"\n{'='*70}")
    print(f"  KONWERSJA DO FORMATU GGUF")
    print(f"{'='*70}")

    gguf_output = Path(temp_dir) / f"{ollama_name.replace(':', '-')}-pazpik.gguf"

    # Użyj gguf_converter.py
    sys.path.insert(0, str(SCRIPT_DIR))
    from gguf_converter import safetensors_to_gguf

    llamacpp_dir = PROJECT_DIR / GGUF_CFG.get("llamacpp_dir", "tools/llama.cpp")

    result = safetensors_to_gguf(
        merged_path,
        gguf_output,
        llamacpp_dir=llamacpp_dir,
    )

    return result


# ============================================================
# TWORZENIE MODELA W OLLAMIE
# ============================================================

def create_ollama_model(ollama_client: OllamaClient, new_model_name: str,
                        gguf_path: Path):
    """Tworzy model w Ollamie z pliku GGUF."""
    from ollama_utils import create_ollama_model_from_gguf

    full_model_name = f"pazpik-{new_model_name.split(':')[0]}"

    print(f"\n{'='*70}")
    print(f"  TWORZENIE MODELA W OLLAMIE")
    print(f"{'='*70}")
    print(f"\n  Nazwa:  {full_model_name}")
    print(f"  Plik:   {gguf_path}")

    success = create_ollama_model_from_gguf(full_model_name, str(gguf_path), ollama_client)

    if success:
        print(f"\n  [SUKCES] Model '{full_model_name}' gotowy!")
        print(f"    ollama run {full_model_name}")
        return True
    else:
        print(f"\n  [!] Nie udało się utworzyć modelu.")
        return False


# ============================================================
# GŁÓWNA FUNKCJA
# ============================================================

def main():
    print_header("PAZPIK - FINE-TUNING (extraction-gguf)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Projekt: {PROJECT_DIR}")

    # ----------------------------------------------------------
    # KROK 0: CUDA
    # ----------------------------------------------------------
    if not check_cuda():
        sys.exit(1)

    # ----------------------------------------------------------
    # KROK 1: Połączenie z Ollama
    # ----------------------------------------------------------
    print_step(1, 7, "ŁĄCZENIE Z OLLAMA API")
    client = OllamaClient(OLLAMA_CFG.get("base_url", "http://localhost:11434"))

    # ----------------------------------------------------------
    # KROK 2: Przetworzenie PDF -> korpus
    # ----------------------------------------------------------
    print_step(2, 7, "EKSTRAKCJA PDF -> KORPUS TRENINGOWY")

    finetune_dir = PROJECT_DIR / PATHS.get("finetune_pdfs", "data/finetune")
    corpus_path = PROJECT_DIR / PATHS.get("corpus_output", "data/finetune_corpus.txt")

    corpus = process_pdfs_to_corpus(
        pdf_dir=str(finetune_dir),
        output_file=str(corpus_path),
        vision_model=VISION.get("model", "gemma4-vision:9b"),
        vision_prompt=VISION.get("prompt"),
        ollama_client=client,
    )

    if not corpus:
        print(f"\n[!] Brak danych treningowych. Przerwano.")
        sys.exit(1)

    # ----------------------------------------------------------
    # KROK 3: Wybór modelu
    # ----------------------------------------------------------
    print_step(3, 7, "WYBÓR MODELA DO TRENOWANIA")

    models = list_ollama_models(client)
    if not models:
        print(f"\n  [!] Brak modeli. Zainstaluj: ollama pull gemma4:31b")
        sys.exit(1)

    while True:
        try:
            choice = input(f"\n  Wybierz model (numer lub nazwa): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected_model = models[idx]
                    break
            else:
                selected = [m for m in models if m["name"] == choice]
                if selected:
                    selected_model = selected[0]
                    break
            print("  Nieprawidłowy wybór.")
        except (ValueError, IndexError):
            print("  Nieprawidłowy wybór.")

    ollama_model_name = selected_model["name"]
    print(f"\n  Wybrano: {ollama_model_name}")

    # ----------------------------------------------------------
    # KROK 4: Ekstrakcja modelu z Ollamy (GGUF -> safetensors)
    # ----------------------------------------------------------
    print_step(4, 7, "EKSTRAKCJA MODELU Z OLLAMY")

    converted_dir = PROJECT_DIR / PATHS.get("converted_model_dir", "models/converted_hf")
    model_dir = extract_model_from_ollama(ollama_model_name, str(converted_dir))

    if model_dir is None:
        print(f"\n  [!] Nie udało się wyodrębnić modelu z Ollamy.")
        print(f"  Spróbuj podejścia automatic-huggingface lub:")
        print(f"  1. Sprawdź czy model istnieje: ollama list")
        print(f"  2. Pobierz model: ollama pull gemma4:31b")
        sys.exit(1)

    # ----------------------------------------------------------
    # KROK 5: QLoRA Fine-tuning
    # ----------------------------------------------------------
    print_step(5, 7, "QLoRA FINE-TUNING")

    temp_dir = PATHS.get("temp_dir", "/tmp/pazpik")
    os.makedirs(temp_dir, exist_ok=True)

    model, tokenizer = setup_qlora_model(model_dir)

    block_size = TRAINING.get("sequence_length", 2048)
    dataset = prepare_dataset(str(corpus_path), tokenizer, block_size)
    if dataset is None:
        sys.exit(1)

    adapter_output = PROJECT_DIR / PATHS.get("adapters_dir", "models/adapters") / ollama_model_name.replace(":", "_")
    _, adapter_path = run_training(model, tokenizer, dataset, str(adapter_output))

    # ----------------------------------------------------------
    # KROK 6: Scalenie + konwersja do GGUF
    # ----------------------------------------------------------
    print_step(6, 7, "SCALENIE ADAPTERA + KONWERSJA DO GGUF")

    gguf_path = merge_and_convert_gguf(
        model=model,
        tokenizer=tokenizer,
        model_dir=model_dir,
        adapter_path=adapter_path,
        ollama_name=ollama_model_name,
        temp_dir=temp_dir,
    )

    if gguf_path is None:
        print(f"\n  [!] Konwersja do GGUF nie powiodła się.")
        print(f"      Adapter zapisany w: {adapter_path}")
        sys.exit(1)

    # ----------------------------------------------------------
    # KROK 7: Tworzenie modelu w Ollamie
    # ----------------------------------------------------------
    print_step(7, 7, "TWORZENIE MODELA W OLLAMIE")

    create_ollama_model(client, ollama_model_name, gguf_path)

    # ----------------------------------------------------------
    # KONIEC
    # ----------------------------------------------------------
    print_header("PROCES ZAKOŃCZONY")
    print(f"  Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n  Podsumowanie:")
    print(f"    - Przetworzone PDFy: {finetune_dir}")
    print(f"    - Wytrenowany model: pazpik-{ollama_model_name.split(':')[0]}")
    print(f"    - Adapter LoRA:      {adapter_output}")
    print(f"\n  Użycie:")
    print(f"    ollama run pazpik-{ollama_model_name.split(':')[0]}")
    print(f"    # lub przez Alpaca")


if __name__ == "__main__":
    main()
