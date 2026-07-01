#!/usr/bin/env python3
"""
=============================================================================
  PAZPIK - Fine-tuning (automatic-huggingface)
  =============================================================================
  Skrypt do QLoRA fine-tuningu modeli językowych na dokumentach PDF
  (lotnictwo cywilne) z automatycznym pobieraniem wag z HuggingFace.

  Przepływ:
    1. Skanuje data/finetune/ → wyciąga tekst + opisuje schematy (vision model)
    2. Pyta użytkownika który model Ollama trenować
    3. Mapuje na model HuggingFace → pobiera wagi
    4. QLoRA fine-tuning (continued pre-training)
    5. Scalenie adaptera → konwersja do GGUF → ollama create
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
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

# Wczytaj konfigurację przed importem reszty
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# Własne moduły
sys.path.insert(0, str(SCRIPT_DIR))
from pdf_utils import process_pdfs_to_corpus, estimate_tokens
from ollama_utils import OllamaClient, list_ollama_models, get_hf_model_name


# ============================================================
# KONFIGURACJA
# ============================================================

PATHS = CONFIG.get("paths", {})
TRAINING = CONFIG.get("training", {})
QUANT = CONFIG.get("quantization", {})
VISION = CONFIG.get("vision", {})
OLLAMA_CFG = CONFIG.get("ollama", {})
MODEL_MAPPING = CONFIG.get("model_mapping", {})


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
    """Sprawdza dostępność CUDA i wyświetla informacje o GPU."""
    print_header("SPRAWDZANIE SPRZĘTU")

    if not torch.cuda.is_available():
        print("\n[!] CUDA NIE JEST DOSTĘPNA!")
        print("    Fine-tuning wymaga kart NVIDIA z CUDA.")
        print("    Sprawdź instalację sterowników NVIDIA:")
        print("      nvidia-smi")
        print("    Sprawdź instalację PyTorch z CUDA:")
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


def ask_huggingface_token() -> str:
    """Pyta użytkownika o token HuggingFace (jeśli nie ma w .env)."""
    env_file = PROJECT_DIR / ".env"
    token = None

    # Sprawdź plik .env
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1]
                    print(f"\n  [OK] Odczytano token HF z pliku .env")
                    break

    if not token:
        token = os.environ.get("HF_TOKEN")

    if not token:
        print(f"\n{'='*70}")
        print("  WYMAGANY TOKEN HUGGINGFACE")
        print("  Model Gemma 4 wymaga akceptacji licencji na HuggingFace.")
        print("  1. Załóż konto na: https://huggingface.co/join")
        print("  2. Wejdź na: https://huggingface.co/google/gemma-4-31b-it")
        print("     i zaakceptuj licencję (przycisk 'Agree and access repository')")
        print("  3. Wygeneruj token: https://huggingface.co/settings/tokens")
        print("     (potrzebujesz token z prawem 'read')")
        print(f"{'='*70}")
        token = input("  Wklej swój HuggingFace token: ").strip()

        if token:
            # Zapisz do .env
            with open(env_file, "w") as f:
                f.write(f"HF_TOKEN={token}\n")
            print(f"  [OK] Token zapisany do {env_file}")
            os.chmod(env_file, 0o600)

    return token


# ============================================================
# PRZYGOTOWANIE DATASETU (continued pre-training)
# ============================================================

def prepare_dataset(corpus_path: str, tokenizer, block_size: int = 2048):
    """
    Przygotowuje dataset do continued pre-training.
    Dzieli tekst na bloki o długości block_size (przewidywanie następnego tokena).
    """
    from datasets import Dataset

    print(f"\n  Tokenizacja korpusu...")
    start_time = time.time()

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"    Długość tekstu: {len(text):,} znaków")

    # Tokenizacja całego tekstu
    tokenized = tokenizer(
        text,
        return_tensors=None,
        truncation=False,
        add_special_tokens=True,
    )

    input_ids = tokenized["input_ids"]
    print(f"    Liczba tokenów: {len(input_ids):,}")

    # Usuń próbkę, aby dopasować do bloków
    total_length = (len(input_ids) // block_size) * block_size
    if total_length == 0:
        print("  [BŁĄD] Korpus jest zbyt krótki na jeden blok.")
        print(f"    Minimum {block_size} tokenów, posiadasz {len(input_ids)}")
        return None

    input_ids = input_ids[:total_length]
    labels = input_ids.copy()

    # Podział na bloki
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
    print(f"    Rozmiar datasetu: {len(dataset)} próbek")

    return dataset


# ============================================================
# QLoRA - KONFIGURACJA MODELE
# ============================================================

def setup_qlora_model(hf_model_name: str, hf_token: str):
    """
    Ładuje model w 4-bit (QLoRA) i przygotowuje do treningu.
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

    print(f"\n  Ładowanie modelu: {hf_model_name}")
    print(f"  Konfiguracja: 4-bit ({QUANT.get('bnb_4bit_quant_type', 'nf4')})")

    # Konfiguracja kwantyzacji
    compute_dtype = torch.bfloat16 if QUANT.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANT.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=QUANT.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=QUANT.get("bnb_4bit_use_double_quant", True),
    )

    print("    Pobieranie tokenizera...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        token=hf_token,
        trust_remote_code=True,
    )

    # Ustaw token padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("    Pobieranie modelu (to może potrwać kilka minut)...")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # Przygotuj model do treningu w k-bit
    model = prepare_model_for_kbit_training(model)

    # Konfiguracja LoRA
    print(f"\n  Konfiguracja LoRA:")
    print(f"    r={TRAINING.get('lora_r', 16)}, alpha={TRAINING.get('lora_alpha', 32)}")

    target_modules = TRAINING.get("target_modules", "all-linear")

    lora_config = LoraConfig(
        r=TRAINING.get("lora_r", 16),
        lora_alpha=TRAINING.get("lora_alpha", 32),
        lora_dropout=TRAINING.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Oblicz liczbę trenowalnych parametrów
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Trenowalne parametry: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"    Wszystkie parametry:  {total_params:,}")

    # Włącz gradient checkpointing (oszczędność VRAM)
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

    # Określ optymalną wielkość batcha
    batch_size = TRAINING.get("batch_size_per_device", 1)
    grad_accum = TRAINING.get("gradient_accumulation_steps", 8)
    effective_batch = batch_size * grad_accum * max(1, torch.cuda.device_count())
    learning_rate = TRAINING.get("learning_rate", 2.0e-4)
    num_epochs = TRAINING.get("num_epochs", 3)
    max_steps = TRAINING.get("max_steps", -1)

    print(f"\n  Parametry treningu:")
    print(f"    Batch size (na GPU):     {batch_size}")
    print(f"    Gradient accumulation:    {grad_accum}")
    print(f"    Effective batch size:     {effective_batch}")
    print(f"    Learning rate:            {learning_rate}")
    print(f"    Liczba epok:              {num_epochs}")
    print(f"    Sequence length:          {TRAINING.get('sequence_length', 2048)}")
    print(f"    Wykryte GPU:              {torch.cuda.device_count()}")

    # Oblicz kroki
    steps_per_epoch = max(1, len(dataset) // effective_batch) if len(dataset) > 0 else 1
    total_steps = steps_per_epoch * num_epochs

    print(f"    Kroki na epokę:           {steps_per_epoch}")
    print(f"    Łączna liczba kroków:     {total_steps}")

    # Szacowany czas (na podstawie 5s na krok dla 31B modelu w 4-bit)
    estimated_hours = (total_steps * 8) / 3600  # ~8s per step dla gemma4:31b
    print(f"    Szacowany czas treningu:   {estimated_hours:.1f}h")

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
    print(f"  Czas zakończenia: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Zapisz adapter LoRA
    adapter_path = output_dir / "lora_adapter"
    print(f"\n  Zapisuję adapter LoRA do: {adapter_path}")
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"  [OK] Adapter zapisany.")

    return trainer.model, adapter_path


# ============================================================
# SCALENIE ADAPTERA + KONWERSJA DO GGUF
# ============================================================

def merge_and_convert(model, tokenizer, hf_model_name: str,
                      adapter_path: Path, ollama_name: str,
                      temp_dir: str, hf_token: str) -> Optional[Path]:
    """
    Scala adapter LoRA z bazowym modelem i konwertuje do GGUF.
    """
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*70}")
    print(f"  SCALANIE ADAPTERA Z BAZOWYM MODELEM")
    print(f"{'='*70}")

    # Wczytaj bazowy model w pełnej precyzji (lub 16-bit)
    print(f"\n  Wczytywanie bazowego modelu w 16-bit (to może potrwać)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Scal adapter
    print(f"  Scalam adapter LoRA...")
    merged_model = model.merge_and_unload()
    del model

    # Zapisz scalony model
    merged_path = Path(temp_dir) / "merged_model"
    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"  Zapisuję scalony model do: {merged_path}")
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"  [OK] Scalony model zapisany.")

    del merged_model
    torch.cuda.empty_cache()

    # Konwersja do GGUF
    print(f"\n{'='*70}")
    print(f"  KONWERSJA DO FORMATU GGUF")
    print(f"{'='*70}")

    gguf_output = Path(temp_dir) / f"{ollama_name.replace(':', '-')}-pazpik.gguf"

    # Szukaj konwertera llama.cpp
    llamacpp_dir = Path(temp_dir) / "llama.cpp"
    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"

    if convert_script.exists():
        print(f"\n  Używam konwertera z llama.cpp")
        cmd = [
            sys.executable, str(convert_script),
            str(merged_path),
            "--outfile", str(gguf_output),
            "--outtype", "q4_k_m",
            "--model-name", ollama_name,
        ]
        print(f"  Komenda: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [OK] Konwersja udana!")
        else:
            print(f"  [BŁĄD] Konwersja nieudana:")
            print(f"    {result.stderr[:500]}")
            return None
    else:
        # Alternatywa: użyj pip install gguf + konwersja
        print(f"\n  Konwerter llama.cpp nie znaleziony w {convert_script}")
        print(f"  Próbuję konwersji przez bibliotekę gguf...")
        try:
            from gguf import GGUFWriter
            # TODO: implementacja GGUFWriter
            print(f"  [BŁĄD] Bezpośrednia konwersja wymaga llama.cpp")
            print(f"  Pobierz go: git clone https://github.com/ggml-org/llama.cpp")
            return None
        except ImportError:
            print(f"  [BŁĄD] Biblioteka gguf nie jest zainstalowana")
            return None

    return gguf_output if gguf_output.exists() else None


# ============================================================
# TWORZENIE MODELA W OLLAMIE
# ============================================================

def create_ollama_model(ollama_client: OllamaClient, new_model_name: str,
                        gguf_path: Path):
    """Tworzy nowy model w Ollamie z pliku GGUF."""
    from ollama_utils import create_ollama_model_from_gguf

    print(f"\n{'='*70}")
    print(f"  TWORZENIE MODELA W OLLAMIE")
    print(f"{'='*70}")

    full_model_name = f"pazpik-{new_model_name.split(':')[0]}"

    print(f"\n  Nazwa nowego modelu: {full_model_name}")
    print(f"  Plik GGUF:           {gguf_path}")

    success = create_ollama_model_from_gguf(full_model_name, str(gguf_path), ollama_client)

    if success:
        print(f"\n  [SUKCES] Model '{full_model_name}' utworzony w Ollamie!")
        print(f"  Możesz go teraz używać:")
        print(f"    ollama run {full_model_name}")
        print(f"    # lub przez Alpaca (automatycznie po odświeżeniu)")
        return True
    else:
        print(f"\n  [!] Nie udało się utworzyć modelu w Ollamie.")
        print(f"      Spróbuj ręcznie:")
        print(f"        ollama create {full_model_name} -f Modelfile")
        return False


# ============================================================
# GŁÓWNA FUNKCJA
# ============================================================

def main():
    print_header("PAZPIK - FINE-TUNING (automatic-huggingface)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Projekt: {PROJECT_DIR}")
    print(f"  Ścieżka konfiguracji: {CONFIG_PATH}")

    # ----------------------------------------------------------
    # KROK 0: Sprawdzenie CUDA
    # ----------------------------------------------------------
    if not check_cuda():
        sys.exit(1)

    # ----------------------------------------------------------
    # KROK 1: Połączenie z Ollama
    # ----------------------------------------------------------
    print_step(1, 7, "ŁĄCZENIE Z OLLAMA API")
    client = OllamaClient(OLLAMA_CFG.get("base_url", "http://localhost:11434"))

    # ----------------------------------------------------------
    # KROK 2: Przetworzenie PDF -> korpus treningowy
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
    # KROK 3: Wybór modelu do trenowania
    # ----------------------------------------------------------
    print_step(3, 7, "WYBÓR MODELA DO TRENOWANIA")

    models = list_ollama_models(client)
    if not models:
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
            print(f"  Nieprawidłowy wybór. Spróbuj ponownie.")
        except (ValueError, IndexError):
            print(f"  Nieprawidłowy wybór. Spróbuj ponownie.")

    ollama_model_name = selected_model["name"]
    print(f"\n  Wybrano: {ollama_model_name}")

    # ----------------------------------------------------------
    # KROK 4: Mapowanie na nazwę HuggingFace + token
    # ----------------------------------------------------------
    print_step(4, 7, "MAPOWANIE NA HUGGINGFACE")

    hf_model_name = get_hf_model_name(client, ollama_model_name, MODEL_MAPPING)
    print(f"  Model HF: {hf_model_name}")

    hf_token = ask_huggingface_token()
    if not hf_token:
        print(f"\n[!] Token HuggingFace jest wymagany.")
        sys.exit(1)

    # ----------------------------------------------------------
    # KROK 5: QLoRA Fine-tuning
    # ----------------------------------------------------------
    print_step(5, 7, "QLoRA FINE-TUNING")

    # Utwórz katalog tymczasowy
    temp_dir = PATHS.get("temp_dir", "/tmp/pazpik")
    os.makedirs(temp_dir, exist_ok=True)

    # Wczytaj model z QLoRA
    model, tokenizer = setup_qlora_model(hf_model_name, hf_token)

    # Przygotuj dataset
    block_size = TRAINING.get("sequence_length", 2048)
    dataset = prepare_dataset(str(corpus_path), tokenizer, block_size)
    if dataset is None:
        print(f"\n[!] Przygotowanie datasetu nie powiodło się.")
        sys.exit(1)

    # Uruchom trening
    output_dir = PROJECT_DIR / PATHS.get("adapters_dir", "models/adapters") / ollama_model_name.replace(":", "_")
    _, adapter_path = run_training(model, tokenizer, dataset, str(output_dir))

    # ----------------------------------------------------------
    # KROK 6: Scalenie i konwersja do GGUF
    # ----------------------------------------------------------
    print_step(6, 7, "SCALENIE ADAPTERA + KONWERSJA DO GGUF")

    gguf_path = merge_and_convert(
        model=model,
        tokenizer=tokenizer,
        hf_model_name=hf_model_name,
        adapter_path=adapter_path,
        ollama_name=ollama_model_name,
        temp_dir=temp_dir,
        hf_token=hf_token,
    )

    if gguf_path is None:
        print(f"\n  [!] Konwersja do GGUF nie powiodła się.")
        print(f"      Adapter LoRA jest zapisany w: {adapter_path}")
        print(f"      Możesz go scalić ręcznie później.")
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
    print(f"    - Adapter LoRA:      {output_dir}")
    print(f"    - Plik GGUF:         {gguf_path}")
    print(f"\n  Użycie:")
    print(f"    ollama run pazpik-{ollama_model_name.split(':')[0]}")
    print(f"    # lub przez Alpaca")


if __name__ == "__main__":
    main()
