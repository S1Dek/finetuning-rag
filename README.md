# PAZPIK - Fine-tuning + RAG dla modeli językowych (Lotnictwo Cywilne)

Projekt do QLoRA fine-tuningu oraz RAG na dokumentach PDF dotyczących lotnictwa cywilnego.

## Struktura projektu

```
PAZPIK/
├── automatic-huggingface/        ← Podejście: pobieranie modeli z HuggingFace
│   ├── scripts/
│   │   ├── 01_finetune.py        ← Fine-tuning (QLoRA continued pre-training)
│   │   ├── 02_rag.py             ← RAG + interaktywny Q&A
│   │   ├── pdf_utils.py          ← Ekstrakcja PDF + opisy schematów
│   │   └── ollama_utils.py       ← Komunikacja z Ollama API
│   ├── config/settings.yaml
│   └── requirements.txt
│
├── extraction-gguf/              ← Podejście: ekstrakcja wag z Ollamy (GGUF)
│   ├── scripts/
│   │   ├── 01_finetune.py        ← Fine-tuning (z konwersją GGUF→HF→GGUF)
│   │   ├── 02_rag.py             ← RAG (identyczny)
│   │   ├── gguf_converter.py     ← Konwersja GGUF ↔ safetensors
│   │   ├── pdf_utils.py          ← Ekstrakcja PDF + opisy schematów
│   │   └── ollama_utils.py       ← Komunikacja z Ollama API
│   ├── config/settings.yaml
│   └── requirements.txt
│
└── README.md
```

## Wymagania

- **Linux** (Ubuntu 22.04+/Zorin) z **NVIDIA CUDA** (dla fine-tuningu)
- **2x RTX 4000 Ada** (20GB) lub podobne karty z ~40GB VRAM
- **Python 3.10+**, **Ollama**, **Alpaca** (interfejs webowy)

## Instalacja

### 1. Pakiety systemowe

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip nvidia-cuda-toolkit git
```

### 2. Środowisko Python (wybierz podejście)

```bash
# automatic-huggingface
cd PAZPIK/automatic-huggingface
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --- LUB ---

# extraction-gguf
cd PAZPIK/extraction-gguf
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Ollama + model

```bash
# Uruchom Ollamę
sudo systemctl start ollama

# Pobierz model do fine-tuningu
ollama pull gemma4:31b

# Pobierz model vision (do opisywania schematów)
ollama pull gemma4-vision:9b
```

### 4. (Dla extraction-gguf) Pobierz llama.cpp

```bash
cd PAZPIK/extraction-gguf/tools
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

## Użycie

### Fine-tuning

#### Podejście automatic-huggingface

```bash
cd PAZPIK/automatic-huggingface
source venv/bin/activate

# 1. Wrzuć PDFy do data/finetune/
# 2. Uruchom:
python scripts/01_finetune.py
```

Skrypt:

1. Przetworzy PDFy (tekst + opisy schematów przez vision model)
2. Pokaże listę modeli Ollama → wybierz `gemma4:31b`
3. Poprosi o HuggingFace token (jednorazowo)
4. Pobierze model z HF → QLoRA training → konwersja → `ollama create`

#### Podejście extraction-gguf

```bash
cd PAZPIK/extraction-gguf
source venv/bin/activate

python scripts/01_finetune.py
```

Skrypt:

1. Przetworzy PDFy (identycznie)
2. Pokaże listę modeli → wybierz
3. Znajdzie plik GGUF w `~/.ollama/models/blobs/`
4. Konwersja GGUF→HF → QLoRA → konwersja HF→GGUF → `ollama create`

### RAG (oba podejścia jednakowo)

```bash
cd PAZPIK/automatic-huggingface   # lub extraction-gguf
source venv/bin/activate

# 1. Wrzuć PDFy do data/rag/
# 2. Uruchom:
python scripts/02_rag.py
```

Skrypt:

1. Indeksuje PDFy do bazy wektorowej ChromaDB
2. Uruchamia interaktywny Q&A
3. Każde pytanie: wyszukanie w dokumentach → augmentacja promptu → odpowiedź z Ollama

## Wybór podejścia

| Cecha              | automatic-huggingface   | extraction-gguf                   |
| ------------------ | ----------------------- | --------------------------------- |
| Źródło modelu      | HuggingFace (pobiera)   | Ollama (lokalne GGUF)             |
| Wymaga HF token    | Tak (jednorazowo)       | Nie                               |
| Konwersja formatów | HF→GGUF (po treningu)   | GGUF→HF→GGUF (przed i po)         |
| Stabilność         | ⭐⭐⭐⭐⭐ Sprawdzona   | ⭐⭐⭐ Zależy od wsparcia Gemma 4 |
| Zalecany dla       | Pierwszego uruchomienia | Gdy nie chcesz zakładać konta HF  |

## Monitorowanie

Podczas treningu skrypt wyświetla:

- Postęp w % i szacowany czas
- Loss (strata) w każdym kroku
- Wykorzystanie VRAM
- Komunikaty o błędach z propozycjami rozwiązań

## Uwagi

- **Ilustracje w PDF**: Skrypt automatycznie wykrywa schematy (mało kolorów) i wysyła je do modelu vision (gemma4-vision:9b) aby opisać. Opisy są wstawiane w odpowiednie miejsca w tekście.
- **2x GPU**: `device_map="auto"` automatycznie rozkłada model na obie karty.
- **VRAM**: Gemma4:31B w 4-bit ~16GB + gradienty/aktywacje ~4GB = ~20GB na 2 GPU → wykonalne.

---KRÓTKO---

# 1. Instalacja zależności

cd ~/PAZPIK/automatic-huggingface
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Wrzucenie PDFy i uruchomienie

cp -r /twoje/pdfy/\* data/finetune/
python scripts/01_finetune.py
