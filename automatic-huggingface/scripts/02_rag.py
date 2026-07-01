#!/usr/bin/env python3
"""
=============================================================================
  PAZPIK - RAG Pipeline
=============================================================================
  Skrypt do Retrieval-Augmented Generation na dokumentach PDF.
  - Indeksuje PDFy z data/rag/ do bazy wektorowej (ChromaDB)
  - Uruchamia interaktywną konsolę Q&A z wykorzystaniem Ollama
=============================================================================
"""

import os
import sys
import yaml
import time
from pathlib import Path
from datetime import datetime

# Wczytaj konfigurację
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_DIR / "config" / "settings.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

sys.path.insert(0, str(SCRIPT_DIR))
from pdf_utils import chunk_text_for_rag
from ollama_utils import OllamaClient


# ============================================================
# KONFIGURACJA
# ============================================================

PATHS = CONFIG.get("paths", {})
OLLAMA_CFG = CONFIG.get("ollama", {})


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


# ============================================================
# INDEKSOWANIE DO CHROMADB
# ============================================================

def create_vector_store(chunks: list[dict], persist_dir: str):
    """
    Tworzy lub aktualizuje bazę wektorową ChromaDB z chunków.
    """
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.schema import Document

    if not chunks:
        print("\n  [!] Brak chunków do indeksowania.")
        return None

    print(f"\n  Inicjalizacja embeddera (sentence-transformers)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Konwersja na dokumenty LangChain
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
            },
        )
        documents.append(doc)

    print(f"  Tworzenie bazy wektorowej w: {persist_dir}")
    print(f"  Liczba dokumentów: {len(documents)}")
    print(f"  To może potrwać (embedding {len(documents)} chunków)...")

    start_time = time.time()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    elapsed = time.time() - start_time
    print(f"  [OK] Baza wektorowa gotowa ({elapsed:.1f}s)")
    print(f"  Liczba wektorów: {vector_store._collection.count()}")

    return vector_store


def load_vector_store(persist_dir: str):
    """Ładuje istniejącą bazę wektorową."""
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    if not Path(persist_dir).exists():
        return None

    print(f"  Ładowanie istniejącej bazy wektorowej...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    count = vector_store._collection.count()
    print(f"  [OK] Załadowano bazę z {count} wektorami")
    return vector_store


# ============================================================
# INTERAKCYJNA KONSOLA Q&A
# ============================================================

def interactive_qa(vector_store, ollama_client: OllamaClient, model_name: str):
    """
    Interaktywna pętla pytań i odpowiedzi z RAG.
    """
    print_header("TRYB Q&A - zadawaj pytania dotyczące dokumentów")
    print(f"  Model: {model_name}")
    print(f"  Aby zakończyć, wpisz: exit, quit lub /bye")
    print(f"{'='*70}")

    history = []

    while True:
        try:
            question = input(f"\n{'─'*70}\n  Twoje pytanie: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Zakończono.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "/bye", "q"):
            print("  Zakończono.")
            break

        # ----------------------------------------------------------
        # 1. Wyszukanie w wektorowej bazie
        # ----------------------------------------------------------
        print(f"\n  Szukam w dokumentach...")
        search_start = time.time()

        try:
            docs = vector_store.similarity_search_with_score(question, k=5)
        except Exception as e:
            print(f"  [BŁĄD] Wyszukiwanie nie powiodło się: {e}")
            continue

        search_time = time.time() - search_start
        print(f"  Znaleziono {len(docs)} fragmentów (czas: {search_time:.2f}s)")

        # ----------------------------------------------------------
        # 2. Zbudowanie kontekstu
        # ----------------------------------------------------------
        context_parts = []
        sources = set()
        for i, (doc, score) in enumerate(docs, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            context_parts.append(f"[Źródło {i}: {source}, strona {page}]\n{doc.page_content}")
            sources.add(source)

        context = "\n\n".join(context_parts)

        # ----------------------------------------------------------
        # 3. Zbudowanie promptu z kontekstem
        # ----------------------------------------------------------
        system_prompt = (
            "Jesteś asystentem ekspertem w dziedzinie lotnictwa cywilnego. "
            "Odpowiadasz na pytania użytkownika wyłącznie na podstawie "
            "dostarczonego kontekstu z dokumentów. Jeśli kontekst nie zawiera "
            "informacji potrzebnych do odpowiedzi, po prostu powiedz, że nie wiesz. "
            "Odpowiadaj w języku polskim."
        )

        full_prompt = (
            f"KONTEKS Z DOKUMENTÓW:\n{context}\n\n"
            f"PYTANIE: {question}\n\n"
            f"ODPOWIEDŹ (na podstawie kontekstu, w języku polskim):"
        )

        # ----------------------------------------------------------
        # 4. Generowanie odpowiedzi przez Ollama
        # ----------------------------------------------------------
        print(f"  Generuję odpowiedź (model: {model_name})...")
        gen_start = time.time()

        response = ollama_client.generate(
            model=model_name,
            prompt=full_prompt,
            system=system_prompt,
            max_tokens=1024,
        )

        gen_time = time.time() - gen_start

        # ----------------------------------------------------------
        # 5. Wyświetlenie odpowiedzi
        # ----------------------------------------------------------
        print(f"\n{'─'*70}")
        print(f"  ODPOWIEDŹ (czas generowania: {gen_time:.1f}s):")
        print(f"{'─'*70}")
        print(f"  {response}")
        print(f"\n  Źródła: {', '.join(sources)}")

        # Zapisz do historii
        history.append({"question": question, "answer": response, "sources": list(sources)})


# ============================================================
# GŁÓWNA FUNKCJA
# ============================================================

def main():
    print_header("PAZPIK - RAG (Retrieval-Augmented Generation)")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Projekt: {PROJECT_DIR}")

    # ----------------------------------------------------------
    # KROK 1: Połączenie z Ollama
    # ----------------------------------------------------------
    print_step(1, 4, "ŁĄCZENIE Z OLLAMA API")
    client = OllamaClient(OLLAMA_CFG.get("base_url", "http://localhost:11434"))

    # ----------------------------------------------------------
    # KROK 2: Wybór modelu
    # ----------------------------------------------------------
    print_step(2, 4, "WYBÓR MODELA DO ODPOWIEDZI")

    models = client.list_models()
    if not models:
        print("\n  [!] Brak modeli w Ollama. Uruchom najpierw fine-tuning lub")
        print("      pobierz model: ollama pull <nazwa>")
        return

    # Filtruj modele pazpik na pierwsze miejsce
    pazpik_models = [m for m in models if "pazpik" in m["name"]]
    other_models = [m for m in models if "pazpik" not in m["name"]]
    sorted_models = pazpik_models + other_models

    print(f"\n  {'Lp.':<5} {'Nazwa modelu':<35} {'Rozmiar':<10}")
    print(f"  {'-'*50}")
    for i, m in enumerate(sorted_models, 1):
        prefix = " ★" if "pazpik" in m["name"] else "  "
        print(f"  {i:<5}{prefix} {m['name']:<33} {m['size_gb']:<10}")

    while True:
        try:
            choice = input(f"\n  Wybierz model do RAG (numer lub nazwa): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(sorted_models):
                    selected = sorted_models[idx]
                    break
            else:
                selected = [m for m in sorted_models if m["name"] == choice]
                if selected:
                    selected = selected[0]
                    break
            print("  Nieprawidłowy wybór.")
        except (ValueError, IndexError):
            print("  Nieprawidłowy wybór.")

    model_name = selected["name"]
    print(f"\n  Wybrano model: {model_name}")

    # ----------------------------------------------------------
    # KROK 3: Indeksowanie PDF do bazy wektorowej
    # ----------------------------------------------------------
    print_step(3, 4, "INDEKSOWANIE DOKUMENTÓW PDF")

    rag_dir = PROJECT_DIR / PATHS.get("rag_pdfs", "data/rag")
    vectordb_dir = PROJECT_DIR / PATHS.get("vectordb_dir", "vectordb")

    # Sprawdź czy już istnieje baza i czy pytac o reindeksację
    vector_store = load_vector_store(str(vectordb_dir))

    if vector_store is not None:
        print(f"\n  Istniejąca baza wektorowa znaleziona.")
        reindex = input("  Czy chcesz przeindeksować PDFy od nowa? (t/N): ").strip().lower()
        if reindex in ("t", "tak", "y", "yes"):
            vector_store = None

    if vector_store is None:
        chunks = chunk_text_for_rag(
            pdf_dir=str(rag_dir),
            chunk_size=1000,
            chunk_overlap=200,
        )

        if not chunks:
            print(f"\n  [!] Brak dokumentów do indeksowania.")
            print(f"      Umieść pliki PDF w: {rag_dir}")
            return

        vector_store = create_vector_store(chunks, str(vectordb_dir))

    # ----------------------------------------------------------
    # KROK 4: Interaktywny Q&A
    # ----------------------------------------------------------
    print_step(4, 4, "TRYB PYTAŃ I ODPOWIEDZI")

    if vector_store is None:
        print("\n  [!] Baza wektorowa nie jest dostępna.")
        return

    interactive_qa(vector_store, client, model_name)

    # ----------------------------------------------------------
    # KONIEC
    # ----------------------------------------------------------
    print_header("RAG ZAKOŃCZONY")
    print(f"  Zakończono: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
