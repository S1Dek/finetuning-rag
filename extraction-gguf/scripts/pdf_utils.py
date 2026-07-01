"""
PAZPIK - pdf_utils.py
Ekstrakcja tekstu i schematów z dokumentów PDF lotnictwa cywilnego.
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from ollama_utils import OllamaClient


# ============================================================
# KONFIGURACJA DOMYŚLNA
# ============================================================

DEFAULT_CONFIG = {
    "vision_model": "gemma4-vision:9b",
    "vision_prompt": (
        "Opisz szczegółowo ten schemat techniczny z dokumentu lotnictwa cywilnego. "
        "Wymień wszystkie elementy, połączenia i ich przeznaczenie. "
        "Jeśli to schemat blokowy, diagram lub rysunek techniczny - opisz go dokładnie."
    ),
    "schematic_color_threshold": 0.30,
    "min_image_width": 100,
    "min_image_height": 100,
}


# ============================================================
# KLASYFIKACJA OBRAZÓW
# ============================================================

def classify_as_schematic(image_path: str, threshold: float = 0.30) -> bool:
    """
    Klasyfikuje obraz jako schemat (a nie zdjęcie).
    Heurystyka: schematy mają mniej unikalnych kolorów w stosunku do liczby pikseli.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        pixels = img.getcolors(maxcolors=img.width * img.height)
        if pixels is None:
            return False
        unique_colors = len(pixels)
        total_pixels = img.width * img.height
        color_ratio = unique_colors / total_pixels
        return color_ratio <= threshold
    except Exception as e:
        print(f"  [OSTRZEŻENIE] Błąd klasyfikacji obrazu: {e}")
        return False


# ============================================================
# EKSTRAKCJA PDF
# ============================================================

def extract_pdf_content(
    pdf_path: str,
    output_images_dir: Optional[str] = None,
    vision_model: str = "gemma4-vision:9b",
    vision_prompt: str = DEFAULT_CONFIG["vision_prompt"],
    schematic_threshold: float = 0.30,
    ollama_client: Optional[OllamaClient] = None,
) -> str:
    """
    Główna funkcja: wyciąga tekst + opisuje schematy z PDF.
    Zwraca pełny tekst strony z wstawionymi opisami schematów.
    """
    print(f"\n{'='*60}")
    print(f"  PRZETWARZANIE PDF: {Path(pdf_path).name}")
    print(f"{'='*60}")

    doc = fitz.open(pdf_path)
    full_text_pages = []
    total_pages = len(doc)

    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)

    for page_num in range(total_pages):
        page = doc[page_num]
        print(f"\n  --- Strona {page_num + 1}/{total_pages} ---")

        # 1. Wyciągnij bloki tekstu z pozycjami
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        image_regions = []

        for block in blocks:
            if block["type"] == 0:  # blok tekstu
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span.get("text", "") + " "
                if text.strip():
                    text_blocks.append({
                        "y0": block["bbox"][1],
                        "text": text.strip()
                    })

            elif block["type"] == 1:  # blok obrazu
                image_regions.append(block)

        # 2. Wyciągnij obrazy i sklasyfikuj jako schematy
        schematic_descriptions = []
        for img_idx, img_block in enumerate(image_regions):
            bbox = img_block["bbox"]
            img_width = bbox[2] - bbox[0]
            img_height = bbox[3] - bbox[1]

            if img_width < DEFAULT_CONFIG["min_image_width"] or img_height < DEFAULT_CONFIG["min_image_height"]:
                continue

            try:
                xref = img_block.get("image", 0)
                if not xref:
                    # image może być w sub-blokach
                    for sub in img_block.get("images", []):
                        xref = sub.get("xref", 0)
                        if xref:
                            break

                if not xref:
                    continue

                pix = fitz.Pixmap(doc, xref)

                # Konwersja do RGB
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_filename = f"page{page_num+1:04d}_img{img_idx+1:03d}.png"
                img_path = os.path.join(output_images_dir or "/tmp/pazpik_images", img_filename)

                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                pix.save(img_path)

                # Klasyfikacja: czy to schemat?
                is_schematic = classify_as_schematic(img_path, schematic_threshold)

                if is_schematic:
                    print(f"    [SCHEMAT] Znaleziono schemat: {img_filename}")
                    if ollama_client:
                        try:
                            description = ollama_client.describe_image(
                                image_path=img_path,
                                prompt=vision_prompt,
                                model=vision_model,
                            )
                            print(f"      Opis: {description[:100]}...")
                            schematic_descriptions.append({
                                "y0": bbox[1],
                                "description": f"[SCHEMAT: {description}]"
                            })
                        except Exception as e:
                            print(f"      [BŁĄD] Nie udało się opisać schematu: {e}")
                            schematic_descriptions.append({
                                "y0": bbox[1],
                                "description": f"[SCHEMAT - nie udało się opisać]"
                            })
                    else:
                        schematic_descriptions.append({
                            "y0": bbox[1],
                            "description": f"[SCHEMAT (do opisania)]"
                        })
                else:
                    print(f"    [ZDJĘCIE] Pomijam (nie jest schematem): {img_filename}")

                pix = None

            except Exception as e:
                print(f"    [BŁĄD] Problem z ekstrakcją obrazu: {e}")

        # 3. Interleaving: połącz tekst i opisy schematów (sortowanie po Y)
        all_elements = []
        for tb in text_blocks:
            all_elements.append({"y0": tb["y0"], "type": "text", "content": tb["text"]})
        for sd in schematic_descriptions:
            all_elements.append({"y0": sd["y0"], "type": "schematic", "content": sd["description"]})

        all_elements.sort(key=lambda x: x["y0"])

        page_text = ""
        for elem in all_elements:
            page_text += elem["content"] + "\n"

        full_text_pages.append(page_text)
        print(f"    Wyodrębniono {len(text_blocks)} bloków tekstu, "
              f"{len(schematic_descriptions)} schematów")

    doc.close()

    result = "\n\n".join(full_text_pages)
    print(f"\n  [OK] Przetworzono {total_pages} stron. Łącznie {len(result)} znaków.")
    return result


def process_pdfs_to_corpus(
    pdf_dir: str,
    output_file: str,
    vision_model: str = "gemma4-vision:9b",
    vision_prompt: str = DEFAULT_CONFIG["vision_prompt"],
    ollama_client: Optional[OllamaClient] = None,
) -> str:
    """
    Przetwarza wszystkie PDFy w katalogu i tworzy jeden korpus tekstowy.
    """
    pdf_dir = Path(pdf_dir)
    output_file = Path(output_file)
    images_dir = str(pdf_dir.parent / "temp_images")

    if not pdf_dir.exists():
        print(f"\n[!] Katalog {pdf_dir} nie istnieje!")
        print("    Utwórz go i umieść w nim pliki PDF do fine-tuningu.")
        return ""

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[!] Brak plików PDF w katalogu: {pdf_dir}")
        print("    Umieść pliki PDF i uruchom ponownie.")
        return ""

    print(f"\n{'#'*60}")
    print(f"  ZNALEZIONO {len(pdf_files)} PLIKÓW PDF:")
    print(f"{'#'*60}")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"    {i}. {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")

    all_text = []
    for pdf_path in pdf_files:
        try:
            text = extract_pdf_content(
                pdf_path=str(pdf_path),
                output_images_dir=str(images_dir),
                vision_model=vision_model,
                vision_prompt=vision_prompt,
                ollama_client=ollama_client,
            )
            if text.strip():
                all_text.append(text)
                print(f"  [OK] Dodano: {pdf_path.name}")
        except Exception as e:
            print(f"\n  [BŁĄD] Nie przetworzono {pdf_path.name}: {e}")

    if not all_text:
        print("\n[!] Nie udało się wyodrębnić tekstu z żadnego PDF.")
        return ""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    full_corpus = "\n\n".join(all_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    total_chars = len(full_corpus)
    total_words = len(full_corpus.split())
    print(f"\n{'='*60}")
    print(f"  KORPUS GOTOWY!")
    print(f"  Plik:    {output_file}")
    print(f"  Znaków:  {total_chars:,}")
    print(f"  Słów:    {total_words:,}")
    print(f"  Szac. tokenów (dla modelu 31B): ~{int(total_chars * 0.35):,}")
    print(f"{'='*60}")

    return full_corpus


def chunk_text_for_rag(
    pdf_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Dzieli PDFy z katalogu na chunki dla RAG.
    Zwraca listę słowników: {"text": ..., "source": ..., "page": ...}
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        print(f"\n[!] Katalog {pdf_dir} nie istnieje!")
        return []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n[!] Brak plików PDF w katalogu RAG: {pdf_dir}")
        return []

    print(f"\n{'='*60}")
    print(f"  PRZETWARZANIE PDF DLA RAG - {len(pdf_files)} plików")
    print(f"{'='*60}")

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\n  Przetwarzanie: {pdf_path.name}")
        try:
            doc = fitz.open(str(pdf_path))
            full_text = ""
            page_map = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text += text + "\n"
                    page_map.append((len(full_text), page_num + 1, pdf_path.name))

            doc.close()

            # Chunkowanie
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

            chunks = splitter.split_text(full_text)
            print(f"    Utworzono {len(chunks)} chunków")

            for chunk_idx, chunk_text in enumerate(chunks):
                # Znajdź stronę
                page_no = 1
                for pos, p, name in page_map:
                    if chunk_idx * chunk_size < pos:
                        page_no = p
                        break

                all_chunks.append({
                    "text": chunk_text,
                    "source": pdf_path.name,
                    "page": page_no,
                    "chunk_id": f"{pdf_path.name}_p{page_no}_c{chunk_idx}",
                })

        except Exception as e:
            print(f"  [BŁĄD] Problem z {pdf_path.name}: {e}")

    print(f"\n  [OK] Łącznie utworzono {len(all_chunks)} chunków z {len(pdf_files)} PDFów")
    return all_chunks


def estimate_tokens(text: str) -> int:
    """Szacuje liczbę tokenów (ok. 0.35 * liczba znaków dla modeli 31B)."""
    return int(len(text) * 0.35)
