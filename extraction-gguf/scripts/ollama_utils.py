"""
PAZPIK - ollama_utils.py
Komunikacja z API Ollamy.
"""

import os
import sys
import json
import base64
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


# ============================================================
# KLIENT OLLAMA
# ============================================================

class OllamaClient:
    """Klient do komunikacji z API Ollamy."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._check_connection()

    def _check_connection(self) -> bool:
        """Sprawdza czy Ollama jest uruchomiona."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            print(f"\n[!] Nie można połączyć się z Ollama API ({self.base_url})")
            print("    Upewnij się, że Ollama jest uruchomiona:")
            print("      sudo systemctl start ollama")
            print("      # lub: ollama serve")
            return False
        except Exception as e:
            print(f"\n[!] Błąd połączenia z Ollama: {e}")
            return False
        return False

    def list_models(self) -> list[dict]:
        """Zwraca listę zainstalowanych modeli Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                result = []
                for m in models:
                    name = m.get("name", "unknown")
                    size_bytes = m.get("size", 0)
                    size_gb = size_bytes / (1024**3)
                    modified = m.get("modified_at", "")[:19]
                    details = m.get("details", {})
                    result.append({
                        "name": name,
                        "size_gb": round(size_gb, 2),
                        "modified": modified,
                        "parameter_size": details.get("parameter_size", "?"),
                        "quantization": details.get("quantization_level", "?"),
                    })
                return result
            return []
        except Exception as e:
            print(f"[!] Błąd pobierania listy modeli: {e}")
            return []

    def show_model(self, model_name: str) -> dict:
        """Pobiera szczegóły modelu (Modelfile, parametry)."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            return {}
        except Exception as e:
            print(f"[!] Błąd pobierania informacji o modelu {model_name}: {e}")
            return {}

    def generate(self, model: str, prompt: str, system: str = "",
                 stream: bool = False, max_tokens: int = 512) -> str:
        """Wysyła prompt do modelu i zwraca odpowiedź."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_predict": max_tokens,
                },
            }
            if system:
                payload["system"] = system

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "")
            return f"[Błąd {resp.status_code}]"
        except Exception as e:
            return f"[Błąd: {e}]"

    def describe_image(self, image_path: str, prompt: str,
                       model: str = "gemma4-vision:9b") -> str:
        """
        Wysyła obraz do modelu vision i zwraca opis.
        Używa API Ollama z obrazem zakodowanym w base64.
        """
        try:
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "images": [img_base64],
                "options": {"num_predict": 512},
            }

            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )

            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
            else:
                return f"[Błąd API: {resp.status_code}]"

        except FileNotFoundError:
            return "[Błąd: plik obrazu nie istnieje]"
        except Exception as e:
            return f"[Błąd: {e}]"

    def pull_model(self, model_name: str) -> bool:
        """Pobiera model z rejestru Ollamy."""
        print(f"\n  Pobieranie modelu {model_name}...")
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600,
            )
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "downloading" in status:
                                pass
                            elif status:
                                print(f"    {status}")
                        except json.JSONDecodeError:
                            pass
                return True
            else:
                print(f"  [BŁĄD] Kod odpowiedzi: {resp.status_code}")
                return False
        except Exception as e:
            print(f"  [BŁĄD] {e}")
            return False

    def create_model(self, model_name: str, modelfile_content: str) -> bool:
        """Tworzy nowy model w Ollamie z Modelfile."""
        print(f"\n  Tworzenie modelu Ollama: {model_name}")
        try:
            resp = requests.post(
                f"{self.base_url}/api/create",
                json={"name": model_name, "modelfile": modelfile_content},
                stream=True,
                timeout=600,
            )
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if status:
                                print(f"    {status}")
                        except json.JSONDecodeError:
                            pass
                return True
            else:
                error_text = resp.text[:200]
                print(f"  [BŁĄD] Kod {resp.status_code}: {error_text}")
                return False
        except Exception as e:
            print(f"  [BŁĄD] {e}")
            return False


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def list_ollama_models(client: OllamaClient) -> list[dict]:
    """Wyświetla i zwraca listę modeli."""
    models = client.list_models()
    if not models:
        print("\n[!] Brak zainstalowanych modeli w Ollama.")
        print("    Zainstaluj model: ollama pull <nazwa_modelu>")
        return []

    print(f"\n{'='*60}")
    print(f"  ZAINSTALOWANE MODELE W OLLAMA ({len(models)}):")
    print(f"{'='*60}")
    print(f"  {'Lp.':<5} {'Nazwa modelu':<30} {'Rozmiar':<10} {'Parametry':<10} {'Kwanty':<8}")
    print(f"  {'-'*63}")
    for i, m in enumerate(models, 1):
        print(f"  {i:<5} {m['name']:<30} {m['size_gb']:<10} {m['parameter_size']:<10} {m['quantization']:<8}")
    return models


def get_hf_model_name(client: OllamaClient, ollama_name: str,
                      model_mapping: dict) -> str:
    """
    Próbuje znaleźć nazwę modelu na HuggingFace na podstawie
    nazwy modelu Ollama.
    """
    # 1. Sprawdź mapowanie konfiguracyjne
    if ollama_name in model_mapping:
        hf_name = model_mapping[ollama_name]
        print(f"\n  [OK] Znaleziono mapowanie: {ollama_name} -> {hf_name}")
        return hf_name

    # 2. Spróbuj z ollama show
    try:
        show_info = client.show_model(ollama_name)
        modelfile = show_info.get("modelfile", "")
        for line in modelfile.split("\n"):
            line = line.strip()
            if line.startswith("FROM "):
                from_val = line[5:].strip()
                if "/" in from_val and ":" in from_val:
                    print(f"\n  [OK] Odczytano z Modelfile: {from_val}")
                    return from_val
    except Exception:
        pass

    # 3. Nie znaleziono - zapytaj użytkownika
    print(f"\n{'='*60}")
    print(f"  NIE ZNALEZIONO MAPOWANIA DLA: {ollama_name}")
    print(f"  Podaj nazwę modelu na HuggingFace")
    print(f"  (np. google/gemma-4-31b-it)")
    print(f"{'='*60}")
    hf_name = input("  > ").strip()
    return hf_name if hf_name else ollama_name


def create_ollama_model_from_gguf(model_name: str, gguf_path: str,
                                   client: OllamaClient) -> bool:
    """Tworzy model w Ollamie z pliku GGUF."""
    modelfile = f"FROM {gguf_path}"
    return client.create_model(model_name, modelfile)


def create_ollama_model_from_dir(model_name: str, model_dir: str,
                                  client: OllamaClient) -> bool:
    """Tworzy model w Ollamie z katalogu (safetensors -> konwersja)."""
    modelfile = f"FROM {model_dir}"
    return client.create_model(model_name, modelfile)
