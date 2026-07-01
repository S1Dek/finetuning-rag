#!/usr/bin/env python3
"""
=============================================================================
  PAZPIK - GGUF Converter
=============================================================================
  Konwersja pomiędzy formatem GGUF (Ollama) a safetensors (HuggingFace).

  Funkcje:
    - find_gguf_blob()  → lokalizacja pliku GGUF w katalogu Ollamy
    - gguf_to_safetensors() → konwersja GGUF → HF (safetensors + config.json)
    - safetensors_to_gguf() → konwersja HF → GGUF (przez llama.cpp)
=============================================================================
"""

import os
import sys
import json
import struct
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ============================================================
# STAŁE GGUF
# ============================================================

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# Typy tensorów w GGUF
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS  = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S   = 19
GGML_TYPE_IQ4_NL  = 20
GGML_TYPE_IQ3_S   = 21
GGML_TYPE_IQ2_S   = 22
GGML_TYPE_IQ4_XS  = 23
GGML_TYPE_I8      = 24
GGML_TYPE_I16     = 25
GGML_TYPE_I32     = 26
GGML_TYPE_I64     = 27
GGML_TYPE_F64     = 28
GGML_TYPE_IQ1_M   = 29
GGML_TYPE_BF16    = 30

TYPE_NAMES = {
    GGML_TYPE_F32: "f32", GGML_TYPE_F16: "f16",
    GGML_TYPE_Q4_0: "q4_0", GGML_TYPE_Q4_1: "q4_1",
    GGML_TYPE_Q5_0: "q5_0", GGML_TYPE_Q5_1: "q5_1",
    GGML_TYPE_Q8_0: "q8_0", GGML_TYPE_Q8_1: "q8_1",
    GGML_TYPE_Q2_K: "q2_K", GGML_TYPE_Q3_K: "q3_K",
    GGML_TYPE_Q4_K: "q4_K", GGML_TYPE_Q5_K: "q5_K",
    GGML_TYPE_Q6_K: "q6_K", GGML_TYPE_Q8_K: "q8_K",
    GGML_TYPE_BF16: "bf16",
}

# ============================================================
# MAPOWANIE NAZW TENSORÓW: GGUF → HuggingFace
# ============================================================

# Nazwy tensorów wspólne dla wszystkich warstw
GLOBAL_TENSOR_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}

# Nazwy tensorów dla poszczególnych warstw (blk.{i}.{nazwa}.weight)
LAYER_TENSOR_MAP = {
    "attn_norm": "input_layernorm",
    "attn_norm_2": "input_layernorm",      # Gemma 2
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_norm": "post_attention_layernorm",
    "ffn_norm_2": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_2": "mlp.gate_proj",          # Gemma 2 MoE
    "ffn_up_2": "mlp.up_proj",              # Gemma 2 MoE
    "ffn_down_2": "mlp.down_proj",          # Gemma 2 MoE
}

# Odwrotne mapowanie: HF → GGUF (dla konwersji powrotnej)
HF_TO_GGUF_GLOBAL = {v: k for k, v in GLOBAL_TENSOR_MAP.items()}
HF_TO_GGUF_LAYER = {v: k for k, v in LAYER_TENSOR_MAP.items()}


# ============================================================
# DEKWANTYZACJA
# ============================================================

def dequantize_q8_0(data: np.ndarray, shape) -> np.ndarray:
    """Dequantyzacja Q8_0: blok 32 wartości, każda int8, skala float16."""
    block_size = 32
    n_blocks = data.shape[0] // (block_size + 1)  # 32 wartości + 2 bajty skali na blok

    result = np.zeros(n_blocks * block_size, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * (block_size + 2)
        scale = struct.unpack("<e", data[offset:offset+2])[0]  # float16
        quantized = np.frombuffer(data[offset+2:offset+2+block_size], dtype=np.int8)
        result[i*block_size:(i+1)*block_size] = quantized.astype(np.float32) * scale

    return result.reshape(shape)


def dequantize_q4_0(data: np.ndarray, shape) -> np.ndarray:
    """Dequantyzacja Q4_0: blok 32 wartości w 4-bitach, skala float16."""
    block_size = 32
    # 2 bajty skali + 16 bajtów danych (po 2 wartości na bajt) = 18 bajtów/blok
    block_bytes = 2 + block_size // 2

    n_blocks = data.shape[0] // block_bytes
    result = np.zeros(n_blocks * block_size, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * block_bytes
        scale = struct.unpack("<e", data[offset:offset+2])[0]
        quantized = np.frombuffer(data[offset+2:offset+2+block_size//2], dtype=np.uint8)

        for j in range(block_size // 2):
            low = quantized[j] & 0x0F
            high = (quantized[j] >> 4) & 0x0F
            result[i*block_size + j*2] = (low - 8) * scale
            result[i*block_size + j*2 + 1] = (high - 8) * scale

    return result.reshape(shape)


def dequantize_q4_k(data: np.ndarray, shape) -> np.ndarray:
    """
    Dequantyzacja Q4_K: blok K-means z 16 podblokami.
    Uproszczona implementacja - dla pełnej dokładności zalecane użycie llama.cpp.
    """
    block_size = 256
    # Q4_K: 6 super-skal, 16 skal podbloków (po 16 elementów), 128 bajtów danych
    # Struktura: 6 bajtów super-skale + 32 bajty skale podbloków + 128 bajtów dane = 166 bajtów
    block_bytes = 166

    n_blocks = data.shape[0] // block_bytes
    n_elements = n_blocks * block_size

    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * block_bytes
        # Odczytaj skale (uproszczenie: traktuj jako 16 float16 skal)
        scales = np.frombuffer(data[offset+6:offset+6+32], dtype=np.float16).astype(np.float32)
        min_vals = np.zeros(16, dtype=np.float32)

        # Odczytaj dane 4-bit
        data_start = offset + 6 + 32
        for sub in range(16):
            sub_vals = np.zeros(16, dtype=np.float32)
            for j in range(8):  # 8 bajtów na 16 wartości 4-bit
                byte_val = data[data_start + sub * 8 + j]
                low = byte_val & 0x0F
                high = (byte_val >> 4) & 0x0F
                sub_vals[j*2] = min_vals[sub] + low * scales[sub] / 16.0
                sub_vals[j*2+1] = min_vals[sub] + high * scales[sub] / 16.0
            result[i*block_size + sub*16:(i+1)*block_size + sub*16] = sub_vals

    return result.reshape(shape)


def dequantize_tensor(tensor_data: np.ndarray, tensor_type: int,
                      shape: tuple) -> torch.Tensor:
    """Dequantyzuje tensor do torch.float16."""
    if tensor_type == GGML_TYPE_F32:
        arr = tensor_data.astype(np.float32)
    elif tensor_type == GGML_TYPE_F16:
        arr = tensor_data.astype(np.float16)
    elif tensor_type == GGML_TYPE_BF16:
        # bfloat16 - konwersja przez uint32
        arr = tensor_data.view(np.uint16).astype(np.float32) * 0.00000095367431640625
    elif tensor_type == GGML_TYPE_Q8_0:
        arr = dequantize_q8_0(tensor_data, shape)
    elif tensor_type == GGML_TYPE_Q4_0:
        arr = dequantize_q4_0(tensor_data, shape)
    elif tensor_type == GGML_TYPE_Q4_K or tensor_type == GGML_TYPE_Q5_K or tensor_type == GGML_TYPE_Q6_K:
        arr = dequantize_q4_k(tensor_data, shape)
    else:
        raise ValueError(f"Niewspierany typ tensora: {tensor_type} ({TYPE_NAMES.get(tensor_type, '?')})")

    return torch.from_numpy(arr).to(torch.float16)


# ============================================================
# ODCZYT PLIKU GGUF
# ============================================================

def read_gguf_header(filepath: str) -> tuple:
    """Odczytuje nagłówek GGUF i zwraca (metadata, tensor_infos)."""
    with open(filepath, "rb") as f:
        # Nagłówek: magic (4B), version (4B), tensor_count (8B), metadata_kv_count (8B)
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Nieprawidłowy magic number GGUF: {hex(magic)}")

        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        print(f"  Wersja GGUF: {version}")
        print(f"  Liczba tensorów: {tensor_count}")
        print(f"  Liczba metadata KV: {metadata_kv_count}")

        # Odczytaj metadata KV
        metadata = {}
        for _ in range(metadata_kv_count):
            key = _read_gguf_string(f)
            value = _read_gguf_value(f)
            metadata[key] = value

        # Odczytaj informacje o tensorach
        tensor_infos = []
        for _ in range(tensor_count):
            name = _read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{'Q' * n_dims}", f.read(8 * n_dims))
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_infos.append({
                "name": name,
                "dims": tuple(dims),
                "ggml_type": ggml_type,
                "offset": offset,
            })

        # Pozycja danych tensorów (po align do 32)
        data_offset = f.tell()
        # Align to 32
        data_offset = (data_offset + 31) & ~31

    print(f"  Offset danych: {data_offset}")
    return metadata, tensor_infos, data_offset


def _read_gguf_string(f) -> str:
    """Odczytuje string w formacie GGUF: długość (8B) + dane UTF-8."""
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_gguf_value(f):
    """Odczytuje wartość metadata w formacie GGUF."""
    value_type = struct.unpack("<I", f.read(4))[0]
    if value_type == 0:  # uint8
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == 1:  # int8
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == 2:  # uint16
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == 3:  # int16
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == 4:  # uint32
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == 5:  # int32
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == 6:  # float32
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == 7:  # bool
        return bool(struct.unpack("<B", f.read(1))[0])
    elif value_type == 8:  # string
        return _read_gguf_string(f)
    elif value_type == 9:  # array
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(arr_len):
            # Dla array, wartości nie mają prefixu typu - używamy arr_type
            if arr_type == 8:  # string array
                arr.append(_read_gguf_string(f))
            elif arr_type == 5:  # int32 array
                arr.append(struct.unpack("<i", f.read(4))[0])
            elif arr_type == 6:  # float32 array
                arr.append(struct.unpack("<f", f.read(4))[0])
            else:
                arr.append(f"<unsupported array type {arr_type}>")
        return arr
    elif value_type == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]
    else:
        return f"<unknown type {value_type}>"


def read_tensor_data(filepath: str, offset: int, ggml_type: int,
                     dims: tuple) -> np.ndarray:
    """Odczytuje surowe dane tensora z pliku GGUF."""
    # Oblicz rozmiar w bajtach
    n_elems = 1
    for d in dims:
        n_elems *= d

    type_size = _ggml_type_size(ggml_type)
    n_bytes = (n_elems * type_size + 31) // 32 * 32  # align to 32

    with open(filepath, "rb") as f:
        f.seek(offset)
        data = np.frombuffer(f.read(n_bytes), dtype=np.uint8)

    return data


def _ggml_type_size(ggml_type: int) -> int:
    """Zwraca rozmiar w bajtach na element dla danego typu GGML."""
    sizes = {
        GGML_TYPE_F32: 4, GGML_TYPE_F16: 2, GGML_TYPE_BF16: 2,
        GGML_TYPE_Q4_0: 18, GGML_TYPE_Q4_1: 20,
        GGML_TYPE_Q5_0: 22, GGML_TYPE_Q5_1: 24,
        GGML_TYPE_Q8_0: 34, GGML_TYPE_Q8_1: 36,
        GGML_TYPE_Q2_K: 70, GGML_TYPE_Q3_K: 104,
        GGML_TYPE_Q4_K: 166, GGML_TYPE_Q5_K: 198,
        GGML_TYPE_Q6_K: 230, GGML_TYPE_Q8_K: 294,
    }
    return sizes.get(ggml_type, 4)


# ============================================================
# MAPOWANIE NAZW TENSORÓW
# ============================================================

def gguf_name_to_hf(tensor_name: str, architecture: str) -> Optional[str]:
    """Konwertuje nazwę tensora z GGUF na format HuggingFace."""
    # Tensory globalne (bez prefiksu blk.)
    if tensor_name in GLOBAL_TENSOR_MAP:
        return GLOBAL_TENSOR_MAP[tensor_name]

    # Tensory per-warstwa: blk.{layer}.{nazwa}.{suffix}
    if tensor_name.startswith("blk."):
        parts = tensor_name.split(".")
        if len(parts) >= 4:
            layer = parts[1]
            name = parts[2]
            suffix = parts[3] if len(parts) > 3 else "weight"

            if name in LAYER_TENSOR_MAP:
                hf_name = LAYER_TENSOR_MAP[name]
                return f"model.layers.{layer}.{hf_name}.{suffix}"

    return None


def hf_name_to_gguf(tensor_name: str) -> Optional[str]:
    """Konwertuje nazwę tensora z HuggingFace na GGUF."""
    # Tensory globalne
    if tensor_name in HF_TO_GGUF_GLOBAL:
        return HF_TO_GGUF_GLOBAL[tensor_name]

    # Tensory per-warstwa
    parts = tensor_name.split(".")
    # Format: model.layers.{layer}.{submodule}.{name}.weight
    if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
        layer = parts[2]
        submodule = ".".join(parts[3:-1])
        suffix = parts[-1]

        for hf_sub, gguf_sub in HF_TO_GGUF_LAYER.items():
            if submodule == hf_sub:
                return f"blk.{layer}.{gguf_sub}.{suffix}"

    return None


# ============================================================
# BUDOWANIE KONFIGURACJI MODELU (config.json)
# ============================================================

def build_hf_config(metadata: dict, tensor_infos: list) -> dict:
    """Tworzy config.json w formacie HuggingFace na podstawie metadata GGUF."""
    arch = metadata.get("general.architecture", "unknown")
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8") if hasattr(arch, 'decode') else str(arch)

    # Mapowanie architektury GGUF na HF
    arch_map = {
        "gemma": "GemmaForCausalLM",
        "gemma2": "Gemma2ForCausalLM",
        "gemma3": "Gemma3ForCausalLM",
        "gemma4": "Gemma4ForCausalLM",
        "llama": "LlamaForCausalLM",
        "llama2": "LlamaForCausalLM",
        "llama3": "LlamaForCausalLM",
        "mistral": "MistralForCausalLM",
        "mixtral": "MixtralForCausalLM",
        "falcon": "FalconForCausalLM",
        "qwen2": "Qwen2ForCausalLM",
        "starcoder2": "Starcoder2ForCausalLM",
        "phi3": "Phi3ForCausalLM",
        "dbrx": "DbrxForCausalLM",
    }

    hf_arch = arch_map.get(arch, f"{arch.capitalize()}ForCausalLM")

    # Wyciągnij hiperparametry
    def get_meta(key: str, default=None):
        val = metadata.get(key, default)
        if isinstance(val, bytes):
            val = val.decode("utf-8") if hasattr(val, 'decode') else str(val)
        return val

    hidden_size = get_meta(f"{arch}.embedding_length") or get_meta(f"{arch}.hidden_size") or 4096
    if isinstance(hidden_size, (list, tuple)):
        hidden_size = hidden_size[0]

    num_layers = get_meta(f"{arch}.block_count") or get_meta(f"{arch}.num_hidden_layers") or 32
    if isinstance(num_layers, (list, tuple)):
        num_layers = num_layers[0]

    num_heads = get_meta(f"{arch}.head_count") or 32
    if isinstance(num_heads, (list, tuple)):
        num_heads = num_heads[0]

    num_kv_heads = get_meta(f"{arch}.head_count_kv") or num_heads
    if isinstance(num_kv_heads, (list, tuple)):
        num_kv_heads = num_kv_heads[0]

    intermediate_size = get_meta(f"{arch}.feed_forward_length") or hidden_size * 4
    if isinstance(intermediate_size, (list, tuple)):
        intermediate_size = intermediate_size[0]

    vocab_size = get_meta(f"{arch}.vocab_size") or 256000
    if isinstance(vocab_size, (list, tuple)):
        vocab_size = vocab_size[0]

    rope_theta = get_meta(f"{arch}.rope.freq_base") or 10000.0
    if isinstance(rope_theta, (list, tuple)):
        rope_theta = rope_theta[0]

    norm_eps = get_meta(f"{arch}.attention.layer_norm_rms_epsilon") or get_meta(f"{arch}.norm_eps") or 1e-6
    if isinstance(norm_eps, (list, tuple)):
        norm_eps = norm_eps[0]

    # Sprawdź typ aktywacji
    act_fn = str(get_meta(f"{arch}.activation_type") or "gelu_pytorch_tanh")

    # Sprawdź czy model używa MoE
    num_experts = get_meta(f"{arch}.expert_count") or 0
    num_experts_per_tok = get_meta(f"{arch}.expert_used_count") or 0
    if isinstance(num_experts, bytes):
        num_experts = 0
    if isinstance(num_experts_per_tok, bytes):
        num_experts_per_tok = 0

    config = {
        "architectures": [hf_arch],
        "model_type": arch,
        "hidden_size": int(hidden_size),
        "num_hidden_layers": int(num_layers),
        "num_attention_heads": int(num_heads),
        "num_key_value_heads": int(num_kv_heads),
        "intermediate_size": int(intermediate_size),
        "vocab_size": int(vocab_size),
        "rope_theta": float(rope_theta),
        "rms_norm_eps": float(norm_eps),
        "hidden_act": act_fn,
        "torch_dtype": "float16",
        "use_cache": True,
        "bos_token_id": get_meta("tokenizer.ggml.bos_token_id") or 1,
        "eos_token_id": get_meta("tokenizer.ggml.eos_token_id") or 2,
        "tie_word_embeddings": False,
    }

    if isinstance(config["bos_token_id"], bytes):
        config["bos_token_id"] = 1
    if isinstance(config["eos_token_id"], bytes):
        config["eos_token_id"] = 2

    if int(num_experts) > 0:
        config["num_local_experts"] = int(num_experts)
        config["num_experts_per_tok"] = int(num_experts_per_tok)

    # Specyficzne dla Gemma
    if arch == "gemma4":
        config["head_dim"] = int(hidden_size) // int(num_heads)
        config["hidden_act"] = "gelu_pytorch_tanh"
        config["sliding_window"] = metadata.get(f"{arch}.attn_sliding_window", 4096)

    return config


# ============================================================
# GŁÓWNE FUNKCJE KONWERSJI
# ============================================================

def find_gguf_blob(model_name: str,
                   ollama_dir: Optional[str] = None) -> Optional[Path]:
    """
    Znajduje plik GGUF w katalogu Ollamy dla podanego modelu.
    """
    if ollama_dir is None:
        ollama_dir = Path.home() / ".ollama" / "models"

    ollama_dir = Path(ollama_dir).expanduser()

    # Szukaj manifestu
    # Format: ~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>
    manifests_dir = ollama_dir / "manifests"
    if not manifests_dir.exists():
        print(f"  [!] Katalog manifestów nie istnieje: {manifests_dir}")
        return None

    # Szukaj we wszystkich podkatalogach
    for manifest_path in manifests_dir.rglob("*"):
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)

                config = data.get("config", {})
                if model_name in config.get("digest", ""):
                    blobs_dir = ollama_dir / "blobs"
                    blob_path = blobs_dir / config["digest"].replace(":", "-")
                    if blob_path.exists():
                        return blob_path

                # Szukaj w layers
                for layer in data.get("layers", []):
                    digest = layer.get("digest", "")
                    if digest:
                        blob_path = ollama_dir / "blobs" / digest.replace(":", "-")
                        if blob_path.exists():
                            # Sprawdź czy to GGUF po rozmiarze (GGUF > 100MB)
                            if blob_path.stat().st_size > 100 * 1024 * 1024:
                                return blob_path
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    print(f"  [!] Nie znaleziono pliku GGUF dla modelu {model_name}")
    print(f"      Szukano w: {manifests_dir}")
    return None


def gguf_to_safetensors(gguf_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Konwertuje plik GGUF na format HuggingFace (safetensors + config.json).
    """
    from safetensors.torch import save_file as sf_save

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Konwersja GGUF -> safetensors")
    print(f"  Źródło: {gguf_path}")
    print(f"  Cel:    {output_dir}")

    size_gb = gguf_path.stat().st_size / (1024**3)
    print(f"  Rozmiar pliku: {size_gb:.2f} GB")

    # ----------------------------------------------------------
    # Krok 1: Odczytaj nagłówek GGUF
    # ----------------------------------------------------------
    print(f"\n  [1/4] Odczyt nagłówka GGUF...")
    metadata, tensor_infos, data_offset = read_gguf_header(str(gguf_path))

    architecture = metadata.get("general.architecture", "unknown")
    if isinstance(architecture, bytes):
        architecture = architecture.decode("utf-8")
    print(f"  Architektura: {architecture}")

    print(f"  Liczba tensorów: {len(tensor_infos)}")

    # ----------------------------------------------------------
    # Krok 2: Konwertuj tensory
    # ----------------------------------------------------------
    print(f"\n  [2/4] Konwersja tensorów...")
    tensor_dict = {}
    converted = 0
    errors = 0
    skipped = 0

    for i, info in enumerate(tensor_infos):
        hf_name = gguf_name_to_hf(info["name"], architecture)
        if hf_name is None:
            print(f"    Pomijam (brak mapowania): {info['name']}")
            skipped += 1
            continue

        type_name = TYPE_NAMES.get(info["ggml_type"], f"type_{info['ggml_type']}")
        dims_str = " x ".join(str(d) for d in info["dims"])

        print(f"    [{i+1}/{len(tensor_infos)}] {hf_name} ({type_name}, {dims_str})", end=" ")

        try:
            # Odczytaj surowe dane
            tensor_offset = data_offset + info["offset"]
            raw_data = read_tensor_data(str(gguf_path), tensor_offset, info["ggml_type"], info["dims"])
            print(f"- odczytano {len(raw_data)} bajtów", end=" ")

            # Dequantyzuj
            tensor = dequantize_tensor(raw_data, info["ggml_type"], info["dims"])
            print(f"- dequantyzowano do {tensor.shape}", end="")

            tensor_dict[hf_name] = tensor
            converted += 1
            print()
        except Exception as e:
            errors += 1
            print(f" [BŁĄD] {e}")

    print(f"\n  Podsumowanie konwersji:")
    print(f"    Przekonwertowano: {converted}")
    print(f"    Pominięto:        {skipped}")
    print(f"    Błędów:           {errors}")

    if not tensor_dict:
        print(f"\n  [BŁĄD] Nie przekonwertowano żadnego tensora!")
        return None

    # ----------------------------------------------------------
    # Krok 3: Zapisz plik safetensors
    # ----------------------------------------------------------
    print(f"\n  [3/4] Zapis pliku safetensors...")

    # Podziel na shardy jeśli > 2GB
    total_bytes = sum(t.element_size() * t.numel() for t in tensor_dict.values())
    print(f"    Łączny rozmiar tensorów: {total_bytes / (1024**3):.2f} GB")

    if total_bytes > 2 * 1024**3:
        # Podziel na shardy po ~2GB
        max_shard_size = 2 * 1024 ** 3
        current_size = 0
        shard_idx = 1
        shard_data = {}
        shard_files = []

        for name, tensor in tensor_dict.items():
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > max_shard_size and shard_data:
                # Zapisz bieżący shard
                shard_path = output_dir / f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                sf_save(shard_data, str(shard_path))
                shard_files.append(shard_path.name)
                print(f"    Zapisano shard {shard_idx}")
                shard_data = {}
                current_size = 0
                shard_idx += 1

            shard_data[name] = tensor
            current_size += tensor_size

        # Zapisz ostatni shard
        if shard_data:
            shard_path = output_dir / f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            sf_save(shard_data, str(shard_path))
            shard_files.append(shard_path.name)
            print(f"    Zapisano shard {shard_idx}")

        # Zaktualizuj nazwy shardów
        total_shards = shard_idx
        for old_name in shard_files:
            old_path = output_dir / old_name
            new_name = old_name.replace("XXXXX", f"{total_shards:05d}")
            old_path.rename(output_dir / new_name)

        # Zapisz model_index
        index_data = {
            "metadata": {"total_size": total_bytes},
            "weight_map": {},
        }

        current_size = 0
        shard_idx = 1
        for name, tensor in tensor_dict.items():
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > max_shard_size:
                current_size = 0
                shard_idx += 1
            index_data["weight_map"][name] = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
            current_size += tensor_size

        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index_data, f, indent=2)
    else:
        tensor_path = output_dir / "model.safetensors"
        sf_save(tensor_dict, str(tensor_path))
        print(f"    Zapisano: {tensor_path}")

    # ----------------------------------------------------------
    # Krok 4: Zapisz config.json
    # ----------------------------------------------------------
    print(f"\n  [4/4] Zapis config.json...")
    config = build_hf_config(metadata, tensor_infos)
    config_path = output_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"    Zapisano: {config_path}")
    print(f"    Architektura HF: {config.get('architectures', ['?'])[0]}")
    print(f"    hidden_size: {config.get('hidden_size', '?')}")
    print(f"    num_hidden_layers: {config.get('num_hidden_layers', '?')}")
    print(f"    num_attention_heads: {config.get('num_attention_heads', '?')}")

    print(f"\n  [OK] Konwersja GGUF -> safetensors zakończona!")
    print(f"      Wynik: {output_dir}")

    return output_dir


def safetensors_to_gguf(input_dir: Path, output_path: Path,
                        llamacpp_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Konwertuje model w formacie HuggingFace (safetensors) na GGUF.
    Używa llama.cpp convert_hf_to_gguf.py jeśli dostępny.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  Konwersja safetensors -> GGUF")
    print(f"  Źródło: {input_dir}")
    print(f"  Cel:    {output_path}")

    # ----------------------------------------------------------
    # Próba 1: Użyj llama.cpp
    # ----------------------------------------------------------
    if llamacpp_dir is None:
        llamacpp_dir = (Path(__file__).resolve().parent.parent / "tools" / "llama.cpp")

    convert_script = Path(llamacpp_dir) / "convert_hf_to_gguf.py"

    if convert_script.exists():
        print(f"\n  [1/2] Używam konwertera llama.cpp...")
        cmd = [
            sys.executable, str(convert_script),
            str(input_dir),
            "--outfile", str(output_path),
            "--outtype", "q4_k_m",
        ]

        print(f"    Komenda: {' '.join(cmd)}")
        print(f"    To może potrwać kilka minut...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.returncode == 0:
                print(f"    [OK] Konwersja udana!")
                if output_path.exists():
                    size_mb = output_path.stat().st_size / (1024**2)
                    print(f"    Rozmiar: {size_mb:.0f} MB")
                    return output_path
            else:
                print(f"    [BŁĄD] Konwersja nieudana (kod: {result.returncode})")
                if result.stderr:
                    print(f"    Szczegóły: {result.stderr[:500]}")
                if result.stdout:
                    print(f"    Stdout: {result.stdout[:300]}")
        except subprocess.TimeoutExpired:
            print(f"    [BŁĄD] Przekroczono limit czasu (1h)")
        except Exception as e:
            print(f"    [BŁĄD] Wyjątek: {e}")
    else:
        print(f"\n  [1/2] Konwerter llama.cpp nie znaleziony w:")
        print(f"      {convert_script}")
        print(f"    Pobierz go:")
        print(f"      git clone https://github.com/ggml-org/llama.cpp {llamacpp_dir}")

    # ----------------------------------------------------------
    # Próba 2: Użyj biblioteki gguf
    # ----------------------------------------------------------
    print(f"\n  [2/2] Próbuję konwersji przez bibliotekę gguf...")
    try:
        from gguf import GGUFWriter
        print(f"    Biblioteka gguf dostępna, ale konwersja safetensors -> GGUF")
        print(f"    przez GGUFWriter wymaga ręcznego mapowania każdego tensora.")
        print(f"    Zalecane: zainstaluj llama.cpp i użyj konwertera.")
        print(f"\n    Alternatywnie, możesz użyć modelu bezpośrednio z safetensors:")
        print(f"    - Utwórz Modelfile: FROM {input_dir}")
        print(f"    - ollama create pazpik-model {input_dir}")
        return None
    except ImportError:
        print(f"    Biblioteka gguf nie jest zainstalowana.")
        print(f"    Zainstaluj: pip install gguf")
        return None

    return None


# ============================================================
# FUNKCJA POMOCNICZA: Pobranie modelu jako HF z Ollamy
# ============================================================

def export_ollama_model_to_hf(model_name: str,
                               output_dir: Path) -> Optional[Path]:
    """
    Eksportuje model z Ollamy (GGUF) do formatu HuggingFace (safetensors).
    """
    print(f"\n{'='*70}")
    print(f"  EKSPORT MODELA: {model_name}")
    print(f"  Format: GGUF (Ollama) -> safetensors (HuggingFace)")
    print(f"{'='*70}")

    # Znajdź plik GGUF
    print(f"\n  Szukam pliku GGUF dla modelu {model_name}...")
    gguf_path = find_gguf_blob(model_name)
    if gguf_path is None:
        print(f"\n  [!] Nie znaleziono pliku GGUF.")
        print(f"      Spróbuj pobrać model: ollama pull {model_name}")
        return None

    print(f"  Znaleziono: {gguf_path}")
    print(f"  Rozmiar: {gguf_path.stat().st_size / (1024**3):.2f} GB")

    # Konwertuj
    return gguf_to_safetensors(gguf_path, output_dir)


# ============================================================
# INTERAKTYWNY TRYB
# ============================================================

def interactive_export():
    """Interaktywny tryb eksportu modelu z Ollamy do HF."""
    print(f"\n{'#'*70}")
    print(f"  PAZPIK - Eksporter modeli GGUF -> HuggingFace")
    print(f"{'#'*70}")

    from ollama_utils import OllamaClient, list_ollama_models

    client = OllamaClient()
    models = list_ollama_models(client)

    if not models:
        print("\n  [!] Brak modeli w Ollamie.")
        return

    while True:
        try:
            choice = input(f"\n  Wybierz model do eksportu (numer lub nazwa): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    break
            else:
                selected = [m for m in models if m["name"] == choice]
                if selected:
                    selected = selected[0]
                    break
            print("  Nieprawidłowy wybór.")
        except (ValueError, IndexError):
            print("  Nieprawidłowy wybór.")

    model_name = selected["name"]
    output_dir = Path.cwd() / "models" / "converted_hf" / model_name.replace(":", "_")

    result = export_ollama_model_to_hf(model_name, output_dir)

    if result:
        print(f"\n  [SUKCES] Model wyeksportowany do:")
        print(f"    {result}")
        print(f"\n  Możesz teraz użyć go do fine-tuningu.")
    else:
        print(f"\n  [!] Eksport nie powiódł się.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Użycie: python gguf_converter.py <model_name> [output_dir]
        model_name = sys.argv[1]
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.cwd() / "models" / "converted_hf" / model_name.replace(":", "_")
        export_ollama_model_to_hf(model_name, output_dir)
    else:
        interactive_export()
