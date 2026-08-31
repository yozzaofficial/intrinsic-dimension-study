#!/usr/bin/env python3
"""
Proof of concept: allena due Transformer da zero, IDENTICI in tutto tranne
la dimensione ambientale (hidden_dim), sullo stesso corpus di coding.

Ipotesi da verificare:
  - Se la dimensione intrinseca del coding è ~13 (misurata in Part 1),
    un modello con hidden_dim=64 (~5× la dim intrinseca) dovrebbe convergere
    a qualità comparabile a un modello con hidden_dim=256 (~20× la dim intrinseca)
    sullo stesso task, a parità di step e dati.

Modelli confrontati:
  - MODEL A (baseline standard):  hidden_dim=256, layers=8   → ~30M param
  - MODEL B (bassa dim nativa):   hidden_dim=64,  layers=16  → ~10M param
    (più layer per compensare la dimensione minore — architettura Narrow+Deep)

Cosa misuriamo:
  - Loss di training e validation nel tempo
  - Loss finale su held-out
  - Un campione qualitativo di generazione con lo stesso prompt

Uso:
    python3 train_low_dim_native.py
"""
import argparse
import json
import math
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


# ─── Config ───────────────────────────────────────────────────────────────────
TOKENIZER_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # riusiamo il tokenizer
SEQ_LEN = 256
BATCH_SIZE = 32


# ─── Corpus di training: coding Python misto ─────────────────────────────────
# Base di 30 pattern diversi, espansi con variazioni multiple per raggiungere
# ~20000 snippet. Il corpus è deliberatamente vario ma tutto Python (dominio
# ristretto — coerente con l'idea di specializzazione).
BASE_SNIPPETS = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    l = merge_sort(arr[:mid])\n    r = merge_sort(arr[mid:])\n    result = []\n    i = j = 0\n    while i < len(l) and j < len(r):\n        if l[i] <= r[j]: result.append(l[i]); i += 1\n        else: result.append(r[j]); j += 1\n    return result + l[i:] + r[j:]",
    "def fibonacci(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]",
    "def is_prime(n):\n    if n < 2: return False\n    if n == 2: return True\n    if n % 2 == 0: return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0: return False\n    return True",
    "def factorial(n):\n    if n < 0: raise ValueError('negative')\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
    "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen: return [seen[target - n], i]\n        seen[n] = i\n    return []",
    "def reverse_string(s):\n    return s[::-1]",
    "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
    "class Stack:\n    def __init__(self):\n        self.data = []\n    def push(self, x):\n        self.data.append(x)\n    def pop(self):\n        if not self.data: raise IndexError('empty')\n        return self.data.pop()\n    def peek(self):\n        return self.data[-1] if self.data else None",
    "class Queue:\n    def __init__(self):\n        self.data = []\n    def enqueue(self, x):\n        self.data.append(x)\n    def dequeue(self):\n        if not self.data: raise IndexError('empty')\n        return self.data.pop(0)",
    "class LinkedList:\n    def __init__(self):\n        self.head = None\n    def add(self, value):\n        node = Node(value)\n        node.next = self.head\n        self.head = node\n    def find(self, value):\n        current = self.head\n        while current:\n            if current.value == value: return current\n            current = current.next\n        return None",
    "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n    def insert(self, value):\n        if value < self.value:\n            if self.left: self.left.insert(value)\n            else: self.left = BinaryTree(value)\n        else:\n            if self.right: self.right.insert(value)\n            else: self.right = BinaryTree(value)",
    "def bfs(graph, start):\n    visited = set([start])\n    queue = [start]\n    while queue:\n        node = queue.pop(0)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return visited",
    "def dfs(graph, start, visited=None):\n    if visited is None: visited = set()\n    visited.add(start)\n    for neighbor in graph.get(start, []):\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited",
    "def count_words(text):\n    from collections import Counter\n    words = text.lower().split()\n    return Counter(words)",
    "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    "def group_by(items, key):\n    result = {}\n    for item in items:\n        k = key(item)\n        result.setdefault(k, []).append(item)\n    return result",
    "def read_json(path):\n    import json\n    with open(path) as f:\n        return json.load(f)\n\ndef write_json(path, data):\n    import json\n    with open(path, 'w') as f:\n        json.dump(data, f, indent=2)",
    "async def fetch_url(url):\n    import aiohttp\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.text()",
    "def retry(fn, attempts=3, delay=1):\n    import time\n    for i in range(attempts):\n        try:\n            return fn()\n        except Exception as e:\n            if i == attempts - 1: raise\n            time.sleep(delay * (2 ** i))",
    "def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper",
    "def timing_decorator(fn):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = fn(*args, **kwargs)\n        print(f'{fn.__name__} took {time.time()-start:.3f}s')\n        return result\n    return wrapper",
    "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
    "def parse_url(url):\n    from urllib.parse import urlparse\n    p = urlparse(url)\n    return {'scheme': p.scheme, 'host': p.hostname, 'path': p.path}",
    "def hash_password(password, salt=None):\n    import hashlib, secrets\n    if salt is None: salt = secrets.token_bytes(16)\n    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)\n    return salt + key",
    "def validate_email(email):\n    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return re.match(pattern, email) is not None",
    "def load_config(path):\n    import configparser\n    config = configparser.ConfigParser()\n    config.read(path)\n    return {section: dict(config[section]) for section in config.sections()}",
    "def batch_iter(items, batch_size):\n    batch = []\n    for item in items:\n        batch.append(item)\n        if len(batch) >= batch_size:\n            yield batch\n            batch = []\n    if batch:\n        yield batch",
]


def build_corpus(n_target=20000):
    """Espande i pattern base con variazioni multiple per creare un corpus grande."""
    random.seed(42)
    substitutions = [
        ("arr", ["data", "items", "lst", "seq", "nums", "elements"]),
        ("target", ["value", "key", "elem", "x", "needle"]),
        ("result", ["output", "res", "acc", "out", "collected"]),
        ("value", ["val", "item", "elem", "entry", "record"]),
        ("node", ["current", "head", "n", "vertex"]),
        ("data", ["records", "input", "payload", "items"]),
        ("count", ["total", "num", "size", "n"]),
        ("cache", ["store", "buffer", "memo", "table"]),
    ]
    out = list(BASE_SNIPPETS)
    seen = set(out)
    attempts = 0
    while len(out) < n_target and attempts < n_target * 10:
        attempts += 1
        s = random.choice(BASE_SNIPPETS)
        for _ in range(random.randint(2, 5)):
            old, news = random.choice(substitutions)
            new = random.choice(news)
            if old in s:
                s = s.replace(old, new, 1)
        s = f"# variant {random.randint(1, 9999999)}\n{s}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─── Modello Transformer decoder minimale ────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        assert d_model % n_heads == 0
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.Wq(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # scaled dot product attention causale
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.Wo(out.transpose(1, 2).reshape(B, T, D))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=100, temperature=0.7):
        self.eval()
        for _ in range(max_new_tokens):
            ids_cropped = ids[:, -self.seq_len:]
            logits = self.forward(ids_cropped)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


# ─── Preparazione dati ───────────────────────────────────────────────────────
def tokenize_corpus(tokenizer, corpus, seq_len):
    """Concatena tutto in una lunga sequenza di token, poi ritaglia in blocchi."""
    all_ids = []
    for text in corpus:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id or 0)
    ids_tensor = torch.tensor(all_ids, dtype=torch.long)
    n_blocks = len(ids_tensor) // seq_len
    return ids_tensor[:n_blocks * seq_len].reshape(n_blocks, seq_len)


def get_batch(data, batch_size, device):
    idx = torch.randint(0, data.shape[0], (batch_size,))
    batch = data[idx].to(device)
    return batch[:, :-1], batch[:, 1:]  # input, target (shift-1)


# ─── Training loop ────────────────────────────────────────────────────────────
def train_model(name, model, train_data, val_data, device, steps, lr, log_every=100):
    print(f"\n{'='*70}\n[{name}] params: {model.n_params()/1e6:.2f}M  training {steps} steps\n{'='*70}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    warmup = min(500, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(s / warmup, 1.0) * max(0.1, 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    )
    model.train()
    history = {"step": [], "train_loss": [], "val_loss": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, BATCH_SIZE, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            # eval veloce
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, BATCH_SIZE, device)
                vloss = F.cross_entropy(model(vx).reshape(-1, logits.shape[-1]), vy.reshape(-1)).item()
            model.train()
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(vloss)
            elapsed = time.time() - t0
            print(f"  [{name}] step {step:5d}/{steps}  train={loss.item():.3f}  val={vloss:.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  t={elapsed:.0f}s", flush=True)
    return history


@torch.no_grad()
def sample_generation(name, model, tokenizer, prompt, device, max_new_tokens=80):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.7)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n[{name}] SAMPLE GENERATION (prompt='{prompt[:60]}...'):")
    print(text)
    return text


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-snippets", type=int, default=20000)
    ap.add_argument("--out", default="low_dim_native_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}", flush=True)

    print(f"[main] loading tokenizer {TOKENIZER_MODEL}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    # NB: tokenizer.vocab_size (151643 per Qwen) NON include i token speciali
    # aggiunti sopra, ma tokenizer(...) puo' emettere id fino a len(tokenizer).
    # Se dimensioniamo nn.Embedding a vocab_size, quei token speciali causano
    # "index out of bounds" in CUDA. Serve il vero massimo id emettibile.
    vocab_size = len(tokenizer)
    print(f"[main] vocab_size (len(tokenizer)): {vocab_size} "
          f"(tokenizer.vocab_size={tokenizer.vocab_size})", flush=True)

    print(f"[main] building corpus ({args.n_snippets} snippets)...", flush=True)
    corpus = build_corpus(args.n_snippets)
    print(f"[main] tokenizing...", flush=True)
    data = tokenize_corpus(tokenizer, corpus, SEQ_LEN)
    print(f"[main] tokenized: {data.shape[0]} blocks of {data.shape[1]} tokens", flush=True)
    # 95/5 split
    n_train = int(data.shape[0] * 0.95)
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"[main] train: {train_data.shape[0]} blocks  val: {val_data.shape[0]} blocks", flush=True)

    # ─── MODEL A: baseline standard (hidden_dim=256, layers=8) ────────────────
    model_a = TinyLM(
        vocab_size=vocab_size,
        d_model=256, n_heads=8, n_layers=8, d_ff=1024,
        seq_len=SEQ_LEN - 1
    ).to(device)

    # ─── MODEL B: bassa dim nativa (hidden_dim=64, layers=16 per compensare) ──
    model_b = TinyLM(
        vocab_size=vocab_size,
        d_model=64, n_heads=4, n_layers=16, d_ff=256,
        seq_len=SEQ_LEN - 1
    ).to(device)

    print(f"\n[main] MODEL A: hidden_dim=256, layers=8  → {model_a.n_params()/1e6:.2f}M params", flush=True)
    print(f"[main] MODEL B: hidden_dim=64,  layers=16 → {model_b.n_params()/1e6:.2f}M params", flush=True)
    ratio = model_a.n_params() / model_b.n_params()
    print(f"[main] MODEL A is {ratio:.1f}× larger than MODEL B", flush=True)

    # Training A
    history_a = train_model("A/256d", model_a, train_data, val_data, device, args.steps, args.lr)
    torch.cuda.empty_cache()

    # Training B
    history_b = train_model("B/64d", model_b, train_data, val_data, device, args.steps, args.lr)
    torch.cuda.empty_cache()

    # Generazione qualitativa sullo stesso prompt
    prompts = [
        "def binary_search",
        "class Stack:",
        "def is_prime(n):",
    ]
    generations = {"A": {}, "B": {}}
    for p in prompts:
        generations["A"][p] = sample_generation("A/256d", model_a, tokenizer, p, device)
        generations["B"][p] = sample_generation("B/64d",  model_b, tokenizer, p, device)

    # Riepilogo
    print(f"\n{'='*70}\nRIEPILOGO FINALE\n{'='*70}", flush=True)
    print(f"MODEL A (hidden_dim=256, {model_a.n_params()/1e6:.1f}M):  final val loss = {history_a['val_loss'][-1]:.3f}")
    print(f"MODEL B (hidden_dim=64,  {model_b.n_params()/1e6:.1f}M):  final val loss = {history_b['val_loss'][-1]:.3f}")
    delta = (history_b['val_loss'][-1] - history_a['val_loss'][-1]) / history_a['val_loss'][-1] * 100
    print(f"Delta val loss B vs A: {delta:+.1f}%")
    print(f"\nInterpretazione:")
    print(f"  |Δ| < 15%:  MODEL B compete con A nonostante {ratio:.1f}× meno parametri → tesi VALIDATA")
    print(f"  |Δ| 15-40%: risultato ambiguo → serve piu training o config diversa")
    print(f"  |Δ| > 40%:  MODEL B non converge bene in bassa dim → tesi NON validata")

    with open(args.out, "w") as f:
        json.dump({
            "config": {
                "steps": args.steps, "lr": args.lr, "n_snippets": args.n_snippets,
                "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "vocab_size": vocab_size,
            },
            "model_a": {
                "hidden_dim": 256, "n_layers": 8, "d_ff": 1024, "n_heads": 8,
                "params": model_a.n_params(),
                "history": history_a,
                "final_val_loss": history_a['val_loss'][-1],
            },
            "model_b": {
                "hidden_dim": 64, "n_layers": 16, "d_ff": 256, "n_heads": 4,
                "params": model_b.n_params(),
                "history": history_b,
                "final_val_loss": history_b['val_loss'][-1],
            },
            "delta_pct": delta,
            "compression_ratio": ratio,
            "sample_generations": generations,
        }, f, indent=2)
    print(f"\n[main] salvato -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
