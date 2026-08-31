#!/usr/bin/env python3
"""
Part 4 — Distillazione da Qwen2.5-Coder-14B (teacher, ~14B param) a un
transformer studente a bassa dimensione (~20M param, hidden_dim=64,
16 layer). Ratio 700×.

Domanda che risponde: il segnale di distillazione da un teacher enorme
a uno studente 700× più piccolo produce apprendimento utile, o si perde
nel rumore?

Se la KL loss scende progressivamente e le generazioni dello studente
migliorano rispetto a un baseline (stesso studente senza distillazione),
il segnale c'è. Se la KL loss resta piatta, sappiamo che l'architettura
studente attuale non ha capacità di seguire il teacher.

Loss combinata:
  L = α · CrossEntropy(student, target) + (1-α) · T² · KL(student/T || teacher/T)

dove T è la temperatura (>1 rende le distribuzioni più morbide, più
informative per la distillazione).

Uso:
    python3 distill_14b_to_lowdim.py
"""
import argparse
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


TEACHER_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"


# ─── Corpus (stesso schema di Parte 3, semplificato) ─────────────────────────
BASE_SNIPPETS = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "def fibonacci(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]",
    "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
    "def two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen: return [seen[target - n], i]\n        seen[n] = i\n    return []",
    "class Stack:\n    def __init__(self): self.data = []\n    def push(self, x): self.data.append(x)\n    def pop(self): return self.data.pop() if self.data else None\n    def peek(self): return self.data[-1] if self.data else None",
    "class Queue:\n    def __init__(self): self.data = []\n    def enqueue(self, x): self.data.append(x)\n    def dequeue(self): return self.data.pop(0) if self.data else None",
    "def is_palindrome(s):\n    s = s.lower()\n    left, right = 0, len(s) - 1\n    while left < right:\n        if s[left] != s[right]: return False\n        left += 1\n        right -= 1\n    return True",
    "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)",
    "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a",
    "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0: return False\n    return True",
    "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()",
    "def parse_json(s):\n    import json\n    try: return json.loads(s)\n    except json.JSONDecodeError: return None",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
    "def retry(func, max_attempts=3):\n    for attempt in range(max_attempts):\n        try: return func()\n        except Exception as e:\n            if attempt == max_attempts - 1: raise\n            time.sleep(2 ** attempt)",
    "def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache: cache[args] = func(*args)\n        return cache[args]\n    return wrapper",
    "class Config:\n    def __init__(self, **kwargs):\n        for key, value in kwargs.items(): setattr(self, key, value)",
    "def validate_email(email):\n    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))",
    "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list): result.extend(flatten(item))\n        else: result.append(item)\n    return result",
    "def http_get(url):\n    import requests\n    response = requests.get(url)\n    response.raise_for_status()\n    return response.json()",
]


def build_corpus(n_target=3000):
    """
    Genera varianti dei pattern base con sostituzioni + commento univoco.
    Il commento garantisce che possiamo sempre raggiungere n_target snippet
    diversi anche se le sostituzioni collidono (lo spazio combinatorio dei
    soli rename di variabili si esaurisce presto).
    """
    import random
    random.seed(42)
    out = list(BASE_SNIPPETS)
    substitutions = [
        ("arr", ["data", "items", "lst", "seq"]),
        ("target", ["value", "key", "elem", "x"]),
        ("result", ["output", "res", "acc"]),
        ("value", ["val", "item", "elem"]),
    ]
    seen = set(out)
    while len(out) < n_target:
        s = random.choice(BASE_SNIPPETS)
        for _ in range(random.randint(1, 3)):
            old, news = random.choice(substitutions)
            new = random.choice(news)
            if old in s:
                s = s.replace(old, new, 1)
        # commento univoco per garantire snippet distinti
        s = f"# variant {random.randint(1, 9_999_999)}\n{s}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─── Studente: piccolo transformer identico a MODEL B di Parte 3 ─────────────
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        return x + self.ff(self.norm2(x))


class StudentTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, n_layers=16, n_heads=4,
                 d_ff=256, seq_len=256, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, n_heads, d_ff, dropout)
                                       for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # tied weights

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.token_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─── Tokenize + batching ─────────────────────────────────────────────────────
def tokenize_corpus(tokenizer, corpus, seq_len):
    """Concatena tutti gli snippet in un flusso di token, poi divide in blocchi."""
    all_ids = []
    for text in corpus:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)
    total = len(all_ids)
    n_blocks = total // seq_len
    trimmed = all_ids[:n_blocks * seq_len]
    return torch.tensor(trimmed, dtype=torch.long).reshape(n_blocks, seq_len)


def get_batch(data, batch_size, device):
    ix = torch.randint(0, data.shape[0], (batch_size,))
    return data[ix].to(device)


# ─── Distillation loss ───────────────────────────────────────────────────────
def distill_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=2.0):
    """
    Loss combinata:
      L = alpha * CE(student, targets)
         + (1-alpha) * T^2 * KL(softmax(student/T) || softmax(teacher/T))
    """
    B, T, V = student_logits.shape
    # cross-entropy classico
    ce = F.cross_entropy(student_logits.reshape(-1, V), targets.reshape(-1))
    # KL divergence tra distribuzioni ammorbidite dalla temperatura
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
    return alpha * ce + (1 - alpha) * kl, ce.item(), kl.item()


# ─── Sample generation ───────────────────────────────────────────────────────
@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device, max_new=100, temperature=0.7):
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    for _ in range(max_new):
        input_ids = ids[:, -model.seq_len:]
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    model.train()
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)  # limitato dal teacher!
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.5, help="peso CE (1-alpha va a KL)")
    ap.add_argument("--temperature", type=float, default=2.0, help="temperatura distillazione")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-snippets", type=int, default=3000)
    ap.add_argument("--out", default="distill_14b_to_lowdim_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}", flush=True)

    # Teacher (congelato) + tokenizer
    print(f"[main] loading teacher {TEACHER_MODEL} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL, dtype=torch.bfloat16, device_map=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[main] teacher pronto ({sum(p.numel() for p in teacher.parameters())/1e9:.2f}B param)", flush=True)

    # Usiamo vocab_size del MODELLO, non del tokenizer:
    # Qwen padda l'embedding a multipli di 128 per efficienza GPU,
    # quindi teacher.config.vocab_size (152064) > len(tokenizer) (151665).
    # Per la KL divergence tra logit teacher e student, i due vocab devono
    # coincidere — quindi lo studente adotta il vocab_size del teacher.
    vocab_size = teacher.config.vocab_size
    print(f"[main] vocab_size (teacher.config): {vocab_size}  "
          f"(len(tokenizer)={len(tokenizer)})", flush=True)

    # Corpus
    corpus = build_corpus(args.n_snippets)
    print(f"[main] corpus: {len(corpus)} snippet unici", flush=True)
    data = tokenize_corpus(tokenizer, corpus, args.seq_len)
    print(f"[main] data: {data.shape[0]} blocchi di {data.shape[1]} token", flush=True)
    split = int(data.shape[0] * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"[main] train: {train_data.shape[0]} blocchi, val: {val_data.shape[0]}", flush=True)

    # Studente
    student = StudentTransformer(
        vocab_size=vocab_size, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, seq_len=args.seq_len,
    ).to(device)
    print(f"[main] studente: {student.n_params()/1e6:.2f}M param "
          f"(hidden_dim={args.hidden_dim}, n_layers={args.n_layers})", flush=True)
    print(f"[main] rapporto teacher/studente: "
          f"{sum(p.numel() for p in teacher.parameters()) / student.n_params():.0f}×", flush=True)

    # Optimizer + scheduler
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    def lr_lambda(step):
        if step < args.warmup:
            return step / args.warmup
        progress = (step - args.warmup) / (args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n=== DISTILLAZIONE: {args.steps} step, alpha={args.alpha}, T={args.temperature} ===\n", flush=True)
    history = []

    student.train()
    t0 = time.time()

    for step in range(1, args.steps + 1):
        batch = get_batch(train_data, args.batch_size, device)  # (B, T)
        # target per la CE: shift-by-one classico
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        # forward teacher (no grad)
        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits.float()  # (B, T-1, V)

        # forward student
        student_logits = student(input_ids)  # (B, T-1, V)

        loss, ce_val, kl_val = distill_loss(student_logits, teacher_logits, targets,
                                              alpha=args.alpha, temperature=args.temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            # val loss dello studente (solo CE, no KL, per confrontabilità con Parte 3)
            student.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(10):
                    vb = get_batch(val_data, args.batch_size, device)
                    vlog = student(vb[:, :-1])
                    val_losses.append(F.cross_entropy(
                        vlog.reshape(-1, vlog.size(-1)), vb[:, 1:].reshape(-1)
                    ).item())
                val_ce = float(np.mean(val_losses))
            student.train()

            elapsed = time.time() - t0
            print(f"[step {step:5d}/{args.steps}] loss={loss.item():.4f}  "
                  f"ce={ce_val:.4f}  kl={kl_val:.4f}  val_ce={val_ce:.4f}  "
                  f"val_ppl={math.exp(val_ce):.2f}  lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"t={elapsed:.0f}s", flush=True)
            history.append({
                "step": step, "total_loss": loss.item(),
                "ce_loss": ce_val, "kl_loss": kl_val,
                "val_ce": val_ce, "val_ppl": math.exp(val_ce),
                "lr": scheduler.get_last_lr()[0], "elapsed_s": elapsed,
            })

    # Generazione di esempio
    print(f"\n=== SAMPLES DOPO DISTILLAZIONE ===\n", flush=True)
    prompts = ["def binary_search", "class Stack:", "def is_prime(n):", "async def fetch"]
    samples = {}
    for p in prompts:
        s = generate_sample(student, tokenizer, p, device, max_new=80)
        print(f"--- prompt: {p!r} ---")
        print(s)
        print()
        samples[p] = s

    # Salva
    results = {
        "config": {
            "teacher": TEACHER_MODEL,
            "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "d_ff": args.d_ff, "seq_len": args.seq_len,
            "student_params": student.n_params(),
            "teacher_params": sum(p.numel() for p in teacher.parameters()),
            "compression_ratio": sum(p.numel() for p in teacher.parameters()) / student.n_params(),
            "vocab_size": vocab_size,
            "n_snippets": len(corpus),
        },
        "training": {
            "steps": args.steps, "batch_size": args.batch_size,
            "lr": args.lr, "alpha": args.alpha, "temperature": args.temperature,
        },
        "history": history,
        "samples": samples,
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main] salvato -> {args.out}", flush=True)

    # Riepilogo
    print(f"\n{'='*60}\nRIEPILOGO — Distillazione 14B → studente {args.hidden_dim}d\n{'='*60}")
    initial_ce = history[0]["ce_loss"]
    final_ce = history[-1]["ce_loss"]
    initial_kl = history[0]["kl_loss"]
    final_kl = history[-1]["kl_loss"]
    initial_val_ppl = history[0]["val_ppl"]
    final_val_ppl = history[-1]["val_ppl"]
    print(f"CE:      {initial_ce:.4f} → {final_ce:.4f}  ({(final_ce/initial_ce-1)*100:+.1f}%)")
    print(f"KL:      {initial_kl:.4f} → {final_kl:.4f}  ({(final_kl/initial_kl-1)*100:+.1f}%)")
    print(f"val PPL: {initial_val_ppl:.2f} → {final_val_ppl:.2f}  ({(final_val_ppl/initial_val_ppl-1)*100:+.1f}%)")
    print(f"\nInterpretazione:")
    print(f"  Se KL scende almeno del 50%: lo studente STA seguendo il teacher")
    print(f"  Se val_ppl < 5: convergenza forte, distillazione funziona")
    print(f"  Se KL piatta o val_ppl > 30: studente troppo piccolo per il teacher")


if __name__ == "__main__":
    main()
