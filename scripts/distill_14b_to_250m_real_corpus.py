#!/usr/bin/env python3
"""
Part 5 — Distillazione da Qwen2.5-Coder-14B (teacher, 14.77B param) a un
studente di ~250M parametri (hidden_dim=768, 20 layer), su corpus reale
di coding (~150K snippet da The Stack: Python + JavaScript + TypeScript).

Differenze rispetto a Part 4:
  - Studente ~24× più grande (10.5M → 250M param)
  - Corpus reale multi-lingua invece di template ripetitivi
  - Più step di training (30K vs 5K) per attraversare bene il corpus
  - Checkpoint automatici ogni 5000 step per non perdere lavoro
  - Valutazione qualitativa periodica su prompt reali di coding

Setup:
  - Teacher: Qwen2.5-Coder-14B-Instruct (bf16, congelato)
  - Studente: 768d, 20 layer, 12 heads, d_ff=3072, tied embeddings (~250M param)
  - Loss: 0.5·CE + 0.5·T²·KL(student/T || teacher/T), T=2.0
  - Optimizer: AdamW, lr=3e-4, cosine schedule con warmup 500 step
  - Batch size 4, seq_len 512

Uso:
    python3 distill_14b_to_250m_real_corpus.py
"""
import argparse
import json
import math
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


TEACHER_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"


# ─── Corpus loader: The Stack (Python + JS + TS) via HuggingFace datasets ────
def load_real_corpus(n_per_lang=50000, min_len=100, max_len=8000):
    """
    Scarica un subset di The Stack (versione dedup, smallest) per Python,
    JS, TS. Filtra per lunghezza minima e massima. Restituisce lista di stringhe.

    Nota: usa `bigcode/the-stack-smol` che è già un subset compatto (poche GB
    totali), scaricato al primo utilizzo. Sostituisce `bigcode/the-stack` che
    è troppo grande (~3TB).
    """
    from datasets import load_dataset
    languages = [("python", "python"), ("javascript", "javascript"), ("typescript", "typescript")]
    corpus = []
    for lang_name, lang_field in languages:
        print(f"[corpus] loading {lang_name} from bigcode/the-stack-smol...", flush=True)
        try:
            ds = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang_field}",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"[corpus] falling back to non-streaming for {lang_name}: {e}", flush=True)
            ds = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang_field}",
                split="train",
            )
        count = 0
        for item in ds:
            content = item.get("content", "")
            if min_len <= len(content) <= max_len:
                corpus.append(content)
                count += 1
                if count >= n_per_lang:
                    break
        print(f"[corpus] {lang_name}: {count} snippet raccolti", flush=True)
    print(f"[corpus] totale: {len(corpus)} snippet", flush=True)
    return corpus


# ─── Studente: transformer ~250M param ───────────────────────────────────────
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
    def __init__(self, vocab_size, hidden_dim=768, n_layers=20, n_heads=12,
                 d_ff=3072, seq_len=512, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_dim, n_heads, d_ff, dropout)
                                       for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # tied

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
    """Concatena tutti gli snippet in flusso token, divide in blocchi."""
    all_ids = []
    print(f"[tokenize] tokenizzo {len(corpus)} snippet...", flush=True)
    t0 = time.time()
    for i, text in enumerate(corpus):
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=seq_len*4)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)
        if (i+1) % 20000 == 0:
            print(f"  [tokenize] {i+1}/{len(corpus)} — {len(all_ids):,} token — t={time.time()-t0:.0f}s", flush=True)
    total = len(all_ids)
    n_blocks = total // seq_len
    trimmed = all_ids[:n_blocks * seq_len]
    print(f"[tokenize] fatto: {total:,} token totali, {n_blocks} blocchi da {seq_len}", flush=True)
    return torch.tensor(trimmed, dtype=torch.long).reshape(n_blocks, seq_len)


def get_batch(data, batch_size, device):
    ix = torch.randint(0, data.shape[0], (batch_size,))
    return data[ix].to(device)


# ─── Distillation loss ───────────────────────────────────────────────────────
def distill_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=2.0):
    """
    Loss combinata con KL calcolata in chunk lungo la dimensione temporale,
    per evitare picchi di memoria sul vocab-size grande (152K).
    Un unico F.kl_div su (B, T, V) alloca ~B·T·V·4 byte di intermedi ×2
    per softmax+log_softmax, che con V=152K esplode a ~2GB anche per B=4.
    Dividendo lungo T in chunk piccoli, il picco cala di molto.
    """
    B, T, V = student_logits.shape
    # cross-entropy classico (già efficiente da solo)
    ce = F.cross_entropy(student_logits.reshape(-1, V), targets.reshape(-1))

    # KL divergence in chunk per contenere il picco di memoria
    T_chunk = max(1, T // 4)  # 4 chunk lungo T
    kl_total = 0.0
    n_chunks = 0
    for start in range(0, T, T_chunk):
        end = min(start + T_chunk, T)
        s_slice = student_logits[:, start:end, :]
        t_slice = teacher_logits[:, start:end, :]
        s_log_probs = F.log_softmax(s_slice / temperature, dim=-1)
        t_probs = F.softmax(t_slice / temperature, dim=-1)
        kl_chunk = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)
        kl_total = kl_total + kl_chunk
        n_chunks += 1
    kl = kl_total / n_chunks

    return alpha * ce + (1 - alpha) * kl, ce.item(), kl.item()


# ─── Sample generation ───────────────────────────────────────────────────────
@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device, max_new=150, temperature=0.7):
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


def save_checkpoint(student, opt, scheduler, step, history, path):
    torch.save({
        "step": step,
        "student": student.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
    }, path)
    print(f"[checkpoint] salvato a step {step} → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-dim", type=int, default=768)
    ap.add_argument("--n-layers", type=int, default=20)
    ap.add_argument("--n-heads", type=int, default=12)
    ap.add_argument("--d-ff", type=int, default=3072)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--sample-every", type=int, default=5000)
    ap.add_argument("--n-per-lang", type=int, default=50000)
    ap.add_argument("--out", default="distill_14b_to_250m_results.json")
    ap.add_argument("--checkpoint-path", default="student_checkpoint.pt")
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
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"[main] teacher pronto ({teacher_params/1e9:.2f}B param)", flush=True)

    vocab_size = teacher.config.vocab_size
    print(f"[main] vocab_size (teacher.config): {vocab_size}  (len(tokenizer)={len(tokenizer)})", flush=True)

    # Corpus reale
    print(f"[main] caricamento corpus (~{args.n_per_lang}×3 snippet)...", flush=True)
    corpus = load_real_corpus(n_per_lang=args.n_per_lang)
    data = tokenize_corpus(tokenizer, corpus, args.seq_len)
    print(f"[main] data: {data.shape[0]} blocchi × {data.shape[1]} token", flush=True)
    split = int(data.shape[0] * 0.98)  # con corpus grande, 2% val basta
    train_data = data[:split]
    val_data = data[split:]
    print(f"[main] train: {train_data.shape[0]} blocchi, val: {val_data.shape[0]}", flush=True)

    # Studente
    student = StudentTransformer(
        vocab_size=vocab_size, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, seq_len=args.seq_len,
    ).to(device)
    student_params = student.n_params()
    print(f"[main] studente: {student_params/1e6:.2f}M param "
          f"(hidden_dim={args.hidden_dim}, n_layers={args.n_layers})", flush=True)
    print(f"[main] rapporto teacher/studente: {teacher_params/student_params:.0f}×", flush=True)

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
    sample_prompts = [
        "def binary_search(arr, target):",
        "class LRUCache:",
        "function useDebounce(",
        "async function fetchData(url) {",
        "interface User {",
    ]

    student.train()
    t0 = time.time()

    for step in range(1, args.steps + 1):
        batch = get_batch(train_data, args.batch_size, device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits.float()

        student_logits = student(input_ids)
        loss, ce_val, kl_val = distill_loss(student_logits, teacher_logits, targets,
                                              alpha=args.alpha, temperature=args.temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1 or step == args.steps:
            student.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    vb = get_batch(val_data, args.batch_size, device)
                    vlog = student(vb[:, :-1])
                    val_losses.append(F.cross_entropy(
                        vlog.reshape(-1, vlog.size(-1)), vb[:, 1:].reshape(-1)
                    ).item())
                val_ce = float(np.mean(val_losses))
            student.train()

            elapsed = time.time() - t0
            eta_s = elapsed / step * (args.steps - step)
            print(f"[step {step:5d}/{args.steps}] loss={loss.item():.4f}  "
                  f"ce={ce_val:.4f}  kl={kl_val:.4f}  val_ce={val_ce:.4f}  "
                  f"val_ppl={math.exp(val_ce):.2f}  lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"t={elapsed:.0f}s eta={eta_s/60:.0f}m", flush=True)
            history.append({
                "step": step, "total_loss": loss.item(),
                "ce_loss": ce_val, "kl_loss": kl_val,
                "val_ce": val_ce, "val_ppl": math.exp(val_ce),
                "lr": scheduler.get_last_lr()[0], "elapsed_s": elapsed,
            })

        # Sample periodici + checkpoint
        if step % args.sample_every == 0 or step == args.steps:
            print(f"\n--- samples at step {step} ---", flush=True)
            for p in sample_prompts[:2]:  # solo 2 per non riempire log
                s = generate_sample(student, tokenizer, p, device, max_new=100)
                print(f"prompt: {p!r}\n{s}\n", flush=True)

        if step % args.checkpoint_every == 0 or step == args.steps:
            save_checkpoint(student, opt, scheduler, step, history, args.checkpoint_path)

    # Sample finali completi
    print(f"\n=== SAMPLES FINALI DOPO DISTILLAZIONE ===\n", flush=True)
    samples = {}
    for p in sample_prompts:
        s = generate_sample(student, tokenizer, p, device, max_new=150)
        print(f"--- prompt: {p!r} ---\n{s}\n", flush=True)
        samples[p] = s

    # Salva
    results = {
        "config": {
            "teacher": TEACHER_MODEL,
            "hidden_dim": args.hidden_dim, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "d_ff": args.d_ff, "seq_len": args.seq_len,
            "student_params": student_params, "teacher_params": teacher_params,
            "compression_ratio": teacher_params / student_params,
            "vocab_size": vocab_size,
            "n_snippets": len(corpus), "n_per_lang": args.n_per_lang,
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
    print(f"\n{'='*60}\nRIEPILOGO — Distillazione 14B → studente {args.hidden_dim}d ({student_params/1e6:.1f}M param)\n{'='*60}")
    initial_ce = history[0]["ce_loss"]
    final_ce = history[-1]["ce_loss"]
    initial_kl = history[0]["kl_loss"]
    final_kl = history[-1]["kl_loss"]
    initial_val_ppl = history[0]["val_ppl"]
    final_val_ppl = history[-1]["val_ppl"]
    print(f"CE:      {initial_ce:.4f} → {final_ce:.4f}  ({(final_ce/initial_ce-1)*100:+.1f}%)")
    print(f"KL:      {initial_kl:.4f} → {final_kl:.4f}  ({(final_kl/initial_kl-1)*100:+.1f}%)")
    print(f"val PPL: {initial_val_ppl:.2f} → {final_val_ppl:.2f}")


if __name__ == "__main__":
    main()
