#!/usr/bin/env python3
"""
Misura la dimensione intrinseca dei modelli Qwen-Coder sul dominio coding.

Cosa fa:
1. Carica il modello (fp16 su GPU) e alcuni tokenizer
2. Passa un dataset di snippet coding e cattura hidden states a ogni layer
3. Stima la dimensione intrinseca layer-per-layer con TwoNN + MLE + PCA
4. Salva risultati in JSON e stampa un riepilogo leggibile

Uso (su VPS con GPU):
    python3 -u measure_intrinsic_dim.py
    python3 -u measure_intrinsic_dim.py --model Qwen/Qwen2.5-Coder-14B-Instruct
"""
import argparse
import json
import math
import time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Dataset — snippet di coding reali, vari linguaggi/pattern ───────────────
SNIPPETS = [
    # Algoritmi classici
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "def fibonacci(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\n    return memo[n]",
    # React/JSX
    "export default function Navbar() {\n  const [open, setOpen] = useState(false);\n  return (\n    <nav className=\"navbar\">\n      <button onClick={() => setOpen(!open)}>Menu</button>\n      {open && <ul>{items.map(i => <li key={i.id}>{i.label}</li>)}</ul>}\n    </nav>\n  );\n}",
    "function useDebounce(value, delay) {\n  const [debounced, setDebounced] = useState(value);\n  useEffect(() => {\n    const handler = setTimeout(() => setDebounced(value), delay);\n    return () => clearTimeout(handler);\n  }, [value, delay]);\n  return debounced;\n}",
    # CSS
    ".navbar {\n  position: fixed;\n  top: 0;\n  width: 100%;\n  background: rgba(255,255,255,0.9);\n  backdrop-filter: blur(10px);\n  z-index: 100;\n  padding: 1rem 2rem;\n  display: flex;\n  justify-content: space-between;\n}",
    "@media (max-width: 768px) {\n  .container { padding: 1rem; }\n  .grid { grid-template-columns: 1fr; }\n  .nav-links { display: none; }\n  .hamburger { display: block; }\n}",
    # SQL
    "SELECT u.id, u.name, COUNT(o.id) AS order_count, SUM(o.total) AS total_spent\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 5\nORDER BY total_spent DESC LIMIT 20;",
    # TypeScript patterns
    "type ApiResponse<T> = {\n  data: T;\n  error: string | null;\n  loading: boolean;\n};\n\nasync function fetchUser(id: string): Promise<ApiResponse<User>> {\n  try {\n    const res = await fetch(`/api/users/${id}`);\n    const data = await res.json();\n    return { data, error: null, loading: false };\n  } catch (e) {\n    return { data: null as any, error: String(e), loading: false };\n  }\n}",
    # Rust
    "fn fibonacci(n: u64) -> u64 {\n    let mut a = 0u64;\n    let mut b = 1u64;\n    for _ in 0..n {\n        let c = a + b;\n        a = b;\n        b = c;\n    }\n    a\n}",
    # Bash
    "#!/bin/bash\nset -euo pipefail\nfor file in *.log; do\n  if [[ $(stat -f%z \"$file\") -gt 1000000 ]]; then\n    gzip \"$file\"\n    echo \"Compressed $file\"\n  fi\ndone",
    # Go concurrency
    "func worker(id int, jobs <-chan int, results chan<- int) {\n    for j := range jobs {\n        fmt.Printf(\"worker %d processing job %d\\n\", id, j)\n        time.Sleep(time.Second)\n        results <- j * 2\n    }\n}",
    # Error handling patterns
    "class DatabaseError(Exception):\n    def __init__(self, message, query=None):\n        super().__init__(message)\n        self.query = query\n\ndef execute_query(conn, sql, params):\n    try:\n        with conn.cursor() as cur:\n            cur.execute(sql, params)\n            return cur.fetchall()\n    except psycopg2.Error as e:\n        raise DatabaseError(f\"Query failed: {e}\", query=sql)",
    # Testing
    "describe('Calculator', () => {\n  let calc;\n  beforeEach(() => { calc = new Calculator(); });\n  \n  test('adds two numbers', () => {\n    expect(calc.add(2, 3)).toBe(5);\n  });\n  \n  test('handles division by zero', () => {\n    expect(() => calc.divide(1, 0)).toThrow('Division by zero');\n  });\n});",
    # Design patterns
    "class Singleton:\n    _instance = None\n    _lock = threading.Lock()\n    \n    def __new__(cls):\n        if cls._instance is None:\n            with cls._lock:\n                if cls._instance is None:\n                    cls._instance = super().__new__(cls)\n        return cls._instance",
]


def build_dataset(n_target=2000):
    """
    Replica gli snippet base con piccole variazioni (rinomina variabili,
    cambia costanti) per raggiungere ~2000 esempi diversi. Serve massa
    per stime affidabili di dimensione intrinseca (TwoNN vuole almeno
    qualche migliaio di punti).
    """
    out = list(SNIPPETS)
    import random
    random.seed(42)
    while len(out) < n_target:
        snip = random.choice(SNIPPETS)
        # variazioni banali ma sufficienti a creare hidden state diversi
        for old, new in [
            ("arr", random.choice(["data", "items", "lst", "seq"])),
            ("target", random.choice(["value", "key", "elem", "x"])),
            ("result", random.choice(["output", "res", "acc", "out"])),
            ("user", random.choice(["client", "member", "acct", "person"])),
        ]:
            if random.random() < 0.5:
                snip = snip.replace(old, new)
        out.append(snip)
    return out[:n_target]


def extract_hidden_states(model, tokenizer, snippets, device, max_len=256):
    """
    Passa gli snippet nel modello, cattura per ogni layer il vettore
    dell'ULTIMO token utile (che riassume il contesto letto fino a lì).
    Ritorna un array (n_layer, n_snippets, hidden_dim).
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n = len(snippets)

    print(f"[extract] layers={n_layers} hidden_dim={hidden_dim} snippets={n}", flush=True)

    # pre-alloco l'array: n_layer x n_snippets x hidden_dim
    all_states = np.zeros((n_layers, n, hidden_dim), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i, snippet in enumerate(snippets):
            inputs = tokenizer(snippet, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
            # outputs.hidden_states è tuple di (n_layer+1) tensori (embedding + ogni layer)
            # saltiamo l'embedding, teniamo l'output di ogni layer
            for li in range(n_layers):
                # ultimo token dell'ultima posizione, in float32 su cpu
                h = outputs.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
                all_states[li, i, :] = h
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{n}] t={time.time()-t0:.1f}s", flush=True)

    print(f"[extract] done in {time.time()-t0:.1f}s", flush=True)
    return all_states


def twonn(X, discard_frac=0.1):
    """
    Two-Nearest-Neighbors estimator (Facco et al. 2017).
    X: (n_points, n_dim). Ritorna la dimensione intrinseca stimata.
    """
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1 = dist[:, 1]
    r2 = dist[:, 2]
    # scarta punti degenerati (distanza zero, duplicati)
    valid = (r1 > 1e-12) & (r2 > r1)
    mu = r2[valid] / r1[valid]
    mu = np.sort(mu)
    # scarta la coda alta (outlier)
    k_keep = int(len(mu) * (1 - discard_frac))
    mu = mu[:k_keep]
    # stima ML: d = n / sum(log(mu))
    return len(mu) / np.sum(np.log(mu))


def mle_dim(X, k=5):
    """MLE estimator (Levina-Bickel) con k vicini."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]  # scarta il punto stesso
    # per ogni punto, stima locale: (k-1) / sum_{j=1..k-1} log(r_k / r_j)
    log_ratios = np.log(dist[:, -1:] / dist[:, :-1])
    valid = np.all(log_ratios > 0, axis=1)
    log_ratios = log_ratios[valid]
    if len(log_ratios) == 0:
        return float("nan")
    local_d = (k - 2) / log_ratios.sum(axis=1)
    return float(np.median(local_d))


def pca_dim(X, threshold=0.95):
    """Numero di componenti PCA per spiegare `threshold` della varianza."""
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, threshold) + 1)


def analyze(all_states):
    """
    Applica TwoNN, MLE e PCA su ogni layer.
    all_states: (n_layer, n_snippets, hidden_dim)
    """
    n_layers = all_states.shape[0]
    results = []
    print(f"\n[analyze] {n_layers} layers", flush=True)
    for li in range(n_layers):
        X = all_states[li]
        t0 = time.time()
        d_twonn = twonn(X)
        d_mle = mle_dim(X, k=5)
        d_pca95 = pca_dim(X, threshold=0.95)
        d_pca99 = pca_dim(X, threshold=0.99)
        results.append({
            "layer": li,
            "twonn": float(d_twonn),
            "mle": float(d_mle),
            "pca95": d_pca95,
            "pca99": d_pca99,
            "hidden_dim": int(X.shape[1]),
        })
        print(f"  layer {li:3d}: twonn={d_twonn:6.1f} mle={d_mle:6.1f} "
              f"pca95={d_pca95:4d} pca99={d_pca99:4d} (out of {X.shape[1]}) "
              f"t={time.time()-t0:.1f}s", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n-snippets", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"[main] device={device} dtype={dtype} model={args.model}", flush=True)

    print(f"[main] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)

    snippets = build_dataset(n_target=args.n_snippets)
    print(f"[main] dataset: {len(snippets)} snippet", flush=True)

    all_states = extract_hidden_states(model, tok, snippets, device, max_len=args.max_len)

    # libera GPU prima delle analisi CPU-bound
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = analyze(all_states)

    out_path = args.out or f"intrinsic_dim_{args.model.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_snippets": len(snippets),
            "max_len": args.max_len,
            "hidden_dim": all_states.shape[2],
            "n_layers": all_states.shape[0],
            "layers": results,
        }, f, indent=2)
    print(f"\n[main] salvato -> {out_path}", flush=True)

    # riepilogo finale
    twonn_vals = [r["twonn"] for r in results]
    pca95_vals = [r["pca95"] for r in results]
    print(f"\n{'='*60}")
    print(f"MODELLO: {args.model}")
    print(f"hidden_dim ambientale: {all_states.shape[2]}")
    print(f"dim intrinseca TwoNN — media: {np.mean(twonn_vals):.1f}  "
          f"min: {min(twonn_vals):.1f}  max: {max(twonn_vals):.1f}")
    print(f"dim intrinseca PCA-95 — media: {np.mean(pca95_vals):.1f}  "
          f"min: {min(pca95_vals)}  max: {max(pca95_vals)}")
    print(f"Fattore di compressione teorico: {all_states.shape[2]/np.mean(twonn_vals):.1f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
