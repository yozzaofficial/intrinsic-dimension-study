#!/usr/bin/env python3
"""
Verifica sperimentale: la dimensione intrinseca bassa misurata sull'ultimo
layer di Qwen2.5-Coder-14B è SFRUTTABILE per compressione via SVD?

Procedura:
  1. Baseline: perplexity di Qwen2.5-Coder-14B su un test set di coding
  2. Per ogni rank in [16, 32, 64]:
     a. Comprimi le matrici FFN dell'ULTIMO layer con SVD rank-k
        (down_proj, up_proj, gate_proj — quelle che pesano di piu')
     b. Misura perplexity SUBITO (a freddo, senza recupero)
     c. Fine-tuning breve (100 steps) dello strato compresso su dati coding
     d. Misura perplexity dopo il recupero
  3. Tabella comparativa: quanto degrada, quanto recupera

Risultato atteso:
  - Se dopo recovery a rank 16-32 la perplexity resta vicina alla baseline
    (delta < 10-15%), la dimensione intrinseca bassa è realmente sfruttabile.
  - Se crolla anche a rank 64 con recovery, il margine teorico non e' pratico.

Hardware richiesto: GPU >=40GB VRAM (A100/A6000/L40).
Tempo stimato: ~40-60 minuti totali sulla VPS.

Uso:
    python3 svd_compress_and_recover.py
"""
import argparse
import copy
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Test set: coding snippet, uguali (~) a quelli dell'esperimento dim ──────
TEST_SNIPPETS = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key not in self.cache: return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache: self.cache.move_to_end(key)\n        elif len(self.cache) >= self.capacity: self.cache.popitem(last=False)\n        self.cache[key] = value",
    "async function fetchUserData(userId) {\n  try {\n    const response = await fetch(`/api/users/${userId}`);\n    if (!response.ok) throw new Error('Failed');\n    return await response.json();\n  } catch (error) {\n    console.error('Error:', error);\n    return null;\n  }\n}",
    "export default function Navbar({ links, onLogout }) {\n  const [open, setOpen] = useState(false);\n  return (\n    <nav className=\"navbar\">\n      <button onClick={() => setOpen(!open)}>Menu</button>\n      {open && <ul>{links.map(l => <li key={l.href}>{l.label}</li>)}</ul>}\n      <button onClick={onLogout}>Logout</button>\n    </nav>\n  );\n}",
    "SELECT u.id, u.name, COUNT(o.id) AS order_count, SUM(o.total) AS spent\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id\nHAVING COUNT(o.id) > 5\nORDER BY spent DESC LIMIT 20;",
    "fn quicksort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {\n    if arr.len() <= 1 { return arr.to_vec(); }\n    let pivot = arr[arr.len() / 2].clone();\n    let less: Vec<T> = arr.iter().filter(|&x| x < &pivot).cloned().collect();\n    let equal: Vec<T> = arr.iter().filter(|&x| x == &pivot).cloned().collect();\n    let greater: Vec<T> = arr.iter().filter(|&x| x > &pivot).cloned().collect();\n    [quicksort(&less), equal, quicksort(&greater)].concat()\n}",
    "type ApiResponse<T> = { data: T; error: string | null; loading: boolean };\n\nfunction useApi<T>(url: string): ApiResponse<T> {\n  const [state, setState] = useState<ApiResponse<T>>({ data: null as any, error: null, loading: true });\n  useEffect(() => {\n    fetch(url).then(r => r.json()).then(data => setState({ data, error: null, loading: false }))\n              .catch(e => setState({ data: null as any, error: String(e), loading: false }));\n  }, [url]);\n  return state;\n}",
    "func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {\n    defer wg.Done()\n    for j := range jobs {\n        fmt.Printf(\"worker %d processing job %d\\n\", id, j)\n        time.Sleep(time.Millisecond * 100)\n        results <- j * 2\n    }\n}",
    ".navbar {\n  position: fixed;\n  top: 0;\n  left: 0;\n  right: 0;\n  z-index: 100;\n  display: flex;\n  justify-content: space-between;\n  align-items: center;\n  padding: 1rem 2rem;\n  background: rgba(255, 255, 255, 0.85);\n  backdrop-filter: blur(12px);\n  border-bottom: 1px solid rgba(0, 0, 0, 0.05);\n}\n\n@media (max-width: 768px) {\n  .navbar { padding: 0.5rem 1rem; }\n  .nav-links { display: none; }\n  .hamburger { display: block; }\n}",
    "class Singleton:\n    _instance = None\n    _lock = threading.Lock()\n    \n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None:\n            with cls._lock:\n                if cls._instance is None:\n                    cls._instance = super().__new__(cls)\n                    cls._instance._init(*args, **kwargs)\n        return cls._instance\n    \n    def _init(self, config):\n        self.config = config",
    "def dijkstra(graph, start):\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]: continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist",
    "impl<T: Clone> BinaryTree<T> {\n    pub fn insert(&mut self, value: T) where T: Ord {\n        let new_node = Box::new(Node { value, left: None, right: None });\n        match &mut self.root {\n            None => self.root = Some(new_node),\n            Some(root) => Self::insert_recursive(root, new_node),\n        }\n    }\n}",
]


# ─── Corpus di fine-tuning (piu' snippet, con variazioni) ────────────────────
def build_finetune_corpus(n_target=500):
    import random
    random.seed(123)
    out = list(TEST_SNIPPETS)
    substitutions = [
        ("arr", ["data", "items", "lst", "seq"]),
        ("target", ["value", "key", "elem", "x"]),
        ("user", ["client", "member", "account"]),
        ("result", ["output", "res", "acc"]),
        ("value", ["val", "item", "data"]),
    ]
    seen = set(out)
    attempts = 0
    while len(out) < n_target and attempts < n_target * 10:
        attempts += 1
        s = random.choice(TEST_SNIPPETS)
        for _ in range(random.randint(2, 3)):
            old, news = random.choice(substitutions)
            new = random.choice(news)
            if old in s:
                s = s.replace(old, new, 1)
        s = f"# variant {random.randint(1, 999999)}\n{s}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─── Perplexity su un testo (o su un corpus, media pesata) ───────────────────
@torch.no_grad()
def compute_perplexity(model, tokenizer, texts, device, max_len=256):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        n_tok = input_ids.shape[1] - 1
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss


# ─── SVD low-rank replacement: sostituisce nn.Linear con due Linear in serie ─
class LowRankLinear(nn.Module):
    def __init__(self, down, up):
        super().__init__()
        self.down = down
        self.up = up

    def forward(self, x):
        return self.up(self.down(x))


def svd_compress_linear(linear, rank, dtype, device):
    """
    Comprime un nn.Linear (peso W di forma (d_out, d_in)) usando SVD rank-k.
    Nuovi parametri totali: rank * (d_in + d_out) invece di d_in * d_out.
    Utile solo se rank < (d_in * d_out) / (d_in + d_out).
    """
    W = linear.weight.data.float().cpu()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    Uk = U[:, :rank]
    Sk = S[:rank]
    Vtk = Vt[:rank, :]
    d_in, d_out = W.shape[1], W.shape[0]
    has_bias = linear.bias is not None
    down = nn.Linear(d_in, rank, bias=False)
    up = nn.Linear(rank, d_out, bias=has_bias)
    down.weight.data = (Vtk * Sk.unsqueeze(1)).to(dtype).to(device)
    up.weight.data = Uk.contiguous().to(dtype).to(device)
    if has_bias:
        up.bias.data = linear.bias.data.to(dtype).to(device)
    return LowRankLinear(down, up)


def compress_last_layer_ffn(model, rank, dtype, device):
    """Sostituisce gate_proj, up_proj, down_proj dell'ULTIMO layer con versioni SVD rank-k."""
    last_layer = model.model.layers[-1]
    mlp = last_layer.mlp
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        if hasattr(mlp, proj_name):
            new_proj = svd_compress_linear(getattr(mlp, proj_name), rank, dtype, device)
            setattr(mlp, proj_name, new_proj)
    return model


def count_ffn_params_last(model):
    last = model.model.layers[-1].mlp
    total = 0
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        if hasattr(last, proj_name):
            total += sum(p.numel() for p in getattr(last, proj_name).parameters())
    return total


# ─── Recovery fine-tuning: aggiorna SOLO le matrici SVD compresse ────────────
def finetune_recover(model, tokenizer, corpus, device, steps=100, lr=1e-4, batch_len=256):
    """
    Fine-tuning breve delle sole matrici SVD dell'ultimo layer.
    Congela tutto il resto — obiettivo: recuperare qualita' persa dalla
    compressione senza toccare la conoscenza del resto del modello.
    """
    model.train()
    # congela tutto
    for p in model.parameters():
        p.requires_grad = False
    # sblocca solo le proiezioni compresse dell'ultimo layer
    last_mlp = model.model.layers[-1].mlp
    trainable = []
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        if hasattr(last_mlp, proj_name):
            proj = getattr(last_mlp, proj_name)
            if isinstance(proj, LowRankLinear):
                for p in proj.parameters():
                    p.requires_grad = True
                    trainable.append(p)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"    [recover] parametri trainable: {n_trainable:,}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr)

    import random
    random.seed(456)
    t0 = time.time()
    for step in range(1, steps + 1):
        text = random.choice(corpus)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=batch_len).to(device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 20 == 0 or step == steps:
            print(f"    [recover] step {step:3d}/{steps}  loss={loss.item():.4f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    model.eval()


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--recover-steps", type=int, default=100)
    ap.add_argument("--recover-lr", type=float, default=1e-4)
    ap.add_argument("--n-finetune", type=int, default=500)
    ap.add_argument("--out", default="svd_recovery_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[main] device={device} dtype={dtype} model={args.model}", flush=True)

    print(f"[main] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)

    hidden_dim = base_model.config.hidden_size
    inter_dim = base_model.config.intermediate_size
    print(f"[main] hidden_dim={hidden_dim} intermediate_dim={inter_dim} "
          f"n_layers={base_model.config.num_hidden_layers}", flush=True)
    ffn_orig = count_ffn_params_last(base_model)
    print(f"[main] FFN ultimo layer: {ffn_orig:,} parametri "
          f"({ffn_orig*2/1e9:.2f} GB in bf16)", flush=True)

    # ─── Baseline ────────────────────────────────────────────────────────────
    print(f"\n=== BASELINE (nessuna compressione) ===", flush=True)
    t0 = time.time()
    ppl_base, loss_base = compute_perplexity(base_model, tok, TEST_SNIPPETS, device)
    print(f"perplexity baseline: {ppl_base:.4f}  loss: {loss_base:.4f}  "
          f"t={time.time()-t0:.1f}s", flush=True)

    corpus = build_finetune_corpus(n_target=args.n_finetune)
    print(f"\n[main] corpus fine-tuning: {len(corpus)} snippet", flush=True)

    results = {"model": args.model, "baseline_ppl": ppl_base,
               "hidden_dim": hidden_dim, "intermediate_dim": inter_dim,
               "ffn_params_orig": ffn_orig, "runs": []}

    for rank in args.ranks:
        print(f"\n{'='*70}\n=== RANK {rank} ===\n{'='*70}", flush=True)

        # copia il modello (per lasciare il baseline intatto tra i test)
        # nota: deepcopy su modelli grandi e' costoso ma necessario
        print(f"  [copy] deepcopy del modello ...", flush=True)
        t0 = time.time()
        m = copy.deepcopy(base_model)
        print(f"  [copy] fatto in {time.time()-t0:.1f}s", flush=True)

        # comprimi ultimo layer
        print(f"  [compress] SVD rank-{rank} su gate_proj/up_proj/down_proj "
              f"dell'ultimo layer ...", flush=True)
        t0 = time.time()
        m = compress_last_layer_ffn(m, rank, dtype, device)
        ffn_new = count_ffn_params_last(m)
        ratio = ffn_orig / ffn_new
        print(f"  [compress] fatto in {time.time()-t0:.1f}s  "
              f"nuovi params: {ffn_new:,}  compressione: {ratio:.1f}x", flush=True)

        # perplexity a freddo
        t0 = time.time()
        ppl_cold, _ = compute_perplexity(m, tok, TEST_SNIPPETS, device)
        delta_cold = (ppl_cold / ppl_base - 1) * 100
        print(f"  [cold]   perplexity: {ppl_cold:.4f}  delta: {delta_cold:+.1f}%  "
              f"t={time.time()-t0:.1f}s", flush=True)

        # recovery fine-tuning
        print(f"  [recover] fine-tuning {args.recover_steps} steps "
              f"su {len(corpus)} snippet ...", flush=True)
        t0 = time.time()
        finetune_recover(m, tok, corpus, device, steps=args.recover_steps, lr=args.recover_lr)
        t_recover = time.time() - t0

        # perplexity dopo recovery
        t0 = time.time()
        ppl_recovered, _ = compute_perplexity(m, tok, TEST_SNIPPETS, device)
        delta_recovered = (ppl_recovered / ppl_base - 1) * 100
        print(f"  [warm]   perplexity: {ppl_recovered:.4f}  delta: {delta_recovered:+.1f}%  "
              f"(recovery: {t_recover:.0f}s)", flush=True)

        results["runs"].append({
            "rank": rank,
            "ffn_params": ffn_new,
            "compression_ratio": ratio,
            "ppl_cold": ppl_cold,
            "delta_cold_pct": delta_cold,
            "ppl_recovered": ppl_recovered,
            "delta_recovered_pct": delta_recovered,
            "recovery_seconds": t_recover,
        })

        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ─── Salva risultati + riepilogo ─────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main] salvato -> {args.out}", flush=True)

    print(f"\n{'='*70}")
    print(f"RIEPILOGO — {args.model}")
    print(f"baseline ppl: {ppl_base:.4f}")
    print(f"{'='*70}")
    print(f"{'RANK':>6} {'RATIO':>8} {'PPL COLD':>10} {'ΔCOLD':>8} {'PPL WARM':>10} {'ΔWARM':>8}")
    for r in results["runs"]:
        print(f"  {r['rank']:>4} {r['compression_ratio']:>6.1f}x "
              f"{r['ppl_cold']:>10.4f} {r['delta_cold_pct']:>+7.1f}% "
              f"{r['ppl_recovered']:>10.4f} {r['delta_recovered_pct']:>+7.1f}%")
    print(f"{'='*70}")
    print(f"\nInterpretazione:")
    print(f"  ΔWARM < +5%: la compressione a quel rank e' PRATICAMENTE SFRUTTABILE")
    print(f"  ΔWARM 5-15%: ambiguo — potrebbe reggere con piu' step di recovery")
    print(f"  ΔWARM > 15%: la compressione ROMPE il modello, teoria non pratica a quel rank")


if __name__ == "__main__":
    main()
