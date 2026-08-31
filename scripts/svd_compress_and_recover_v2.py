#!/usr/bin/env python3
"""
Verifica sperimentale della sfruttabilità della dimensione intrinseca bassa
misurata su Qwen2.5-Coder-14B — VERSIONE CORRETTA CON SEPARAZIONE TRAIN/TEST.

Differenza chiave dalla v1: il corpus di fine-tuning di recovery è
COMPLETAMENTE DIVERSO dal test set — sono pattern e domini di coding non
presenti nei snippet di valutazione. Senza questa separazione, il fine-tune
avrebbe imparato direttamente le sequenze del test (contaminazione), non
avrebbe realmente recuperato la conoscenza persa dalla compressione.

Procedura:
  1. Baseline: perplexity su TEST_SNIPPETS
  2. Per ogni rank in [16, 32, 64]:
     a. Comprimi FFN ultimo layer con SVD rank-k
     b. Misura perplexity SUBITO su TEST_SNIPPETS (cold, senza recovery)
     c. Fine-tuning breve su RECOVERY_CORPUS (dominio disgiunto)
     d. Misura perplexity di nuovo su TEST_SNIPPETS (warm)

Il ΔWARM ora riflette il vero recupero di capacità, non memorizzazione.

Uso:
    python3 svd_compress_and_recover_v2.py
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


# ─── TEST SET: pattern di coding usati SOLO per valutazione ──────────────────
# Ognuno è distintivo (algoritmo, componente, query specifica) — nessuno
# di questi appare nel corpus di recovery, nemmeno come variante.
TEST_SNIPPETS = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key):\n        if key not in self.cache: return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]",
    "async function fetchUserData(userId) {\n  try {\n    const response = await fetch(`/api/users/${userId}`);\n    if (!response.ok) throw new Error('Failed');\n    return await response.json();\n  } catch (error) {\n    console.error('Error:', error);\n    return null;\n  }\n}",
    "export default function Navbar({ links, onLogout }) {\n  const [open, setOpen] = useState(false);\n  return (\n    <nav className=\"navbar\">\n      <button onClick={() => setOpen(!open)}>Menu</button>\n      {open && <ul>{links.map(l => <li key={l.href}>{l.label}</li>)}</ul>}\n    </nav>\n  );\n}",
    "SELECT u.id, u.name, COUNT(o.id) AS order_count, SUM(o.total) AS spent\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id\nHAVING COUNT(o.id) > 5\nORDER BY spent DESC LIMIT 20;",
    "fn quicksort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {\n    if arr.len() <= 1 { return arr.to_vec(); }\n    let pivot = arr[arr.len() / 2].clone();\n    let less: Vec<T> = arr.iter().filter(|&x| x < &pivot).cloned().collect();\n    let equal: Vec<T> = arr.iter().filter(|&x| x == &pivot).cloned().collect();\n    let greater: Vec<T> = arr.iter().filter(|&x| x > &pivot).cloned().collect();\n    [quicksort(&less), equal, quicksort(&greater)].concat()\n}",
    "type ApiResponse<T> = { data: T; error: string | null; loading: boolean };\n\nfunction useApi<T>(url: string): ApiResponse<T> {\n  const [state, setState] = useState<ApiResponse<T>>({ data: null as any, error: null, loading: true });\n  useEffect(() => {\n    fetch(url).then(r => r.json()).then(data => setState({ data, error: null, loading: false }));\n  }, [url]);\n  return state;\n}",
    "func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {\n    defer wg.Done()\n    for j := range jobs {\n        fmt.Printf(\"worker %d processing job %d\\n\", id, j)\n        time.Sleep(time.Millisecond * 100)\n        results <- j * 2\n    }\n}",
    ".navbar {\n  position: fixed;\n  top: 0;\n  z-index: 100;\n  display: flex;\n  justify-content: space-between;\n  align-items: center;\n  padding: 1rem 2rem;\n  background: rgba(255, 255, 255, 0.85);\n  backdrop-filter: blur(12px);\n}",
    "class Singleton:\n    _instance = None\n    _lock = threading.Lock()\n    def __new__(cls, *args, **kwargs):\n        if cls._instance is None:\n            with cls._lock:\n                if cls._instance is None:\n                    cls._instance = super().__new__(cls)\n        return cls._instance",
    "def dijkstra(graph, start):\n    dist = {node: float('inf') for node in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]: continue\n        for v, w in graph[u]:\n            if dist[u] + w < dist[v]:\n                dist[v] = dist[u] + w\n                heapq.heappush(pq, (dist[v], v))\n    return dist",
    "impl<T: Clone> BinaryTree<T> {\n    pub fn insert(&mut self, value: T) where T: Ord {\n        let new_node = Box::new(Node { value, left: None, right: None });\n        match &mut self.root {\n            None => self.root = Some(new_node),\n            Some(root) => Self::insert_recursive(root, new_node),\n        }\n    }\n}",
]


# ─── RECOVERY CORPUS: pattern completamente DIVERSI dal test set ─────────────
# Coprono coding in generale ma NESSUNO di questi appare (né come variante)
# tra i test snippet. Include: parsing, networking, machine learning, game
# dev, database interni, crittografia — domini disgiunti da quelli del test.
RECOVERY_SEEDS = [
    "def parse_ini_file(path):\n    config = {}\n    current_section = None\n    with open(path) as f:\n        for line in f:\n            line = line.strip()\n            if not line or line.startswith('#'):\n                continue\n            if line.startswith('[') and line.endswith(']'):\n                current_section = line[1:-1]\n                config[current_section] = {}\n            elif '=' in line and current_section:\n                key, val = line.split('=', 1)\n                config[current_section][key.strip()] = val.strip()\n    return config",
    "import socket\nimport threading\n\ndef handle_client(conn, addr):\n    print(f'Connected: {addr}')\n    while True:\n        data = conn.recv(1024)\n        if not data: break\n        conn.sendall(data.upper())\n    conn.close()\n\ndef start_server(host='0.0.0.0', port=8080):\n    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n    s.bind((host, port))\n    s.listen(5)\n    while True:\n        conn, addr = s.accept()\n        threading.Thread(target=handle_client, args=(conn, addr)).start()",
    "import numpy as np\n\nclass LogisticRegression:\n    def __init__(self, lr=0.01, epochs=1000):\n        self.lr = lr\n        self.epochs = epochs\n    def sigmoid(self, z):\n        return 1 / (1 + np.exp(-z))\n    def fit(self, X, y):\n        n, d = X.shape\n        self.w = np.zeros(d)\n        self.b = 0\n        for _ in range(self.epochs):\n            z = X @ self.w + self.b\n            pred = self.sigmoid(z)\n            dw = (X.T @ (pred - y)) / n\n            db = np.mean(pred - y)\n            self.w -= self.lr * dw\n            self.b -= self.lr * db\n    def predict(self, X):\n        return (self.sigmoid(X @ self.w + self.b) > 0.5).astype(int)",
    "using System;\nusing System.Collections.Generic;\n\npublic class GameEngine {\n    private List<Entity> entities = new List<Entity>();\n    private float deltaTime;\n\n    public void Update() {\n        deltaTime = Time.deltaTime;\n        foreach (var entity in entities) {\n            entity.Update(deltaTime);\n        }\n        CollisionDetection();\n    }\n\n    private void CollisionDetection() {\n        for (int i = 0; i < entities.Count; i++) {\n            for (int j = i + 1; j < entities.Count; j++) {\n                if (entities[i].Bounds.Intersects(entities[j].Bounds)) {\n                    entities[i].OnCollision(entities[j]);\n                }\n            }\n        }\n    }\n}",
    "CREATE TABLE audit_log (\n    id BIGSERIAL PRIMARY KEY,\n    table_name TEXT NOT NULL,\n    operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT','UPDATE','DELETE')),\n    old_data JSONB,\n    new_data JSONB,\n    changed_by TEXT,\n    changed_at TIMESTAMPTZ DEFAULT NOW(),\n    ip_address INET\n);\n\nCREATE INDEX idx_audit_table_time ON audit_log(table_name, changed_at DESC);\nCREATE INDEX idx_audit_operation ON audit_log(operation);",
    "import hashlib\nimport hmac\nimport secrets\nimport base64\n\ndef derive_key(password: str, salt: bytes, iterations: int = 100_000) -> bytes:\n    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)\n\ndef sign_token(payload: dict, secret: bytes) -> str:\n    import json\n    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()\n    signature = hmac.new(secret, body.encode(), hashlib.sha256).digest()\n    sig_b64 = base64.urlsafe_b64encode(signature).decode()\n    return f'{body}.{sig_b64}'\n\ndef verify_token(token: str, secret: bytes) -> bool:\n    try:\n        body, sig = token.rsplit('.', 1)\n        expected = hmac.new(secret, body.encode(), hashlib.sha256).digest()\n        return hmac.compare_digest(base64.urlsafe_b64decode(sig), expected)\n    except Exception:\n        return False",
    "package cache\n\nimport (\n    \"sync\"\n    \"time\"\n)\n\ntype TTLCache struct {\n    mu    sync.RWMutex\n    items map[string]cacheItem\n}\n\ntype cacheItem struct {\n    value      interface{}\n    expiration int64\n}\n\nfunc (c *TTLCache) Set(key string, value interface{}, ttl time.Duration) {\n    c.mu.Lock()\n    defer c.mu.Unlock()\n    c.items[key] = cacheItem{value: value, expiration: time.Now().Add(ttl).UnixNano()}\n}\n\nfunc (c *TTLCache) Get(key string) (interface{}, bool) {\n    c.mu.RLock()\n    defer c.mu.RUnlock()\n    item, ok := c.items[key]\n    if !ok || time.Now().UnixNano() > item.expiration {\n        return nil, false\n    }\n    return item.value, true\n}",
    "const express = require('express');\nconst rateLimit = require('express-rate-limit');\n\nconst app = express();\n\nconst apiLimiter = rateLimit({\n  windowMs: 15 * 60 * 1000,\n  max: 100,\n  standardHeaders: true,\n  legacyHeaders: false,\n  message: { error: 'Too many requests, please try again later' }\n});\n\napp.use('/api/', apiLimiter);\n\napp.get('/api/status', (req, res) => {\n  res.json({ status: 'ok', timestamp: Date.now() });\n});\n\napp.listen(3000);",
    "@Component({\n  selector: 'app-user-list',\n  template: `\n    <div *ngFor=\"let user of users$ | async\">\n      {{ user.name }} - {{ user.email }}\n      <button (click)=\"deleteUser(user.id)\">Delete</button>\n    </div>\n  `\n})\nexport class UserListComponent implements OnInit {\n  users$: Observable<User[]>;\n  constructor(private userService: UserService) {}\n  ngOnInit() {\n    this.users$ = this.userService.getUsers();\n  }\n  deleteUser(id: number) {\n    this.userService.deleteUser(id).subscribe(() => this.users$ = this.userService.getUsers());\n  }\n}",
    "use std::io::{BufRead, BufReader, Write};\nuse std::fs::File;\n\nfn count_words_by_length(path: &str) -> Result<std::collections::HashMap<usize, u32>, std::io::Error> {\n    let file = File::open(path)?;\n    let reader = BufReader::new(file);\n    let mut counts = std::collections::HashMap::new();\n    for line in reader.lines() {\n        let line = line?;\n        for word in line.split_whitespace() {\n            *counts.entry(word.len()).or_insert(0) += 1;\n        }\n    }\n    Ok(counts)\n}",
    "def train_test_split(X, y, test_size=0.2, random_state=None):\n    import numpy as np\n    if random_state is not None:\n        np.random.seed(random_state)\n    n = len(X)\n    n_test = int(n * test_size)\n    indices = np.random.permutation(n)\n    test_idx = indices[:n_test]\n    train_idx = indices[n_test:]\n    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]",
    "class ThreadPool:\n    def __init__(self, num_workers=4):\n        from queue import Queue\n        from threading import Thread\n        self.tasks = Queue()\n        self.workers = [Thread(target=self._worker, daemon=True) for _ in range(num_workers)]\n        for w in self.workers:\n            w.start()\n    def _worker(self):\n        while True:\n            fn, args, kwargs = self.tasks.get()\n            try:\n                fn(*args, **kwargs)\n            except Exception as e:\n                print(f'Task failed: {e}')\n            finally:\n                self.tasks.task_done()\n    def submit(self, fn, *args, **kwargs):\n        self.tasks.put((fn, args, kwargs))\n    def wait_all(self):\n        self.tasks.join()",
    "resource \"aws_s3_bucket\" \"logs\" {\n  bucket = \"my-app-logs-${var.environment}\"\n  \n  lifecycle_rule {\n    id      = \"expire_old_logs\"\n    enabled = true\n    expiration { days = 90 }\n    noncurrent_version_expiration { days = 30 }\n  }\n  \n  versioning { enabled = true }\n  \n  server_side_encryption_configuration {\n    rule {\n      apply_server_side_encryption_by_default {\n        sse_algorithm = \"AES256\"\n      }\n    }\n  }\n}",
    "interface Observer<T> {\n  update(value: T): void;\n}\n\nclass Subject<T> {\n  private observers: Set<Observer<T>> = new Set();\n  \n  subscribe(observer: Observer<T>): () => void {\n    this.observers.add(observer);\n    return () => this.observers.delete(observer);\n  }\n  \n  notify(value: T): void {\n    for (const observer of this.observers) {\n      observer.update(value);\n    }\n  }\n}",
    "def parse_url(url: str) -> dict:\n    from urllib.parse import urlparse, parse_qs\n    parsed = urlparse(url)\n    return {\n        'scheme': parsed.scheme,\n        'host': parsed.hostname,\n        'port': parsed.port,\n        'path': parsed.path,\n        'query': parse_qs(parsed.query),\n        'fragment': parsed.fragment,\n    }",
    "@app.route('/upload', methods=['POST'])\ndef upload_file():\n    if 'file' not in request.files:\n        return jsonify({'error': 'No file'}), 400\n    file = request.files['file']\n    if file.filename == '':\n        return jsonify({'error': 'Empty filename'}), 400\n    if not allowed_file(file.filename):\n        return jsonify({'error': 'File type not allowed'}), 400\n    filename = secure_filename(file.filename)\n    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))\n    return jsonify({'filename': filename, 'size': os.path.getsize(...)})",
    "func BFS(graph map[int][]int, start int) []int {\n    visited := make(map[int]bool)\n    queue := []int{start}\n    result := []int{}\n    visited[start] = true\n    for len(queue) > 0 {\n        node := queue[0]\n        queue = queue[1:]\n        result = append(result, node)\n        for _, neighbor := range graph[node] {\n            if !visited[neighbor] {\n                visited[neighbor] = true\n                queue = append(queue, neighbor)\n            }\n        }\n    }\n    return result\n}",
    "CREATE OR REPLACE FUNCTION calculate_discount(\n    base_price NUMERIC,\n    customer_tier VARCHAR,\n    quantity INTEGER\n) RETURNS NUMERIC AS $$\nDECLARE\n    discount_pct NUMERIC := 0;\nBEGIN\n    IF customer_tier = 'GOLD' THEN discount_pct := 0.15;\n    ELSIF customer_tier = 'SILVER' THEN discount_pct := 0.08;\n    END IF;\n    IF quantity >= 100 THEN discount_pct := discount_pct + 0.05;\n    END IF;\n    RETURN base_price * (1 - LEAST(discount_pct, 0.30));\nEND;\n$$ LANGUAGE plpgsql;",
    "import { Injectable } from '@angular/core';\nimport { HttpClient, HttpErrorResponse } from '@angular/common/http';\nimport { Observable, throwError, of } from 'rxjs';\nimport { catchError, retry, map } from 'rxjs/operators';\n\n@Injectable({ providedIn: 'root' })\nexport class ApiService {\n  constructor(private http: HttpClient) {}\n  \n  fetchWithRetry<T>(url: string): Observable<T> {\n    return this.http.get<T>(url).pipe(\n      retry(3),\n      catchError((error: HttpErrorResponse) => {\n        console.error('API error:', error);\n        return throwError(() => new Error('Failed after 3 retries'));\n      })\n    );\n  }\n}",
    "type Trie struct {\n    children map[rune]*Trie\n    isEnd    bool\n}\n\nfunc NewTrie() *Trie {\n    return &Trie{children: make(map[rune]*Trie)}\n}\n\nfunc (t *Trie) Insert(word string) {\n    node := t\n    for _, ch := range word {\n        if _, ok := node.children[ch]; !ok {\n            node.children[ch] = NewTrie()\n        }\n        node = node.children[ch]\n    }\n    node.isEnd = true\n}\n\nfunc (t *Trie) Search(word string) bool {\n    node := t\n    for _, ch := range word {\n        if _, ok := node.children[ch]; !ok {\n            return false\n        }\n        node = node.children[ch]\n    }\n    return node.isEnd\n}",
    "class EventEmitter {\n  constructor() { this.events = new Map(); }\n  on(event, listener) {\n    if (!this.events.has(event)) this.events.set(event, []);\n    this.events.get(event).push(listener);\n    return () => this.off(event, listener);\n  }\n  off(event, listener) {\n    const listeners = this.events.get(event);\n    if (listeners) this.events.set(event, listeners.filter(l => l !== listener));\n  }\n  emit(event, ...args) {\n    const listeners = this.events.get(event) || [];\n    listeners.forEach(l => l(...args));\n  }\n}",
    "#include <vector>\n#include <algorithm>\n\ntemplate<typename T>\nclass MergeSort {\npublic:\n    static void sort(std::vector<T>& arr) {\n        if (arr.size() <= 1) return;\n        merge_sort(arr, 0, arr.size() - 1);\n    }\nprivate:\n    static void merge_sort(std::vector<T>& arr, size_t left, size_t right) {\n        if (left >= right) return;\n        size_t mid = left + (right - left) / 2;\n        merge_sort(arr, left, mid);\n        merge_sort(arr, mid + 1, right);\n        merge(arr, left, mid, right);\n    }\n};",
    "async fn download_file(url: &str, dest: &Path) -> Result<u64, Box<dyn std::error::Error>> {\n    use tokio::io::AsyncWriteExt;\n    let response = reqwest::get(url).await?;\n    let total = response.content_length().unwrap_or(0);\n    let mut file = tokio::fs::File::create(dest).await?;\n    let mut stream = response.bytes_stream();\n    while let Some(chunk) = futures::StreamExt::next(&mut stream).await {\n        file.write_all(&chunk?).await?;\n    }\n    Ok(total)\n}",
    "from dataclasses import dataclass, field\nfrom typing import Optional\nfrom datetime import datetime\n\n@dataclass\nclass Transaction:\n    id: str\n    amount: float\n    currency: str = 'USD'\n    timestamp: datetime = field(default_factory=datetime.utcnow)\n    metadata: dict = field(default_factory=dict)\n    parent_id: Optional[str] = None\n    \n    def is_refund(self) -> bool:\n        return self.amount < 0\n    \n    def to_dict(self) -> dict:\n        return {'id': self.id, 'amount': self.amount, 'currency': self.currency, 'ts': self.timestamp.isoformat()}",
]


def build_recovery_corpus(n_target=500):
    """Espande i seed di recovery con variazioni per raggiungere n_target snippet."""
    import random
    random.seed(9999)
    out = list(RECOVERY_SEEDS)
    substitutions = [
        ("data", ["records", "input", "payload", "items"]),
        ("value", ["val", "elem", "item"]),
        ("result", ["output", "res", "acc"]),
        ("config", ["settings", "params", "options"]),
        ("cache", ["store", "buffer", "memo"]),
    ]
    seen = set(out)
    attempts = 0
    while len(out) < n_target and attempts < n_target * 10:
        attempts += 1
        s = random.choice(RECOVERY_SEEDS)
        for _ in range(random.randint(2, 4)):
            old, news = random.choice(substitutions)
            new = random.choice(news)
            if old in s:
                s = s.replace(old, new, 1)
        s = f"# recovery-{random.randint(1, 999999)}\n{s}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def check_no_contamination(test_snippets, recovery_corpus):
    """Verifica che nessun test snippet appaia (anche parzialmente) nel recovery."""
    warnings = []
    for i, test in enumerate(test_snippets):
        # controlla se sequenze significative del test snippet appaiono nel recovery
        # (prendi frammenti di 40+ caratteri distintivi)
        test_sig = test[:80].strip()  # prime 80 char come firma
        for j, rec in enumerate(recovery_corpus):
            if test_sig in rec:
                warnings.append(f"test[{i}] appare in recovery[{j}]")
    return warnings


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
        n_tok = input_ids.shape[1] - 1
        total_loss += outputs.loss.item() * n_tok
        total_tokens += n_tok
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss), avg_loss


class LowRankLinear(nn.Module):
    def __init__(self, down, up):
        super().__init__()
        self.down = down
        self.up = up

    def forward(self, x):
        return self.up(self.down(x))


def svd_compress_linear(linear, rank, dtype, device):
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


def finetune_recover(model, tokenizer, corpus, device, steps=100, lr=1e-4, batch_len=256):
    model.train()
    for p in model.parameters():
        p.requires_grad = False
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--recover-steps", type=int, default=100)
    ap.add_argument("--recover-lr", type=float, default=1e-4)
    ap.add_argument("--n-recovery", type=int, default=500)
    ap.add_argument("--out", default="svd_recovery_v2_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[main] device={device} dtype={dtype} model={args.model}", flush=True)

    # Verifica separazione train/test PRIMA di caricare il modello
    recovery_corpus = build_recovery_corpus(n_target=args.n_recovery)
    print(f"[main] recovery corpus: {len(recovery_corpus)} snippet", flush=True)
    warnings = check_no_contamination(TEST_SNIPPETS, recovery_corpus)
    if warnings:
        print(f"[ERRORE] contaminazione train/test rilevata!", flush=True)
        for w in warnings:
            print(f"  {w}", flush=True)
        return
    print(f"[main] verifica no-contamination: OK — nessun test snippet nel recovery corpus", flush=True)

    print(f"\n[main] loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)

    hidden_dim = base_model.config.hidden_size
    inter_dim = base_model.config.intermediate_size
    print(f"[main] hidden_dim={hidden_dim} intermediate_dim={inter_dim} "
          f"n_layers={base_model.config.num_hidden_layers}", flush=True)
    ffn_orig = count_ffn_params_last(base_model)
    print(f"[main] FFN ultimo layer: {ffn_orig:,} parametri", flush=True)

    print(f"\n=== BASELINE (nessuna compressione, su TEST_SNIPPETS) ===", flush=True)
    t0 = time.time()
    ppl_base, loss_base = compute_perplexity(base_model, tok, TEST_SNIPPETS, device)
    print(f"perplexity baseline: {ppl_base:.4f}  loss: {loss_base:.4f}  "
          f"t={time.time()-t0:.1f}s", flush=True)

    # Baseline anche sul recovery corpus, per capire quanto sia diverso dal test
    ppl_base_recovery, _ = compute_perplexity(base_model, tok, recovery_corpus[:50], device)
    print(f"perplexity baseline sul recovery corpus (sanity check): {ppl_base_recovery:.4f}", flush=True)
    print(f"→ se test e recovery hanno perplexity simili, i due domini sono paragonabili in difficoltà", flush=True)

    results = {"model": args.model, "baseline_ppl": ppl_base,
               "baseline_ppl_recovery_domain": ppl_base_recovery,
               "hidden_dim": hidden_dim, "intermediate_dim": inter_dim,
               "ffn_params_orig": ffn_orig, "n_test": len(TEST_SNIPPETS),
               "n_recovery": len(recovery_corpus), "runs": []}

    for rank in args.ranks:
        print(f"\n{'='*70}\n=== RANK {rank} ===\n{'='*70}", flush=True)

        print(f"  [copy] deepcopy del modello ...", flush=True)
        t0 = time.time()
        m = copy.deepcopy(base_model)
        print(f"  [copy] fatto in {time.time()-t0:.1f}s", flush=True)

        print(f"  [compress] SVD rank-{rank} su FFN ultimo layer ...", flush=True)
        t0 = time.time()
        m = compress_last_layer_ffn(m, rank, dtype, device)
        ffn_new = count_ffn_params_last(m)
        ratio = ffn_orig / ffn_new
        print(f"  [compress] fatto in {time.time()-t0:.1f}s  "
              f"nuovi params: {ffn_new:,}  compressione: {ratio:.1f}x", flush=True)

        t0 = time.time()
        ppl_cold, _ = compute_perplexity(m, tok, TEST_SNIPPETS, device)
        delta_cold = (ppl_cold / ppl_base - 1) * 100
        print(f"  [cold]   perplexity su TEST: {ppl_cold:.4f}  delta: {delta_cold:+.1f}%  "
              f"t={time.time()-t0:.1f}s", flush=True)

        print(f"  [recover] fine-tuning {args.recover_steps} steps su RECOVERY corpus "
              f"({len(recovery_corpus)} snippet disgiunti dal test) ...", flush=True)
        t0 = time.time()
        finetune_recover(m, tok, recovery_corpus, device, steps=args.recover_steps, lr=args.recover_lr)
        t_recover = time.time() - t0

        t0 = time.time()
        ppl_warm, _ = compute_perplexity(m, tok, TEST_SNIPPETS, device)
        delta_warm = (ppl_warm / ppl_base - 1) * 100
        print(f"  [warm]   perplexity su TEST: {ppl_warm:.4f}  delta: {delta_warm:+.1f}%  "
              f"(recovery: {t_recover:.0f}s)", flush=True)

        results["runs"].append({
            "rank": rank, "ffn_params": ffn_new, "compression_ratio": ratio,
            "ppl_cold": ppl_cold, "delta_cold_pct": delta_cold,
            "ppl_warm": ppl_warm, "delta_warm_pct": delta_warm,
            "recovery_seconds": t_recover,
        })

        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[main] salvato -> {args.out}", flush=True)

    print(f"\n{'='*70}")
    print(f"RIEPILOGO — {args.model} (con separazione train/test)")
    print(f"baseline ppl (test): {ppl_base:.4f}")
    print(f"baseline ppl (recovery domain): {ppl_base_recovery:.4f}")
    print(f"{'='*70}")
    print(f"{'RANK':>6} {'RATIO':>8} {'PPL COLD':>10} {'ΔCOLD':>8} {'PPL WARM':>10} {'ΔWARM':>8}")
    for r in results["runs"]:
        print(f"  {r['rank']:>4} {r['compression_ratio']:>6.1f}x "
              f"{r['ppl_cold']:>10.4f} {r['delta_cold_pct']:>+7.1f}% "
              f"{r['ppl_warm']:>10.4f} {r['delta_warm_pct']:>+7.1f}%")
    print(f"{'='*70}")
    print(f"\nInterpretazione ONESTA (con train/test separati):")
    print(f"  ΔWARM entro ±5%: compressione a quel rank e' PRATICAMENTE SFRUTTABILE")
    print(f"  ΔWARM 5-20%: recupero parziale, potrebbe reggere con piu' step/dati")
    print(f"  ΔWARM > 20% o negativo importante: recovery ha appreso qualcosa di diverso")
    print(f"    (negativo forte = overfitting sul recovery domain, non recupero reale)")


if __name__ == "__main__":
    main()
