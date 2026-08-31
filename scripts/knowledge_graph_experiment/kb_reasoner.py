"""
kb_reasoner.py — ragionatore che USA la knowledge base.

Loop esplicito plan → retrieve → answer, pilotato da qwen2.5-coder:14b via ollama.

Non è un chain-of-thought spontaneo: il modello viene chiamato TRE volte con ruoli precisi:

  1) PLANNER: data la richiesta utente + lista dei tipi/pattern nella KB, decide
     quali termini cercare (una o più query fulltext) e quali nodi espandere
     (traversal a partire da un nodo).
  2) [server] esegue le query sulla KB, raccoglie sottografo.
  3) ANSWERER: riceve richiesta + sottografo strutturato → risposta finale.

Il grafo non è mai "invisibile": il planner vede l'inventario tipi/pattern,
l'answerer vede i nodi CONCRETI con id, title, summary, content, e le relazioni
tra loro.

Uso:
    from kb_reasoner import KBReasoner
    r = KBReasoner()
    out = r.answer("come faccio un hamburger menu accessibile?")
    print(out["response"])
    print(out["trace"])       # cosa ha cercato, cosa ha trovato
"""
import json
import re
import time
import urllib.request
import urllib.error
from typing import Optional

from kb import KnowledgeBase


OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:14b-instruct-q8_0"
FAST_MODEL = "qwen2.5:3b-instruct-q4_K_M"


def _ollama_chat(model: str, messages: list, temperature: float = 0.2,
                  max_tokens: int = 512, timeout: int = 120) -> str:
    """Chiama ollama, ritorna il testo della risposta. Blocking."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["message"]["content"]


def _extract_json(text: str) -> Optional[dict]:
    """Estrae il primo blocco JSON dal testo. Tollera fence ```json."""
    # prova prima con fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # fallback: primo { ... } bilanciato
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    start = None
    return None


class KBReasoner:
    def __init__(self, model: str = DEFAULT_MODEL, planner_model: Optional[str] = None):
        """
        model      : usato per la risposta finale (14B raccomandato).
        planner_model : se diverso, usato per il planning (default = stesso).
        """
        self.kb = KnowledgeBase()
        self.model = model
        self.planner_model = planner_model or model

    def close(self):
        self.kb.close()

    # ── introspezione KB per il planner ─────────────────────────────────
    def _kb_inventory(self) -> dict:
        """Riassunto compatto della KB: tipi disponibili + tutti i nodi con id+title."""
        s = self.kb.stats()
        # lista di tutti i nodi (per pilota piccola, 34 nodi → cap.)
        with self.kb.driver.session() as sess:
            r = sess.run("MATCH (n:Node) RETURN n.id AS id, n.type AS type, n.title AS title ORDER BY n.type, n.id")
            nodes = [dict(rec) for rec in r]
        return {"stats": s, "nodes": nodes}

    # ── STEP 1: planner ────────────────────────────────────────────────
    def _plan(self, user_request: str) -> dict:
        inv = self._kb_inventory()
        # inventario compatto: raggruppa per tipo
        by_type = {}
        for n in inv["nodes"]:
            by_type.setdefault(n["type"], []).append(f'{n["id"]} ("{n["title"]}")')

        inventory_text = "\n".join(
            f"{t} ({len(ids)}):\n  " + "\n  ".join(ids)
            for t, ids in by_type.items()
        )

        system = (
            "You are a planner that decides how to query a knowledge graph to answer a user's "
            "question about web UI development.\n\n"
            "You will be given: (1) the user's question, (2) the complete inventory of nodes in "
            "the graph, grouped by type.\n\n"
            "Your task: output a JSON plan with:\n"
            '  - "search_queries": list of 1–3 short text queries for fulltext search over the graph.\n'
            '  - "expand_nodes": list of node IDs (from the inventory) whose neighbors (depth 1–2) '
            "should be fetched. Pick the most directly relevant nodes.\n"
            '  - "reasoning": one short sentence explaining what you\'re looking for.\n\n'
            "Output ONLY the JSON, nothing else. Example:\n"
            '{"search_queries": ["hamburger menu", "keyboard navigation"], '
            '"expand_nodes": ["pattern-hamburger-menu"], '
            '"reasoning": "Need the hamburger pattern and its accessibility requirements."}'
        )
        user = f"USER QUESTION:\n{user_request}\n\nGRAPH INVENTORY:\n{inventory_text}"

        raw = _ollama_chat(
            self.planner_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        plan = _extract_json(raw)
        if not plan:
            # fallback: nessun plan, il retrieve userà solo la richiesta come query
            plan = {
                "search_queries": [user_request],
                "expand_nodes": [],
                "reasoning": "fallback: planner did not return valid JSON",
            }
        plan.setdefault("search_queries", [])
        plan.setdefault("expand_nodes", [])
        return plan

    # ── STEP 2: retrieve dalla KB ──────────────────────────────────────
    def _retrieve(self, plan: dict) -> dict:
        """Esegue il plan sulla KB. Ritorna dict con nodi trovati + relazioni."""
        found_ids: set[str] = set()

        # (a) fulltext search
        search_hits = []
        for q in plan.get("search_queries", [])[:5]:
            if not q.strip():
                continue
            try:
                hits = self.kb.find_by_text(q, limit=5)
            except Exception:
                hits = []
            for h in hits:
                if h["id"] not in found_ids:
                    found_ids.add(h["id"])
                    search_hits.append({"query": q, **h})

        # (b) expand vicini dei nodi indicati
        for node_id in plan.get("expand_nodes", [])[:5]:
            try:
                neigh = self.kb.neighbors(node_id, depth=1)
            except Exception:
                neigh = []
            for n in neigh:
                found_ids.add(n["id"])
            # includi anche il nodo stesso
            found_ids.add(node_id)

        # carica dati completi dei nodi trovati + le relazioni tra di essi
        nodes: list[dict] = []
        for nid in found_ids:
            n = self.kb.get_node(nid)
            if n:
                nodes.append(n)

        # relazioni interne al sottografo trovato
        rels: list[dict] = []
        if found_ids:
            with self.kb.driver.session() as sess:
                r = sess.run(
                    """
                    MATCH (a:Node)-[r]->(b:Node)
                    WHERE a.id IN $ids AND b.id IN $ids
                    RETURN a.id AS src, type(r) AS rel, b.id AS dst
                    """,
                    ids=list(found_ids),
                )
                rels = [dict(rec) for rec in r]

        return {
            "search_hits": search_hits,
            "nodes": nodes,
            "relations": rels,
        }

    # ── STEP 3: answerer ───────────────────────────────────────────────
    def _answer(self, user_request: str, retrieved: dict) -> str:
        # formatta il sottografo per il modello: nodi con content, poi relazioni
        nodes = retrieved["nodes"]
        rels = retrieved["relations"]

        if not nodes:
            context_block = "(the knowledge graph returned no matching nodes)"
        else:
            lines = []
            for n in nodes:
                lines.append(f"── {n['id']} [{n['type']}] — {n.get('title','')}")
                if n.get("summary"):
                    lines.append(f"   summary: {n['summary']}")
                if n.get("content"):
                    content = n["content"][:600]
                    lines.append(f"   content: {content}")
                if n.get("code_snippet"):
                    snippet = n["code_snippet"][:400]
                    lines.append(f"   code:\n{snippet}")
                if n.get("source"):
                    lines.append(f"   source: {n['source']}")
                lines.append("")
            if rels:
                lines.append("── relations in subgraph ──")
                for r in rels:
                    lines.append(f"   {r['src']}  --[{r['rel']}]-->  {r['dst']}")
            context_block = "\n".join(lines)

        system = (
            "You are a web UI expert. Answer the user's question using ONLY the knowledge graph "
            "excerpt provided below. If the excerpt is insufficient, say so honestly — do NOT "
            "invent APIs, properties, or examples.\n\n"
            "When you use a fact from the graph, cite the node id in brackets like [pattern-hamburger-menu].\n"
            "Be concrete and code-oriented. Prefer showing the pattern with a minimal working snippet "
            "over abstract prose.\n"
            "Respond in the same language as the user's question."
        )
        user = f"USER QUESTION:\n{user_request}\n\nKNOWLEDGE GRAPH EXCERPT:\n{context_block}"

        return _ollama_chat(
            self.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=600,
            timeout=600,
        )

    # ── entry point ────────────────────────────────────────────────────
    def answer(self, user_request: str) -> dict:
        trace = {"phases": []}
        t0 = time.time()

        t = time.time()
        plan = self._plan(user_request)
        trace["phases"].append({"phase": "plan", "seconds": round(time.time() - t, 1), "plan": plan})

        t = time.time()
        retrieved = self._retrieve(plan)
        trace["phases"].append({
            "phase": "retrieve",
            "seconds": round(time.time() - t, 1),
            "n_nodes": len(retrieved["nodes"]),
            "n_relations": len(retrieved["relations"]),
            "node_ids": [n["id"] for n in retrieved["nodes"]],
        })

        t = time.time()
        response = self._answer(user_request, retrieved)
        trace["phases"].append({"phase": "answer", "seconds": round(time.time() - t, 1)})
        trace["total_seconds"] = round(time.time() - t0, 1)

        return {"response": response, "trace": trace, "retrieved": retrieved, "plan": plan}


# ── CLI di test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "come faccio un hamburger menu accessibile con keyboard?"

    r = KBReasoner()
    try:
        out = r.answer(question)
    finally:
        r.close()

    print("=" * 72)
    print("QUESTION:", question)
    print("=" * 72)
    print()
    print("── PLAN ─────────────────────────────────────────────")
    print(json.dumps(out["plan"], indent=2, ensure_ascii=False))
    print()
    print("── RETRIEVED ────────────────────────────────────────")
    print(f"nodes ({len(out['retrieved']['nodes'])}):", [n["id"] for n in out["retrieved"]["nodes"]])
    print(f"relations: {len(out['retrieved']['relations'])}")
    print()
    print("── ANSWER ───────────────────────────────────────────")
    print(out["response"])
    print()
    print("── TIMING ───────────────────────────────────────────")
    for p in out["trace"]["phases"]:
        print(f"  {p['phase']:10s} {p['seconds']:>5.1f}s")
    print(f"  {'TOTAL':10s} {out['trace']['total_seconds']:>5.1f}s")
