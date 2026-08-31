"""
kb_navigator.py — ragionatore iterativo che NAVIGA il grafo.

Differenza fondamentale da kb_reasoner.py:
- kb_reasoner: pianifica UNA volta, retrieve tutto, risponde (contesto passivo)
- kb_navigator: LOOP di navigazione — il modello decide a ogni step quale nodo
  visitare, quale relazione seguire, quando fermarsi. Il grafo è il substrato
  del ragionamento, non un blob di contesto.

Ogni step:
    1. Modello vede: domanda + stato attuale (nodi visitati, ultimo nodo)
    2. Modello sceglie UN'AZIONE tra:
         - SEARCH(query)   : cerca fulltext nella KB
         - VISIT(node_id)  : leggi contenuto completo di un nodo
         - FOLLOW(node_id, rel_type) : segui una relazione da un nodo
         - ANSWER          : ho abbastanza per rispondere, elabora la risposta
    3. Sistema esegue, feed-back al modello
    4. Loop finché ANSWER o max_steps

Output finale: risposta + catena di nodi visitati (tracciabile).

Include modalità "no_graph" per test comparativo: stessa domanda, stesso modello,
zero contesto — così vediamo se il grafo aggiunge valore.
"""
import json
import re
import time
import urllib.request
from typing import Optional

from kb import KnowledgeBase


OLLAMA_URL = "http://localhost:11434/api/chat"
REASONING_MODEL = "deepseek-r1:7b"


def _ollama_chat(model: str, messages: list, temperature: float = 0.3,
                  max_tokens: int = 800, timeout: int = 300) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["message"]["content"]


def _strip_thinking(text: str) -> tuple[str, str]:
    """Separa <think>...</think> dal resto. Ritorna (thought, response)."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        thought = m.group(1).strip()
        response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thought, response
    return "", text.strip()


def _parse_action(text: str) -> Optional[dict]:
    """Estrae la PRIMA azione valida dal testo del modello.

    Tollera: modelli che emettono più azioni, azioni miste a prosa,
    fence markdown, campi extra.
    """
    valid_actions = {"SEARCH", "VISIT", "FOLLOW", "ANSWER"}

    # tutti i {..} bilanciati nel testo
    candidates = []
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
                candidates.append(text[start : i + 1])
                start = None

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        act = str(obj.get("action", "")).upper()
        if act in valid_actions:
            obj["action"] = act
            return obj
    return None


class KBNavigator:
    def __init__(self, model: str = REASONING_MODEL, max_steps: int = 8):
        self.kb = KnowledgeBase()
        self.model = model
        self.max_steps = max_steps

    def close(self):
        self.kb.close()

    # ── introspezione minima per l'inventario ──────────────────────────
    def _inventory(self) -> str:
        """Riassunto compatto di cosa contiene la KB (tipi + conteggi)."""
        s = self.kb.stats()
        lines = ["Available node types in the graph:"]
        for t, c in s["nodes_by_type"].items():
            lines.append(f"  - {t}: {c} nodes")
        lines.append("Available relation types:")
        for r in s["relations_by_type"]:
            lines.append(f"  - {r}")
        return "\n".join(lines)

    # ── esecuzione delle azioni ────────────────────────────────────────
    def _do_search(self, query: str) -> str:
        hits = self.kb.find_by_text(query, limit=5)
        if not hits:
            return f"SEARCH('{query}') → no results"
        lines = [f"SEARCH('{query}') → {len(hits)} results:"]
        for h in hits:
            lines.append(f"  [{h['id']}] ({h['labels'][-1] if h['labels'] else '?'}) {h.get('title','')}")
            if h.get("summary"):
                lines.append(f"      {h['summary']}")
        return "\n".join(lines)

    def _do_visit(self, node_id: str) -> str:
        n = self.kb.get_node(node_id)
        if not n:
            return f"VISIT({node_id}) → node not found"
        lines = [f"VISIT({node_id}):", f"  type: {n.get('type')}", f"  title: {n.get('title')}"]
        if n.get("summary"):
            lines.append(f"  summary: {n['summary']}")
        if n.get("content"):
            lines.append(f"  content: {n['content'][:800]}")
        if n.get("code_snippet"):
            lines.append(f"  code_snippet:\n{n['code_snippet'][:500]}")
        # relazioni uscenti
        with self.kb.driver.session() as sess:
            r = sess.run(
                "MATCH (a:Node {id: $id})-[r]->(b:Node) RETURN type(r) AS rel, b.id AS dst, b.title AS title",
                id=node_id,
            )
            rels = [dict(rec) for rec in r]
        if rels:
            lines.append(f"  outgoing relations ({len(rels)}):")
            for rel in rels:
                lines.append(f"    --[{rel['rel']}]--> [{rel['dst']}] {rel['title']}")
        return "\n".join(lines)

    def _do_follow(self, node_id: str, rel_type: str) -> str:
        try:
            neigh = self.kb.neighbors(node_id, rel_type=rel_type, depth=1)
        except Exception as e:
            return f"FOLLOW({node_id}, {rel_type}) → error: {e}"
        if not neigh:
            return f"FOLLOW({node_id}, {rel_type}) → no neighbors of that type"
        lines = [f"FOLLOW({node_id}, {rel_type}) → {len(neigh)} neighbors:"]
        for n in neigh:
            lines.append(f"  [{n['id']}] ({n['labels'][-1] if n['labels'] else '?'}) {n.get('title','')}")
            if n.get("summary"):
                lines.append(f"      {n['summary']}")
        return "\n".join(lines)

    # ── loop principale ────────────────────────────────────────────────
    def navigate(self, question: str, verbose: bool = True) -> dict:
        system = (
            "You are an agent that navigates a knowledge graph to gather evidence, ONE STEP at a time.\n\n"
            "At EACH turn you output EXACTLY ONE JSON action. Never output multiple actions.\n"
            "Never output code, prose, or explanations after the JSON. The system executes your\n"
            "action and returns the result — THEN you get another turn to decide the next action.\n\n"
            "Available actions (pick one per turn):\n"
            '  {"action": "SEARCH", "query": "<text>"}\n'
            '  {"action": "VISIT", "node_id": "<id from a previous result>"}\n'
            '  {"action": "FOLLOW", "node_id": "<id>", "rel_type": "<REL from list>"}\n'
            '  {"action": "ANSWER"}\n\n'
            "Rules:\n"
            "  - Never invent node IDs. Only use IDs that appeared in previous results.\n"
            "  - Never invent relation types. Only use those in the inventory.\n"
            "  - Typical flow: SEARCH → VISIT the best hit → FOLLOW a useful relation → ANSWER.\n"
            "  - 2 to 5 steps are usually enough. Then output ANSWER.\n\n"
            "You may think inside <think>...</think> tags. After </think>, output ONLY the JSON.\n"
            "STOP as soon as the JSON object is complete. Do not write anything after the closing }."
        )

        history: list[dict] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n\n"
                    f"{self._inventory()}\n\n"
                    "What is your first action? Reply with ONE JSON action."
                ),
            },
        ]

        trace: list[dict] = []
        visited_nodes: list[str] = []
        actions_taken: list[dict] = []
        t_start = time.time()

        for step in range(self.max_steps):
            t = time.time()
            raw = _ollama_chat(self.model, history, temperature=0.2, max_tokens=2500)
            thought, response = _strip_thinking(raw)
            action = _parse_action(response)
            step_time = round(time.time() - t, 1)

            step_record = {
                "step": step + 1,
                "seconds": step_time,
                "thought": thought[:400] if thought else "",
                "raw_response": response[:400],
                "action": action,
            }

            if verbose:
                print(f"\n── STEP {step+1} ({step_time}s) ──")
                if thought:
                    print(f"THINK: {thought[:300]}...")
                print(f"ACTION: {action}")

            if action is None:
                # modello ha sbagliato format — forziamo ANSWER
                step_record["error"] = "no valid action parsed"
                trace.append(step_record)
                # forza risposta finale con ciò che abbiamo
                actions_taken.append({"action": "ANSWER_FORCED"})
                break

            actions_taken.append(action)
            trace.append(step_record)
            act = action.get("action", "").upper()

            if act == "ANSWER":
                break

            # esegui azione
            if act == "SEARCH":
                result = self._do_search(action.get("query", ""))
            elif act == "VISIT":
                nid = action.get("node_id", "")
                if nid and nid not in visited_nodes:
                    visited_nodes.append(nid)
                result = self._do_visit(nid)
            elif act == "FOLLOW":
                result = self._do_follow(action.get("node_id", ""), action.get("rel_type", ""))
            else:
                result = f"unknown action: {act}"

            if verbose:
                print(f"RESULT: {result[:300]}...")

            # feedback al modello
            history.append({"role": "assistant", "content": response})
            history.append({
                "role": "user",
                "content": f"{result}\n\nNext action?"
            })

        # ── STEP FINALE: elabora la risposta con la storia completa ─────
        history.append({
            "role": "user",
            "content": (
                f"Now write the final answer to the original question:\n\n"
                f"QUESTION: {question}\n\n"
                "Use ONLY the information you gathered from the graph in the previous steps. "
                "For every claim, cite the node id in brackets like [pattern-hamburger-menu]. "
                "If information is insufficient, say so explicitly.\n"
                "Respond in the same language as the question."
            ),
        })
        t = time.time()
        final_raw = _ollama_chat(self.model, history, temperature=0.3, max_tokens=1000, timeout=600)
        final_thought, final_response = _strip_thinking(final_raw)

        return {
            "question": question,
            "answer": final_response,
            "final_thought": final_thought,
            "visited_nodes": visited_nodes,
            "n_steps": len(trace),
            "trace": trace,
            "total_seconds": round(time.time() - t_start, 1),
            "answer_seconds": round(time.time() - t, 1),
        }

    # ── modalità no-graph per confronto ────────────────────────────────
    def answer_no_graph(self, question: str) -> dict:
        """Stesso modello, zero contesto. Per confronto."""
        t = time.time()
        raw = _ollama_chat(
            self.model,
            [
                {"role": "system", "content": "You are a web UI expert. Answer concretely with code when relevant. Respond in the same language as the question."},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=600,
        )
        thought, response = _strip_thinking(raw)
        return {
            "question": question,
            "answer": response,
            "final_thought": thought,
            "seconds": round(time.time() - t, 1),
        }


# ── CLI comparativa ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "come faccio un hamburger menu accessibile con la tastiera in React?"
    mode = "compare"  # 'graph' | 'no_graph' | 'compare'
    if question.startswith("--"):
        parts = question.split(None, 1)
        mode = parts[0].lstrip("-")
        question = parts[1] if len(parts) > 1 else question

    nav = KBNavigator()
    try:
        if mode in ("compare", "no_graph"):
            print("=" * 72)
            print("MODE: NO GRAPH (baseline — modello senza aiuti)")
            print("=" * 72)
            r0 = nav.answer_no_graph(question)
            print(f"\n[time: {r0['seconds']}s]")
            print(r0["answer"])
            print()

        if mode in ("compare", "graph"):
            print("=" * 72)
            print("MODE: WITH GRAPH (navigazione iterativa)")
            print("=" * 72)
            r1 = nav.navigate(question, verbose=True)
            print()
            print("=" * 72)
            print("── FINAL ANSWER ──")
            print("=" * 72)
            print(r1["answer"])
            print()
            print(f"visited nodes: {r1['visited_nodes']}")
            print(f"n steps: {r1['n_steps']}, total: {r1['total_seconds']}s (answer {r1['answer_seconds']}s)")
    finally:
        nav.close()
