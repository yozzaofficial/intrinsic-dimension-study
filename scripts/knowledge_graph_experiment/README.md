# Knowledge Graph Experiment

Terzo esperimento della serie, pivot dopo i risultati misti della distillazione 14B→250M.

## Ipotesi

Se il problema del "modello piccolo" è **la conoscenza compressa nei pesi**, allora un modello piccolo + **grafo curato esterno** potrebbe rispondere meglio di un modello grande senza contesto.

Test empirico: dominio ristretto (React navigation UI), knowledge graph su Neo4j popolato da MDN, ragionatore che deve **usare il grafo per rispondere**.

## Componenti

| File | Ruolo |
|---|---|
| `schema.py` | Tipi di nodo (CSSProperty, HTMLElement, AriaRole, Pattern, Technique, Problem, Solution, ...) + relazioni tipizzate (USES, REQUIRES, SOLVES, ...) |
| `kb.py` | Wrapper Neo4j: `add_node`, `find_by_text` (fulltext), `neighbors` (traversal typed), `path` (shortestPath) |
| `seed_navigation.py` | Popola KB con 34 nodi + 44 relazioni curati dal dominio "navigation UI" |
| `kb_reasoner.py` | Loop **plan → retrieve → answer**: LLM pianifica, sistema esegue query, LLM risponde con contesto |
| `kb_navigator.py` | Loop **iterativo** SEARCH/VISIT/FOLLOW/ANSWER: LLM decide ogni passo, grafo come substrato del ragionamento |

## Come rifarlo

```bash
# 1. Neo4j attivo
brew services start neo4j
# password iniziale: alter a 'revolutionkb' (cfr kb.py)

# 2. Ollama con almeno un modello caricato
ollama pull qwen2.5:3b-instruct-q4_K_M
# opzionale reasoning:
ollama pull deepseek-r1:7b

# 3. Seed graph
python3 seed_navigation.py --clear

# 4. Test retrieval semplice
python3 -c "from kb import KnowledgeBase; kb=KnowledgeBase(); print(kb.find_by_text('hamburger'))"

# 5. Test reasoner (una-query)
python3 kb_reasoner.py

# 6. Test navigator (multi-turn) — richiede num_predict alto
python3 kb_navigator.py 'come faccio un hamburger menu accessibile?'
```

## Risultati chiave

Vedi `../../data/knowledge_graph_experiment/kb_experiment_results.json` per dettaglio.

**Baseline (7B senza grafo):** allucinazioni gravi ("freccia del pavimento" per ArrowUp), codice non-React, testo cinese misto.

**Con grafo (3B + kb_reasoner):** codice React accessibile corretto (aria-expanded, aria-controls, useState, useRef), ma il modello non cita gli id dei nodi e ignora tecniche del contesto (keyboard handling reale).

**Con grafo iterativo (7B + kb_navigator):** protocollo agentico multi-turn non seguito dal modello — emette più azioni per turno, inventa relation types. Il reasoning tuning di R1 lavora contro il protocollo JSON strutturato.

## Conclusione onesta

L'infrastruttura del grafo funziona (query, traversal, fulltext). Il collo di bottiglia è:

- **modelli reasoning-tuned esistenti** (DeepSeek-R1-7B) non sono function-calling-tuned
- **modelli small function-calling-tuned** (Qwen 3B) non usano il grafo come autorità
- **modelli grandi** (14B) sono troppo lenti su M2 16GB per essere pratici in loop multi-turno

Il prossimo passo naturale non è un modello più grande, è un **router 30-70M encoder-only** (MiniLM/DistilBERT) fine-tuned sul dominio: capisce la richiesta, seleziona la sotto-query del grafo, restituisce il sottografo, e solo alla fine passa a un LLM generico per la formulazione della risposta. Il ragionamento resta guidato dal codice, non dall'LLM.

## Files pesanti esclusi (`.gitignore`)

- `__pycache__/`
- Cache Neo4j (fuori repo: `/opt/homebrew/var/neo4j/`)
- Cache HuggingFace (fuori repo: `~/.cache/huggingface/`)
- Ollama models (fuori repo: `~/.ollama/models/`)
- Test outputs verbose (`.log`, `.txt` in `outputs/`)
