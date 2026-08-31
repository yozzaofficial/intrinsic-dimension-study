"""
Schema del knowledge graph — definisce tipi di nodo, relazioni,
proprietà attese. Punto di riferimento unico per tutto il resto.
"""

# Tipi di nodo — categoria semantica dell'unità di conoscenza.
NODE_TYPES = {
    # Concetti/proprietà fondamentali (spesso 1:1 con pagine MDN)
    "CSSProperty",    # es. position, transform, z-index — atomici, riusabili
    "HTMLElement",    # es. nav, header, dialog, button — elementi semantici
    "AriaAttribute",  # es. aria-expanded, aria-haspopup — accessibility
    "AriaRole",       # es. navigation, menubar, menu — semantic roles

    # Costrutti compositi
    "Pattern",        # es. hamburger-menu, sticky-header — pattern UI ricorrenti
    "Technique",      # es. focus-trap, keyboard-navigation — tecniche specifiche
    "Problem",        # es. "menu inaccessibile da tastiera" — problema descritto
    "Solution",       # es. "gestisci onKeyDown Escape" — soluzione a un problema

    # Esempi concreti
    "Example",        # snippet di codice reale, completo, funzionante
    "AntiPattern",    # es. "dropdown senza aria-expanded" — cosa NON fare
}

# Tipi di relazione tra nodi — direzionale, tipizzata
RELATION_TYPES = {
    "USES",           # A usa B come dipendenza/primitivo
    "IMPLEMENTS",     # A è un'implementazione concreta del pattern/concetto B
    "COMPOSES",       # A è composto da B come parte strutturale
    "REQUIRES",       # A ha bisogno di B per funzionare
    "SOLVES",         # A risolve il problema B
    "ALTERNATIVE_TO", # A è un modo alternativo di risolvere lo stesso problema di B
    "AVOIDS",         # A evita l'anti-pattern B
    "SEE_ALSO",       # generico "vedi anche" (fallback quando non c'è relazione precisa)
    "DEFINED_IN",     # A è definito nella specifica/documento B (per tracciabilità)
}

# Proprietà attese per ogni tipo di nodo (schema informale, non enforced).
# Ogni nodo ha almeno: id, type, title, summary, content, source.
NODE_PROPERTIES = {
    "required": ["id", "type", "title", "summary", "content"],
    "optional": [
        "source",           # URL della fonte MDN/spec (per tracciabilità)
        "code_snippet",     # snippet di codice tipico (per Pattern/Technique/Example)
        "concepts_used",    # lista di CSS/HTML/API principali usati
        "tags",             # keyword per matching semantico
        "browser_support",  # dove rilevante (bleeding-edge features)
    ],
}


def validate_node(node: dict) -> list[str]:
    """Ritorna lista di errori (vuota se ok)."""
    errors = []
    if node.get("type") not in NODE_TYPES:
        errors.append(f"unknown type: {node.get('type')} (valid: {NODE_TYPES})")
    for f in NODE_PROPERTIES["required"]:
        if not node.get(f):
            errors.append(f"missing required field: {f}")
    if node.get("id") and not node["id"].replace("-", "").replace("_", "").isalnum():
        errors.append(f"id must be alphanumeric with -/_ only: {node['id']}")
    return errors


def validate_relation(source_id: str, rel_type: str, target_id: str) -> list[str]:
    errors = []
    if rel_type not in RELATION_TYPES:
        errors.append(f"unknown relation: {rel_type} (valid: {RELATION_TYPES})")
    if source_id == target_id:
        errors.append(f"self-reference not allowed: {source_id}")
    return errors
