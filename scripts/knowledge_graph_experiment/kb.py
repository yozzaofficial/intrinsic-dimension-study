"""
Knowledge base API su Neo4j.

Un'unica classe KnowledgeBase espone:
  - add_node(dict)                         # inserisce/aggiorna un nodo
  - add_relation(src, rel_type, dst, note?)# collega due nodi con relazione tipizzata
  - get_node(id) -> dict
  - find_by_text(query, limit=10)           # ricerca fulltext
  - neighbors(id, rel_type?, depth=1)       # nodi collegati (traversal)
  - path(from_id, to_id, max_hops=3)        # trova un cammino tra due nodi
  - stats()                                 # conteggi per tipo/relazione
  - clear()                                 # svuota tutto (per test)
  - init_schema()                           # crea vincoli univoci + fulltext index

Uso base:
    from kb import KnowledgeBase
    kb = KnowledgeBase()
    kb.init_schema()
    kb.add_node({"id": "css-position", "type": "CSSProperty", ...})
    kb.close()
"""
import json
from typing import Optional
from neo4j import GraphDatabase

from schema import validate_node, validate_relation, NODE_TYPES, RELATION_TYPES


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_AUTH = ("neo4j", "revolutionkb")


class KnowledgeBase:
    def __init__(self, uri: str = DEFAULT_URI, auth: tuple = DEFAULT_AUTH):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    # ── schema/init ─────────────────────────────────────────────────────────
    def init_schema(self):
        """Crea vincoli e indici. Idempotente."""
        with self.driver.session() as s:
            # ogni nodo ha id univoco (a prescindere dal tipo, per semplicità)
            s.run("CREATE CONSTRAINT node_id_unique IF NOT EXISTS "
                  "FOR (n:Node) REQUIRE n.id IS UNIQUE")
            # index fulltext su title + summary + content per ricerca testuale
            s.run("""
                CREATE FULLTEXT INDEX node_fulltext IF NOT EXISTS
                FOR (n:Node) ON EACH [n.title, n.summary, n.content, n.tags]
            """)

    def clear(self):
        """Cancella tutti i nodi/relazioni. Usare con cautela."""
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # ── nodi ───────────────────────────────────────────────────────────────
    def add_node(self, node: dict):
        errors = validate_node(node)
        if errors:
            raise ValueError(f"invalid node {node.get('id')}: {errors}")

        # normalizza campi: lista → JSON string (Neo4j supporta liste solo di tipi
        # semplici; per liste di dict le serializziamo, per liste di str vanno bene)
        props = dict(node)
        for key in ("concepts_used", "tags", "browser_support"):
            if key in props and props[key] is None:
                del props[key]

        with self.driver.session() as s:
            # usiamo MERGE per idempotenza (aggiorna se esiste, crea se no)
            # e etichette double: :Node (per unicità e fulltext) + :Type (per query per tipo)
            type_label = node["type"]
            s.run(f"""
                MERGE (n:Node {{id: $id}})
                SET n:{type_label}
                SET n += $props
            """, id=node["id"], props=props)

    def get_node(self, node_id: str) -> Optional[dict]:
        with self.driver.session() as s:
            r = s.run("MATCH (n:Node {id: $id}) RETURN properties(n) AS n", id=node_id)
            rec = r.single()
            return dict(rec["n"]) if rec else None

    # ── relazioni ─────────────────────────────────────────────────────────
    def add_relation(self, source_id: str, rel_type: str, target_id: str,
                      note: Optional[str] = None):
        errors = validate_relation(source_id, rel_type, target_id)
        if errors:
            raise ValueError(f"invalid relation: {errors}")

        with self.driver.session() as s:
            # MERGE per evitare relazioni duplicate
            props_set = "SET r.note = $note" if note else ""
            s.run(f"""
                MATCH (src:Node {{id: $src}}), (dst:Node {{id: $dst}})
                MERGE (src)-[r:{rel_type}]->(dst)
                {props_set}
            """, src=source_id, dst=target_id, note=note)

    # ── ricerca ────────────────────────────────────────────────────────────
    def find_by_text(self, query: str, limit: int = 10) -> list[dict]:
        """Ricerca fulltext tramite l'index creato in init_schema."""
        with self.driver.session() as s:
            r = s.run("""
                CALL db.index.fulltext.queryNodes('node_fulltext', $q) YIELD node, score
                RETURN node.id AS id, labels(node) AS labels,
                       node.title AS title, node.summary AS summary, score
                ORDER BY score DESC
                LIMIT $limit
            """, q=query, limit=limit)
            return [dict(rec) for rec in r]

    def neighbors(self, node_id: str, rel_type: Optional[str] = None,
                   depth: int = 1) -> list[dict]:
        """Nodi raggiungibili da node_id entro `depth` hop, opzionalmente filtrando per relazione."""
        rel_pattern = f":{rel_type}" if rel_type else ""
        with self.driver.session() as s:
            r = s.run(f"""
                MATCH (start:Node {{id: $id}})-[r{rel_pattern}*1..{depth}]->(other:Node)
                RETURN DISTINCT other.id AS id, labels(other) AS labels,
                       other.title AS title, other.summary AS summary
            """, id=node_id)
            return [dict(rec) for rec in r]

    def path(self, from_id: str, to_id: str, max_hops: int = 3) -> list[dict]:
        """Cammino più breve tra due nodi (se esiste)."""
        with self.driver.session() as s:
            r = s.run(f"""
                MATCH p = shortestPath((a:Node {{id: $from_id}})-[*1..{max_hops}]-(b:Node {{id: $to_id}}))
                RETURN [n in nodes(p) | n.id] AS nodes,
                       [rel in relationships(p) | type(rel)] AS rels
            """, from_id=from_id, to_id=to_id)
            rec = r.single()
            return [dict(rec)] if rec else []

    # ── statistiche ────────────────────────────────────────────────────────
    def stats(self) -> dict:
        with self.driver.session() as s:
            total_nodes = s.run("MATCH (n:Node) RETURN count(n) AS c").single()["c"]
            total_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            by_type = {}
            for t in NODE_TYPES:
                c = s.run(f"MATCH (n:{t}) RETURN count(n) AS c").single()["c"]
                if c > 0:
                    by_type[t] = c

            by_rel = {}
            for r in RELATION_TYPES:
                c = s.run(f"MATCH ()-[rel:{r}]->() RETURN count(rel) AS c").single()["c"]
                if c > 0:
                    by_rel[r] = c

            return {
                "total_nodes": total_nodes,
                "total_relations": total_rels,
                "nodes_by_type": by_type,
                "relations_by_type": by_rel,
            }
