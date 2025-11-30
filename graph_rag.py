# graph_rag_extended.py

from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import re
import os
from datasets import load_dataset
from embeddings import LocalEmbeddingManager

try:
    from langchain.schema import Document
except Exception:
    Document = None


class GraphRag:
    def __init__(self):    
        load_dotenv()
        self.NEO4J_URI = os.environ.get("NEO4J_URI")
        self.NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
        self.NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
        # optional db name
        self.NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", None)
        self.AUTH = (self.NEO4J_USERNAME, self.NEO4J_PASSWORD)

        # Initiate graph connection object (langchain_neo4j wrapper)
        self.graph = None
        self.temp_ds = None

        # Local embedder
        self.embedder = LocalEmbeddingManager("BAAI/bge-m3")

    def initiate_graph(self):
        """Instantiate the Neo4jGraph wrapper (langchain_neo4j)."""
        self.graph = Neo4jGraph(
            url=self.NEO4J_URI,
            username=self.NEO4J_USERNAME,
            password=self.NEO4J_PASSWORD
        )

    def load_temp_dataset(self, name="naver-clova-ix/cord-v2"):
        """Load the dataset you asked to use for testing."""
        self.temp_ds = load_dataset(name)

    def get_sample_images(self, num=10):
        """Return first 10 images (or [] if dataset not loaded)."""
        if self.temp_ds is not None:
            return self.temp_ds['train'][:num]['image']
        return []

    def build_embedding_text(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """Deterministic text builder for an entity's embedding."""
        # Sorted keys for determinism
        parts = [f"{k}: {properties[k]}" for k in sorted(properties.keys())]
        props_text = " ; ".join(parts)
        return f"{entity_type} | {props_text}"

    def create_entity_node(self, entity_id, entity_type, properties):
        """
        Creates an entity node with both the specific type label and
        a universal :Entity label so we can vector-index everything globally.
        """

        text = " ".join([str(v) for v in properties.values() if v is not None])
        embedding = self.embedder.get_embedding(text)

        query = f"""
        MERGE (e:Entity:{entity_type} {{id: $id}})
        SET e += $props,
            e.embedding = $embedding,
            e.embedding_text = $embedding_text
        RETURN e
        """

        return self.graph.query(
            query,
            {
                "id": entity_id,
                "props": properties,
                "embedding": embedding,
                "embedding_text": text,
            }
        )


    def create_relationship(self, relationship: dict):
        """Create relationship and store an embedding for the relationship's text."""
        from_id = relationship.get("from")
        to_id = relationship.get("to")
        rel_type = relationship.get("type")
        rel_properties = relationship.get("properties", {})

        if not from_id or not to_id or not rel_type:
            print(f"Skipping relationship with missing data: {relationship}")
            return

        # Build relationship text for embedding: include type and key props
        rel_text_parts = [f"type: {rel_type}"]
        for k, v in sorted(rel_properties.items()):
            rel_text_parts.append(f"{k}: {v}")
        rel_text = " | ".join(rel_text_parts)

        # compute embedding
        rel_embedding = self.embedder.get_embedding(rel_text)

        # attach to rel properties
        rel_properties["embedding_text"] = rel_text
        rel_properties["embedding"] = rel_embedding

        if rel_properties:
            properties_str = " { " + ", ".join([f"{key}: ${key}" for key in rel_properties.keys()]) + " }"
        else:
            properties_str = ""

        query = f"""
        MATCH (from {{id: $from_id}}), (to {{id: $to_id}})
        MERGE (from)-[r:{rel_type}{properties_str}]->(to)
        RETURN r
        """

        params = {"from_id": from_id, "to_id": to_id, **rel_properties}
        try:
            self.graph.query(query, params)
            print(f"Created relationship: {from_id} -[{rel_type}]-> {to_id}")
        except Exception as e:
            print(f"Error creating relationship {from_id} -> {to_id}: {e}")

    def cleanup_empty_embeddings(self):
        """Remove empty embeddings so we can start fresh"""
        print("Cleaning up empty embeddings...")

        cleanup_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL AND size(n.embedding) = 0
        REMOVE n.embedding, n.embedding_text
        RETURN count(n) as cleaned_count
        """
        result = self.graph.query(cleanup_query)
        if result:
            print(f"Removed {result[0].get('cleaned_count', 0)} empty embeddings")
        else:
            print("No cleanup results returned.")

        cleanup_text_query = """
        MATCH (n)
        WHERE n.embedding_text IS NOT NULL AND n.embedding_text IN ['. .', ' . ', '.  .', '']
        REMOVE n.embedding_text
        RETURN count(n) as cleaned_text_count
        """
        result_text = self.graph.query(cleanup_text_query)
        if result_text:
            print(f"Removed {result_text[0].get('cleaned_text_count', 0)} meaningless embedding texts")
        else:
            print("No cleanup text results returned.")

    # Index creation helper (run once manually)
    def neo4j_create_vector_index_cypher(self):
        dim = self.embedder.dimensions
        return f"""
        CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
        FOR (n:Entity)
        ON (n.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}
        }};
        """


    # Retrieval functions
    def search_similar(self, query_text: str, top_k: int = 5, index_name: str = "entity_embedding_index"):
        """Vector search using Neo4j vector index; returns list of (node_props, score)."""
        query_vec = self.embedder.get_embedding(query_text)

        cypher = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $top_k,
            $query_vec
        ) YIELD node, score
        RETURN node, score
        """

        results = self.graph.query(cypher, {"top_k": top_k, "query_vec": query_vec})
        # Normalize results into a friendly list
        out = []
        for r in results:
            node = r["node"]
            score = r["score"]
            # node could be a dict-like record; convert metadata if needed
            out.append({"node": node, "score": score})
        return out

    def get_neighbors(self, entity_id: str, depth: int = 1, limit: int = 25):
        """Return neighborhood subgraph around an entity (basic cypher)."""
        cypher = f"""
        MATCH (n {{id: $entity_id}})-[*1..{depth}]-(m)
        RETURN DISTINCT m LIMIT $limit
        """
        results = self.graph.query(cypher, {"entity_id": entity_id, "limit": limit})
        # results are list of records with 'm' keys in typical wrapper; normalize:
        nodes = []
        for r in results:
            # try both 'm' and 'node' keys depending on driver
            if 'm' in r:
                nodes.append(r['m'])
            elif 'node' in r:
                nodes.append(r['node'])
            else:
                # try the first value
                nodes.append(list(r.values())[0])
        return nodes

    # Combined Graph + Embedding Retrieval (Hybrid)
    def hybrid_retrieve(self, query: str, top_k_vector: int = 5, expand_neighbors: int = 3) -> List[Dict[str, Any]]:
        """
        1) Vector search -> get top_k_vector nodes
        2) For each returned node, optionally expand via graph neighbors (hybrid)
        3) Return merged list of documents (node metadata + score + neighbor context)
        """
        vector_hits = self.search_similar(query, top_k=top_k_vector)

        merged_results = []
        seen_ids = set()

        for hit in vector_hits:
            node = hit["node"]
            score = hit["score"]
            # try to find id prop
            node_id = node.get("id") if isinstance(node, dict) else None

            # Add base node result
            merged_results.append({"type": "node", "node": node, "score": score})
            if node_id:
                seen_ids.add(node_id)

            # Expand by neighbors if requested
            if node_id and expand_neighbors > 0:
                neighbors = self.get_neighbors(node_id, depth=1, limit=expand_neighbors)
                for n in neighbors:
                    n_id = n.get("id") if isinstance(n, dict) else None
                    if n_id and n_id in seen_ids:
                        continue
                    merged_results.append({"type": "neighbor", "node": n, "score": None})
                    if n_id:
                        seen_ids.add(n_id)

        return merged_results

    # Query router (heuristic + DB check)
    def route_query(self, query: str, vector_k: int = 5, neighbor_expand: int = 2):
        """
        Heuristic router:
        - If the query contains explicit entity ids or matches a node name in DB: use graph neighborhood
        - Else use vector retrieval (hybrid)
        """
        # heuristic 1: quoted token -> treat as entity name
        quoted = re.findall(r'"([^"]+)"', query)
        search_tokens = quoted if quoted else query.split()

        # Quick DB check: see if any token matches a node property 'name' exactly (case-insensitive)
        for token in search_tokens:
            token_norm = token.strip().lower()
            if len(token_norm) < 2:
                continue
            # Query nodes by name or id (case-insensitive)
            cypher = """
            MATCH (n)
            WHERE toLower(n.name) = $token OR toLower(n.id) = $token
            RETURN n LIMIT 1
            """
            try:
                res = self.graph.query(cypher, {"token": token_norm})
            except Exception:
                res = []

            if res:
                # found an entity -> route to graph neighborhood
                node_record = res[0].get("n") if isinstance(res[0], dict) and "n" in res[0] else res[0]
                node_id = node_record.get("id") if isinstance(node_record, dict) else None
                if node_id:
                    neighbors = self.get_neighbors(node_id, depth=1, limit=neighbor_expand)
                    return {"mode": "graph", "root_node": node_record, "neighbors": neighbors}

        # else fall back to hybrid vector+graph retrieval
        hybrid = self.hybrid_retrieve(query, top_k_vector=vector_k, expand_neighbors=neighbor_expand)
        return {"mode": "hybrid", "results": hybrid}

    # Pipeline to embed relationships at scale
    def embed_existing_relationships(self, batch_size: int = 200):
        """
        Walk all relationships and add embedding (if missing). Processes in batches.
        NOTE: This method uses simple pagination via cypher SKIP/LIMIT. For large DBs consider
        using streaming or server-side procedures.
        """
        offset = 0
        while True:
            cypher = """
            MATCH ()-[r]-()
            WHERE r.embedding IS NULL OR size(r.embedding) = 0
            RETURN id(r) AS rid, type(r) AS rtype, r, startNode(r).id AS from_id, endNode(r).id AS to_id
            SKIP $skip LIMIT $limit
            """
            rows = self.graph.query(cypher, {"skip": offset, "limit": batch_size})
            if not rows:
                break

            for row in rows:
                # extract relationship data
                rid = row.get("rid")
                rtype = row.get("rtype")
                r_props = row.get("r") or {}
                # Build a textual representation for the relationship
                rel_text = f"{rtype} | " + " ; ".join([f"{k}: {v}" for k, v in sorted((r_props or {}).items())])
                emb = self.embedder.get_embedding(rel_text)

                # Update relationship by internal id (most Neo4j drivers allow using 'id(r)')
                update_cypher = """
                MATCH ()-[r]->()
                WHERE id(r) = $rid
                SET r.embedding = $emb, r.embedding_text = $text
                RETURN count(r) AS updated
                """
                try:
                    self.graph.query(update_cypher, {"rid": rid, "emb": emb, "text": rel_text})
                except Exception as e:
                    print(f"Failed to update relationship id={rid}: {e}")

            offset += batch_size
            # If fewer than batch_size rows returned, we're done
            if len(rows) < batch_size:
                break

        print("Finished embedding existing relationships.")

