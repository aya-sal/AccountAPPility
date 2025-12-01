# GraphRAG + Neo4j + Google ADK

A complete **end-to-end Retrieval-Augmented Generation (RAG) pipeline** built on **Neo4j**, **local embeddings (BGE-M3)**, **Gemini 2.5 Flash**, and **Google ADK tool-calling**.

This project turns receipt images into a structured knowledge graph and enables **hybrid retrieval** through **vector search + graph expansion**, all accessible through a clean ADK Retrieval Agent.

## Features

* **Receipt Image → Entities + Relationships** (Gemini 2.5 Flash)

* Local embeddings using `BAAI/bge-m3`

* Full **GraphRAG pipeline** on **Neo4j**

* Vector indexing & retrieval using Neo4j native vector indexes

* **Hybrid retrieval** (vector + graph neighbors)

* Google ADK `LlmAgent` with **Tool Calling**

* Image ingestion as a callable tool

* Dataset ingestion (CORD-v2)

## Installation Guide

### 1. Clone the repository

### Install Neo4j

You need **Neo4j 5.x** (Neo4j Aura or Desktop both work).

**Option A — Install Neo4j Desktop**

1. Download Neo4j Desktop from:
   <https://neo4j.com/download/>

2. Create a new project → new local DB → start it.

**Option B — Install Neo4j locally via tarball**

1. Download Neo4j Community 5.x:
   <https://neo4j.com/download-center/#community>

2. Extract and run:
```bash
bin/neo4j start
```

### 🔌 Enable APOC in Neo4j

This project requires **APOC** (standard procedures).

**1. Download APOC matching your Neo4j version**

* Visit APOC releases:
  <https://github.com/neo4j/apoc/releases>

* Download the latest `apoc-x.y.z-core.jar` for your Neo4j version.

**2. Move APOC JAR into plugins directory**

* For Neo4j Desktop:
```bash
~/.config/Neo4j Desktop/Application/relate-data/dbmss/<db-id>/plugins
```

* For Server installation:
```
neo4j/plugins/
```

**3. Enable APOC in `neo4j.conf`**

Add these lines:
```
dbms.security.procedures.unrestricted=apoc.* 
dbms.security.procedures.allowlist=apoc.*
```

> **Note:** If using Neo4j Desktop: Go to **Settings** → add the above lines → **Restart DB**.

### Python Dependencies

1. Create a virtual environment:
```
python3 -m venv venv source venv/bin/activate
```

2. Install required packages:
```
pip install -r requirements.txt
```

### Environment Variables

Create a file named `.env`:
```bash
GOOGLE_API_KEY=your-key 
NEO4J_URI=bolt://localhost:7687 
NEO4J_USERNAME=neo4j 
NEO4J_PASSWORD=yourpassword 
NEO4J_DATABASE=neo4j
```

## Project Structure

```bash
/project 
│ 
├── data_input_agent.py 
├── graph_rag.py 
├── embeddings.py 
├── full_run.ipynb 
├── README.md 
└── .env
```

## Core Modules

Below is a description of the core Python modules included in the system.

### `embeddings.py`

Implements:

* `LocalEmbeddingManager` (BGE-M3)

**Features:**

* Loads model `BAAI/bge-m3`

* CPU/GPU auto detection

* Returns normalized embeddings (**1024 dimensions**)

* Used for all **Neo4j vector embeddings**

### `graph_rag.py`

Implements `GraphRag`, which includes:

* **Graph Setup**

  * Neo4j initialization

  * Dataset loading (CORD-v2)

* **Entity & Relationship Insertion**

  * `create_entity_node()`

  * `create_relationship()`

* **Vector Index Definition**

  * `neo4j_create_vector_index_cypher()`

* **Retrieval Methods**

  * `search_similar()` → vector search

  * `get_neighbors()` → graph expansion

  * `hybrid_retrieve()` → vector + graph hybrid

  * `route_query()` → heuristic router

* Relationship embedding pipeline

  * Handles embeddings, graph writes, retrieval logic.

## Ingesting Dataset Samples

The notebook defines:
```python
async def ingest_dataset_samples(gr, num_samples=10)
```

This function:

1. Loads images from CORD-v2

2. Sends each image to **Gemini 2.5 Flash**

3. Extracts entities + relationships

4. Writes them to **Neo4j**

5. Embeds all nodes + relationships

## ADK Tools

The notebook defines three tools callable by the ADK agent:
```python
ingest_dataset_samples_tool(image_input) 
get_query_embedding(query) 
retrieve_from_graph(query, top_k)
```

These are registered with the retrieval agent.

## Root Retrieval agent

The core ADK agent:
```python
retrieval_agent = LlmAgent(
    name="retrieval_agent",
    model=Gemini(...),
    instruction=""" 
        1. get_query_embedding
        2. retrieve_from_graph
        3. ingest image if provided
    """,
    tools=[get_query_embedding, retrieve_from_graph, ingest_dataset_samples_tool],
    output_key="answer"
)
```

Run it via:

```python
from google.adk.runners import InMemoryRunner 
await app_runner.run_debug("Did I order any coffee?")
```

## Multi-Agent RAG System

This parts uses a hierarchical multi-agent architecture, where a top-level orchestrator (RootRagAgent) delegates tasks to specialized sub-agents. Each agent has a single responsibility, and the root agent coordinates the workflow for ingestion, embedding, and retrieval.

## Agents Overview

### IngestionAgent
- Responsibility: Ingests receipt images and extracts structured entities and relationships.
- Tool used: `ingest_dataset_samples_tool(image_input)`
- Instruction: Only ingest images; do not answer queries directly.

### EmbeddingAgent
- Responsibility: Generates vector embeddings for user queries.
- Tool used: `get_query_embedding(query)`
- Instruction: Only generate embeddings; called by root agent, not directly by the user.

### RetrievalAgent
- Responsibility: Performs hybrid Neo4j retrieval using embeddings.
- Tool used: `retrieve_from_graph(query)`
- Instruction: Only retrieve graph results; called by root agent after embedding is ready.

### RootRagAgent
- Responsibility: Orchestrates the workflow.
- Logic:
  1. If the user provides an image, call `IngestionAgent`.
  2. If the user asks a question:
     - Call `EmbeddingAgent` → compute embeddings
     - Call `RetrievalAgent` → fetch relevant nodes from Neo4j
  3. After receiving outputs from sub-agents, generate a final, concise answer.



## How Retrieval Works

1. **Embed query** using **BGE-M3**

2. Perform **vector search**

3. Expand neighbors using **graph traversal**

4. Merge results

5. LLM composes final answer

This enables structured and semantic retrieval across receipts, entities, prices, stores, and items.

## Troubleshooting

### APOC not loading?

* Ensure correct `apoc-x.y.z-core.jar`

* Ensure placed in `/plugins`

### Vector index errors?

* Make sure you ran the Cypher from:

```python
gr.neo4j_create_vector_index_cypher()
```