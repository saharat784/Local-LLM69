import os
import glob
import re
import json
import ollama
import chromadb
import networkx as nx
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
try:
    import pypdf
except ImportError:
    pypdf = None

EMBEDDING_MODEL = "bge-m3"
CHROMA_PERSIST_DIR = ".chroma_db"
GRAPH_PERSIST_FILE = ".knowledge_graph.graphml"
COLLECTION_NAME = "knowledge_base"

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def _extract_embeddings(self, res) -> list:
        if hasattr(res, 'embeddings') and res.embeddings:
            return res.embeddings
        if isinstance(res, dict) and "embeddings" in res and res["embeddings"]:
            return res["embeddings"]
        return []

    def _extract_single_embedding(self, res) -> list:
        embs = self._extract_embeddings(res)
        if embs:
            return embs[0]
        if hasattr(res, 'embedding') and res.embedding:
            return res.embedding
        if isinstance(res, dict) and "embedding" in res and res["embedding"]:
            return res["embedding"]
        return []

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        try:
            res = ollama.embed(model=self.model_name, input=input)
            embs = self._extract_embeddings(res)
            if embs:
                return embs
        except Exception:
            pass

        embeddings = []
        for text in input:
            try:
                res = ollama.embed(model=self.model_name, input=text)
                emb = self._extract_single_embedding(res)
            except Exception:
                try:
                    res = ollama.embeddings(model=self.model_name, prompt=text)
                    emb = self._extract_single_embedding(res)
                except Exception:
                    emb = []
            embeddings.append(emb)
        return embeddings

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        break_pos = text.rfind('\n\n', start, end)
        if break_pos == -1 or break_pos < start + (chunk_size // 2):
            break_pos = text.rfind('\n', start, end)
        if break_pos == -1 or break_pos < start + (chunk_size // 2):
            break_pos = text.rfind('. ', start, end)

        if break_pos != -1 and break_pos > start:
            chunk = text[start:break_pos].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, break_pos - overlap)
        else:
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

    return [c for c in chunks if c]

def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.md', '.txt']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.pdf':
        if not pypdf:
            return ""
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    return ""

def extract_coding_entities(text: str) -> list[str]:
    """Extracts coding-related entities (CamelCase, snake_case, ALL_CAPS) for Knowledge Graph."""
    entities = set()
    # CamelCase (e.g., MyClass, DataBaseManager)
    camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    # snake_case (e.g., my_function_name)
    snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', text)
    # ALL_CAPS (e.g., CONSTANT_VALUE)
    all_caps = re.findall(r'\b[A-Z]+(?:_[A-Z]+)+\b', text)
    # Common tech keywords (Fallback for non-code text)
    tech_keywords = re.findall(r'\b(API|JSON|XML|HTML|CSS|SQL|Database|Server|Client|Frontend|Backend|Python|JavaScript|Java|C\+\+|Go|Rust|Docker|Kubernetes|AWS|Cloud)\b', text, re.IGNORECASE)
    
    for e in camel_case + snake_case + all_caps + tech_keywords:
        entities.add(e)
    return list(entities)

class RAGEngine:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.embedding_fn = OllamaEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        # Initialize Graph
        self.graph = nx.Graph()
        if os.path.exists(GRAPH_PERSIST_FILE):
            try:
                self.graph = nx.read_graphml(GRAPH_PERSIST_FILE)
            except Exception:
                pass

    def set_embedding_model(self, model_name: str):
        """Update the embedding model used by the engine."""
        self.embedding_fn.model_name = model_name

    def index_knowledge_base(self, force_reindex: bool = False, progress_callback=None) -> dict:
        if force_reindex:
            try:
                self.chroma_client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self.graph.clear()
            if os.path.exists(GRAPH_PERSIST_FILE):
                os.remove(GRAPH_PERSIST_FILE)

        supported_extensions = ['*.md', '*.txt', '*.pdf']
        files = []
        for ext in supported_extensions:
            files.extend(glob.glob(os.path.join(self.knowledge_dir, ext)))
            files.extend(glob.glob(os.path.join(self.knowledge_dir, "**", ext), recursive=True))
        files.extend(glob.glob("*.pdf"))
        files = sorted(list(set(files)))
        
        all_ids = []
        all_chunks = []
        all_metadatas = []
        indexed_files_count = 0

        for file_path in files:
            file_name = os.path.basename(file_path)
            content = load_document(file_path)
            if not content.strip():
                continue

            chunks = chunk_text(content)
            if not chunks:
                continue

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{file_name}_chunk_{idx}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadatas.append({"source": file_name, "file_path": file_path, "chunk_index": idx})
                
                # --- GRAPH RAG: Entity Extraction and Graph Building ---
                entities = extract_coding_entities(chunk)
                # Link all entities found in the same chunk
                for i in range(len(entities)):
                    self.graph.add_node(entities[i], type="entity")
                    for j in range(i + 1, len(entities)):
                        self.graph.add_node(entities[j], type="entity")
                        # Add or update edge with chunk reference
                        if self.graph.has_edge(entities[i], entities[j]):
                            # Append source if not already present
                            existing_sources = self.graph[entities[i]][entities[j]].get("sources", "")
                            if file_name not in existing_sources:
                                self.graph[entities[i]][entities[j]]["sources"] = existing_sources + f", {file_name}"
                        else:
                            self.graph.add_edge(entities[i], entities[j], sources=file_name, chunk_snippet=chunk[:100] + "...")

            indexed_files_count += 1

        batch_size = 32
        total_chunks = len(all_chunks)
        total_batches = max(1, (total_chunks + batch_size - 1) // batch_size)

        for b_idx, i in enumerate(range(0, total_chunks, batch_size)):
            self.collection.upsert(
                ids=all_ids[i:i + batch_size],
                documents=all_chunks[i:i + batch_size],
                metadatas=all_metadatas[i:i + batch_size]
            )
            if progress_callback:
                progress = min(1.0, (b_idx + 1) / total_batches)
                current_done = min(i + batch_size, total_chunks)
                progress_callback(progress, current_done, total_chunks)

        # Save Graph to disk
        if self.graph.number_of_nodes() > 0:
            nx.write_graphml(self.graph, GRAPH_PERSIST_FILE)

        return {
            "status": "success",
            "files_indexed": indexed_files_count,
            "total_chunks": total_chunks,
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges()
        }

    def retrieve(self, query: str, top_k: int = 4, mode: str = "Hybrid") -> list[dict]:
        if self.collection.count() == 0:
            self.index_knowledge_base()

        if self.collection.count() == 0:
            return []

        retrieved = []
        seen_content = set()
        
        # 1. Vector Retrieval
        if "Vector" in mode or "Hybrid" in mode:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count())
            )
            
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, distances):
                    retrieved.append({
                        "content": doc,
                        "source": meta.get("source", "Unknown"),
                        "score": round(1.0 - dist, 4) if dist is not None else 1.0,
                        "type": "Vector"
                    })
                    seen_content.add(doc)

        # 2. Graph Retrieval
        if "Graph" in mode or "Hybrid" in mode:
            query_entities = extract_coding_entities(query)
            for entity in query_entities:
                if self.graph.has_node(entity):
                    # Get neighbors
                    neighbors = list(self.graph.neighbors(entity))[:3] # Limit to top 3 neighbors
                    for neighbor in neighbors:
                        edge_data = self.graph.get_edge_data(entity, neighbor)
                        relation_info = f"Graph Relationship: '{entity}' is connected to '{neighbor}' (Found in: {edge_data.get('sources', 'Unknown')})"
                        
                        if relation_info not in seen_content:
                            retrieved.append({
                                "content": relation_info,
                                "source": edge_data.get("sources", "Graph Knowledge"),
                                "score": 0.95, # High pseudo-score for direct graph hits
                                "type": "Graph"
                            })
                            seen_content.add(relation_info)

        # Sort combined results by score (descending)
        retrieved = sorted(retrieved, key=lambda x: x['score'], reverse=True)
        return retrieved[:top_k + 2] # Return top_k vector + some graph context

    def build_rag_system_prompt(self, base_system_prompt: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return base_system_prompt

        context_str = "\n\n".join([
            f"--- Document Source ({chunk.get('type', 'Vector')}): {chunk['source']} ---\n{chunk['content']}"
            for chunk in retrieved_chunks
        ])

        augmented_prompt = f"""{base_system_prompt}

You are provided with relevant knowledge documents and Graph relationships below to accurately answer the user's request.

=== INSTRUCTIONS ===
1. Base your answer strictly on the provided context where applicable.
2. Respond in the user's language (e.g., Thai), synthesizing information from the context documents.
3. If the information is found in the context, cite the relevant document source filename.

=== RETRIEVED HYBRID KNOWLEDGE CONTEXT ===
{context_str}
===================================
"""
        return augmented_prompt
