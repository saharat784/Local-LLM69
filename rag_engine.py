import os
import glob
import time
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    import pypdf
except ImportError:
    pypdf = None

BM25_INDEX_FILE = ".bm25_index.pkl"
EMBEDDING_MODEL = "nomic-embed-text:latest"
FAISS_INDEX_DIR = ".faiss_index"
QUERY_HISTORY_FILE = "query_history.log"
KUZU_DB_DIR = ".kuzu_db"

import kuzu
import re

def extract_coding_entities(text: str) -> list[str]:
    # Regex to find camelCase, PascalCase, or snake_case words
    entities = set(re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[a-z]+(?:_[a-z]+)+\b', text))
    return list(entities)[:10]

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

import pickle
from langchain_community.retrievers import BM25Retriever

class HybridRetriever:
    """ห่อหุ้ม BM25 + FAISS + Kuzu Graph + RRF เป็น interface เดียว"""
    def __init__(self, faiss_store, bm25_retriever, kuzu_conn):
        self.faiss = faiss_store
        self.bm25 = bm25_retriever
        self.kuzu_conn = kuzu_conn

    def retrieve(self, query: str, top_k: int = 5, k_rrf: int = 60, metadata_filter=None):
        # 1. FAISS Search
        filter_kwargs = {}
        if metadata_filter:
            filter_kwargs['filter'] = lambda metadata: metadata.get("source") in metadata_filter
        
        try:
            faiss_results = self.faiss.similarity_search_with_score(query, k=top_k*2, **filter_kwargs)
        except Exception:
            faiss_results = []
            
        # 2. BM25 Search
        try:
            bm25_all = self.bm25.invoke(query)
            if metadata_filter:
                bm25_results = [doc for doc in bm25_all if doc.metadata.get("source") in metadata_filter][:top_k*2]
            else:
                bm25_results = bm25_all[:top_k*2]
        except Exception:
            bm25_results = []

        # 3. Kuzu Graph Search (Requirement 7: MATCH ... RETURN)
        graph_results = []
        try:
            query_entities = extract_coding_entities(query)
            for entity in query_entities:
                res = self.kuzu_conn.execute(
                    "MATCH (a:Concept {name: $q})-[:RELATES_TO]->(b:Concept) RETURN b.name, b.source",
                    {"q": entity}
                )
                for row in res:
                    b_name, b_source = row[0], row[1]
                    if not metadata_filter or b_source in metadata_filter:
                        graph_doc = Document(
                            page_content=f"Graph Knowledge: '{entity}' is related to '{b_name}'", 
                            metadata={"source": b_source}
                        )
                        graph_results.append(graph_doc)
        except Exception:
            pass
            
        # 4. RRF Fusion (Reciprocal Rank Fusion)
        faiss_sorted = sorted(faiss_results, key=lambda x: x[1])
        rrf_scores = {}
        
        def add_to_rrf(doc, rank, source_name, dist=999.0):
            doc_id = doc.page_content
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0, "sources": [], "faiss_dist": dist}
            rrf_scores[doc_id]["score"] += 1.0 / (k_rrf + rank + 1)
            if source_name not in rrf_scores[doc_id]["sources"]:
                rrf_scores[doc_id]["sources"].append(source_name)
        
        for rank, (doc, dist) in enumerate(faiss_sorted):
            add_to_rrf(doc, rank, "FAISS", dist)
            
        for rank, doc in enumerate(bm25_results):
            add_to_rrf(doc, rank, "BM25")
            
        # Graph gets top rank boost because it matches specific concepts directly
        for rank, doc in enumerate(graph_results):
            add_to_rrf(doc, 0, "KuzuGraph")
                
        fused = sorted(list(rrf_scores.values()), key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

class RAGEngine:
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = knowledge_dir
        self.embedding_model_name = EMBEDDING_MODEL
        self.embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self.vectorstore = None
        self.bm25_retriever = None
        self.hybrid_retriever = None
        
        # Initialize Kuzu (Requirement 1: Kuzu embedded)
        self.kuzu_db = kuzu.Database(KUZU_DB_DIR)
        self.kuzu_conn = kuzu.Connection(self.kuzu_db)
        
        # Requirement 2 & 4: Table creation with explicit FROM and TO
        try:
            self.kuzu_conn.execute("CREATE NODE TABLE Concept (name STRING, source STRING, PRIMARY KEY (name))")
        except RuntimeError:
            pass
        try:
            self.kuzu_conn.execute("CREATE REL TABLE RELATES_TO (FROM Concept TO Concept)")
        except RuntimeError:
            pass

        self._load_vectorstore()
        
        # Requirement 7: Keep query history
        if not os.path.exists(QUERY_HISTORY_FILE):
            with open(QUERY_HISTORY_FILE, 'w', encoding='utf-8') as f:
                f.write("timestamp\tquery\ttop_rrf_score\trelevant_chunks\n")

    def set_embedding_model(self, model_name: str):
        if self.embedding_model_name != model_name:
            self.embedding_model_name = model_name
            self.embeddings = OllamaEmbeddings(model=model_name)
            self.vectorstore = None # Needs reindex

    def _load_vectorstore(self):
        """Requirement 4: Save/Load FAISS and BM25 index"""
        if os.path.exists(FAISS_INDEX_DIR) and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
            try:
                self.vectorstore = FAISS.load_local(
                    FAISS_INDEX_DIR, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                if os.path.exists(BM25_INDEX_FILE):
                    with open(BM25_INDEX_FILE, 'rb') as f:
                        self.bm25_retriever = pickle.load(f)
                
                if self.vectorstore and self.bm25_retriever:
                    self.hybrid_retriever = HybridRetriever(self.vectorstore, self.bm25_retriever, self.kuzu_conn)
            except Exception:
                self.vectorstore = None
                self.bm25_retriever = None
                self.hybrid_retriever = None

    def get_available_sources(self) -> list[str]:
        """Utility for UI to get list of source files for metadata filtering"""
        files = []
        for ext in ['*.md', '*.txt', '*.pdf']:
            files.extend(glob.glob(os.path.join(self.knowledge_dir, ext)))
            files.extend(glob.glob(os.path.join(self.knowledge_dir, "**", ext), recursive=True))
        return sorted(list(set([os.path.basename(f) for f in files])))

    def index_knowledge_base(self, force_reindex: bool = False, progress_callback=None) -> dict:
        """Requirement 2: FAISS.from_documents() does embed + index automatically"""
        supported_extensions = ['*.md', '*.txt', '*.pdf']
        files = []
        for ext in supported_extensions:
            files.extend(glob.glob(os.path.join(self.knowledge_dir, ext)))
            files.extend(glob.glob(os.path.join(self.knowledge_dir, "**", ext), recursive=True))
        files.extend(glob.glob("*.pdf"))
        files = sorted(list(set(files)))
        
        documents = []
        indexed_files_count = 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
        )

        for i, file_path in enumerate(files):
            file_name = os.path.basename(file_path)
            content = load_document(file_path)
            if not content.strip():
                continue
            
            # Include source in metadata for filtering
            doc = Document(page_content=content, metadata={"source": file_name, "file_path": file_path})
            
            chunks = text_splitter.split_documents([doc])
            documents.extend(chunks)
            indexed_files_count += 1
            
            if progress_callback:
                progress_callback(min(1.0, (i+1)/len(files)), i+1, len(files))

        if not documents:
            return {"status": "success", "files_indexed": 0, "total_chunks": 0}

        # --- Populate Kuzu Graph DB ---
        for doc in documents:
            file_name = doc.metadata.get("source", "Unknown")
            entities = extract_coding_entities(doc.page_content)
            
            # Requirement 3 & 5: Insert nodes first using CREATE and params
            for entity in entities:
                check_node = self.kuzu_conn.execute("MATCH (c:Concept {name: $n}) RETURN c.name", {"n": entity})
                if not check_node.has_next():
                    self.kuzu_conn.execute("CREATE (c:Concept {name: $n, source: $s})", {"n": entity, "s": file_name})
                    
            # Requirement 6: Insert relationships using MATCH + CREATE pattern
            for idx_ent in range(len(entities) - 1):
                na, nb = entities[idx_ent], entities[idx_ent+1]
                check_rel = self.kuzu_conn.execute(
                    "MATCH (a:Concept {name: $na})-[r:RELATES_TO]->(b:Concept {name: $nb}) RETURN r",
                    {"na": na, "nb": nb}
                )
                if not check_rel.has_next():
                    self.kuzu_conn.execute(
                        "MATCH (a:Concept {name: $na}), (b:Concept {name: $nb}) CREATE (a)-[:RELATES_TO]->(b)",
                        {"na": na, "nb": nb}
                    )

        # Requirement 1: LangChain wraps FAISS
        # Batch processing to prevent Ollama runner from crashing (OOM)
        batch_size = 32
        self.vectorstore = None
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(batch_docs, self.embeddings)
            else:
                self.vectorstore.add_documents(batch_docs)
            time.sleep(0.1) # ป้องกันปัญหา Windows Socket เต็ม (WSAENOBUFS)
            
                
        self.vectorstore.save_local(FAISS_INDEX_DIR)
        
        # Requirement 1 (BM25): วัด "ความสำคัญ" ของคำ
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        with open(BM25_INDEX_FILE, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)
            
        self.hybrid_retriever = HybridRetriever(self.vectorstore, self.bm25_retriever, self.kuzu_conn)

        return {
            "status": "success",
            "files_indexed": indexed_files_count,
            "total_chunks": len(documents)
        }

    def retrieve(self, query: str, top_k: int = 4, source_filters: list[str] = None) -> list[dict]:
        """Requirement 3: Filter with metadata, Requirement 6: Check FAISS score"""
        if self.hybrid_retriever is None:
            self.index_knowledge_base()
            if self.hybrid_retriever is None:
                return []

        # RRF Hybrid Retrieve
        results = self.hybrid_retriever.retrieve(query, top_k=top_k, metadata_filter=source_filters)

        # Requirement 6: FAISS always returns results, we must check distance score.
        # Threshold for L2 distance (Lower is better).
        SCORE_THRESHOLD = 1.3
        
        retrieved = []
        top_score = None
        for result in results:
            doc = result["doc"]
            rrf_score = result["score"]
            faiss_dist = result["faiss_dist"]
            sources = result["sources"]
            
            if top_score is None:
                top_score = rrf_score
            
            # Check if relevant (either passed FAISS threshold, or was heavily boosted by BM25)
            if faiss_dist > SCORE_THRESHOLD and "BM25" not in sources:
                continue

            retrieved.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "score": round(float(rrf_score), 4),
                "type": " + ".join(sources)
            })
            
        # Requirement 7: Save query history
        self._log_query(query, top_score, len(retrieved))

        return retrieved

    def _log_query(self, query: str, top_score: float, num_relevant: int):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        ts_score = f"{top_score:.4f}" if top_score is not None else "N/A"
        # Sanitize query for single-line log
        clean_query = query.replace('\n', ' ').replace('\r', '').strip()
        log_line = f"{timestamp}\t{clean_query}\t{ts_score}\t{num_relevant}\n"
        with open(QUERY_HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(log_line)

    def build_rag_system_prompt(self, base_system_prompt: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return base_system_prompt

        context_str = "\n\n".join([
            f"--- SOURCE: {chunk['source']} (Distance: {chunk['score']}) ---\n{chunk['content']}"
            for chunk in retrieved_chunks
        ])

        # Requirement 5: Source attribution (tell the user where the answer comes from)
        augmented_prompt = f"""{base_system_prompt}

You are provided with relevant knowledge documents below to accurately answer the user's request.

=== INSTRUCTIONS ===
1. Base your answer strictly on the provided context where applicable.
2. Respond in the user's language (e.g., Thai).
3. SOURCE ATTRIBUTION: You MUST explicitly cite the source filename in your answer when referencing its content (e.g., "อ้างอิงจากไฟล์ `filename.md`...").

=== RETRIEVED KNOWLEDGE CONTEXT ===
{context_str}
===================================
"""
        return augmented_prompt
