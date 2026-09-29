import os
from rag_engine import RAGEngine

def run_setup():
    print("🚀 Starting Setup: Building Knowledge Database...")
    print("This script separates indexing from serving (app.py) as required.")
    
    # Initialize engine (creates Kuzu DB if not exists)
    engine = RAGEngine()
    
    print("\n⏳ Indexing documents into FAISS, BM25, and Kuzu Graph...")
    stats = engine.index_knowledge_base()
    
    if stats['status'] == 'success':
        print(f"✅ Successfully indexed {stats['files_indexed']} files ({stats['total_chunks']} chunks)!")
        print("✅ You can now run `uv run streamlit run app.py` to start the web app.")
    else:
        print("⚠️ Failed to index knowledge base. Please check your knowledge/ directory.")

if __name__ == "__main__":
    run_setup()
