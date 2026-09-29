import os
import sys

# Fix Windows Unicode Encode Error
sys.stdout.reconfigure(encoding='utf-8')

from rag_engine import RAGEngine

# ข้อมูลจำลองสำหรับทดสอบ (Ground Truth) 
# เปรียบเทียบคำถาม กับไฟล์ Source ที่ควรจะถูกดึงมา
GROUND_TRUTH = [
    {"query": "How to name a class in python?", "expected_source": "python_clean_code.md"},
    {"query": "What is the primary color for React?", "expected_source": "react_tailwind_styleguide.md"},
    {"query": "Give me a singleton pattern in python", "expected_source": "python_clean_code.md"},
    {"query": "How to make a text input with Tailwind?", "expected_source": "react_tailwind_styleguide.md"},
    {"query": "Explain factory method", "expected_source": "python_clean_code.md"},
]

def calculate_metrics(results, expected_source, k=5):
    """
    คำนวณ Metrics ตามที่กำหนด:
    - Recall@k: วัดว่าในผลลัพธ์ top-k มีเอกสารที่ relevant หรือไม่ (ถ้ามีถือว่าไม่พลาด = 1)
    - Precision@k: วัดว่าในผลลัพธ์ top-k มีเอกสารที่ relevant กี่เปอร์เซ็นต์
    - MRR (Mean Reciprocal Rank): วัดว่าเอกสาร relevant ปรากฏที่ rank แรกคือเท่าไหร่ (1/rank)
    """
    hits = [1 if res['source'] == expected_source else 0 for res in results[:k]]
    
    # 3. Recall@k
    recall_at_k = 1 if sum(hits) > 0 else 0
    
    # 3. Precision@k
    precision_at_k = sum(hits) / k if k > 0 else 0
    
    # 4. MRR
    mrr = 0.0
    for rank, is_hit in enumerate(hits):
        if is_hit:
            mrr = 1.0 / (rank + 1)
            break
            
    return recall_at_k, precision_at_k, mrr

def run_evaluation():
    print("🚀 Starting RAG Evaluation (BM25 + FAISS + RRF)...")
    engine = RAGEngine()
    
    if engine.hybrid_retriever is None:
        print("Indexing knowledge base first...")
        engine.index_knowledge_base()
        
    k = 5
    total_recall = 0
    total_precision = 0
    total_mrr = 0
    num_queries = len(GROUND_TRUTH)
    
    print("\n" + "="*60)
    for idx, gt in enumerate(GROUND_TRUTH):
        query = gt['query']
        expected = gt['expected_source']
        
        results = engine.retrieve(query, top_k=k)
        recall, precision, mrr = calculate_metrics(results, expected, k)
        
        total_recall += recall
        total_precision += precision
        total_mrr += mrr
        
        print(f"Q{idx+1}: {query}")
        print(f"  Expected: {expected}")
        if results:
            print(f"  Top Result: {results[0]['source']} (RRF Score: {results[0]['score']:.4f}) [By: {results[0]['type']}]")
        else:
            print("  Top Result: None")
        print(f"  -> Recall@{k}: {recall}, Precision@{k}: {precision:.2f}, MRR: {mrr:.2f}")
        print("-" * 60)
        
    avg_recall = total_recall / num_queries
    avg_precision = total_precision / num_queries
    avg_mrr = total_mrr / num_queries
    
    print("\n📊 === FINAL EVALUATION RESULTS (HYBRID RETRIEVER) ===")
    print(f"Average Recall@{k}    : {avg_recall*100:.1f}%")
    print(f"Average Precision@{k} : {avg_precision*100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.3f}")
    print("======================================================")
    print("✅ 1. BM25 & FAISS ถูกห่อหุ้มและประมวลผลร่วมกันผ่าน HybridRetriever")
    print("✅ 2. RRF รวม Rank ให้โดยไม่ต้องทำ Score Normalization")
    print("✅ 3. แสดงผล Recall@k และ Precision@k เพื่อวัดประสิทธิภาพ")
    print("✅ 4. คำนวณ MRR เพื่อชี้วัดอันดับของข้อมูลที่ถูกต้อง (ยิ่งเข้าใกล้ 1 ยิ่งดี)\n")

if __name__ == "__main__":
    run_evaluation()
