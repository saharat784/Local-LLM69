import streamlit as st
import time
import ollama
import importlib
import rag_engine as rag_module
importlib.reload(rag_module)
from rag_engine import RAGEngine

st.set_page_config(
    page_title="Local LLM Studio",
    page_icon="🤖",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Requirement 4 & 5: โหลด heavy objects ด้วย @st.cache_resource เสมอ 
# เพื่อป้องกัน model reload ทุกครั้งที่ Streamlit rerun (เพราะ Streamlit รัน script ตั้งแต่ต้นทุกครั้ง)
@st.cache_resource
def load_rag():
    try:
        engine = RAGEngine()
        # Requirement 7: handle error ใน load_rag()
        if engine.hybrid_retriever is None:
            return None # Database might be missing
        return engine
    except Exception as e:
        return None

rag_engine = load_rag()

# Helper to fetch installed models
@st.cache_data(ttl=10)
def get_installed_models():
    try:
        models_res = ollama.list()
        embed_models = []
        llm_models = []
        for m in models_res.models:
            is_embed = 'embed' in m.model.lower() or 'bge-' in m.model.lower()
            if hasattr(m, 'details') and m.details.families:
                if any('bert' in f.lower() for f in m.details.families):
                    is_embed = True
            
            if is_embed:
                embed_models.append(m.model)
            else:
                llm_models.append(m.model)
        
        return {"llm": llm_models, "embed": embed_models}
    except Exception as e:
        return {"llm": [], "embed": []}

st.markdown('<div class="main-header">Local LLM Studio</div>', unsafe_allow_html=True)
st.caption("Powered by Local Ollama Models, LangChain & FAISS RAG")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    
    models_dict = get_installed_models()
    available_llms = models_dict["llm"]
    available_embeds = models_dict["embed"]
    
    if not available_llms and not available_embeds:
        st.error("Cannot connect to Ollama or no models found.")
        st.info("Make sure Ollama is running (`ollama serve`) and you have pulled models.")
        selected_model = None
        selected_embedding = None
    else:
        selected_model = st.selectbox(
            "Select LLM Model",
            options=available_llms,
            index=0 if "qwen2.5-coder:7b" not in available_llms else available_llms.index("qwen2.5-coder:7b")
        )
        
        selected_embedding = st.selectbox(
            "Select Embedding Model",
            options=available_embeds,
            index=0 if not available_embeds else (available_embeds.index([m for m in available_embeds if "nomic-embed-text" in m][0]) if any("nomic-embed-text" in m for m in available_embeds) else 0)
        )
        
        # Requirement 1 & 2: ใช้ session_state เป็น "ความจำ" ของ Streamlit
        if 'current_embedding' not in st.session_state:
            st.session_state.current_embedding = selected_embedding
            if selected_embedding and rag_engine:
                rag_engine.set_embedding_model(selected_embedding)
            
        if selected_embedding and selected_embedding != st.session_state.current_embedding:
            if rag_engine:
                rag_engine.set_embedding_model(selected_embedding)
            st.session_state.current_embedding = selected_embedding
            st.warning("You changed the Embedding Model. Please click 'Re-Index Database' below to apply changes.")

        st.success(f"Connected: **{selected_model}**")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    st.divider()
    st.header("Knowledge Base (FAISS)")
    enable_rag = st.toggle("Enable Knowledge Base Search", value=True)
    
    # Requirement 3: Metadata filtering UI
    available_sources = rag_engine.get_available_sources() if rag_engine else []
    selected_sources = st.multiselect(
        "Filter by Source File (Leave empty for all)",
        options=available_sources,
        default=[],
        disabled=not enable_rag or not rag_engine
    )
    
    top_k = st.slider("Max Context Chunks", min_value=1, max_value=8, value=4)

    # นำปุ่ม Re-Index กลับมาตามคำขอผู้ใช้
    if st.button("Re-Index Database", use_container_width=True):
        progress_bar = st.progress(0, text="Starting indexing (FAISS/BM25/Kuzu)...")
        def on_progress(pct, done, total):
            progress_bar.progress(pct, text=f"Processing: {done}/{total} files ({int(pct * 100)}%)...")

        if rag_engine is not None:
            stats = rag_engine.index_knowledge_base(force_reindex=True, progress_callback=on_progress)
        else:
            temp_engine = RAGEngine()
            stats = temp_engine.index_knowledge_base(force_reindex=True, progress_callback=on_progress)
            st.cache_resource.clear()
            
        progress_bar.empty()
        st.success(f"Indexed {stats['files_indexed']} files ({stats['total_chunks']} chunks) successfully!")
        time.sleep(1)
        st.rerun()
    
    st.divider()

    system_prompt = st.text_area(
        "System Prompt",
        value="You are an Expert Software Engineer and Coding Assistant. Write clean, efficient, and well-documented code. Explain your thought process briefly before providing the code.",
        height=100
    )
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Requirement 7: แสดงข้อความที่ actionable อย่าให้ app crash โดยไม่บอกเหตุ
if rag_engine is None:
    st.error("🚨 ไม่พบฐานข้อมูล RAG (FAISS/BM25/Kuzu)!")
    st.info("💡 ข้อแนะนำ (Actionable): กรุณากดปุ่ม **'Re-Index Database'** ในเมนูด้านซ้าย หรือรัน `uv run python setup.py` ใน Terminal เพื่อสร้างฐานข้อมูลก่อนเริ่มใช้งานครับ")
    st.stop()

# Requirement 1 & 2: ใช้ session_state จำประวัติแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "generation_time" in msg:
                st.caption(f"Generated in {msg['generation_time']:.2f} seconds")
                
            if "sources" in msg and msg["sources"]:
                with st.expander("Referenced Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**[{src.get('type', 'FAISS')}] Source:** `{src['source']}` (Distance: {src['score']})")
                        st.caption(src['content'])

# Chat Input
if prompt := st.chat_input("Ask for code, concepts, or query your knowledge base..."):
    if not selected_model:
        st.error("Please select a model first.")
        st.stop()

    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Retrieval
    retrieved_chunks = []
    effective_system_prompt = system_prompt

    if enable_rag:
        with st.status("Searching FAISS Vectorstore...", expanded=False) as status:
            # Pass source_filters
            retrieved_chunks = rag_engine.retrieve(prompt, top_k=top_k, source_filters=selected_sources)
            if retrieved_chunks:
                status.update(label=f"Found {len(retrieved_chunks)} relevant contexts (L2 Distance Threshold applied)", state="complete")
            else:
                status.update(label="No relevant context found (or all exceeded distance threshold).", state="complete")

        effective_system_prompt = rag_engine.build_rag_system_prompt(system_prompt, retrieved_chunks)

    # Requirement 9: ส่ง chat history ไปใน prompt ด้วย — ทำให้ LLM รู้ context ก่อนหน้า
    api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state.messages

    # Stream Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            start_time = time.time()
            
            # Requirement 8: streaming ทำให้ user เห็น LLM พิมพ์ทีละตัวอักษร แทนที่จะรอนาน
            stream = ollama.chat(
                model=selected_model,
                messages=api_messages,
                options={"temperature": temperature},
                stream=True
            )

            timer_placeholder = st.empty()
            last_update_time = start_time

            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                full_response += content
                message_placeholder.markdown(full_response + "▌")
                
                # Realtime timer update
                current_time = time.time()
                if current_time - last_update_time > 0.1:
                    timer_placeholder.caption(f"Thinking... {current_time - start_time:.1f}s")
                    last_update_time = current_time
            
            message_placeholder.markdown(full_response)
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Final timer display
            timer_placeholder.caption(f"Generated in {generation_time:.2f} seconds")

            # Requirement 6: แสดง sources ให้ user เห็นว่า RAG retrieve มาจากไฟล์ไหน
            if retrieved_chunks:
                with st.expander("Referenced Sources"):
                    for src in retrieved_chunks:
                        st.markdown(f"**[{src.get('type', 'FAISS')}] Source:** `{src['source']}` (Distance: {src['score']})")
                        st.caption(src['content'])

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_chunks,
                "generation_time": generation_time
            })

        except Exception as e:
            st.error(f"Error generating response: {e}")
