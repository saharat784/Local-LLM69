import streamlit as st
import time
import ollama
import importlib
import rag_engine as rag_module
importlib.reload(rag_module)
from rag_engine import RAGEngine

st.set_page_config(
    page_title="Local LLM",
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
    .rag-badge {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Engine
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

# Ensure fresh RAGEngine instance if code changed
rag_engine = RAGEngine()

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

st.markdown('<div class="main-header">Local LLM Coding Assistant</div>', unsafe_allow_html=True)
st.caption("Powered by Local Ollama Models & Hybrid RAG")

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
            index=0 if not available_embeds else (available_embeds.index([m for m in available_embeds if "bge-m3" in m][0]) if any("bge-m3" in m for m in available_embeds) else 0)
        )
        
        if 'current_embedding' not in st.session_state:
            st.session_state.current_embedding = selected_embedding
            if selected_embedding:
                rag_engine.set_embedding_model(selected_embedding)
            
        if selected_embedding and selected_embedding != st.session_state.current_embedding:
            rag_engine.set_embedding_model(selected_embedding)
            st.session_state.current_embedding = selected_embedding
            st.warning("You changed the Embedding Model. Please click 'Re-Index Database' below to apply changes.")

        st.success(f"Connected: **{selected_model}**")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    st.divider()
    st.header("Knowledge Base")
    enable_rag = st.toggle("Enable Knowledge Base Search", value=True)
    
    rag_mode = st.radio(
        "Retrieval Mode",
        options=["Hybrid (Vector + Graph)", "Vector Only", "Graph Only"],
        index=0,
        disabled=not enable_rag
    )
    
    top_k = st.slider("Max Context Chunks", min_value=1, max_value=8, value=4)

    if st.button("Re-Index Database", use_container_width=True):
        progress_bar = st.progress(0, text="Starting document indexing...")
        def on_progress(pct, done, total):
            progress_bar.progress(pct, text=f"Indexing: {done}/{total} chunks ({int(pct * 100)}%)...")

        stats = rag_engine.index_knowledge_base(force_reindex=True, progress_callback=on_progress)
        progress_bar.empty()
        st.success(f"Indexed {stats['files_indexed']} files ({stats['total_chunks']} chunks).")

    st.divider()

    system_prompt = st.text_area(
        "System Prompt",
        value="You are an Expert Software Engineer and Coding Assistant. Write clean, efficient, and well-documented code. Explain your thought process briefly before providing the code.",
        height=100
    )
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
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
                        st.markdown(f"**[{src.get('type', 'Vector')}] Source:** `{src['source']}` (Score: {src['score']})")
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
        mode_text = rag_mode.split(" ")[0]
        with st.status(f"Searching Knowledge Base ({mode_text} Mode)...", expanded=False) as status:
            retrieved_chunks = rag_engine.retrieve(prompt, top_k=top_k, mode=rag_mode)
            if retrieved_chunks:
                vector_count = sum(1 for c in retrieved_chunks if 'Vector' in c.get('type', ''))
                graph_count = sum(1 for c in retrieved_chunks if 'Graph' in c.get('type', ''))
                status.update(label=f"Found {len(retrieved_chunks)} relevant contexts ({vector_count} Vector, {graph_count} Graph)", state="complete")
            else:
                status.update(label="No relevant context found.", state="complete")

        effective_system_prompt = rag_engine.build_rag_system_prompt(system_prompt, retrieved_chunks)

    # Prepare payload for Ollama
    api_messages = [{"role": "system", "content": effective_system_prompt}] + st.session_state.messages

    # Stream Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            start_time = time.time()
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

            if retrieved_chunks:
                with st.expander("Referenced Sources"):
                    for src in retrieved_chunks:
                        st.markdown(f"**[{src.get('type', 'Vector')}] Source:** `{src['source']}` (Score: {src['score']})")
                        st.caption(src['content'])

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": retrieved_chunks,
                "generation_time": generation_time
            })

        except Exception as e:
            st.error(f"Error generating response: {e}")
