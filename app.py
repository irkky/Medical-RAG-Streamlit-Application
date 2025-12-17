import os
import streamlit as st
from dotenv import load_dotenv
from src.rag_chain import get_rag_chain
from src.ingestion import ingest_new_file
from src.utils import get_chat_history
import requests
from streamlit_lottie import st_lottie

load_dotenv()

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #2E86C1, #1ABC9C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .stSpinner > div {
        border-top-color: #2E86C1 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_medical = load_lottieurl("https://lottie.host/5a830953-6110-4fa4-8395-581d63603d36/8X9yU5Z2Z.json")
if not lottie_medical:
     lottie_medical = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_wakubj1j.json")


col1, col2 = st.columns([1, 4])
with col1:
    if lottie_medical:
        st_lottie(lottie_medical, height=100, key="header_logo")
    else:
        st.write("🩺")

with col2:
    st.markdown('<div class="main-header">Medical Assistant Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by RAG, Pinecone, and Llama 3.1</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Settings")
    
    st.info("Using Meta Llama 3.1 (via Hugging Face)")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF Medical Document", type=["pdf"])
    
    if uploaded_file:
        if st.button("Process Document"):
            processing_placeholder = st.empty()
            with processing_placeholder.container():
                st_lottie(lottie_medical, height=150, key="process_anim")
                st.info("Ingesting document... this may take a moment.")
            
            try:
                temp_path = os.path.join("data", "raw", uploaded_file.name)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                success, msg = ingest_new_file(temp_path)
                
                processing_placeholder.empty()
                
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            except Exception as e:
                processing_placeholder.empty()
                st.error(f"An error occurred during upload: {str(e)}")

    st.markdown("---")
    st.markdown("### How to use")
    st.write("1. Upload PDFs in the sidebar.")
    st.write("2. Adjust settings if needed.")
    st.write("3. Ask questions with context awareness!")

# --- Chat Logic ---

if "chain" not in st.session_state:
    init_placeholder = st.empty()
    with init_placeholder.container():
        st.info("Initializing Llama model... Please wait.")
        if lottie_medical:
            st_lottie(lottie_medical, height=150, key="init_anim")
    
    try:
        st.session_state.chain = get_rag_chain()
        init_placeholder.empty()
    except Exception as e:
        init_placeholder.empty()
        st.error(f"Failed to initialize model: {str(e)}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("👋 Hello! I'm your Medical AI Assistant. Upload a medical document to get started, or ask general questions (though I work best with context!).")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def stream_generator():
            try:
                history = get_chat_history(st.session_state.messages)
                response_stream = st.session_state.chain.stream({
                    "input": prompt,
                    "chat_history": history
                })
                
                for chunk in response_stream:
                    if "answer" in chunk:
                        yield chunk["answer"]
            except Exception as e:
                yield f"An error occurred while generating response: {str(e)}"
        
        response_text = st.write_stream(stream_generator())
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})