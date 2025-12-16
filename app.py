import streamlit as st
from dotenv import load_dotenv
from src.rag_chain import get_rag_chain

load_dotenv()

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺"
)

st.title("🩺 Medical Assistant Chatbot")
st.caption("Powered by RAG, Pinecone, and Llama 3.1 / Gemini")

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio(
        "Choose your LLM:",
        ["llama", "gemini"],
        captions=["Meta Llama 3.1 (via Hugging Face)", "Google Gemini 1.5 Flash"]
    )
    st.markdown("---")
    st.markdown("### How to use")
    st.write("1. Upload PDFs to `data/raw/` and run `ingestion.py`.")
    st.write("2. Select a model above.")
    st.write("3. Ask a question about your documents.")

# We store the chain in session state so we don't reload it on every interaction
if "chain" not in st.session_state or st.session_state.get("current_model") != model_choice:
    with st.spinner(f"Initializing {model_choice} model..."):
        st.session_state.chain = get_rag_chain(model_type=model_choice)
        st.session_state.current_model = model_choice

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            response = st.session_state.chain.invoke({"input": prompt})
            answer = response["answer"]
            
            message_placeholder.markdown(answer)
            
            context_docs = response["context"]
            
            sources = set()
            for doc in context_docs:
                page_num = doc.metadata.get("page", "Unknown")
                source_file = doc.metadata.get("source", "Unknown").split("/")[-1]
                sources.add(f"{source_file} (Page {page_num})")
            
            with st.expander("📚 View Sources"):
                for source in sources:
                    st.write(f"- {source}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            message_placeholder.markdown(f"Error: {str(e)}")