import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_docs():
    data_path = os.path.join("data", "raw")
    
    if not os.path.exists(data_path):
        print(f"Error: Directory '{data_path}' not found. Please create 'data/raw/' and add PDFs.")
        return

    print(f"--- Loading Documents from {data_path} ---")
    
    loader = PyPDFDirectoryLoader(data_path)
    raw_docs = loader.load()
    
    if not raw_docs:
        print("No documents found in 'data/raw/'. Exiting.")
        return

    print(f"Loaded {len(raw_docs)} document pages.")

    print("--- Splitting Text ---")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(documents)} text chunks.")

    print("--- Embedding and Upserting to Pinecone ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not index_name:
        print("Error: PINECONE_INDEX_NAME not found in .env file.")
        return

    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    print("--- Ingestion Complete! Data is now in Pinecone. ---")

if __name__ == "__main__":
    ingest_docs()