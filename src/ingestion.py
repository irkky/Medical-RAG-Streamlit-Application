import os
import concurrent.futures
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone 

load_dotenv()

def load_pdf(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def ingest_docs():
    data_path = os.path.join("data", "raw")
    
    if not os.path.exists(data_path):
        print(f"Error: Directory '{data_path}' not found.")
        return

    print(f"--- Loading Documents from {data_path} ---")
    
    pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF documents found. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF files. Starting parallel loading...")
    
    raw_docs = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(load_pdf, pdf_files)
        for result in results:
            raw_docs.extend(result)
    
    if not raw_docs:
        print("No documents loaded. Exiting.")
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
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        print("Error: PINECONE_INDEX_NAME not found.")
        return

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    batch_size = 100
    total_docs = len(documents)
    
    print(f"Total documents to ingest: {total_docs}")
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}")
        
        try:
            vectorstore.add_documents(batch)
            print(f"Batch {i // batch_size + 1} added.")
        except Exception as e:
            print(f"Error uploading batch {i // batch_size + 1}: {e}")
            
            
    print("--- Ingestion Complete! ---")

def ingest_new_file(file_path):
    print(f"--- Ingesting new file: {file_path} ---")
    
    raw_docs = load_pdf(file_path)
    if not raw_docs:
        return False, "Failed to load PDF."
        
    print(f"Loaded {len(raw_docs)} pages.")

    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(raw_docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    try:
        vectorstore.add_documents(documents)
        print("File ingested successfully.")
        return True, f"Successfully ingested {len(documents)} chunks."
    except Exception as e:
        print(f"Error ingesting file: {e}")
        return False, str(e)

if __name__ == "__main__":
    ingest_docs()