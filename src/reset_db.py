import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def reset_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    index = pc.Index(index_name)
    
    try:
        index.delete(delete_all=True, namespace="")
        print(f"✅ Successfully deleted all vectors from index '{index_name}'.")
    except Exception as e:
        print(f"Error deleting vectors: {e}")

if __name__ == "__main__":
    confirm = input("⚠️  Are you sure you want to delete ALL data? (yes/no): ")
    if confirm.lower() == "yes":
        reset_pinecone_index()
    else:
        print("Operation cancelled.")