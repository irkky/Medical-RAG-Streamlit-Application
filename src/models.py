import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
def get_llm():
    print("--- Initializing Llama 3.1 (via Hugging Face) ---")
    
    llama_temperature = float(os.getenv("LLAMA_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("MAX_TOKENS", "512"))
    
    llm_engine = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=max_tokens,
        do_sample=False,
        temperature=llama_temperature,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    return ChatHuggingFace(llm=llm_engine)