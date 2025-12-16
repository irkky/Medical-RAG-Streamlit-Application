import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_type="llama"):
    if model_type == "llama":
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

    elif model_type == "gemini":
        print("--- Initializing Google Gemini ---")
        
        gemini_temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))
        
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=gemini_temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    else:
        raise ValueError(f"Invalid model type: {model_type}. Choose 'llama' or 'gemini'.")