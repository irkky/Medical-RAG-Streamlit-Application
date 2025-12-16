import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.models import get_llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(model_type="llama"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    
    retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))
    # "k" means retrieve the top k most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

    llm = get_llm(model_type)

    system_prompt = (
        "You are an advanced medical assistant designed to help users understand complex medical documents. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "\n\n"
        "**Guidelines:**\n"
        "1. **Strict Context Adherence:** Answer ONLY based on the provided documents. If the answer is not in the context, say 'I cannot find this information in the documents.' Do not attempt to answer from outside knowledge.\n"
        "2. **Safety First:** Do not provide medical diagnoses or treatment recommendations. Always advise the user to consult a healthcare professional.\n"
        "3. **Clarity:** Explain medical jargon in simple terms if possible.\n"
        "4. **Format:** Use bullet points for lists (like symptoms or steps) to make the answer easy to read.\n"
        "\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext: {context}"),
        ("human", "{input}"),
    ])

    # RAG Chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["input"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    class RAGChainWrapper:
        def __init__(self, chain):
            self.chain = chain
        
        def invoke(self, inputs):
            if isinstance(inputs, dict) and "input" in inputs:
                query = inputs["input"]
            else:
                query = inputs
            answer = self.chain.invoke({"input": query})
            return {"answer": answer}
    
    return RAGChainWrapper(rag_chain)