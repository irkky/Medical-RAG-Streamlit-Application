import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.models import get_llm

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    
    retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})

    llm = get_llm()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

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

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    class RAGChainWrapper:
        def __init__(self, chain):
            self.chain = chain
        
        def _prepare_input(self, inputs):
            if isinstance(inputs, str):
                return {"input": inputs, "chat_history": []}
            if isinstance(inputs, dict):
                if "input" not in inputs:
                    raise ValueError("Input dict must contain 'input' key")
                if "chat_history" not in inputs:
                    inputs["chat_history"] = []
                return inputs
            raise ValueError(f"Invalid input type: {type(inputs)}")

        def invoke(self, inputs):
            prepared_input = self._prepare_input(inputs)
            return self.chain.invoke(prepared_input)
            
        def stream(self, inputs):
            prepared_input = self._prepare_input(inputs)
            return self.chain.stream(prepared_input)
    
    return RAGChainWrapper(rag_chain)