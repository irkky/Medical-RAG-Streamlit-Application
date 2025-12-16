try:
    from langchain.chains.retrieval import create_retrieval_chain
    print("Found create_retrieval_chain in langchain.chains.retrieval")
except Exception as e:
    print(f"Not in langchain.chains.retrieval: {e}")

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    print("Found create_stuff_documents_chain in langchain.chains.combine_documents")
except Exception as e:
    print(f"Not in langchain.chains.combine_documents: {e}")

try:
    from langchain_core.runnables import RunnablePassthrough
    print("Found RunnablePassthrough in langchain_core.runnables")
except Exception as e:
    print(f"Not in langchain_core.runnables: {e}")
