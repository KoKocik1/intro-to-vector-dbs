from chains.basic_llm_chain import basic_llm_chain
from chains.retrieval_qa_pipeline import retrieval_qa_pipeline
from chains.custom_rag_pinecone import custom_rag_pinecone


if __name__ == "__main__":
    query = "What is vector database?"
    basic_llm_chain(query)
    # retrieval_qa_pipeline(query)
    # custom_rag_pinecone(query)
