import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
load_dotenv()


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def custom_rag_pinecone(query):
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )

    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know.
    Don't make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}
    Question: {question}
    
    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_dict = {"context": vectorstore.as_retriever() | format_docs,
                "question": RunnablePassthrough()}

    rag_chain = rag_dict | custom_rag_prompt | llm
    result = rag_chain.invoke(query)
    print(result)


if __name__ == "__main__":
    custom_rag_pinecone("What is vector database?")
