import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("Loading data...")
    loader = TextLoader(
        "/Users/krzysztofkokot/Projects/LLM/myOwnProject/intro-to-vector-dbs/mediumblog1.txt"
    )
    documents = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )
    print("Data ingested successfully.")
