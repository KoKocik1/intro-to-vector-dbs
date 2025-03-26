from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def basic_llm_chain(query):
    print("Loading data...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    chain = PromptTemplate.from_template(query) | llm | StrOutputParser()
    result = chain.invoke(input={})
    print(result)


if __name__ == "__main__":
    basic_llm_chain("What is vector database?")
