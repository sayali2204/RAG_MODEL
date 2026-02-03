import os
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def build_rag_chain():
    # 1️⃣ Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    # 2️⃣ Vector Store (Pinecone)
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3️⃣ LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.5
    )

    # 4️⃣ Prompt Template
    prompt = ChatPromptTemplate.from_template("""
You are a Deep Learning expert.

Answer the question ONLY using the context below.
If the answer is not present, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}
""")

    # 5️⃣ LCEL RAG Chain
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def chat():
    rag_chain, retriever = build_rag_chain()

    while True:
        question = input("\nAsk me anything → ")
        if question.lower() in ["exit", "quit"]:
            break

        # Retrieve source documents
        source_docs = retriever.invoke(question)

        # Generate answer
        answer = rag_chain.invoke(question)

        print("\nAnswer:\n", answer)

        print("\nSources:")
        for doc in source_docs:
            print("-", doc.metadata.get("source", "Unknown"))


if __name__ == "__main__":
    chat()
