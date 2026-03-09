
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

qa_chain = None

def ask_question(question: str):
    global qa_chain
    if qa_chain is None:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file or environment.")

        # Sample documents
        docs = [
            Document(page_content="Python is a programming language."),
            Document(page_content="FastAPI is a modern web framework for APIs."),
            Document(page_content="RAG stands for Retrieval Augmented Generation."),
        ]

        # Split documents
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        documents = splitter.split_documents(docs)

        # Embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Vector database
        vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="./vector_store")

        # Retriever
        retriever = vectorstore.as_retriever()

        # LLM
        llm = ChatGroq( model="llama-3.1-8b-instant",)

        # RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

    try:
        response = qa_chain.run(question)
        return response
    except Exception as e:
        return f"Error processing question: {str(e)}"

if __name__ == "__main__":
    question = "What is Python?"
    print(ask_question(question))