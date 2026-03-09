
from os import path

from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load document
PERSIST_DIRECTORY = "/vector_store"
UPLOAD_DIRECTORY = "/uploads_pdfs"
def load_vectorstore(uploaded_file):
    file_path=[]
    for file in uploaded_file:
        saved_path =os.path.join(UPLOAD_DIRECTORY, file.filename)
        with open(saved_path, "wb") as f:
            f.write(file.file.read())
        file_path.append(saved_path)
    documents = []
    for file in file_path:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(sentence_transformer_model="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        vectorstore = Chroma(collection_name="pdf_docs", embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)
        vectorstore.add_documents(split_docs)
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=PERSIST_DIRECTORY, collection_name="pdf_docs")
    return vectorstore  