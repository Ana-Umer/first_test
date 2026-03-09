from json import load

from fastapi import FastAPI, UploadFile, File ,Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.ingest import load_vectorstore
from modules.handle_pdf import save_uploaded_file
from modules.model import get_llm
from modules.query import query_chain
from modules.logger import logger
app = FastAPI(title="LABU RAG API")

app.add_middleware(
    CORSMiddleware,     
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.middleware("http")
async def catch_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
@app.post("/upload_pdf/")
async def upload_pdf(uploaded_file: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(uploaded_file)} files for upload")
        load.vectorstore(save_uploaded_file(uploaded_file))
        return {"message": f"Successfully uploaded {len(uploaded_file)} files"}
    except Exception as e:
        logger.error(f"Error occurred while uploading PDF: {e}")
        return JSONResponse(status_code=500, content={"message": "Error uploading PDF"})

@app.post("/ask")
async def ask_question(question: str):
    try:
        logger.info(f"Received question: {question}")
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from modules.ingest import PERSIST_DIRECTORY
        vectorestore = Chroma(
            collection_name="pdf_docs",
            embedding_function=HuggingFaceEmbeddings(sentence_transformer_model="sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=PERSIST_DIRECTORY
        )
        chain = get_llm(vectorestore)
        response = query_chain(chain, question)
        return response
    except Exception as e:
        logger.error(f"Error occurred while asking question: {e}")
        return JSONResponse(status_code=500, content={"message": "Error processing question"})

@app.get("/RAG")
def get_rag():
    return {"message": "Welcome to the RAG API"}