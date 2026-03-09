from fastapi import FastAPI
from models import QueryRequest, QueryResponse
from setup import ask_question

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):

    answer = ask_question(request.question)

    return QueryResponse(answer=answer)