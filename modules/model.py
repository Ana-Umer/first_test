from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

def get_llm(vectotore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
    )
    retrieval_qa = vectotore.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(
        llm=llm,
        retriever=retrieval_qa
    )
