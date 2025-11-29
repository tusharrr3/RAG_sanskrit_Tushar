from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import query_rag

app = FastAPI(title="Sanskrit RAG API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Ready for Sanskrit Q&A!"}

@app.post("/query")
def ask(request: QueryRequest):
    result = query_rag(request.question)
    return {"question": request.question, "answer": result["answer"], "context_words": result["context_words"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)