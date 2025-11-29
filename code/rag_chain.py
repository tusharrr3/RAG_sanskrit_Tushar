from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db_path = "../chroma_db"
if not os.path.exists(db_path):
    print("❌ DB missing—run ingest!")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=db_path)
else:
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print(f"DB loaded: {len(vectorstore.get()['ids'])} chunks.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(temperature=0.1, model_name="qwen/qwen3-32b")

prompt = ChatPromptTemplate.from_template(
    """You are a Sanskrit expert. Provide detailed response using context:
    1. Summary. 2. Translation. 3. Context. 4. Themes.

    Context: {context}
    Query: {query}
    Answer:"""
)

def format_docs(docs):
    if not docs:
        print("⚠️ No retrieval.")
        return "No context."
    formatted = ""
    for i, doc in enumerate(docs):
        score = doc.metadata.get('score', 'N/A') if doc.metadata else 'N/A'
        print(f"🔍 Chunk {i+1}: Score {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print("-" * 50)
        formatted += f"\n--- Chunk {i+1} ---\n{doc.page_content}"
    context_len = len(formatted.split())
    print(f"Total: {len(docs)} chunks | {context_len} words")
    return formatted

rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

def query_rag(question: str):
    try:
        docs = retriever.invoke(question)
        context_len = sum(len(doc.page_content.split()) for doc in docs)
        answer = rag_chain.invoke(question)
        return {"answer": answer, "context_words": context_len}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "context_words": 0}

if __name__ == "__main__":
    print(query_rag("मुखभूयायः कथा"))