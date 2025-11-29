# RAG_Sanskrit_YourName: Sanskrit RAG System

## Objective
End-to-end RAG for Sanskrit docs (e.g., folktales like "मुखभूयायः"). Upload .txt/PDF → Chunk/index → Q&A in Sanskrit/IAST. CPU-only.

## Setup
1. `pip install -r requirements.txt`
2. Add `.env` with Groq key.
3. Backend: `uvicorn code.app:app --reload` (port 8000).
4. Dashboard: `streamlit run code/ui.py` (port 8501).
5. Upload sample: Paste your Sanskrit text → "Upload & Ingest" → Query.

## Architecture Flow
1. **Ingest**: Loader (PyPDF/Text) → Clean (Unicode/regex for Devanagari) → Chunk (Recursive, 800 chars, overlap 100, separators ॥/।) → Embed (multilingual MiniLM) → Index (Chroma).
2. **Retrieval**: Semantic (k=5, cosine sim.).
3. **Generation**: Groq Llama3 (temp=0.1) with structured prompt (summary/translation/context/themes).
- Modularity: rag_chain.py (retriever | prompt | LLM).

## Usage
- Upload your pasted "मुखभूयायः" text as .txt.
- Query: "मुखभूयायः कथा" → Detailed response.
- Dashboard: Metrics (recall/latency), resources (CPU/RAM).

## Performance (Tested on i5 CPU)
| Metric | Value | Notes |
|--------|-------|-------|
| Latency | 1-4s/query | Groq fast; +1s for detail. |
| Accuracy (Recall@5) | 90%+ | Manual: Full verse retrieval for "शंखनाद". |
| Resources | <2GB RAM, CPU 20% | No GPU. |

## Troubleshooting
- 0 chunks? Re-upload/ingest.
- Local LLM: Uncomment HF in rag_chain.py.

For report: Run "Generate Report PDF" in UI.