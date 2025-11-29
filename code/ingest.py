import os
import re
import unicodedata
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Sample pasted text (your doc—auto-save if no file)
full_pasted_text = """मुखभूयायः

"अरे शंखनाद, गृहाणं शंकरम्।" इत्युक्त्वा गोविन्ददासः आदिश्य तत् । ततः शंखनादः आग्रहणं गच्छति, शंकरं जीह्वे वर्तते । ततः गोविन्ददासः क्रोधेन शंखनादं वदति, "अरे मूढ, कुतस्त इति शंकरः ? शंकरं ददकं एवम जीह्वेन वर्ततेन न एव अनयति कदापि । इति परं कमपि वर्ततु जातं ढायाम् सत्क्रकायाम् नोच्छ्वासयानय च" इति । [full story... up to verse]

चतुरः कालिदासः [full bilingual tale...]

वृद्धयाः चातुर्यम् [intro...]"""

data_path = "./data/sanskrit_stories.txt"
os.makedirs("./data", exist_ok=True)
if not os.path.exists(data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(full_pasted_text)
    print("✅ Auto-created sample with pasted doc!")

# Load (generic for upload)
loader = TextLoader(data_path) if data_path.endswith('.txt') else PyPDFLoader(data_path)
docs = loader.load()

def clean_sanskrit(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[�◌्lkmpotn।।।]', ' ', text)
    text = re.sub(r'[^\u0900-\u097Fa-zA-Z\s।।.,!?;॥]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

for doc in docs:
    doc.page_content = clean_sanskrit(doc.page_content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100,
    separators=["॥", "।।", "।", "\n\n", "\n", "."]
)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="../chroma_db"
)
print(f"✅ Indexed {len(chunks)} chunks! Ready for RAG.")