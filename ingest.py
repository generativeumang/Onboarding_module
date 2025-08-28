import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google.generativeai import configure, embed_content
from tqdm import tqdm


load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY"))

DOC_FOLDER = "data/uploaded_docs"

def load_docs():
    all_docs = []
    for file in os.listdir(DOC_FOLDER):
        path = os.path.join(DOC_FOLDER, file)
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(path)
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)


class GeminiEmbeddingFunction:
    def embed_documents(self, texts):
        embeddings = []
        for t in tqdm(texts, desc="[Embedding chunks with Gemini]"):
            try:
                emb = embed_content(
                    model="models/embedding-001",
                    content=t,
                    task_type="retrieval_document"
                )["embedding"]
                embeddings.append(emb)
            except Exception as e:
                print(f"[ERROR] Embedding failed for chunk: {e}")
                embeddings.append([0.0] * 768) 
            time.sleep(1.2) 
        return embeddings

    def embed_query(self, text):
        time.sleep(1.2)
        return embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

def store_chunks(chunks):
    embedding = GeminiEmbeddingFunction()
    vectordb = Chroma.from_documents(chunks, embedding, persist_directory="vectorstore")
    vectordb.persist()

def ingest():
    docs = load_docs()
    print(f"[INFO] Loaded {len(docs)} documents.")
    chunks = chunk_docs(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    store_chunks(chunks)
    print("[INFO] Embeddings stored using Gemini and saved in Chroma.")

if __name__ == "__main__":
    ingest()
