import streamlit as st
import os
from ingest import ingest
from rag_chain import chain_run 

UPLOAD_DIR = "data/uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Onboarding RAG - Gemini", layout="centered")
st.title("📄 Onboarding Assistant (Gemini + Local Embeddings)")


uploaded_files = st.file_uploader(
    "📤 Upload Onboarding Documents (PDF/DOCX):",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.success("✅ Documents uploaded.")
    if st.button("🔄 Ingest & Generate Embeddings"):
        ingest()
        st.success("✅ Documents ingested into Chroma vector DB.")

st.markdown("---")
st.header("🧠 Ask a Question About Your Docs")

user_input = st.text_input("🔍 Ask a question:")

if user_input:
    response = chain_run({"query": user_input})
 
    st.subheader("📌 Answer:")
    st.markdown(response["result"]) 

