from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import os


prompt_template = PromptTemplate.from_template("""
You are an intelligent onboarding assistant. Use the following context to answer the user's question.
Answer only in concise, clear bullet points using "•" for each point.
Do not include any document names, numbers, or metadata.

Context:
{context}

Queey:
{question}

Answer in bullet points:
""")


def get_rag_chain():
    persist_directory = "chroma_db"
    docs_path = "data/uploaded_docs"

    all_docs = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, filename))
            all_docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key="AIzaSyBBgGb7teAEPg6EkZphpMNH2OVzdbEYCpA"
    )

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain


def chain_run(inputs):
    chain = get_rag_chain()
    response = chain({"query": inputs["query"]})
    return response
