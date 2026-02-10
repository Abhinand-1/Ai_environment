import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Environmental RAG Assistant", layout="wide")
st.title("🌍 Environmental Indices RAG Assistant")
st.write("Ask questions based on the uploaded satellite & GIS reference documents.")

# -----------------------------
# OpenAI API Key
# -----------------------------
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ OpenAI API key not found. Add it in Streamlit Secrets.")
    st.stop()

# -----------------------------
# Load PDFs
# -----------------------------
DATA_PATH = "data"
PERSIST_DIR = "rag_db"

pdf_files = [
    "code_for traing.pdf",
    "Satellite Spectral Indices Reference For Rag Models.pdf"
]

documents = []

for pdf in pdf_files:
    loader = PyPDFLoader(os.path.join(DATA_PATH, pdf))
    documents.extend(loader.load())

# -----------------------------
# Split Documents
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunked_docs = text_splitter.split_documents(documents)

# -----------------------------
# Embeddings
# -----------------------------
embeddings = OpenAIEmbeddings(
    openai_api_key=openai_api_key
)

# -----------------------------
# Chroma Vector Store (Persistent)
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# GPT-mini Model
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# -----------------------------
# User Query
# -----------------------------
query = st.text_input("💬 Ask your question:")

if query:
    with st.spinner("Searching documents..."):
        response = qa_chain.run(query)
    st.success("Answer:")
    st.write(response)
