import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ---------------- UI ----------------
st.set_page_config(page_title="Environmental RAG Assistant", layout="wide")
st.title("🌍 Environmental RAG Assistant")

# ---------------- API KEY ----------------
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found")
    st.stop()

# ---------------- LOAD PDFs ----------------
pdf_paths = [
    "data/code_for traing.pdf",
    "data/Satellite Spectral Indices Reference For Rag Models.pdf"
]

documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())

# ---------------- SPLIT TEXT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

# ---------------- EMBEDDINGS + VECTOR STORE ----------------
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="rag_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------------- LLM (GPT-mini) ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=openai_api_key
)

# ---------------- QUERY ----------------
query = st.text_input("💬 Ask a question based on the documents:")

if query:
    with st.spinner("Searching documents..."):
        docs = retriever.get_relevant_documents(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an expert in remote sensing, GIS, and satellite data.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

    st.success("Answer:")
    st.write(response.content)
