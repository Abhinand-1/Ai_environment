import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Environmental RAG Assistant", layout="wide")
st.title("🌍 RAG Chat with Your Satellite & Code Docs")

# ---- Load API key ----
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key missing!")
    st.stop()

# ---- Load your PDFs ----
pdf_files = [
    "data/code_for traing.pdf",
    "data/Satellite Spectral Indices Reference For Rag Models.pdf"
]

documents = []
for path in pdf_files:
    try:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    except Exception as e:
        st.error(f"Error loading {path}: {e}")

if not documents:
    st.error("No documents loaded")
    st.stop()

# ---- Split into chunks ----
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunked_docs = splitter.split_documents(documents)

# ---- Build vector store ----
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings,
    persist_directory="rag_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---- Setup LLM (GPT-mini) ----
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

# ---- UI query box ----
query = st.text_input("💬 Ask a question about these docs:")

if query:
    with st.spinner("Searching..."):
        answer = qa_chain.run(query)
    st.write("**Answer:**")
    st.write(answer)
