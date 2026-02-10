import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Environmental RAG Assistant", layout="wide")
st.title("🌍 Environmental & Remote Sensing RAG Assistant")

# -------------------------------
# Load API Key (from Streamlit Secrets)
# -------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API Key not found.")
    st.stop()

# -------------------------------
# Load & Process Documents
# -------------------------------
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("data/docs.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunked_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory="./rag_db"
    )

    return vectorstore


vectorstore = load_vectorstore()

# -------------------------------
# LLM (GPT-mini)
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",   # ✅ GPT-mini model
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

# -------------------------------
# Retrieval QA Chain
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff"
)

# -------------------------------
# User Input
# -------------------------------
query = st.text_input("Ask a question about Remote Sensing, GIS, or Environmental Indices:")

if query:
    with st.spinner("🔍 Searching knowledge base..."):
        response = qa_chain.run(query)
    st.success("✅ Answer")
    st.write(response)
