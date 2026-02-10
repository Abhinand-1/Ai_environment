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
    st.error("❌ OpenAI API key not found")
    st.stop()

# ---------------- LOAD PDFs (FIXED PATHS) ----------------
pdf_files = [
    "code_for traing.pdf",
    "Satellite Spectral Indices Reference For Rag Models.pdf"
]

documents = []

for pdf in pdf_files:
    if not os.path.exists(pdf):
        st.error(f"❌ File not found: {pdf}")
        st.stop()

    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

st.success(f"Loaded {len(documents)} document pages")

# ---------------- SPLIT TEXT ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

# ---------------- CACHE VECTOR STORE ----------------
@st.cache_resource(show_spinner="Building vector database...")
def build_vectorstore(_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory="rag_db"
    )

vectorstore = build_vectorstore(chunks)
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
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are an expert in remote sensing, GIS, and satellite data.

Answer the question using ONLY the context below.
If the answer is not present, say "Information not found in the documents".

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)

    st.success("Answer:")
    st.write(response.content)


    # ---------------- SHOW SOURCES ----------------
    with st.expander("📚 View source document chunks"):
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content[:1200])
