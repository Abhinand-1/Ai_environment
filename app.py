import streamlit as st
import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI

# ------------------------------
# OpenAI API Key
# ------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found in Streamlit Secrets.")
    st.stop()

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Environmental RAG Assistant",
    layout="wide"
)

st.title("🌍 Environmental Remote Sensing RAG Assistant")

st.markdown(
    """
    This assistant answers **environmental condition queries** by:
    - Selecting the most suitable satellite index  
    - Providing the equation  
    - Recommending the best satellite  
    - Returning a computation template  

    The knowledge base is built from **fixed reference PDFs**.
    """
)

# ------------------------------
# Fixed PDF paths (from GitHub)
# ------------------------------
DECISION_PDF = "data/Satellite Spectral Indices Reference For Rag Models.pdf"
IMPLEMENTATION_PDF = "data/code_for traing.pdf"

if not os.path.exists(DECISION_PDF) or not os.path.exists(IMPLEMENTATION_PDF):
    st.error("❌ Reference PDFs not found in /data folder.")
    st.stop()

# ------------------------------
# Helper: Load PDF with metadata
# ------------------------------
def load_pdf(path, doc_type):
    loader = PyPDFLoader(path)
    pages = loader.load()

    docs = []
    for page in pages:
        docs.append(
            Document(
                page_content=page.page_content,
                metadata={"doc_type": doc_type}
            )
        )
    return docs

# ------------------------------
# Build Vector Store (cached)
# ------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    decision_docs = load_pdf(DECISION_PDF, "index_registry")
    impl_docs = load_pdf(IMPLEMENTATION_PDF, "implementation_reference")

    documents = decision_docs + impl_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0
    )
    chunked_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory="./rag_db"
    )

    return vectorstore

# ------------------------------
# RAG Prompt (from notebook)
# ------------------------------
RAG_PROMPT = """
You are a remote sensing expert.

Task:
- Identify the environmental condition
- Recommend ONE suitable index
- Provide its formula
- Recommend the best satellite
- Provide a computation template

Rules:
- Use ONLY the given context
- Do NOT invent indices or formulas
- Do NOT mix multiple indices
- Output must be VALID JSON

Context:
{context}

Question:
{question}

Return JSON only.
"""

# ------------------------------
# Initialize Vector Store
# ------------------------------
with st.spinner("🔄 Loading environmental knowledge base..."):
    vectorstore = build_vectorstore()

st.success("✅ Knowledge base loaded")

# Metadata-filtered retrievers
decision_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4, "filter": {"doc_type": "index_registry"}}
)

impl_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4, "filter": {"doc_type": "implementation_reference"}}
)

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ------------------------------
# User Query
# ------------------------------
st.subheader("🔎 Ask an environmental question")

query = st.text_input(
    "Example: How to assess vegetation stress using satellite data?"
)

if query:
    with st.spinner("🧠 Running RAG pipeline..."):
        decision_docs = decision_retriever.invoke(query)
        impl_docs = impl_retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in decision_docs + impl_docs]
        )

        prompt = RAG_PROMPT.format(
            context=context,
            question=query
        )

        response = llm.invoke(prompt)

        try:
            output = json.loads(response.content)
            st.success("✅ Structured Answer")
            st.json(output)
        except json.JSONDecodeError:
            st.error("⚠️ Model returned invalid JSON")
            st.text(response.content)
