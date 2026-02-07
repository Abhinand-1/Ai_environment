import os
import json
import streamlit as st

# ------------------------------
# Disable telemetry (important for Streamlit Cloud)
# ------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["CHROMA_TELEMETRY"] = "false"

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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
    - Identifying the environmental condition  
    - Selecting the most suitable satellite index  
    - Providing the mathematical equation  
    - Recommending the appropriate satellite sensor  
    - Returning a computation template  

    The system is based on a **Retrieval-Augmented Generation (RAG)** framework
    grounded in **fixed scientific reference documents**.
    """
)

# ------------------------------
# PDF paths
# ------------------------------
DOMAIN_PDF = "Satellite Spectral Indices Reference For Rag Models.pdf"
EXECUTION_PDF = "code_for traing.pdf"

if not os.path.exists(DOMAIN_PDF) or not os.path.exists(EXECUTION_PDF):
    st.error("❌ Reference PDFs not found in the repository.")
    st.stop()

# ------------------------------
# Load PDF with metadata
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
# Build Vector Store (IN-MEMORY, CLOUD SAFE)
# ------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    domain_docs = load_pdf(DOMAIN_PDF, "domain_knowledge")
    execution_docs = load_pdf(EXECUTION_PDF, "execution_reference")

    documents = domain_docs + execution_docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=30
    )
    chunked_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings
    )

    return vectorstore

# ------------------------------
# RAG Prompt
# ------------------------------
RAG_PROMPT = """
You are a remote sensing and environmental analysis expert.

Your task:
1. Identify the environmental condition
2. Recommend ONE suitable satellite index
3. Provide its mathematical formula
4. Recommend the most appropriate satellite sensor
5. Provide a computation template

STRICT SCIENTIFIC RULES:
- Use ONLY the provided context
- Do NOT invent indices, equations, or satellites
- Do NOT mix multiple indices
- Flood or surface water problems MUST use water-related indices
- Built-up indices are NOT valid for flood or water assessment
- Output MUST be valid JSON only
- Be scientifically conservative and precise

====================
DOMAIN KNOWLEDGE
====================
{domain_context}

====================
EXECUTION REFERENCE
====================
{execution_context}

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

# ------------------------------
# Retrievers
# ------------------------------
domain_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4, "filter": {"doc_type": "domain_knowledge"}}
)

execution_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4, "filter": {"doc_type": "execution_reference"}}
)

# ------------------------------
# LLM
# ------------------------------
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
    "Example: How can flood risk be assessed using satellite data?"
)

if query:
    with st.spinner("🧠 Running RAG pipeline..."):

        # Soft retrieval hint (NOT rule-based)
        domain_query = query + " water flood drought vegetation index"

        domain_docs = domain_retriever.invoke(domain_query)
        execution_docs = execution_retriever.invoke(query)

        domain_context = "\n\n".join(doc.page_content for doc in domain_docs)
        execution_context = "\n\n".join(doc.page_content for doc in execution_docs)

        prompt = RAG_PROMPT.format(
            domain_context=domain_context,
            execution_context=execution_context,
            question=query
        )

        response = llm.invoke(prompt)

        try:
            output = json.loads(response.content)
            st.success("✅ Structured Scientific Answer")
            st.json(output)
        except json.JSONDecodeError:
            st.error("⚠️ Model returned invalid JSON")
            st.text(response.content)
