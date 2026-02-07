from langchain_core.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader

#--------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="RAG Demo App", layout="wide")
st.title("📄 Retrieval Augmented Generation (RAG) App")

st.markdown(
    """
    This app demonstrates **RAG (Retrieval Augmented Generation)** using  
    document embeddings + vector search + LLM.
    """
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("⚙️ Configuration")

OPENAI_API_KEY = st.sidebar.text_input(
    "OpenAI API Key",
    type="password"
)

doc_type = st.sidebar.selectbox(
    "Select Document Type",
    ["Text (.txt)", "PDF (.pdf)"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a document",
    type=["txt", "pdf"]
)

# ------------------------------
# Load & Process Document
# ------------------------------
@st.cache_resource
def load_vectorstore(file_path, file_type):
    if file_type == "Text (.txt)":
        loader = TextLoader(file_path)
    else:
        loader = PyPDFLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ------------------------------
# Main Logic
# ------------------------------
if uploaded_file and OPENAI_API_KEY:
    with st.spinner("Processing document..."):
        os.makedirs("temp", exist_ok=True)
        file_path = f"temp/{uploaded_file.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        vectorstore = load_vectorstore(file_path, doc_type)

    st.success("✅ Document indexed successfully!")

    # LLM
    llm = OpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    st.subheader("🔎 Ask a question")
    query = st.text_input("Enter your query")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)

        st.markdown("### 🧠 Answer")
        st.write(result["result"])

        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)

else:
    st.info("⬅️ Upload a document and enter API key to begin")
