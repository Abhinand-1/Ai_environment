import streamlit as st
import ee
import leafmap.foliumap as geemap
import json

from geopy.geocoders import Nominatim

# ===== LangChain (modern imports) =====
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate


# =================================================
# STREAMLIT CONFIG
# =================================================
st.set_page_config(page_title="AI Environmental Analysis", layout="wide")
st.title("🌍 AI-Powered Environmental Analysis (RAG + GEE)")
st.caption("Natural language → Satellite maps, charts & explanation")


# =================================================
# EARTH ENGINE INITIALIZATION (SERVICE ACCOUNT)
# =================================================
@st.cache_resource
def initialize_gee():
    credentials = ee.ServiceAccountCredentials(
        st.secrets["GEE_SERVICE_ACCOUNT"],
        key_data=st.secrets["GEE_PRIVATE_KEY"]  # string JSON (correct)
    )
    ee.Initialize(credentials)

    # health check
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).getInfo()

initialize_gee()


# =================================================
# LOAD & INDEX RAG DOCUMENTS
# =================================================
@st.cache_resource
def load_vectorstore():
    loaders = [
        PyPDFLoader("data/Satellite_Indices_RAG.pdf"),
        PyPDFLoader("data/GEE_Code_Reference.pdf"),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    return FAISS.from_documents(chunks, embeddings)

vectorstore = load_vectorstore()


# =================================================
# LLM + PROMPT
# =================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

science_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are an environmental remote sensing expert.

Using ONLY the context below, decide:
- appropriate spectral index
- bands
- sensor
- visualization parameters

Return STRICT JSON ONLY (no explanation text):

{{
  "index": "NDVI",
  "index_type": "normalized_difference",
  "bands": ["B8", "B4"],
  "sensor": "Sentinel-2",
  "visualization": {{
    "min": -0.2,
    "max": 0.8,
    "palette": ["blue", "white", "green"]
  }}
}}

Context:
{context}

Query:
{query}
"""
)


# =================================================
# RAG DECISION FUNCTION (ROBUST)
# =================================================
def get_plan_from_query(query, retries=2):
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = science_prompt.format(context=context, query=query)

    for attempt in range(retries + 1):
        response = llm.invoke(prompt).content.strip()

        # try to extract JSON safely
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            return json.loads(response[start:end])
        except Exception:
            if attempt == retries:
                st.error("❌ AI failed to return valid JSON. Please rephrase your query.")
                st.stop()


# =================================================
# LOCATION → GEOMETRY (ROBUST)
# =================================================
def get_geometry_from_location(location_name):
    # simple normalization for common mistakes
    location_name = location_name.lower()
    location_name = location_name.replace("wayand", "wayanad")

    geolocator = Nominatim(user_agent="rag-gee-app")
    loc = geolocator.geocode(location_name, timeout=10)

    if loc is None:
        st.error("❌ Location not found. Please be more specific.")
        st.stop()

    st.info(f"📍 Location identified: **{loc.address}**")

    return ee.Geometry.Point([loc.longitude, loc.latitude]).buffer(20000)


# =================================================
# GEE ANALYSIS ENGINE
# =================================================
def run_environmental_analysis(plan, geometry, year):
    index = plan["index"]
    bands = plan["bands"]

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )

    def compute_index(image):
        return image.normalizedDifference(bands).rename(index)

    indexed = collection.map(compute_index)
    mean_img = indexed.mean().clip(geometry)

    Map = geemap.Map()
    Map.center_object(geometry, zoom=9)   # ✅ FIXED
    Map.addLayer(mean_img, plan["visualization"], f"{index} {year}")

    chart = geemap.chart.image.series(
        indexed,
        geometry,
        ee.Reducer.mean(),
        scale=10
    )

    return Map, chart



# =================================================
# STREAMLIT UI
# =================================================
query = st.text_input(
    "Enter your analysis request",
    "Analyze vegetation condition in Wayanad district, Kerala, India for 2023"
)

year = st.selectbox(
    "Select Year",
    ["2021", "2022", "2023", "2024"],
    index=2
)

if st.button("🚀 Run Analysis"):
    with st.spinner("Running RAG + Satellite analysis..."):
        location = query.split("in")[-1].split("for")[0].strip()

        geometry = get_geometry_from_location(location)
        plan = get_plan_from_query(query)

        m, chart = run_environmental_analysis(plan, geometry, year)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗺️ Spatial Map")
            geemap.static_map(m)

        with col2:
            st.subheader("📈 Time Series")
            st.pyplot(chart)

        st.subheader("🧠 Selected Index & Parameters")
        st.json(plan)

        st.success("✅ Analysis completed successfully")
