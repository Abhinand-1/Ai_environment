
import streamlit as st
import ee
import geemap
import json
import re
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Geospatial Assistant",
    layout="wide"
)

st.title("🌍 AI Geospatial Analysis Assistant")
st.write("Ask a natural language query to analyze satellite data.")

# -----------------------------
# EARTH ENGINE INIT
# -----------------------------
import json
import ee
import streamlit as st

service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
private_key = st.secrets["GEE_PRIVATE_KEY"]

key_dict = {
    "type": "service_account",
    "client_email": service_account,
    "private_key": private_key,
    "token_uri": "https://oauth2.googleapis.com/token"
}

credentials = ee.ServiceAccountCredentials(
    service_account,
    key_data=json.dumps(key_dict)
)

ee.Initialize(credentials)

# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# KNOWLEDGE BASE
# -----------------------------
knowledge_base = [

{
"domain": "drought",
"text": """
NDMI (Normalized Difference Moisture Index)

Purpose:
Detects vegetation moisture stress and drought conditions.

Formula:
(NIR - SWIR) / (NIR + SWIR)

Satellite:
Sentinel-2

Bands:
NIR: B8
SWIR: B11
"""
},

{
"domain": "vegetation",
"text": """
NDVI (Normalized Difference Vegetation Index)

Purpose:
Measures vegetation health and greenness.

Formula:
(NIR - Red) / (NIR + Red)

Satellite:
Sentinel-2

Bands:
NIR: B8
Red: B4
"""
},

{
"domain": "urban",
"text": """
NDBI (Normalized Difference Built-up Index)

Purpose:
Detects built-up areas.

Formula:
(SWIR - NIR) / (SWIR + NIR)

Satellite:
Sentinel-2

Bands:
SWIR: B11
NIR: B8
"""
},

{
"domain": "water",
"text": """
NDWI (Normalized Difference Water Index)

Purpose:
Detects surface water bodies.

Formula:
(Green - NIR) / (Green + NIR)

Satellite:
Sentinel-2

Bands:
Green: B3
NIR: B8
"""
}

]

# -----------------------------
# VECTOR SEARCH (RAG)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [doc["text"] for doc in knowledge_base]

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

def retrieve_context(query, k=2):

    query_vector = model.encode([query])

    D, I = index.search(query_vector, k)

    results = [documents[i] for i in I[0]]

    return results


# -----------------------------
# PLAN GENERATION
# -----------------------------
def generate_plan(query):

    context = retrieve_context(query)

    prompt = f"""
You are a remote sensing expert.

Choose the correct satellite index based on the query.

Return ONLY JSON:

{{
 "index": "",
 "satellite": "",
 "bands": []
}}

Context:
{context}

Query:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    json_text = re.search(r'\{.*\}', result, re.DOTALL).group()

    plan = json.loads(json_text)

    DATASETS = {
        "Sentinel-2": "COPERNICUS/S2_SR_HARMONIZED",
        "MODIS": "MODIS/061/MOD11A2",
        "Sentinel-5P": "COPERNICUS/S5P/OFFL/L3_NO2"
    }

    plan["collection"] = DATASETS.get(plan["satellite"])

    return plan


# -----------------------------
# METADATA EXTRACTION
# -----------------------------
def extract_metadata(query):

    prompt = f"""
Extract metadata.

Return JSON:

location
start_date
end_date
analysis_type

Query:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    json_text = re.search(r'\{.*\}', result, re.DOTALL).group()

    data = json.loads(json_text)

    return data


# -----------------------------
# REGION EXTRACTION
# -----------------------------
def get_roi(location):

    districts = ee.FeatureCollection("FAO/GAUL/2015/level2")

    roi_fc = districts.filter(
        ee.Filter.Or(
            ee.Filter.eq("ADM2_NAME", location),
            ee.Filter.eq("ADM1_NAME", location),
            ee.Filter.eq("ADM0_NAME", location)
        )
    )

    roi = ee.Algorithms.If(
        roi_fc.size().gt(0),
        roi_fc.geometry(),
        ee.Geometry.Point([76.4, 9.6]).buffer(30000)
    )

    return ee.Geometry(roi)


# -----------------------------
# ANALYSIS ENGINE
# -----------------------------
def run_analysis(plan, region, start, end):

    index_name = plan["index"]

    if index_name == "LST":

        collection = (
            ee.ImageCollection("MODIS/061/MOD11A2")
            .filterDate(start, end)
            .filterBounds(region)
            .select("LST_Day_1km")
        )

        image = collection.mean()

        lst = image.multiply(0.02).subtract(273.15).rename("LST")

        return lst.clip(region)

    elif index_name == "NO2":

        collection = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
            .filterDate(start, end)
            .filterBounds(region)
            .select("tropospheric_NO2_column_number_density")
        )

        image = collection.mean().rename("NO2")

        return image.clip(region)

    else:

        collection = (
            ee.ImageCollection(plan["collection"])
            .filterDate(start, end)
            .filterBounds(region)
            .median()
        )

        bands = plan["bands"]

        index_img = collection.normalizedDifference(bands).rename(index_name)

        return index_img.clip(region)


# -----------------------------
# VISUALIZATION
# -----------------------------
def visualize(index_img, region, index_name):

    Map = geemap.Map()
    Map.centerObject(region, 7)

    palettes = {
        "NDVI": ["brown","yellow","green"],
        "NDMI": ["8c510a","d8b365","f6e8c3","c7eae5","5ab4ac","01665e"],
        "NDWI": ["white","cyan","blue"],
        "NDBI": ["black","gray","red"],
        "LST": ["blue","cyan","yellow","orange","red"],
        "NO2": ["blue","cyan","yellow","orange","red"]
    }

    palette = palettes.get(index_name)

    vis = {"min": -1, "max": 1, "palette": palette}

    if index_name == "LST":
        vis = {"min": 20, "max": 45, "palette": palette}

    if index_name == "NO2":
        vis = {"min": 0, "max": 0.0002, "palette": palette}

    Map.addLayer(index_img.clip(region), vis, index_name)

    Map.addLayer(region, {}, "ROI")

    Map.addLayerControl()

    return Map


# -----------------------------
# STREAMLIT UI
# -----------------------------
query = st.text_input(
    "Enter your analysis query",
    "Monitor vegetation health in Karnataka during 2023"
)

if st.button("Run Analysis"):

    with st.spinner("Running AI geospatial analysis..."):

        metadata = extract_metadata(query)

        location = metadata["location"]
        start = metadata["start_date"]
        end = metadata["end_date"]
        analysis = metadata["analysis_type"]

        plan = generate_plan(analysis)

        roi = get_roi(location)

        index_img = run_analysis(plan, roi, start, end)

        Map = visualize(index_img, roi, plan["index"])

        Map.to_streamlit(height=600)

