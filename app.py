

# ---- cell ----

import ee
import geemap
import streamlit as st
from openai import OpenAI
import json

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



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

ee.Initialize(credentials, project="civic-nation-478705-d3")

# ---- cell ----

"""
Create Satellite Index Knowledge Dataset
"""

# ---- cell ----

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

Keywords:
drought, vegetation moisture, water stress
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

Keywords:
vegetation health, greenness, biomass
"""
},

{
"domain": "urban",
"text": """
NDBI (Normalized Difference Built-up Index)

Purpose:
Detects built-up and urban areas.

Formula:
(SWIR - NIR) / (SWIR + NIR)

Satellite:
Sentinel-2

Bands:
SWIR: B11
NIR: B8

Keywords:
urban expansion, built-up area
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

Keywords:
water mapping, lake detection
"""
}

]

# ---- cell ----

"""
Build Embeddings + FAISS Vector Store
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [doc["text"] for doc in knowledge_base]

embeddings = model.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# ---- cell ----

"""
RAG Retrieval Function
"""

def retrieve_context(query, k=2):

    query_vector = model.encode([query])

    D, I = index.search(query_vector, k)

    results = [documents[i] for i in I[0]]

    return results

# ---- cell ----

"""
Query Intent Classifier
"""

def classify_intent(query):

    query = query.lower()

    if "drought" in query or "moisture" in query:
        return "drought"

    if "vegetation" in query or "crop" in query:
        return "vegetation"

    if "urban" in query or "built-up" in query:
        return "urban"

    if "water" in query or "lake" in query:
        return "water"

    return "unknown"

# ---- cell ----

"""
Generate Analysis Plan
"""

def generate_plan(query):

    context = retrieve_context(query)

    prompt = f"""
You are a remote sensing expert.

Choose the correct satellite index based on the query.

Rules:
Drought → NDMI
Vegetation → NDVI
Urban → NDBI
Water → NDWI

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
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    import re

    json_text = re.search(r'\{.*\}', result, re.DOTALL).group()

    plan = json.loads(json_text)

    DATASETS = {
        "Sentinel-2": "COPERNICUS/S2_SR_HARMONIZED",
        "MODIS": "MODIS/061/MOD11A2",
        "Sentinel-5P": "COPERNICUS/S5P/OFFL/L3_NO2"
    }

    satellite = plan.get("satellite")

    if satellite not in DATASETS:
        raise ValueError(f"Unsupported satellite returned by LLM: {satellite}")

    plan["collection"] = DATASETS[satellite]

    return plan
# ---- cell ----

"""
Extract Location and Date
"""

def extract_metadata(query):

    prompt = f"""
Extract:

location
start_date
end_date
analysis_type

Return JSON only.

Query:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    result = response.choices[0].message.content.strip()

    import re

    json_text = re.search(r'\{.*\}', result, re.DOTALL).group()

    data = json.loads(json_text)

    return data

# ---- cell ----

"""
Get Boundary
"""
import geemap

def get_roi(location):

    location = location.title()

    districts = ee.FeatureCollection("FAO/GAUL/2015/level2")

    roi_fc = districts.filter(
        ee.Filter.Or(
            ee.Filter.eq("ADM2_NAME", location),
            ee.Filter.eq("ADM1_NAME", location),
            ee.Filter.eq("ADM0_NAME", location)
        )
    )

    # Check if boundary exists
    if roi_fc.size().getInfo() > 0:

        roi = roi_fc.geometry()

    else:

        # Convert location name → coordinates
        coords = geemap.geocode(location)

        if coords:

            lon, lat = coords[0][0], coords[0][1]

            roi = ee.Geometry.Point([lon, lat]).buffer(50000)

        else:

            # final fallback
            roi = ee.Geometry.Point([0,0]).buffer(50000)

    return roi

#------------------------
st.write("ROI:", roi)
st.write("Index:", plan["index"])
st.write("Dataset:", plan["collection"])
"""
Run GEE Analysis
"""
# ---- cell ----
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

        lst = image.multiply(0.02).subtract(273.15)

        return lst.clip(region)

    elif index_name == "NO2":

        collection = (
            ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
            .filterDate(start, end)
            .filterBounds(region)
            .select("tropospheric_NO2_column_number_density")
        )

        image = collection.mean()

        return image.clip(region)

    else:

        collection = (
            ee.ImageCollection(plan["collection"])
            .filterDate(start, end)
            .filterBounds(region)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        )

        # ensure images exist
        count = collection.size()

        image = ee.Image(
            ee.Algorithms.If(
                count.gt(0),
                collection.median(),
                ee.Image.constant(0)
            )
        )

        bands = plan["bands"]

        image = image.select(bands)

        index_img = image.normalizedDifference(bands)

        return index_img.clip(region)


        

# ---- cell ----

"""
Visualization
"""

def visualize(index_img, region, index_name):

    Map = geemap.Map()

    Map.centerObject(region, 8)

    if index_name in ["NDVI","NDWI","NDMI","NDBI"]:
        vis = {"min": -1, "max": 1, "palette": ["blue","white","green"]}

    elif index_name == "LST":
        vis = {"min": 20, "max": 40, "palette": ["blue","yellow","red"]}

    elif index_name == "NO2":
        vis = {"min": 0, "max": 0.0002, "palette": ["black","purple","red"]}

    else:
        vis = {}

    Map.addLayer(index_img, vis, index_name)

    Map.addLayer(region, {}, "ROI")

    Map.addLayerControl()

    return Map

# ---- cell ----

"""
Run Pipeline
"""

st.title("GeoAI Environmental Analysis")

query = st.text_input(
    "Enter your environmental analysis query",
    "Assess air pollution in Kerala during 2022"
)

if query:

    metadata = extract_metadata(query)

    location = metadata["location"]
    start = metadata["start_date"]
    end = metadata["end_date"]
    analysis = metadata["analysis_type"]

    plan = generate_plan(query)
    
    st.write("Metadata:", metadata)
    st.write("Plan:", plan)

    roi = get_roi(location)

    index_img = run_analysis(plan, roi, start, end)

    
    if index_img is None:
        st.error("Analysis failed. No image generated.")
        st.stop()

    Map = visualize(index_img, roi, plan["index"])

    Map.to_streamlit(height=600)
