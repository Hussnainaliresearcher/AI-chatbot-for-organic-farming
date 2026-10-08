import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Global variables to cache vectorstore AND QA chain
_location_vectorstore = None
_location_qa_chain = None
_current_location_zone = None

def load_agro_zones_geojson():
    """Load the GeoJSON file for agro-ecological zones"""
    try:
        return gpd.read_file("ali_try3_colors.geojson")
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None

def get_location_name(lat, lon):
    """Get short location name from coordinates"""
    try:
        geolocator = Nominatim(user_agent="agro_zone_app")
        location = geolocator.reverse((lat, lon), timeout=10)
        if location and location.raw and "address" in location.raw:
            address = location.raw["address"]
            return address.get("city") or address.get("town") or address.get("village") or address.get("county") or "Unknown"
        return "Unknown"
    except GeocoderTimedOut:
        return "Geocoder timeout"
    except Exception as e:
        print(f"Error getting location name: {e}")
        return "Unknown"

def find_agro_zone_from_location(lat, lon):
    """Find agro-ecological zone from coordinates"""
    try:
        agro_zones = load_agro_zones_geojson()
        if agro_zones is None:
            return None
        
        user_point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        matched = gpd.sjoin(user_point, agro_zones, how="left", predicate="within")
        
        if not matched.empty and not pd.isna(matched.iloc[0]["zone_name"]):
            return matched.iloc[0]["zone_name"]
        return None
    except Exception as e:
        print(f"Error finding agro zone: {e}")
        return None

def load_location_zone_data(zone=None):
    """Load and filter agro-ecological data by zone (optimized version)"""
    try:
        # Debug: Check if file exists
        if not os.path.exists("zone wise data.xlsx"):
            print("Error: zone wise data.xlsx file not found")
            return []
            
        # Load agro zones data with optimized pandas settings
        df1 = pd.read_excel("zone wise data.xlsx", sheet_name="agro zones")
        df1.columns = df1.columns.str.strip()
        
        print(f"Loaded {len(df1)} rows from agro zones sheet")
        print(f"Available zones: {df1['Names'].unique().tolist()}")
        
        # Load general organic farming data
        df2 = pd.read_excel("zone wise data.xlsx", sheet_name="organic farming")
        df2.columns = df2.columns.str.strip()
        
        documents = []

        # Process zone-specific data
        if zone:
            print(f"Searching for zone: {zone}")
            zone_norm = zone.strip().lower()
            # Search in 'Names' column instead of 'Zones' column
            zone_data = df1[df1['Names'].str.strip().str.lower().str.contains(zone_norm, na=False)]

            if zone_data.empty:
                zone_data = df1[df1['Names'].str.strip().str.lower() == zone_norm]
                
            print(f"Found {len(zone_data)} matching rows for zone: {zone}")

            if not zone_data.empty:
                for _, row in zone_data.iterrows():
                    zone_info = f"Zone: {row.get('Zones', 'N/A')}\n"
                    zone_info += f"Zone Name: {row.get('Names', 'N/A')}\n"
                    zone_info += f"Climate: {row.get('Climate', 'N/A')}\n"
                    zone_info += f"Districts: {row.get('Districts', 'N/A')}\n"
                    zone_info += f"Soil Types: {row.get('Soil Types', 'N/A')}\n"
                    zone_info += f"Major crops: {row.get('Major crops', 'N/A')}\n"
                    zone_info += f"Rainfall: {row.get('Rain fall', 'N/A')}\n"
                    
                    # Add alternative phrasings for better matching
                    zone_info += f"\nSoil information: {row.get('Soil Types', 'N/A')}\n"
                    zone_info += f"Crop information: {row.get('Major crops', 'N/A')}\n"
                    zone_info += f"Weather information: {row.get('Climate', 'N/A')}\n"

                    print(f"Created document with content: {zone_info[:100]}...")
                    documents.append(Document(
                        page_content=zone_info,
                        metadata={"source": "agro_zones", "zone": zone}
                    ))

        # Process general organic farming Q&A data
        for _, row in df2.iterrows():
            question = str(row.iloc[0]).strip() if len(row) > 0 and pd.notna(row.iloc[0]) else ""
            answer = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ""
            
            if question and answer and question.lower() != 'nan' and answer.lower() != 'nan':
                qa_pair = f"Q: {question}\nA: {answer}"
                documents.append(Document(
                    page_content=qa_pair,
                    metadata={"source": "organic_farming", "type": "general"}
                ))

        print(f"Total documents created: {len(documents)}")
        return documents

    except Exception as e:
        print(f"Error loading zone data: {e}")
        return []

def create_location_vectorstore(documents):
    """Create FAISS vectorstore for location-based zone data (optimized)"""
    if not documents:
        print("No documents to create vectorstore")
        return None
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks for vectorstore")
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("Successfully created FAISS vectorstore")
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

def create_location_qa_chain(vectorstore):
    """Create QA chain for location-based zone queries (optimized)"""
    try:
        llm = ChatOpenAI(
            temperature=0, 
            model='gpt-4-turbo', 
            max_tokens=300,
            request_timeout=30  # Add timeout
        )
        
        template = """You are an agricultural assistant specialized in organic farming. Answer questions ONLY using the information from the attached data about agro-ecological zones and organic farming. Do not answer questions beyond the scope of the provided data or speculate on missing information.

Context: {context}
Question: {question}

INSTRUCTIONS:

1. **Answer Source Restriction**:
- ONLY use information found in the provided context or explicit organic farming data.
- If a user asks for information not present in the context/data, reply: "This information is not available in the data for [City Name] in [Zone Name]."

2. **Zone-specific Questions** (climate, soil, crops, rainfall, districts):
- Reference ONLY the data for the mentioned zone and city from context.
- Always start with: "In [City Name] (located in [Zone Name]), ..."
- If data is missing or marked "N/A", state this explicitly.

3. **General Organic Farming Questions**:
- Answer using only general knowledge from provided organic farming data.
- When possible, add: "For your location in [City Name] ([Zone Name]), ..."y

4. **Crop Suitability Questions**:
- Check if the crop appears in 'Major crops' for the zone. If not, use available climate/soil data to assess suitability.
- If the data is inconclusive, state: "Insufficient information to determine suitability for [crop name] in [City Name] ([Zone Name])."

5. **Format for Responses**:
- Always start with both city and zone names, where available.
- Be direct and concise; use bullet points for lists.
- Refer to the exact zone name as found in the context data.

6. **Strict Prohibitions**:
- Do NOT guess or fabricate any answer.
- Do NOT provide crop calendars, timings, or months unless found in the data.
- Do NOT answer questions unrelated to agriculture or organic farming. For such queries, say: "I can only help with questions about agro-ecological zones and organic farming."


7. **Unavailable Data Handling**:
- If no relevant data is found, say: "This information is not available in the data for [City Name] in [Zone Name]."
- If the requested location or zone is fully unmatched, reply: "No data available for this specific location."

8. **Polite Interaction**:
- Respond naturally, include brief greetings or thanks if appropriate, and offer to help further with farming questions.

After preparing your answer, review to ensure only supplied data is used, response format matches guidelines, and no speculation is present. If validation fails, self-correct before outputting the answer.

Ensure you do not answer any question unless information is directly found in the supplied data.

Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        
        print("Successfully created QA chain")
        return qa_chain
        
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None

def preload_location_zone_data(zone):
    """Preload agricultural data AND QA chain for the location-detected zone"""
    global _location_vectorstore, _location_qa_chain, _current_location_zone
    
    try:
        # Only reload if zone has changed
        if _current_location_zone != zone or _location_vectorstore is None or _location_qa_chain is None:
            print(f"Preloading data for zone: {zone}")
            
            # Load documents
            documents = load_location_zone_data(zone)
            if not documents:
                print(f"No documents loaded for zone: {zone}")
                return False
            
            # Create vectorstore
            _location_vectorstore = create_location_vectorstore(documents)
            if _location_vectorstore is None:
                print("Failed to create vectorstore")
                return False
            
            # Create QA chain
            _location_qa_chain = create_location_qa_chain(_location_vectorstore)
            if _location_qa_chain is None:
                print("Failed to create QA chain")
                return False
            
            _current_location_zone = zone
            print(f"Successfully preloaded all components for zone: {zone}")
            return True
        else:
            print(f"Data already loaded for zone: {zone}")
            return True
            
    except Exception as e:
        print(f"Error in preload_location_zone_data: {e}")
        return False

def get_location_zone_response(query, zone=None, city_name="Unknown"):
    """Get response for location-based zone queries (optimized for speed)"""
    global _location_vectorstore, _location_qa_chain, _current_location_zone
    
    try:
        # Debug: Check if zone is provided
        if not zone:
            return "Unable to detect your agro-ecological zone. Please ensure location access is enabled."
        
        # Check if data is preloaded
        if (_current_location_zone != zone or 
            _location_vectorstore is None or 
            _location_qa_chain is None):
            print(f"Data not preloaded for zone: {zone}. Attempting to preload...")
            if not preload_location_zone_data(zone):
                return f"Unable to load data for zone: {zone}. Please check if the 'zone wise data.xlsx' file is available."
        
        print(f"Processing query: '{query}' for zone: {zone} in city: {city_name}")
        
        # Add zone and city context to the query to help with retrieval
        contextual_query = f"In {city_name} located in {zone} zone: {query}"
        
        # Use the cached QA chain for fast response
        result = _location_qa_chain({"query": contextual_query})
        answer = result.get('result', 'Unable to find relevant information.')
        
        # Ensure both city and zone names are mentioned in response if not already included
        if city_name.lower() not in answer.lower() and zone.lower() not in answer.lower():
            answer = f"For {city_name} in {zone}: {answer}"
        elif city_name.lower() not in answer.lower():
            answer = f"In {city_name} ({zone}): {answer}"
        elif zone.lower() not in answer.lower():
            answer = f"For your location in {city_name} ({zone}): {answer}"
        
        print(f"Generated answer: {answer[:100]}...")
        return answer
        
    except Exception as e:
        error_msg = f"Error processing your query: {str(e)}"
        print(error_msg)

        return error_msg