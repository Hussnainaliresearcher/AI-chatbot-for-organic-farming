import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Agro-ecological zones list
AGRO_ZONES = [
    "Indus Delta",
    "Southern Irrigated Plain", 
    "Sandy Desert",
    "Northern Irrigated Plain",
    "Barani (rainfed) Lands",
    "Wet Mountains",
    "Northern Dry Mountains",
    "Western Dry Mountains",
    "Dry Western Plateau",
    "Sulaiman Piedmont"
]

# Global variables to cache vectorstore
_zone_vectorstore = None
_current_zone = None

def get_agro_zones():
    """Return the list of agro-ecological zones"""
    return AGRO_ZONES

def load_zone_data(zone=None):
    """Load and filter agro-ecological data by zone"""
    try:
        # Debug: Check if file exists
        if not os.path.exists("zone wise data.xlsx"):
            print("Error: zone wise data.xlsx file not found")
            return []
            
        # Load agro zones data
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

def create_zone_vectorstore(documents):
    """Create FAISS vectorstore"""
    if not documents:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

def preload_zone_data(zone):
    """Preload agricultural data for the specified zone"""
    global _zone_vectorstore, _current_zone
    
    if _current_zone != zone:
        documents = load_zone_data(zone)
        if documents:
            _zone_vectorstore = create_zone_vectorstore(documents)
            _current_zone = zone
            return True
    return _zone_vectorstore is not None

def create_zone_qa_chain(vectorstore):
    """Create QA chain for zone queries"""
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo', max_tokens=300)
    
    template = """You are an agricultural assistant specialized in organic farming. Answer questions ONLY about agro-ecological zones and organic farming based on the provided data.

Context: {context}
Question: {question}

INSTRUCTIONS:

1. **Zone-specific queries** (climate, soil types, crops, rainfall, districts for a specific zone):
   - Use ONLY the zone data from the context
   - ALWAYS include the specific zone name from the context in your answer (look for "Zone Name:" in the context)
   - Format: "The soil types in [Zone Name] are..."
   - If data shows "N/A" or is missing, state: "This information is not available in the data for [Zone Name]"

2. **General organic farming queries** (definitions, practices, methods):
   - Use the "GENERAL ORGANIC FARMING" Q&A data
   - Provide clear, informative answers based on the available data

3. **Crop suitability questions**:
   - Check if the crop is listed in "Major crops" for the zone
   - If not listed, assess suitability based on available climate/soil data only
   - Do NOT provide sowing/harvesting times unless explicitly mentioned in data

4. **Prohibited responses**:
   - Do NOT guess or assume information not in the context
   - Do NOT provide crop calendars/timings unless specifically mentioned
   - Do NOT answer non-agricultural questions (like "what is chatbots")
   - For non-agricultural questions, respond: "I can only help with questions about agro-ecological zones and organic farming."

5. **Polite interactions** (greetings, thanks):
   - Respond naturally and offer to help with farming questions

6. **Data availability**:
   - If no relevant data found: "This information is not available in the data"
   - If zone not found: "No data available for this specific zone"

7. **Response format**:
   - Be direct, concise and specific
   - Use bullet points for multiple items when appropriate
   - ALWAYS include the actual zone name when discussing zone-specific information

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

def get_zone_prep_response(query, zone=None):
    """Get response for zone-based queries"""
    global _zone_vectorstore
    
    # Debug: Check if zone is provided
    if not zone:
        return "Please select an agro-ecological zone first."
    
    # Debug: Try to preload data
    print(f"Loading data for zone: {zone}")
    if not preload_zone_data(zone):
        return f"Unable to load data for zone: {zone}. Please check if the 'zone wise data.xlsx' file is available and the zone name is correct."
    
    if _zone_vectorstore is None:
        return "Vectorstore is None after preloading data."
    
    try:
        qa_chain = create_zone_qa_chain(_zone_vectorstore)
        print(f"Querying: {query}")
        result = qa_chain({"query": query})
        answer = result.get('result', 'Unable to find relevant information.')
        print(f"Answer: {answer}")
        return answer
        
    except Exception as e:
        return f"Error processing your query: {str(e)}"