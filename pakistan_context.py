import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import re

# Global variables to cache vectorstore and raw data
_pakistan_vectorstore = None
_pakistan_data_loaded = False
_raw_agro_data = None

def load_pakistan_context_data():
    """Load all agro-ecological data for Pakistan-wide queries"""
    global _raw_agro_data
    
    try:
        # Debug: Check if file exists
        if not os.path.exists("Agro ecological data district dominant.xlsx"):
            print("Error: Agro ecological data district dominant.xlsx file not found")
            return []
            
        # Load agro zones data (Sheet 1)
        df1 = pd.read_excel("Agro ecological data district dominant.xlsx", sheet_name="agro zones")
        df1.columns = df1.columns.str.strip()
        
        # Store raw data for direct crop queries
        _raw_agro_data = df1
        
        print(f"Loaded {len(df1)} rows from agro zones sheet")
        
        # Load general organic farming data (Sheet 2)
        df2 = pd.read_excel("Agro ecological data district dominant.xlsx", sheet_name="organic farming")
        df2.columns = df2.columns.str.strip()
        
        print(f"Loaded {len(df2)} rows from organic farming sheet")
        
        documents = []

        # Process all agro-ecological zones data
        for _, row in df1.iterrows():
            # Create comprehensive document for each zone/district
            zone_info = f"PAKISTAN AGRO-ECOLOGICAL DATA:\n\n"
            zone_info += f"Zone: {row.get('Zones', 'N/A')}\n"
            zone_info += f"Zone Name: {row.get('Names', 'N/A')}\n"
            zone_info += f"Province: {row.get('Province', 'N/A')}\n"
            zone_info += f"District: {row.get('District', 'N/A')}\n"
            zone_info += f"Climate: {row.get('Climate', 'N/A')}\n"
            zone_info += f"Soil Types: {row.get('Soil Types', 'N/A')}\n"
            zone_info += f"Major Crops: {row.get('Major crops', 'N/A')}\n"
            zone_info += f"Rainfall: {row.get('Rain fall', 'N/A')}\n\n"
            
            # Add search-friendly variations for crop queries
            major_crops = str(row.get('Major crops', '')).lower()
            district = str(row.get('District', 'N/A'))
            province = str(row.get('Province', 'N/A'))
            zone_name = str(row.get('Names', 'N/A'))
            
            zone_info += f"Crop cultivation information:\n"
            zone_info += f"Crops grown in {district}: {row.get('Major crops', 'N/A')}\n"
            zone_info += f"Agricultural crops in {district}, {province}: {row.get('Major crops', 'N/A')}\n"
            zone_info += f"Crops suitable for {zone_name}: {row.get('Major crops', 'N/A')}\n"
            
            # Add specific crop mentions for better search
            if major_crops and major_crops != 'n/a':
                crops_list = [crop.strip() for crop in major_crops.split(',')]
                for crop in crops_list:
                    if crop and len(crop) > 2:  # Avoid empty or very short strings
                        zone_info += f"{crop.title()} cultivation: {district}, {province} - {zone_name}\n"
                        zone_info += f"Where to grow {crop}: {district} district in {province}\n"
            
            zone_info += f"\nLocation details:\n"
            zone_info += f"Climate conditions in {district}: {row.get('Climate', 'N/A')}\n"
            zone_info += f"Soil information for {district}: {row.get('Soil Types', 'N/A')}\n"
            zone_info += f"Rainfall pattern in {district}: {row.get('Rain fall', 'N/A')}\n"

            documents.append(Document(
                page_content=zone_info,
                metadata={
                    "source": "pakistan_agro_zones", 
                    "zone": row.get('Names', 'N/A'),
                    "district": district,
                    "province": province,
                    "crops": row.get('Major crops', 'N/A')
                }
            ))

        # Process general organic farming Q&A data
        for _, row in df2.iterrows():
            question = str(row.iloc[0]).strip() if len(row) > 0 and pd.notna(row.iloc[0]) else ""
            answer = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ""
            
            if question and answer and question.lower() != 'nan' and answer.lower() != 'nan':
                qa_pair = f"GENERAL ORGANIC FARMING KNOWLEDGE:\n\nQ: {question}\nA: {answer}\n\n"
                qa_pair += f"Keywords: organic farming, sustainable agriculture, farming practices, Pakistan agriculture"
                
                documents.append(Document(
                    page_content=qa_pair,
                    metadata={"source": "organic_farming", "type": "general"}
                ))

        print(f"Total documents created for Pakistan context: {len(documents)}")
        return documents

    except Exception as e:
        print(f"Error loading Pakistan context data: {e}")
        return []

def is_agricultural_query(query):
    """Check if query is related to agriculture/farming"""
    agricultural_keywords = [
        # Crops and farming
        'crop', 'crops', 'farming', 'agriculture', 'cultivation', 'grow', 'growing',
        'plant', 'planting', 'harvest', 'harvesting', 'seed', 'seeds', 'organic',
        
        # Soil and land
        'soil', 'land', 'earth', 'field', 'farm', 'irrigation', 'fertilizer',
        'compost', 'manure', 'pesticide', 'herbicide',
        
        # Weather and climate
        'climate', 'weather', 'rainfall', 'rain', 'temperature', 'season', 'seasonal',
        'monsoon', 'drought', 'water',
        
        # Specific crops (common ones)
        'wheat', 'rice', 'cotton', 'sugarcane', 'maize', 'corn', 'mango', 'mangoes',
        'citrus', 'banana', 'apple', 'grapes', 'dates', 'onion', 'potato', 'tomato',
        'chili', 'garlic', 'ginger', 'turmeric', 'vegetables', 'fruits',
        
        # Locations (Pakistan geography)
        'pakistan', 'province', 'district', 'zone', 'punjab', 'sindh', 'balochistan',
        'khyber pakhtunkhwa', 'kpk', 'karachi', 'lahore', 'islamabad', 'faisalabad',
        
        # Agricultural practices
        'sowing', 'reaping', 'tillage', 'plowing', 'rotation', 'intercropping',
        'livestock', 'dairy', 'poultry', 'cattle', 'buffalo', 'goat', 'sheep'
    ]
    
    query_lower = query.lower()
    
    # Check if query contains any agricultural keywords
    return any(keyword in query_lower for keyword in agricultural_keywords)

def is_general_query(query):
    """Check if query is asking for general information that should be specific"""
    general_keywords = [
        'soil types', 'soil type', 'climate', 'rainfall', 'rain fall', 
        'weather', 'temperature', 'what are the', 'list all', 
        'all soil', 'all climate', 'all rainfall'
    ]
    
    query_lower = query.lower()
    
    # Check if it's a general query
    for keyword in general_keywords:
        if keyword in query_lower:
            # Allow only if it mentions specific districts or zones
            # Province-level queries are still considered too general
            if any(spec in query_lower for spec in ['district', 'zone']):
                return False  # It's specific enough
            else:
                return True   # It's too general (includes province-level)
    
    return False

def is_crop_location_query(query):
    """Check if query is asking where to grow specific crops"""
    crop_location_keywords = [
        'where can i grow', 'where to grow', 'which district', 'which province',
        'districts for', 'areas for', 'suitable for', 'grow in', 'cultivation of',
        'districts where', 'provinces where'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in crop_location_keywords)

def search_crop_in_all_districts(crop_name):
    """Search for a specific crop in all districts using raw data"""
    global _raw_agro_data
    
    if _raw_agro_data is None:
        return []
    
    crop_name = crop_name.lower()
    matching_districts = []
    
    for _, row in _raw_agro_data.iterrows():
        major_crops = str(row.get('Major crops', '')).lower()
        if crop_name in major_crops:
            district = str(row.get('District', 'N/A'))
            province = str(row.get('Province', 'N/A'))
            zone = str(row.get('Names', 'N/A'))
            
            matching_districts.append({
                'district': district,
                'province': province,
                'zone': zone,
                'crops': row.get('Major crops', 'N/A'),
                'climate': row.get('Climate', 'N/A'),
                'soil': row.get('Soil Types', 'N/A')
            })
    
    return matching_districts

def extract_crop_from_query(query):
    """Extract crop name from location query"""
    # Common patterns for crop queries
    patterns = [
        r'where can i grow (\w+)',
        r'where to grow (\w+)',
        r'districts for (\w+)',
        r'areas for (\w+)',
        r'cultivation of (\w+)',
        r'grow (\w+) in',
        r'suitable for (\w+)',
        r'(\w+) cultivation'
    ]
    
    query_lower = query.lower()
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return match.group(1)
    
    # If no pattern matches, try to find crop names in the query
    common_crops = [
        'wheat', 'rice', 'cotton', 'sugarcane', 'maize', 'mango', 'mangoes',
        'citrus', 'banana', 'apple', 'grapes', 'dates', 'onion', 'potato',
        'tomato', 'chili', 'garlic', 'ginger', 'turmeric'
    ]
    
    for crop in common_crops:
        if crop in query_lower:
            return crop
    
    return None

def create_pakistan_vectorstore(documents):
    """Create FAISS vectorstore for Pakistan-wide queries"""
    if not documents:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks to keep district info together
        chunk_overlap=200,
        separators=["\n\n", "\n", ":", ".", " "]
    )
    
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

def preload_pakistan_context_data():
    """Preload all Pakistan agricultural data"""
    global _pakistan_vectorstore, _pakistan_data_loaded
    
    if not _pakistan_data_loaded:
        print("Loading Pakistan context data...")
        documents = load_pakistan_context_data()
        if documents:
            _pakistan_vectorstore = create_pakistan_vectorstore(documents)
            _pakistan_data_loaded = True
            print("Pakistan context data loaded successfully!")
            return True
        else:
            print("Failed to load Pakistan context data")
            return False
    return _pakistan_vectorstore is not None

def post_process_pakistan_response(response):
    """Post-process response to ensure sentences are complete"""
    if not response:
        return response
    
    # Find the last complete sentence
    sentences = response.split('.')
    if len(sentences) > 1:
        # If the last element is empty or very short, it's likely an incomplete sentence
        if not sentences[-1].strip() or len(sentences[-1].strip()) < 15:
            # Remove the incomplete sentence and keep complete ones
            complete_response = '.'.join(sentences[:-1]) + '.'
            return complete_response
        else:
            # Check if the last sentence seems incomplete (no proper ending)
            last_sentence = sentences[-1].strip()
            if last_sentence and not any(last_sentence.endswith(ending) for ending in ['.', '!', '?', ':']):
                # Remove incomplete sentence
                complete_response = '.'.join(sentences[:-1]) + '.'
                return complete_response
    
    return response

def create_pakistan_qa_chain(vectorstore):
    """Create QA chain for Pakistan-wide queries with improved token management"""
    # SOLUTION 1: Increased token limit from 400 to 650
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo', max_tokens=650)
    
    # SOLUTION 2: Added instruction for complete sentences
    template = """You are an expert agricultural consultant with comprehensive knowledge of Pakistan's agro-ecological zones and organic farming. You have access to data covering all provinces, districts, and agro-ecological zones of Pakistan.

IMPORTANT: You MUST ONLY answer questions related to agriculture, farming, crops, soil, climate, and Pakistan's agricultural zones. If a question is not related to agriculture, you MUST respond with: "I can only provide information about agriculture, farming, crops, soil and climate with the district name or Pakistan's agro-ecological zones. Please ask questions related to these topics."

Context: {context}
Question: {question}

INSTRUCTIONS:

1. **FIRST CHECK**: Is this question about agriculture, farming, crops, soil, climate, or Pakistan's agro-ecological zones?
   - If NO: Respond with the standard message above
   - If YES: Proceed with the answer using the instructions below

2. **Crop location queries** (e.g., "where can I grow mangoes", "which districts are suitable for wheat"):
   - Search through ALL available district and zone data
   - List ALL districts/provinces where the crop is mentioned in "Major crops"
   - Format: "You can grow [crop] in the following locations: District1 (Province1), District2 (Province2)..."
   - If crop is not found in any major crops list, state data unavailability

3. **Comparative queries** (e.g., "best districts for cotton", "compare wheat growing areas"):
   - Compare multiple locations based on available data
   - Mention climate, soil, and rainfall differences when relevant
   - Rank or categorize based on the data provided

4. **Zone-specific information** (climate, soil, crops for specific zones):
   - Use the agro-ecological zone data
   - Include zone name, provinces, and districts covered
   - Provide comprehensive information about climate, soil, and crops

5. **General organic farming queries**:
   - Use the "GENERAL ORGANIC FARMING KNOWLEDGE" data
   - Provide detailed, informative answers

6. **Multi-location queries** (e.g., "climate of northern districts"):
   - Aggregate information from multiple relevant locations
   - Group by provinces when appropriate

7. **Data handling**:
   - If information is not available, clearly state: "This information is not available in the current data"
   - Always base answers ONLY on the provided context
   - Do NOT guess or assume information not in the data

8. **Response format**:
   - Be comprehensive but organized and concise
   - Use bullet points or numbered lists for multiple locations
   - Include province names with district names for clarity
   - For crop queries, prioritize showing ALL possible locations
   - CRITICAL: Always complete your sentences. If space is limited, provide fewer but complete points rather than cutting off mid-sentence.

9. **Prohibited responses**:
   - Do NOT provide sowing/harvesting calendars unless explicitly mentioned in data
   - Do NOT answer non-agricultural questions
   - Do NOT guess climate or soil information not provided
   - Do NOT provide medical, health, or non-farming advice

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),  # Retrieve more documents for comprehensive answers
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

def get_pakistan_context_response(query):
    """Get response for Pakistan-wide queries with improved sentence completion"""
    global _pakistan_vectorstore
    
    # Check if data is loaded
    if not _pakistan_data_loaded:
        if not preload_pakistan_context_data():
            return "Unable to load Pakistan agricultural data. Please check if the 'agro ecological data.xlsx' file is available."
    
    if _pakistan_vectorstore is None:
        return "Pakistan context data is not available. Please try again."
    
    try:
        # FIRST CHECK: Is this an agricultural query?
        if not is_agricultural_query(query):
            return "I can only provide information about agriculture, farming, crops, soil, climate, and Pakistan's agro-ecological zones. Please ask questions related to these topics."
        
        # Check if it's a general query that should be specific
        if is_general_query(query):
            return "To provide you with precise and specific information, please specify:\n\n" \
                   "• A specific agro-ecological zone (e.g., 'soil types in Zone III - Sandy Desert')\n" \
                   "• A specific district (e.g., 'climate in Lahore district')\n\n" \
                   "Province-level queries are too broad. Please ask about specific districts or agro-ecological zones for detailed and accurate information."
        
        # Check if it's a crop location query
        if is_crop_location_query(query):
            crop_name = extract_crop_from_query(query)
            if crop_name:
                # Search directly in raw data for comprehensive results
                matching_districts = search_crop_in_all_districts(crop_name)
                
                if matching_districts:
                    response = f"You can grow {crop_name} in the following locations:\n\n"
                    
                    # Group by province for better organization
                    by_province = {}
                    for district_info in matching_districts:
                        province = district_info['province']
                        if province not in by_province:
                            by_province[province] = []
                        by_province[province].append(district_info)
                    
                    for province, districts in by_province.items():
                        response += f"**{province} Province:**\n"
                        for district_info in districts:
                            response += f"• {district_info['district']}\n"
                        response += "\n"
                    
                    response += f"\nTotal districts where {crop_name} can be grown: {len(matching_districts)}\n"
                    response += f"\nFor detailed climate, soil, and other agricultural information about any specific district, please ask about that district specifically."
                    
                    return response
                else:
                    return f"Based on the available data, {crop_name} is not listed as a major crop in any of the covered districts. The data might not be comprehensive for all crops, or this crop might be grown as a minor crop in some areas."
        
        # For other queries, use the QA chain
        qa_chain = create_pakistan_qa_chain(_pakistan_vectorstore)
        print(f"Pakistan context query: {query}")
        result = qa_chain({"query": query})
        answer = result.get('result', 'Unable to find relevant information in the Pakistan agricultural data.')
        
        # SOLUTION 3: Post-process to ensure complete sentences
        answer = post_process_pakistan_response(answer)
        
        print(f"Pakistan context answer: {answer[:200]}...")
        return answer
        
    except Exception as e:
        return f"Error processing your Pakistan context query: {str(e)}"