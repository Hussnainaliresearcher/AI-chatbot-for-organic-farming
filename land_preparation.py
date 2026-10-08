from common import load_excel_with_unstructured, create_vectorstore, get_qa_chain

# Province-District mapping (keeping your original)
province_districts = {
    "Punjab": ["Rahim Yar Khan", "Bahawalpur", "Bahawalnagar", "Muzaffargarh", "Lodhran", 
               "Multan (southern parts)", "Khanewal (southern parts)", "Vehari (southern parts)", 
               "Rahim Yar Khan Sandy Desert", "Mianwali", "Sargodha", "Faisalabad", "Lahore", 
               "Kasur", "Okara", "Sahiwal", "Pakpattan", "Jhang", "Chiniot", "Sheikhupura", 
               "Nankana Sahib", "Gujranwala", "Gujrat", "Toba Tek Singh", "Multan (Northern)", 
               "Khanewal (North)", "Vehari (North)", "Attock", "Rawalpindi", "Jhelum", "Chakwal", 
               "Khushab", "Bhakkar", "Layyah", "Dera Ghazi Khan (D.G. Khan)", "Sialkot", 
               "Narowal", "Rawalpindi (Wet Mountain)", "Muree", "Dera Ghazi Khan", "Rajanpur"],
    
    "Sindh": ["Hyderabad", "Badin", "Thatta", "Tharparkar", "Sanghar", "Dadu", "Khairpur", 
              "Larkana", "Shaheed Benazirabad", "Jacobabad", "Sukkur", "Shikarpur", 
              "Tharparkar Sandy Desert", "Khairpur Sandy Desert", "Shaheed Benazirabad Sandy Desert"],
    
    "Khyber Pakhtunkhwa (KPK)": ["Peshawar", "Mardan", "Abbottabad", "Mansehra", "Battagram", 
                                  "Torghar", "Shangla", "Swat", "Upper Dir", "Lower Dir", "Buner", 
                                  "Malakand", "Chitral (parts)", "Kohistan (parts)", "Chitral (dry parts)", 
                                  "Kohistan (dry parts)", "Khyber", "Kurram", "Orakzai", "North Waziristan", 
                                  "South Waziristan", "Hangu", "Kohat (parts)", "Bannu (parts)", 
                                  "Lakki Marwat (parts)", "Dera Ismail Khan (parts)", "Tank", "Dera Ismail Khan"],
    
    "Balochistan": ["Sibi", "Zhob", "Sherani", "Killa", "Saifullah", "Loralai", "Musakhel", 
                    "Barkhan", "Duki", "Ziarat", "Pishin (parts)", "Qila Abdullah (parts)", 
                    "Killa Saifullah (parts)", "Quetta", "Mastung", "Kalat", "Khuzdar", "Nushki", 
                    "Chagai", "Kharan", "Washuk", "Panjgur", "Kech", "Gwadar", "Awaran", 
                    "Lasbela", "Bolan", "Jhal Magsi", "Sibi Dry Western", "Dera Bugti", "Kohlu", 
                    "Harnai", "Kachhi"],
    
    "Azad Jammu & Kashmir (AJK)": ["Muzaffarabad sandy desert", "Muzaffarabad", "Bagh", "Poonch", 
                                   "Neelum", "Sudhnoti", "Kotli (upper parts)", "Haveli"],
    
    "Gilgit-Baltistan": ["Astore", "Diamer", "Gilgit", "Skardu", "Ghanche", "Ghizer", "Hunza", 
                         "Nagar", "Diamer (dry parts)"]
}

def get_land_preparation_response(query, province=None, district=None):
    """Get land preparation response with improved location-specific data handling"""
    
    # Load Excel data using UnstructuredExcelLoader approach
    documents = load_excel_with_unstructured("agro ecological data.xlsx", province, district)
    
    # Debug information
    print(f"\n=== DEBUG INFO for {district}, {province} ===")
    print(f"Total documents loaded: {len(documents)}")
    for i, doc in enumerate(documents[:2]):  # Show first 2 docs
        print(f"Document {i+1} content preview:")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        print("-" * 50)
    
    if not documents:
        return f"No agricultural data available for {district}, {province}. Please check if the location is correct."
    
    # Create vectorstore
    vectorstore = create_vectorstore(documents)
    
    # Create improved prompt template
    custom_prompt = f"""You are an expert agricultural advisor for {province} province, {district} district in Pakistan. 

Your task is to provide helpful farming advice based on the agricultural data provided in the context below.

INSTRUCTIONS:
1. Use the provided context to answer questions about agriculture, farming, crops, climate, soil, rainfall, and land preparation
2. The context contains both location-specific data and general farming knowledge
3. Look for relevant information using flexible matching - don't just look for exact field labels
4. If you find relevant information, provide a comprehensive and helpful answer
5. Include practical advice when possible
6. If the specific information isn't available, provide general guidance if you can, or suggest related topics that are covered
7. Always end with the data source reference

Context Information:
{{context}}

User Question: {{question}}

Please provide a helpful and informative response based on the available agricultural data.

Data Source: Agricultural Database for {district}, {province}"""
    
    # Get QA chain with improved settings
    qa_chain = get_qa_chain(vectorstore, custom_prompt, use_simple_chain=True)
    
    # Get response
    try:
        if hasattr(qa_chain, 'run'):
            response = qa_chain.run(query)
        elif hasattr(qa_chain, 'invoke'):
            result = qa_chain.invoke({"query": query})
            response = result.get('result', result.get('answer', "Unable to process the query."))
        else:
            result = qa_chain({'query': query})
            response = result.get('result', result.get('answer', "Unable to process the query."))
            
    except Exception as e:
        print(f"Error in QA chain: {str(e)}")
        response = f"I encountered an issue processing your question about {district}, {province}. Please try rephrasing your question or ask about specific topics like crops, climate, or soil conditions."
    
    return response

def get_available_topics(province=None, district=None):
    """Get available topics for a specific location"""
    documents = load_excel_with_unstructured("agro ecological data.xlsx", province, district)
    
    if not documents:
        return "No data available for this location."
    
    # Extract topics from document content
    topics = set()
    for doc in documents:
        content = doc.page_content.lower()
        if 'climate' in content or 'weather' in content:
            topics.add("Climate and Weather")
        if 'crop' in content or 'farming' in content:
            topics.add("Crops and Farming")
        if 'soil' in content:
            topics.add("Soil Information")
        if 'rain' in content or 'precipitation' in content:
            topics.add("Rainfall and Irrigation")
        if 'compost' in content or 'organic' in content:
            topics.add("Organic Farming and Composting")
    
    return f"Available topics for {district}, {province}: {', '.join(sorted(topics))}"