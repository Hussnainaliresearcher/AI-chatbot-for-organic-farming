from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import pickle
from datetime import datetime, timedelta

# Global variables for caching
_vectorstore = None
_last_update = None
_cache_duration = timedelta(hours=24)  # Cache for 24 hours

def load_or_create_vectorstore():
    """Load cached vectorstore or create new one"""
    global _vectorstore, _last_update
    
    cache_file = "pakorganic_vectorstore.pkl"
    
    # Check if we have cached vectorstore and it's still valid
    if (_vectorstore is not None and 
        _last_update is not None and 
        datetime.now() - _last_update < _cache_duration):
        return _vectorstore
    
    # Try to load from file cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                if datetime.now() - cache_data['timestamp'] < _cache_duration:
                    _vectorstore = cache_data['vectorstore']
                    _last_update = cache_data['timestamp']
                    return _vectorstore
        except:
            pass  # If loading fails, create new one
    
    # Create new vectorstore
    urls = [
        "https://pakorganics.store/",
        
    ]
    
    # Load web content with a standard browser User-Agent and higher timeout
    try:
        loader = WebBaseLoader(
            web_paths=urls,
            header_template={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            },
            requests_kwargs={
                "timeout": 30
            }
        )
        documents = loader.load()
    except Exception as e:
        print(f"Warning: Failed to scrape the website due to: {e}")
        documents = []  # Fallback to empty if the site is completely offline
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(documents) if documents else []
    
    # Create vectorstore
    embeddings = OpenAIEmbeddings()
    
    if texts:
        vectorstore = FAISS.from_documents(texts, embeddings)
    else:
        # Fallback to prevent app crashes if the website is down
        vectorstore = FAISS.from_texts(
            ["The website pakorganics.store is currently unreachable or unavailable."], 
            embeddings
        )
    
    # Cache the vectorstore
    _vectorstore = vectorstore
    _last_update = datetime.now()
    
    # Save to file
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'vectorstore': vectorstore,
                'timestamp': _last_update
            }, f)
    except:
        pass  # If saving fails, still continue
    
    return vectorstore

def preload_web_store_data():
    """Preload web store data to avoid delay on first query"""
    load_or_create_vectorstore()

def scrape_web_store(query):
    """Scrape pakorganic.com and provide intelligent responses"""
    
    # Get cached vectorstore
    vectorstore = load_or_create_vectorstore()
    
    # Define template with clickable source
    template = """You are a helpful assistant. Use the following web content as your primary reference to answer the user's question. If the answer is not clearly available, make a reasonable guess or summarize based on what is available. If you still cannot find anything useful, say: 'Sorry, this is out of my knowledge domain.'

Try to respond clearly, especially for questions like:
- What are the available products?
- Show product names and prices.
- Give me a list of items available.
- Location and contact information.

Always add this at the end (in HTML): <br><br>Source: <a href='https://www.pakorganics.store' target='_blank'>www.pakorganics.store</a>

Context: {context}
Question: {question}

Answer:"""
    
    # Create prompt and QA chain
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm = ChatOpenAI(temperature=0, model='gpt-4-turbo', max_tokens=200)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    
    # Get response using .invoke() which is the modern LangChain standard
    result = qa_chain.invoke({"query": query})
    return result.get('result', 'Unable to find relevant information.')

def get_web_scraper_response(query):
    """Get response from pakorganic.com content"""
    return scrape_web_store(query)

def clear_cache():
    """Clear the vectorstore cache"""
    global _vectorstore, _last_update
    _vectorstore = None
    _last_update = None
    
    cache_file = "pakorganic_vectorstore.pkl"
    if os.path.exists(cache_file):
        os.remove(cache_file)