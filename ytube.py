# organic_farming_chatbot_with_videos.py
import pandas as pd
import streamlit as st
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Organic Farming Assistant")
st.title("🌱 Organic Farming Chatbot")
st.write("Ask questions about organic farming and get practical guidance.")

# -------------------------------
# READ API KEY
# -------------------------------
openai_key = st.secrets["OPENAI_API_KEY"]

# -------------------------------
# EMBEDDING MODEL (single instance)
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# LOAD KNOWLEDGE BASE (SHEET: organic)
# -------------------------------
@st.cache_resource
def load_knowledge_db():
    df = pd.read_excel("knowledge.xlsx", sheet_name="organic")
    documents = []
    for _, row in df.iterrows():
        text = f"""
        Question: {row['question']}
        Answer: {row['answer']}
        """
        documents.append(Document(page_content=text))
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

knowledge_db = load_knowledge_db()
retriever = knowledge_db.as_retriever(search_kwargs={"k": 2})

# -------------------------------
# LOAD VIDEO DATABASE (SHEET: video)
# We return:
#  - video_store (FAISS index if you want to use it)
#  - video_urls (list aligned with video_texts)
#  - video_texts (list of strings used to compute vectors)
#  - video_vectors (numpy array of vectors aligned with video_texts)
# -------------------------------
@st.cache_resource
def load_video_db(_embeddings):
    try:
        df = pd.read_excel("knowledge.xlsx", sheet_name="video")
    except Exception:
        # if sheet missing or empty, return empties
        return None, [], [], np.empty((0, _embeddings.embedding_dim if hasattr(_embeddings, 'embedding_dim') else 0))

    video_texts = []
    video_urls = []
    video_docs = []

    for _, row in df.iterrows():
        topic = str(row.get("topic", "")).strip()
        desc = str(row.get("description", "")).strip()
        url = str(row.get("video_url", "")).strip()

        if topic == "" and desc == "":
            continue

        # Build a descriptive text that captures topic+description for embedding
        combined_text = f"Topic: {topic}\nDescription: {desc}"
        video_texts.append(combined_text)
        video_urls.append(url)
        video_docs.append(Document(page_content=combined_text))

    # Build a FAISS index of videos as well (optional; not used for final similarity decision)
    if video_docs:
        video_store = FAISS.from_documents(video_docs, _embeddings)
    else:
        video_store = None

    # Compute embeddings for every video_text (returns list of vectors)
    if video_texts:
        # HuggingFaceEmbeddings usually exposes embed_documents
        try:
            vectors = _embeddings.embed_documents(video_texts)
        except Exception:
            # fallback - some wrappers use embed_query for single items only
            vectors = [ _embeddings.embed_query(t) for t in video_texts ]
    else:
        vectors = []

    # convert to numpy array for cosine computations
    if vectors:
        video_vectors = np.array(vectors, dtype=float)
    else:
        video_vectors = np.empty((0, 0))

    return video_store, video_urls, video_texts, video_vectors

video_db, video_urls, video_texts, video_vectors = load_video_db(embeddings)

# -------------------------------
# GPT MODEL
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=openai_key
)

# -------------------------------
# CHAT MEMORY
# -------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# -------------------------------
# Utilities: cosine similarity
# -------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # a, b are 1-D numpy arrays
    if a.size == 0 or b.size == 0:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def find_best_video_by_cosine(query: str, vectors: np.ndarray, urls: list, texts: list, threshold: float = 0.55):
    """
    Compute query embedding, compare with video vectors using cosine similarity,
    return (best_url, best_score, best_text) or (None, None, None) if not found or below threshold.
    """
    if vectors is None or vectors.size == 0:
        return None, None, None

    # get query vector
    try:
        qv = embeddings.embed_query(query)
    except Exception:
        qv = embeddings.embed_documents([query])[0]

    qv = np.array(qv, dtype=float)

    # compute cosine similarities
    sims = []
    for v in vectors:
        sims.append(cosine_similarity(qv, v))
    sims = np.array(sims, dtype=float)

    # get best index
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    # only return if above threshold
    if best_score >= threshold:
        return urls[best_idx], best_score, texts[best_idx]
    else:
        return None, best_score, None

# -------------------------------
# SESSION STATE
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)

# -------------------------------
# USER INPUT
# -------------------------------
user_query = st.chat_input("Ask your farming question...")

if user_query:
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append(("user", user_query))

    # -------------------------------
    # GENERATE ANSWER FROM RAG
    # -------------------------------
    with st.spinner("Generating answer..."):
        result = qa_chain({"question": user_query})
        answer = result["answer"]

    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append(("assistant", answer))

    # -------------------------------
    # RETRIEVE MOST RELEVANT CURATED VIDEO (cosine on video descriptions)
    # -------------------------------
    st.subheader("🎥 Practical Organic Farming Video")

    # You can tune this threshold: 0.55 is a reasonable starting value.
    VIDEO_SIMILARITY_THRESHOLD = 0.55

    video_url, score, matched_text = find_best_video_by_cosine(user_query, video_vectors, video_urls, video_texts, threshold=VIDEO_SIMILARITY_THRESHOLD)

    if video_url:
        # Show video + small info (title/description if available)
        try:
            # If URL is valid, display video
            st.video(video_url)
            # Optionally show matched text and similarity for debugging (comment out in production)
            #st.caption(f"Matched (score={score:.3f}): {matched_text}")
        except Exception:
            st.info("Video found but could not be played here. You can open the link: " + video_url)
    else:
        # friendly fallback message
        st.info("No relevant organic farming video found for this question. If you'd like, you can add a suitable tutorial link to the 'video' sheet in your knowledge.xlsx so future users will see it.")
