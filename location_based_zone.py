"""
location_based_zone.py
──────────────────────
Core architectural fix in this version:
  Zone data (soil, crops, climate, rainfall) is ALWAYS injected directly into
  the LLM context — it is never put in FAISS. FAISS only holds Q&A docs from
  the organic-farming sheet. This guarantees soil types, crop lists, climate
  are always available regardless of similarity ranking.

Speed fixes retained:
  gpt-4o-mini, FAISS disk cache (versioned), GeoJSON cache, module-level
  PromptTemplate, similarity_search k=4.
"""

import os
import re
import time
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from shapely.geometry import Point

# ── versioned cache dir (v2 = Q&A-only FAISS, invalidates old mixed cache) ──
FAISS_CACHE_DIR = "faiss_cache_v2"

# ── module-level cache ────────────────────────────────────────────────────────
_qa_vectorstore: Optional[FAISS] = None      # FAISS holds Q&A docs only
_zone_context: str = ""                       # zone data always injected directly
_location_llm: Optional[ChatOpenAI] = None
_embeddings: Optional[OpenAIEmbeddings] = None
_current_location_zone: Optional[str] = None
_location_memory: Optional[ConversationBufferMemory] = None
_agro_zones_gdf: Optional[gpd.GeoDataFrame] = None  # GeoJSON cached (Fix 5)

# ── prompt (built once at module level) ───────────────────────────────────────
_PROMPT_TEMPLATE = """You are an agricultural assistant for organic farming in Pakistan.

Farmer location: {city_name} (Agro-ecological zone: {zone})

Previous conversation:
{chat_history}

=== ZONE DATA (always complete — use this for any zone-specific question) ===
{zone_context}

=== ORGANIC FARMING Q&A (use this for technique/definition questions) ===
{qa_context}

Farmer's question: {question}

════════════════════════════════════════════════
STRICT RULES
════════════════════════════════════════════════

RULE 1 — ONLY USE THE CONTEXT ABOVE. NEVER USE YOUR OWN TRAINING KNOWLEDGE.
  Every single fact in your answer must come from the ZONE DATA or Q&A above.
  If it is not written there, do not say it.

RULE 2 — CROP AND SOIL NAMES: COPY VERBATIM, NEVER EXPAND OR RENAME.
  When the zone data lists crops or soil types, reproduce those exact words only.
  FORBIDDEN: expanding shorthand into specific species or variety names.
  If the zone data says "pulses" → write "pulses". Never "lentils, chickpeas".
  If the zone data says "oilseeds" → write "oilseeds". Never "sunflower, canola".
  If the zone data says "loamy" → write "loamy". Never add soil science detail.
  Copy the words. Do not interpret, expand, or categorise them.

RULE 3 — ZONE-SPECIFIC QUESTIONS (crops, soil types, climate, rainfall, districts).
  Answer using ONLY the ZONE DATA section above.
  Always name {city_name} and {zone} in your answer.
  If the field exists in ZONE DATA, the answer is available — do NOT say unavailable.

RULE 4 — GENERAL ORGANIC FARMING QUESTIONS (definitions, techniques, practices).
  Answer using the Q&A section above.
  After answering, you may add one sentence connecting it to the farmer's zone
  if genuinely relevant.

RULE 5 — COMBINED QUESTIONS (technique + zone context).
  Use both sections. Apply the technique from Q&A, reference zone crops/soil
  from ZONE DATA.

RULE 6 — TECHNIQUES FOR A CROP NOT IN THE ZONE CROP LIST.
  If the farmer asks for organic farming techniques for a crop that does NOT
  appear in the zone's Major crops list:
  a) Provide the technique from Q&A if available.
  b) Add: "Note: [crop] is not listed as a major crop for {city_name} in the
     {zone} zone and may not be well-suited for your region."

RULE 7 — TRULY MISSING INFORMATION.
  Only say "This information is not available in the database for {city_name}
  in {zone}." when the relevant field is literally absent or blank in ZONE DATA
  AND there is no matching Q&A answer.
  Never say this when the field IS present in ZONE DATA.

RULE 8 — MISSING OR OFF-TOPIC QUESTIONS.
  If the Q&A section above says "NO RELEVANT Q&A FOUND IN DATABASE FOR THIS
  QUESTION" AND the ZONE DATA also does not answer the question, then:
  - If the question is about agriculture/farming but the specific topic is not
    in the database, say:
    "This information is not available in the database for {city_name} in {zone}."
  - If the question has nothing to do with agriculture or organic farming at all,
    say: "I can only help with questions about agro-ecological zones and organic farming."
  NEVER answer using your own training knowledge when Q&A context is absent.
  NEVER expand crop names, animal types, or farming subtopics from your own knowledge.

RULE 9 — FORMAT.
  Write in clear, natural sentences.
  Use a bullet list ONLY when the answer genuinely contains 3 or more separate
  items (e.g. listing crop names, listing soil types).
  Do NOT use bullets for single-fact answers, definitions, or short explanations.
  Keep answers concise.

Answer:"""

_PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=[
        "city_name", "zone", "chat_history",
        "zone_context", "qa_context", "question",
    ],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    global _location_llm
    if _location_llm is None:
        _location_llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            max_tokens=500,
            request_timeout=30,
        )
    return _location_llm


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings()
    return _embeddings


def _zone_to_folder(zone: str) -> str:
    return re.sub(r"[^\w\-]", "_", zone.strip().lower())


# ── memory ────────────────────────────────────────────────────────────────────

def reset_location_memory() -> None:
    global _location_memory
    _location_memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True,
    )
    print("location_based_zone: memory reset.")


def _get_memory() -> ConversationBufferMemory:
    global _location_memory
    if _location_memory is None:
        reset_location_memory()
    return _location_memory


# ── geo utilities ─────────────────────────────────────────────────────────────

def load_agro_zones_geojson() -> Optional[gpd.GeoDataFrame]:
    """Cached GeoJSON — read from disk only once per process."""
    global _agro_zones_gdf
    if _agro_zones_gdf is not None:
        return _agro_zones_gdf
    try:
        _agro_zones_gdf = gpd.read_file("ali_try3_colors.geojson")
        print("location_based_zone: GeoJSON cached.")
        return _agro_zones_gdf
    except Exception as exc:
        print(f"Error loading GeoJSON: {exc}")
        return None


def get_location_name(lat: float, lon: float, max_retries: int = 3) -> str:
    geolocator = Nominatim(user_agent="agro_zone_app", timeout=10)
    for attempt in range(max_retries):
        try:
            location = geolocator.reverse((lat, lon), timeout=10)
            if location and location.raw and "address" in location.raw:
                address = location.raw["address"]
                return (
                    address.get("city") or address.get("town")
                    or address.get("village") or address.get("county") or "Unknown"
                )
            return "Unknown"
        except (GeocoderTimedOut, GeocoderServiceError):
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return "Unknown"
        except Exception:
            return "Unknown"
    return "Unknown"


def find_agro_zone_from_location(lat: float, lon: float) -> Optional[str]:
    try:
        agro_zones = load_agro_zones_geojson()
        if agro_zones is None:
            return None
        user_point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        matched = gpd.sjoin(user_point, agro_zones, how="left", predicate="within")
        if not matched.empty and not pd.isna(matched.iloc[0]["zone_name"]):
            return matched.iloc[0]["zone_name"]
        return None
    except Exception as exc:
        print(f"Error finding agro zone: {exc}")
        return None


# ── data loading ──────────────────────────────────────────────────────────────

def _build_zone_context(zone: str) -> str:
    """
    Read the agro-zones sheet and return the zone fields as a plain text block.
    This is injected directly into every prompt — never put into FAISS.
    """
    try:
        df1 = pd.read_excel("zone wise data.xlsx", sheet_name="agro zones")
        df1.columns = df1.columns.str.strip()

        zone_norm = zone.strip().lower()
        zone_data = df1[
            df1["Names"].str.strip().str.lower().str.contains(zone_norm, na=False)
        ]
        if zone_data.empty:
            zone_data = df1[df1["Names"].str.strip().str.lower() == zone_norm]

        if zone_data.empty:
            print(f"No zone rows found for: {zone}")
            return f"(No zone data found for {zone})"

        parts = []
        for _, row in zone_data.iterrows():
            parts.append(
                f"Zone identifier : {row.get('Zones', 'N/A')}\n"
                f"Zone name       : {row.get('Names', 'N/A')}\n"
                f"Climate         : {row.get('Climate', 'N/A')}\n"
                f"Districts       : {row.get('Districts', 'N/A')}\n"
                f"Soil types      : {row.get('Soil Types', 'N/A')}\n"
                f"Major crops     : {row.get('Major crops', 'N/A')}\n"
                f"Rainfall        : {row.get('Rain fall', 'N/A')}"
            )
        return "\n\n".join(parts)

    except Exception as exc:
        print(f"Error building zone context: {exc}")
        return "(Error reading zone data)"


def _load_qa_documents() -> List[Document]:
    """Load only the organic-farming Q&A sheet as documents for FAISS."""
    try:
        df2 = pd.read_excel("zone wise data.xlsx", sheet_name="organic farming")
        df2.columns = df2.columns.str.strip()

        docs: List[Document] = []
        for _, row in df2.iterrows():
            question = str(row.iloc[0]).strip() if len(row) > 0 and pd.notna(row.iloc[0]) else ""
            answer   = str(row.iloc[1]).strip() if len(row) > 1 and pd.notna(row.iloc[1]) else ""
            if question and answer and question.lower() != "nan" and answer.lower() != "nan":
                docs.append(
                    Document(
                        page_content=f"Q: {question}\nA: {answer}",
                        metadata={"source": "organic_farming"},
                    )
                )
        print(f"Loaded {len(docs)} Q&A documents.")
        return docs
    except Exception as exc:
        print(f"Error loading Q&A docs: {exc}")
        return []


# ── FAISS (Q&A only) with disk cache ─────────────────────────────────────────

def _cache_path(zone: str) -> str:
    return os.path.join(FAISS_CACHE_DIR, _zone_to_folder(zone))


def _load_qa_vectorstore_from_disk(zone: str) -> Optional[FAISS]:
    path = _cache_path(zone)
    if not os.path.exists(path):
        return None
    try:
        vs = FAISS.load_local(
            path, _get_embeddings(), allow_dangerous_deserialization=True
        )
        print(f"FAISS Q&A index loaded from disk: {path}")
        return vs
    except Exception as exc:
        print(f"Could not load FAISS cache ({exc}), will rebuild.")
        return None


def _save_qa_vectorstore_to_disk(vs: FAISS, zone: str) -> None:
    path = _cache_path(zone)
    try:
        os.makedirs(path, exist_ok=True)
        vs.save_local(path)
        print(f"FAISS Q&A index saved: {path}")
    except Exception as exc:
        print(f"Could not save FAISS cache: {exc}")


def _build_qa_vectorstore(zone: str) -> Optional[FAISS]:
    docs = _load_qa_documents()
    if not docs:
        return None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        texts = splitter.split_documents(docs)
        vs = FAISS.from_documents(texts, _get_embeddings())
        _save_qa_vectorstore_to_disk(vs, zone)
        print(f"Built FAISS Q&A index ({len(texts)} chunks).")
        return vs
    except Exception as exc:
        print(f"Error building vectorstore: {exc}")
        return None


# ── preload ───────────────────────────────────────────────────────────────────

def preload_location_zone_data(zone: str) -> bool:
    global _qa_vectorstore, _zone_context, _current_location_zone

    try:
        if _current_location_zone == zone and _qa_vectorstore is not None:
            print(f"Already loaded for zone: {zone}")
            return True

        print(f"Preloading for zone: {zone}")
        reset_location_memory()

        # Always rebuild zone context string (cheap — just reads xlsx)
        _zone_context = _build_zone_context(zone)

        # FAISS: try disk cache first, build if missing
        vs = _load_qa_vectorstore_from_disk(zone)
        if vs is None:
            vs = _build_qa_vectorstore(zone)
        if vs is None:
            print("Failed to create Q&A vectorstore")
            return False

        _qa_vectorstore = vs
        _current_location_zone = zone
        print(f"Preload complete for zone: {zone}")
        return True

    except Exception as exc:
        print(f"Error in preload: {exc}")
        return False


# ── response entry-point ──────────────────────────────────────────────────────

# Relevance threshold for Q&A retrieval.
# OpenAI embeddings are unit-normalised so FAISS returns L2 distance where:
#   L2 = sqrt(2 - 2*cosine_similarity)
# Highly relevant (cosine ≥ 0.90) → L2 ≤ 0.45  → passes threshold
# Loosely related  (cosine ≈ 0.80) → L2 ≈ 0.63  → blocked
# A threshold of 0.50 accepts only genuinely relevant Q&A chunks and blocks
# queries like "chicken farming" or "gardening" that have no dataset match.
_QA_RELEVANCE_THRESHOLD = 0.50


def get_location_zone_response(
    query: str, zone: Optional[str] = None, city_name: str = "Unknown"
) -> str:
    global _qa_vectorstore, _zone_context, _current_location_zone

    try:
        if not zone:
            return "Unable to detect your agro-ecological zone. Please ensure location access is enabled."

        if _current_location_zone != zone or _qa_vectorstore is None:
            if not preload_location_zone_data(zone):
                return (
                    f"Unable to load data for zone: {zone}. "
                    "Please check if 'zone wise data.xlsx' is available."
                )

        # Zone data always injected — never filtered by similarity
        zone_ctx = _zone_context

        # Retrieve Q&A chunks with L2 scores and apply relevance threshold.
        # Only chunks whose L2 distance ≤ _QA_RELEVANCE_THRESHOLD are kept.
        # This prevents loosely-related topics (chicken farming, gardening, etc.)
        # from leaking training knowledge through partial context matches.
        raw_hits = _qa_vectorstore.similarity_search_with_score(query, k=6)
        relevant = [doc for doc, score in raw_hits if score <= _QA_RELEVANCE_THRESHOLD]

        if relevant:
            qa_ctx = "\n\n---\n\n".join([doc.page_content for doc in relevant])
        else:
            qa_ctx = "NO RELEVANT Q&A FOUND IN DATABASE FOR THIS QUESTION."

        # Chat history
        memory = _get_memory()
        raw_history = memory.load_memory_variables({}).get("chat_history", [])
        history_lines = []
        for msg in raw_history:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"Farmer: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"Assistant: {msg.content}")
        chat_history_str = "\n".join(history_lines) if history_lines else "None"

        full_prompt = _PROMPT.format(
            city_name=city_name,
            zone=zone,
            chat_history=chat_history_str,
            zone_context=zone_ctx,
            qa_context=qa_ctx,
            question=query,
        )

        response = _get_llm().invoke(full_prompt)
        answer = response.content.strip()

        memory.save_context({"input": query}, {"output": answer})
        print(f"Answer: {answer[:100]}…")
        return answer

    except Exception as exc:
        print(f"Error: {exc}")
        return f"Error processing your query: {str(exc)}"