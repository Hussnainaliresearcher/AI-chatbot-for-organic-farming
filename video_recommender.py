"""
video_recommender.py
--------------------
Recommends a relevant farming video using an LLM judge.

Fix 2: model changed to gpt-4o-mini — 10x faster and 30x cheaper than
       gpt-4-turbo for this trivial single-digit classification task.

Reads the 'video' sheet from 'zone wise data.xlsx'.
Columns expected: topic | description | video_url
"""

import pandas as pd
from typing import List, Optional, Tuple
from langchain_openai import ChatOpenAI

# ── module-level cache ────────────────────────────────────────────────────────
_video_records: List[Tuple[str, str, str]] = []   # (topic, description, url)
_video_data_loaded: bool = False
_llm: Optional[ChatOpenAI] = None

DATA_FILE   = "zone wise data.xlsx"
VIDEO_SHEET = "video"


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        # Fix 2: gpt-4o-mini is fast enough for outputting one digit or "none"
        _llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=10)
    return _llm


# ── public API ────────────────────────────────────────────────────────────────

def load_video_data() -> bool:
    """
    Load the 'video' sheet from zone wise data.xlsx and cache records.
    Safe to call multiple times — returns immediately if already loaded.
    """
    global _video_records, _video_data_loaded

    if _video_data_loaded:
        return True

    try:
        df = pd.read_excel(DATA_FILE, sheet_name=VIDEO_SHEET)
        df.columns = df.columns.str.strip()

        records = []
        for _, row in df.iterrows():
            topic = str(row.get("topic", "")).strip()
            desc  = str(row.get("description", "")).strip()
            url   = str(row.get("video_url", "")).strip()
            if not topic and not desc:
                continue
            records.append((topic, desc, url))

        if not records:
            print("video_recommender: no rows found in 'video' sheet.")
            return False

        _video_records     = records
        _video_data_loaded = True
        print(f"video_recommender: loaded {len(records)} video records.")
        return True

    except Exception as exc:
        print(f"video_recommender: error loading video data — {exc}")
        return False


def find_best_video(query: str) -> Optional[str]:
    """
    Ask the LLM which video (if any) directly answers *query*.
    Returns the matching URL or None.
    """
    global _video_records

    if not _video_data_loaded:
        load_video_data()

    if not _video_records:
        return None

    video_list_text = "\n".join(
        f"{i + 1}. Topic: {topic} | Description: {desc}"
        for i, (topic, desc, _) in enumerate(_video_records)
    )

    prompt = (
        "You are a strict video-relevance judge for an organic farming assistant.\n\n"
        f"User question: \"{query}\"\n\n"
        "Available videos:\n"
        f"{video_list_text}\n\n"
        "Task: Reply with ONLY the number of the video whose topic and description "
        "directly and specifically address the user's question. "
        "If no video is a direct match, reply with exactly: none\n\n"
        "Rules:\n"
        "- Do NOT pick a video just because it is vaguely related to farming.\n"
        "- The video must clearly cover the specific subject the user asked about.\n"
        "- If the closest video is only loosely related, reply: none\n\n"
        "Reply (a single number or the word none):"
    )

    try:
        result = _get_llm().invoke(prompt)
        raw    = result.content.strip().lower()

        print(f"video_recommender: LLM judge replied '{raw}' for query: {query[:60]}")

        if raw == "none" or raw == "":
            return None

        digits = "".join(ch for ch in raw if ch.isdigit())
        if not digits:
            return None

        idx = int(digits) - 1
        if 0 <= idx < len(_video_records):
            url = _video_records[idx][2]
            print(f"video_recommender: selected video #{idx + 1} — {_video_records[idx][0]}")
            return url if url else None

        return None

    except Exception as exc:
        print(f"video_recommender: LLM judge error — {exc}")
        return None