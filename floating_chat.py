"""
floating_chat.py
──────────────────
A small, reusable "floating chat" widget for Streamlit.
Framework-agnostic: pass a title, an `on_ask(question) -> answer` callback, 
and optionally a few suggested quick-reply questions.
"""

import time
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components


def _container(key: str):
    """Safely attempts to create a container with a CSS-targetable key."""
    try:
        # Streamlit >= 1.35 supports this natively
        return st.container(key=key)
    except TypeError:
        return st.container()


def _inject_css(namespace: str):
    fab, panel, msgs = f"{namespace}_fab", f"{namespace}_panel", f"{namespace}_messages"
    st.markdown(f"""
        <style>
            /* Force the floating action button to the bottom right */
            .st-key-{fab} {{ 
                position: fixed !important; 
                bottom: 24px !important; 
                right: 24px !important; 
                z-index: 99998 !important; 
                width: auto !important;
                height: auto !important;
            }}
            .st-key-{fab} [data-testid="stButton"] button {{
                width: auto !important; 
                height: auto !important; 
                border-radius: 30px !important;  /* Stylish Pill Shape */
                border: none !important;
                background: linear-gradient(135deg, #1b5e20 0%, #4caf50 100%) !important;
                color: #fff !important; 
                padding: 12px 24px !important;
                box-shadow: 0 8px 22px rgba(27,94,32,0.4) !important;
                transition: transform 0.15s ease, background 0.15s ease !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }}
            .st-key-{fab} [data-testid="stButton"] button p {{
                font-size: 16px !important;
                font-weight: 700 !important;
                margin: 0 !important;
                line-height: 1 !important;
                white-space: nowrap !important;
            }}
            .st-key-{fab} [data-testid="stButton"] button:hover {{ 
                transform: translateY(-2px) !important; 
                background: linear-gradient(135deg, #1b5e20 0%, #66bb6a 100%) !important; 
                box-shadow: 0 12px 28px rgba(27,94,32,0.5) !important;
            }}

            /* Chat Panel Styles */
            .st-key-{panel} {{
                position: fixed !important; 
                bottom: 92px !important; 
                right: 24px !important; 
                width: 360px !important; 
                max-width: 90vw !important;
                background: #ffffff !important; 
                border-radius: 16px !important; 
                box-shadow: 0 16px 48px rgba(0,0,0,0.28) !important;
                z-index: 99997 !important; 
                overflow: hidden !important; 
                border: 1px solid #d8ecd8 !important; 
                padding-bottom: 6px !important;
            }}
            .{namespace}-panel-header {{
                background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #66bb6a 100%);
                color: #fff; padding: 14px 16px; font-weight: 700; font-size: 15px;
            }}
            .{namespace}-panel-sub {{ font-weight: 400; font-size: 12px; opacity: 0.85; margin-top: 2px; }}

            .st-key-{msgs} {{ max-height: 320px; overflow-y: auto; padding: 12px 14px 4px 14px; background:#fafffe; scroll-behavior: smooth; }}

            .{namespace}-empty {{ color:#5a7a5a; font-size: 0.9rem; text-align:center; padding: 16px 6px 6px 6px; }}

            .{namespace}-bubble-you {{ background:#a5d6a7; border-radius:14px 14px 4px 14px; padding:7px 12px; font-size:0.94rem; color:#1b5e20; margin:4px 0 4px auto; max-width:88%; width:fit-content; }}
            .{namespace}-bubble-ai  {{ background:#ffffff; border:1.5px solid #c8e6c9; border-radius:14px 14px 14px 4px; padding:7px 12px; font-size:0.94rem; color:#2d4a2d; max-width:92%; width:fit-content; margin:4px 0; }}
            .{namespace}-label {{ font-size:0.66rem; font-weight:700; letter-spacing:0.4px; text-transform:uppercase; color:#7bab7b; margin-bottom:2px; }}

            @keyframes {namespace}-dot-pulse {{
                0%, 80%, 100% {{ transform: scale(0.6); opacity: 0.35; }}
                40% {{ transform: scale(1.2); opacity: 1; }}
            }}
            .{namespace}-typing {{ display:inline-flex; align-items:center; gap:5px; padding: 4px 2px 10px 2px; }}
            .{namespace}-typing .dot {{ width:7px; height:7px; border-radius:50%; background:#4caf50; animation:{namespace}-dot-pulse 1.4s ease-in-out infinite; }}
            .{namespace}-typing .dot:nth-child(2) {{ animation-delay:0.2s; }}
            .{namespace}-typing .dot:nth-child(3) {{ animation-delay:0.4s; }}

            /* quick-reply suggestion chips */
            .st-key-{panel} div[data-testid="stVerticalBlockBorderWrapper"] .stButton>button,
            .st-key-{panel} .stButton>button {{
                background:#e8f5e8 !important; color:#1b5e20 !important; border:1px solid #a5d6a7 !important; border-radius:14px !important;
                font-size:12px !important; padding:4px 10px !important; font-weight:600 !important; white-space:normal !important; height:auto !important;
            }}
            .st-key-{panel} .stButton>button:hover {{ background:#c8e6c9 !important; }}

            /* send button */
            .st-key-{panel} .stFormSubmitButton>button {{
                background:#a5d6a7 !important; color:#000 !important; border:none !important; border-radius:8px !important; font-weight:700 !important;
            }}
            .st-key-{panel} .stFormSubmitButton>button:hover {{ background:#4caf50 !important; color:#fff !important; }}
        </style>
    """, unsafe_allow_html=True)


def render_floating_chat(
    namespace: str,
    title: str,
    on_ask,
    subtitle: Optional[str] = None,
    suggested_questions: Optional[list[str]] = None,
    empty_message: str = "Ask a question to get started…",
    placeholder: str = "Type your question…",
):
    _inject_css(namespace)

    open_key, chat_key, pending_key = f"{namespace}_open", f"{namespace}_chat", f"{namespace}_pending_q"
    for k, v in [(open_key, False), (chat_key, []), (pending_key, "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── floating toggle button ──
    with _container(f"{namespace}_fab"):
        label = "✕ Close Chat" if st.session_state[open_key] else "💬 Ask a question to get more insights"
        if st.button(label, key=f"{namespace}_fab_btn"):
            st.session_state[open_key] = not st.session_state[open_key]
            st.rerun()

    if not st.session_state[open_key]:
        return

    chat = st.session_state[chat_key]
    pending = st.session_state[pending_key]

    # ── chat panel ──
    with _container(f"{namespace}_panel"):
        sub_html = f'<div class="{namespace}-panel-sub">{subtitle}</div>' if subtitle else ""
        st.markdown(f'<div class="{namespace}-panel-header">{title}{sub_html}</div>', unsafe_allow_html=True)

        with _container(f"{namespace}_messages"):
            if not chat and not pending:
                st.markdown(f'<div class="{namespace}-empty">{empty_message}</div>', unsafe_allow_html=True)
            else:
                pairs = list(zip(chat[0::2], chat[1::2]))
                for (_, q), (_, a) in pairs:
                    st.markdown(f'<div class="{namespace}-bubble-you"><div class="{namespace}-label">You</div>{q}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="{namespace}-bubble-ai"><div class="{namespace}-label">🌿 AI</div>{a}</div>', unsafe_allow_html=True)
            if pending:
                st.markdown(
                    f'<div class="{namespace}-typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
                    unsafe_allow_html=True,
                )

        if suggested_questions and not chat and not pending:
            cols = st.columns(len(suggested_questions))
            for i, sq in enumerate(suggested_questions):
                if cols[i].button(sq, key=f"{namespace}_chip_{i}", use_container_width=True):
                    st.session_state[pending_key] = sq
                    st.rerun()

        with st.form(key=f"{namespace}_form", clear_on_submit=True):
            fc1, fc2 = st.columns([5, 1])
            with fc1:
                q_input = st.text_input("", placeholder=placeholder, label_visibility="collapsed", key=f"{namespace}_input")
            with fc2:
                submitted = st.form_submit_button("➤")
        if submitted and q_input.strip():
            st.session_state[pending_key] = q_input.strip()
            st.rerun()

    # ── Dynamic Auto-Scroll Hack ──
    # The unique timestamp ensures Streamlit runs this JS every single time the DOM updates.
    # The 150ms delay gives Streamlit time to actually draw the AI's text onto the screen.
    current_time = time.time()
    components.html(
        f"""
        <script>
            // Execution block: {current_time}
            setTimeout(function() {{
                var parent = window.parent.document;
                var msgs = parent.querySelector('.st-key-{namespace}_messages');
                if (msgs) {{ msgs.scrollTop = msgs.scrollHeight; }}
            }}, 150);
            setTimeout(function() {{
                var parent = window.parent.document;
                var msgs = parent.querySelector('.st-key-{namespace}_messages');
                if (msgs) {{ msgs.scrollTop = msgs.scrollHeight; }}
            }}, 400); // Backup guarantee 
        </script>
        """,
        height=0, width=0
    )

    if pending:
        answer = on_ask(pending)
        st.session_state[chat_key].extend([("you", pending), ("ai", answer)])
        st.session_state[pending_key] = ""
        st.rerun()


def reset_floating_chat_state(namespace: str):
    """Clears a floating chat's conversation + open/closed state."""
    for suffix in ("_open", "_chat", "_pending_q"):
        key = f"{namespace}{suffix}"
        if key in st.session_state:
            del st.session_state[key]