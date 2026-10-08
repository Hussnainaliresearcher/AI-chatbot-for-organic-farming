import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, ViTForImageClassification
from openai import OpenAI

def render_leaf_detection():
    st.markdown("""
    <style>
    [data-testid="stFileUploader"] { border: 2px dashed #a5d6a7; border-radius: 12px; padding: 4px 12px; background: #f1f8e9; }
    .img-card { border-radius: 16px; overflow: hidden; box-shadow: 0 8px 32px rgba(30,80,30,0.13); background: #e8f5e9; padding: 8px; }
    .disease-badge { display: inline-block; background: linear-gradient(135deg,#2e7d32,#66bb6a); color: white; font-size: 1.25rem; font-weight: 700; padding: 7px 18px; border-radius: 50px; margin-bottom: 5px; }
    .healthy-badge { display: inline-block; background: linear-gradient(135deg,#1565c0,#42a5f5); color: white; font-size: 1.25rem; font-weight: 700; padding: 7px 18px; border-radius: 50px; margin-bottom: 5px; }
    .conf-pill { display: inline-block; background: #f1f8e9; color: #388e3c; border: 1.5px solid #a5d6a7; font-size: 1.4rem; padding: 2px 12px; border-radius: 50px; margin-bottom: 10px; }
    .advice-box { background: #fafffe; border-left: 4px solid #66bb6a; border-radius: 10px; padding: 12px 16px; font-size: 1.1rem; color: #2d4a2d; line-height: 1.65; }
    .healthy-box { background: #e3f2fd; border-left: 4px solid #42a5f5; border-radius: 10px; padding: 12px 16px; font-size: 1.1rem; color: #0d47a1; line-height: 1.65; }
    .bubble-you { background: #a5d6a7; border-radius: 14px 14px 4px 14px; padding: 7px 13px; font-size: 1.1rem; color: #1b5e20; margin: 4px 0 4px auto; max-width: 86%; }
    .bubble-ai  { background: #ffffff; border: 1.5px solid #c8e6c9; border-radius: 14px 14px 14px 4px; padding: 7px 13px; font-size: 1.1rem; color: #2d4a2d; max-width: 93%; margin: 4px 0; }
    .chat-label { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; color: #81c784; margin-bottom: 2px; }
    .stTextInput > div > div > input { border: 2px solid #4caf50 !important; border-radius: 8px !important; }
    .stTextInput > div > div > input:focus { border-color: #2e7d32 !important; box-shadow: 0 0 0 3px rgba(46,125,50,0.15) !important; }
    .stFormSubmitButton > button { background: #a5d6a7 !important; color: black !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    .stFormSubmitButton > button:hover { background: linear-gradient(135deg,#1b5e20,#4caf50) !important; }
    /* ── Pulsing dots loader ── */
    @keyframes dot-pulse {
        0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
        40%            { transform: scale(1.2); opacity: 1;    }
    }
    .typing-bubble {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #66bb6a 100%);
        padding: 12px 18px; border-radius: 20px 20px 20px 4px;
        box-shadow: 0 4px 14px rgba(27,94,32,0.35);
        margin: 18px 0 28px 0; max-width: 40%;
    }
    .typing-bubble .dot {
        width: 9px; height: 9px; border-radius: 50%;
        background: #ffffff;
        animation: dot-pulse 1.4s ease-in-out infinite;
    }
    .typing-bubble .dot:nth-child(1) { animation-delay: 0s;    }
    .typing-bubble .dot:nth-child(2) { animation-delay: 0.2s;  }
    .typing-bubble .dot:nth-child(3) { animation-delay: 0.4s;  }
    .typing-label {
        color: #ffffff; font-style: italic; font-weight: 600;
        font-size: 14px; margin-left: 4px; letter-spacing: 0.3px;
    }
    </style>
    """, unsafe_allow_html=True)

    CONFIDENCE_THRESHOLD = 50.0
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    uploaded_file = st.session_state.get("leaf_upload")

    for key, val in [("leaf_chat", []), ("leaf_last_file", None), ("leaf_advice", ""),
                     ("leaf_disease", ""), ("leaf_confidence", 0), ("leaf_pending_q", ""),
                     ("leaf_invalid", False), ("leaf_invalid_reason", ""), ("leaf_healthy", False)]:
        if key not in st.session_state:
            st.session_state[key] = val

    if uploaded_file != st.session_state.leaf_last_file:
        st.session_state.update(leaf_chat=[], leaf_advice="", leaf_disease="", leaf_pending_q="",
                                leaf_last_file=uploaded_file, leaf_invalid=False,
                                leaf_invalid_reason="", leaf_healthy=False)

    @st.cache_resource
    def load_model():
        proc = AutoImageProcessor.from_pretrained("wambugu71/crop_leaf_diseases_vit")
        mdl  = ViTForImageClassification.from_pretrained("wambugu71/crop_leaf_diseases_vit")
        return proc, mdl

    def get_advice(disease):
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.3,
            messages=[{"role": "user", "content":
                f"Disease: {disease}\nGive in exactly this format:\n"
                "<b>Description:</b> (one sentence)\n<b>Organic Treatment:</b>\n• point 1\n• point 2\n"
                "Simple language for farmers. Max 5 lines total. Use HTML <b> tags for bold."}])
        return r.choices[0].message.content

    def ask_chatbot(disease, question):
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.4,
            messages=[{"role": "user", "content":
                f"You are an agriculture expert specializing in crop diseases and organic farming. "
                f"Detected condition: {disease}. Farmer asks: {question}\n"
                "RULES: Only answer if the question is about agriculture, crop diseases, or organic farming. "
                "If unrelated, reply exactly: '⚠️ I can only answer questions about crop diseases and organic farming.' "
                "Otherwise answer in 4-5 lines max, focus on organic solutions, be direct."}])
        return r.choices[0].message.content

    if st.session_state.leaf_pending_q:
        q = st.session_state.leaf_pending_q
        st.session_state.leaf_pending_q = ""
        answer = ask_chatbot(st.session_state.leaf_disease, q)
        st.session_state.leaf_chat.extend([("You", q), ("AI", answer)])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # ── Run detection only once per upload ──
        # The spinner covers the FULL processing time including the GPT
        # get_advice() call. st.rerun() ensures the image columns only
        # render in the NEXT pass — so the spinner is never behind the image.
        if not st.session_state.leaf_disease and not st.session_state.leaf_invalid and not st.session_state.leaf_healthy:
            st.markdown("""
                <div style="display:flex; justify-content:center;
                            align-items:center; padding: 60px 0 40px 0;">
                    <div class="typing-bubble" style="max-width:none;">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <span class="typing-label">🔍 Analyzing image...</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            processor, model = load_model()

            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_id    = logits.argmax(-1).item()
            confidence = F.softmax(logits, dim=1)[0][pred_id].item() * 100
            label      = model.config.id2label[pred_id].replace("___", " ").replace("_", " ")

            if "invalid" in label.lower():
                st.session_state.leaf_invalid        = True
                st.session_state.leaf_invalid_reason = "not_crop"
                st.session_state.leaf_confidence     = confidence
            elif "healthy" in label.lower():
                st.session_state.leaf_healthy    = True
                st.session_state.leaf_disease    = label
                st.session_state.leaf_confidence = confidence
            elif confidence < CONFIDENCE_THRESHOLD:
                st.session_state.leaf_invalid        = True
                st.session_state.leaf_invalid_reason = "low_confidence"
                st.session_state.leaf_confidence     = confidence
            else:
                st.session_state.leaf_disease    = label
                st.session_state.leaf_confidence = confidence
                # get_advice() (GPT call) runs while dots are still visible
                st.session_state.leaf_advice     = get_advice(label)

            # Rerun now — results render in a clean pass, spinner never
            # overlaps the image columns
            st.rerun()

        # ── Only reached after processing is complete ──
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # ── Case A: Not a crop image ──
            if st.session_state.leaf_invalid and st.session_state.leaf_invalid_reason == "not_crop":
                st.markdown("""
                <div style="background:#fff3e0;border-left:4px solid #ff9800;border-radius:10px;padding:18px 20px;margin-top:20px;">
                    <div style="font-size:1.4rem;font-weight:700;color:#e65100;">⚠️ Not a Crop Leaf</div>
                    <div style="margin-top:8px;font-size:1.05rem;color:#bf360c;">
                        This image does not appear to be a crop or plant leaf.<br><br>
                        <b>Supported crops:</b> Rice, Wheat, Corn, Potato<br><br>
                        Please upload a clear leaf image from a supported crop.
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Case B: Low confidence (unrecognised leaf) ──
            elif st.session_state.leaf_invalid and st.session_state.leaf_invalid_reason == "low_confidence":
                st.markdown(f"""
                <div style="background:#fff3e0;border-left:4px solid #ff9800;border-radius:10px;padding:18px 20px;margin-top:20px;">
                    <div style="font-size:1.4rem;font-weight:700;color:#e65100;">⚠️ Unable to Identify</div>
                    <div style="margin-top:8px;font-size:1.05rem;color:#bf360c;">
                        The leaf could not be identified with enough certainty.<br><br>
                        <b>Confidence:</b> {st.session_state.leaf_confidence:.1f}% (minimum required: {CONFIDENCE_THRESHOLD:.0f}%)<br><br>
                        <b>Supported crops:</b> Rice, Wheat, Corn, Potato<br><br>
                        Please upload a clearer, well-lit leaf image.
                    </div>
                </div>""", unsafe_allow_html=True)

            # ── Case C: Healthy leaf ──
            elif st.session_state.leaf_healthy:
                with st.container(height=360, border=False):
                    if not st.session_state.leaf_chat:
                        st.markdown(f'<div class="healthy-badge">✅ {st.session_state.leaf_disease}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="conf-pill" style="margin-top:6px;">Confidence: {st.session_state.leaf_confidence:.1f}%</div>', unsafe_allow_html=True)
                        st.markdown("""
                        <div class="healthy-box" style="margin-top:8px;">
                            <b>Great news! 🎉</b> This leaf appears to be <b>healthy</b>.<br><br>
                            No disease or treatment is needed at this time.<br>
                            Continue good farming practices to maintain plant health.
                        </div>""", unsafe_allow_html=True)
                        st.markdown('<div style="color:red;font-size:0.99rem;margin-top:8px;">💬 Ask anything about healthy crop maintenance below…</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#388e3c;text-transform:uppercase;margin-bottom:4px;">🤖 Follow-up Questions</div>', unsafe_allow_html=True)
                        pairs = list(zip(st.session_state.leaf_chat[0::2], st.session_state.leaf_chat[1::2]))
                        for (_, q), (_, a) in reversed(pairs):
                            st.markdown(f'<div class="bubble-you"><div class="chat-label">You</div>{q}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="bubble-ai"><div class="chat-label">🌿 AI</div>{a}</div>', unsafe_allow_html=True)
                            st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)
                with st.form(key="leaf_chat_form", clear_on_submit=True):
                    fc1, fc2 = st.columns([5, 1])
                    with fc1:
                        question = st.text_input("", placeholder="e.g. How do I keep my crop healthy?", label_visibility="collapsed")
                    with fc2:
                        submitted = st.form_submit_button("Send")
                if submitted and question.strip():
                    st.session_state.leaf_pending_q = question.strip()
                    st.rerun()

            # ── Case D: Disease detected ──
            elif st.session_state.leaf_disease:
                with st.container(height=370, border=False):
                    if not st.session_state.leaf_chat:
                        st.markdown(f'<div class="disease-badge">🌱 {st.session_state.leaf_disease}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="conf-pill" style="margin-top:6px;">Confidence: {st.session_state.leaf_confidence:.1f}%</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="advice-box" style="margin-top:8px;">{st.session_state.leaf_advice.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                        st.markdown('<div style="color:red;font-size:0.99rem;margin-top:1px;">💬 Ask anything about this disease below…</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size:0.78rem;font-weight:600;color:#388e3c;text-transform:uppercase;margin-bottom:4px;">🤖 Follow-up Questions</div>', unsafe_allow_html=True)
                        pairs = list(zip(st.session_state.leaf_chat[0::2], st.session_state.leaf_chat[1::2]))
                        for (_, q), (_, a) in reversed(pairs):
                            st.markdown(f'<div class="bubble-you"><div class="chat-label">You</div>{q}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="bubble-ai"><div class="chat-label">🌿 AI</div>{a}</div>', unsafe_allow_html=True)
                            st.markdown("<hr style='margin:5px 0;'>", unsafe_allow_html=True)
                with st.form(key="leaf_chat_form", clear_on_submit=True):
                    fc1, fc2 = st.columns([5, 1])
                    with fc1:
                        question = st.text_input("", placeholder="e.g. How can I prevent this organically?", label_visibility="collapsed")
                    with fc2:
                        submitted = st.form_submit_button("Send")
                if submitted and question.strip():
                    st.session_state.leaf_pending_q = question.strip()
                    st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;">
            <div style="font-size:4rem;">🌿</div>
            <div style="font-size:1.2rem;color:#388e3c;margin-top:12px;">Upload a crop leaf image from the sidebar to detect diseases</div>
            <div style="font-size:0.88rem;margin-top:8px;color:#a5d6a7;">Supports JPG · JPEG · PNG &nbsp;|&nbsp; Crops: Rice · Wheat · Corn · Potato</div>
        </div>
        """, unsafe_allow_html=True)