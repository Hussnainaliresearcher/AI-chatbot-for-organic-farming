import streamlit as st
from land_prep import get_land_prep_response, preload_agro_data, get_province_districts
from prep_zone import get_zone_prep_response, preload_zone_data, get_agro_zones
from web_scraper import get_web_scraper_response, preload_web_store_data
from pakistan_context import get_pakistan_context_response, preload_pakistan_context_data
from location_based_zone import (
    get_location_zone_response,
    preload_location_zone_data,
    find_agro_zone_from_location,
    get_location_name,
    load_agro_zones_geojson,
    reset_location_memory,          # ← new import
)
from video_recommender import find_best_video, load_video_data  # ← new import
from weather_forecasting import render_weather_forecast
from leaf import render_leaf_detection
from satellite_insights import (               # ← satellite farm insights
    render_satellite_sidebar,
    render_satellite_main,
    reset_satellite_result_state,
)
from datetime import datetime, timedelta
import time
import random
from concurrent.futures import ThreadPoolExecutor
import textwrap
import base64
import folium
from streamlit_folium import st_folium
from streamlit_javascript import st_javascript
import geopandas as gpd
from shapely.geometry import Point
import requests

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_location_details(lat, lon):
    district_name = "Unknown District"
    exact_location = "Unknown Location"
    try:
        WEATHER_API_KEY = "fdec585c9ae956ef0e6e8c6e88016663"
        WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
        response = requests.get(WEATHER_URL, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            district_name = data.get("name", "Unknown District")
    except Exception as e:
        print(f"Error getting district from OpenWeatherMap: {e}")
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&addressdetails=1"
        response = requests.get(url, headers={'User-Agent': 'OrganicFarmingApp/1.0'}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            exact_location = (address.get('suburb') or address.get('neighbourhood') or
                              address.get('village') or address.get('town') or
                              address.get('hamlet') or address.get('locality') or district_name)
    except Exception as e:
        print(f"Error getting exact location from Nominatim: {e}")
        exact_location = district_name
    return district_name, exact_location

def inject_custom_css():
    st.markdown("""
        <style>
            .main-container { margin-left: 23rem; }
            .fixed-header {
                position: fixed; top: 0; left: 23rem; right: 2rem;
                background-color: #a5d6a7; z-index: 1000;
                padding: 2.25rem 1.25rem 0.75rem 0.15rem;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                border-radius: 0 0 10px 10px; display: flex;
                align-items: center; justify-content: space-between;
            }
            .header-content { flex: 1; }
            .fixed-header h1 { margin-bottom: 0; font-size: 28px; color: #000; }
            .tag-label { font-size: 14px; font-weight: 600; color: #1b5e20; margin-bottom: -1px; margin-top: -22px; }
            .tag-strip { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: -10px; }
            .tag-pill { background: #f5f5f5; color: #000; padding: 6px 11px; border-radius: 16px; font-weight: 500; font-size: 14px; }
            .header-logo { height: 68px; width: auto; max-width: 110px; object-fit: contain;margin-top: 25px; margin-left: 12px; border-radius: 6px; }
            .main-content { margin-top: 130px; padding: 0 1rem; }
            section[data-testid="stSidebar"] {
                background-color: #a5d6a7 !important;
                border-right: 1px solid #d2e9d1;
                min-width: 370px !important; max-width: 370px !important;
            }
            section[data-testid="stSidebar"] > div:first-child { padding: 6px 8px !important; }
            section[data-testid="stSidebar"] .element-container { margin-bottom: 4px !important; }
            section[data-testid="stSidebar"] .stSelectbox,
            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] .stAlert { margin-bottom: 4px !important; }
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] h4 { margin: 2px 0 !important; padding: 0 !important; line-height: 1.15; }
            section[data-testid="stSidebar"] .stButton>button {
                background-color: #f5f5f5; color: #000000; border-radius: 10px;
                font-weight: 700; padding: 10px 14px; width: 100%;
                margin: 6px 0 4px 0; border: 1px solid rgba(0,0,0,0.08);
            }
            section[data-testid="stSidebar"] .stButton>button:hover { background-color: #4caf50; color: #fff; }
            .sb-map-title {
                color: black; background: #ffcc80; font-weight: 700; font-size: 16px;
                padding: 6px 8px; border-radius: 6px; border-left: 4px solid #4caf50;
                text-align: left; box-sizing: border-box; line-height: 1.4;
            }
            .sb-map-wrap { margin: 0; padding: 0; background: #a5d6a7 !important; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
            .sb-map-wrap iframe { display: block; width: 100% !important; height: 300px !important; border: 0 !important; margin: 0 !important; padding: 0 !important; background: #a5d6a7 !important; }
            .main-chat-area { max-height: calc(100vh - 150px); overflow-y: auto; padding-bottom: 1.25rem; }
            .chat-wrapper { display: flex; flex-direction: column; gap: 10px; }
            .chat-container { display: flex; align-items: flex-start; max-width: 85%; border-radius: 10px; padding: 8px 12px; word-wrap: break-word; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 16px; line-height: 1.45; }
            .user-message { background-color: #a5d6a7; border: 1px solid #b5e7c4; margin-left: auto; }
            .bot-message { background-color: #f5f5f5; border: 1px solid #ddd; margin-right: auto; }
            .avatar { font-size: 24px; margin: 2px 8px; }
            .timestamp { font-size: 14px; color: #888; margin-top: 4px; }
            .current-box { background: linear-gradient(180deg,#bbf7d0 0%,#86efac 100%); margin-top: -145px; padding:10px;border-radius:9px;box-shadow:0 6px 18px rgba(4,120,87,0.12); text-align:center;border:1px solid rgba(0,0,0,0.06); }
            .current-temp {font-size:1.9rem;font-weight:700;color:#064e3b;margin:6px 0;}
            .current-emoji {font-size:50px;line-height:1;margin-bottom:4px;}
            .hour-box, .day-box {padding:10px;border-radius:12px;text-align:center;box-shadow:0 6px 18px rgba(2,6,23,0.04);}
            .muted {color:#334155;font-size:0.95rem;margin-top:2px;}
            @media (max-width: 768px) {
                .main-container { margin-left: 0; }
                .fixed-header { left: 0; right: 0; width: 100%; border-radius: 0; padding: 12px; }
                .header-logo { height: 48px; margin-left: 6px; }
                .fixed-header h1 { font-size: 22px; }
                .main-content { margin-top: 120px; padding: 0 0.5rem; }
                .chat-container { max-width: 100%; font-size: 15px; }
                .sb-map-wrap iframe { height: 240px !important; }
            }
            /* ── Video expander green styling ── */
            div[data-testid="stExpander"] {
                background-color: #a5d6a7 !important;
                border: 1px solid #4caf50 !important;
                border-radius: 8px !important;
            }
            div[data-testid="stExpander"] summary {
                background-color: #a5d6a7 !important;
                color: #1b5e20 !important;
                font-weight: 700 !important;
                border-radius: 8px !important;
            }
            div[data-testid="stExpander"] summary:hover {
                background-color: #81c784 !important;
            }
            div[data-testid="stExpander"] details[open] summary {
                border-radius: 8px 8px 0 0 !important;
            }
            /* ── Custom spinner ── */
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
                margin: 6px 0; max-width: 25%;
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

def render_chat_message(role, content, timestamp):
    avatar = "👨‍🌾" if role == "user" else "🤖"
    css_class = "user-message" if role == "user" else "bot-message"
    st.markdown(f"""
        <div class="chat-wrapper">
            <div class="chat-container {css_class}">
                <div class="avatar">{avatar}</div>
                <div>
                    <div>{content}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_fixed_header():
    try:
        logo_base64 = get_base64_of_bin_file("images.png")
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="header-logo" alt="Organization Logo">'
    except FileNotFoundError:
        logo_html = '<div class="header-logo" style="background-color:#f0f0f0;display:flex;align-items:center;justify-content:center;color:#666;">LOGO</div>'
    st.markdown(f"""
        <div class="fixed-header">
            <div class="header-content">
                <h1>🌱 AI-Chatbot 🤖 for Organic Farming in Pakistan🌿</h1>
                <div class="tag-label">Ask Questions About:</div>
                <div class="tag-strip">
                    <span class="tag-pill">Organic Farming</span>
                    <span class="tag-pill">Major Crops</span>
                    <span class="tag-pill">Soil Types</span>
                    <span class="tag-pill">Climate</span>
                    <span class="tag-pill">Rain Fall</span>
                    <span class="tag-pill">Pakistan Context</span>
                </div>
            </div>
            {logo_html}
        </div>
    """, unsafe_allow_html=True)

def render_sidebar_map(lat, lon, zone_name, district_name, exact_location):
    try:
        agro_zones = load_agro_zones_geojson()
        if agro_zones is None:
            st.error("Unable to load agro-ecological zones data.")
            return
        location_display = f"📍 {exact_location}, {district_name}" if district_name != exact_location else f"📍 {district_name}"
        st.markdown(f'<div class="sb-map-title">🌍 {location_display}<br/><small style="font-size:13px;">Agro Zone: {zone_name}</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-map-wrap">', unsafe_allow_html=True)
        m = folium.Map(location=[lat, lon], zoom_start=8, width='100%', height=300, tiles='OpenStreetMap')
        popup_text = f"Location: {exact_location}\nDistrict: {district_name}\nZone: {zone_name}" if district_name != exact_location else f"District: {district_name}\nZone: {zone_name}"
        tooltip_text = f"{exact_location}, {district_name}" if district_name != exact_location else district_name
        folium.Marker([lat, lon], tooltip=tooltip_text, popup=popup_text, icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        user_point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
        matched = gpd.sjoin(user_point, agro_zones, how="left", predicate="within")
        if not matched.empty:
            zone_geom = agro_zones[agro_zones['zone_name'] == zone_name].geometry.iloc[0]
            folium.GeoJson(zone_geom, style_function=lambda x: {"fillColor": "#4daf4a", "color": "#2c7fb8", "weight": 2, "fillOpacity": 0.3}, tooltip=f"Agro Zone: {zone_name}").add_to(m)
        st_folium(m, width=None, height=300, returned_objects=[], key=f"sidebar_map_{zone_name}_{district_name}_{exact_location}")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering map: {str(e)}")

def show_location_detection_message():
    st.markdown("""
    <div style="text-align:center;padding:28px;margin:48px auto;background:#e8f5e8;border-radius:10px;border-left:4px solid #4caf50;max-width:600px;">
        <h3 style="color:#1b5e20;margin:0 0 6px 0;">📍 Location Based Detection</h3>
        <p style="color:#2e7d32;margin:0;">📡 Please turn on your location in the browser to detect your agro-ecological zone</p>
    </div>
    """, unsafe_allow_html=True)

def handle_option_change():
    st.session_state.chat_history = []
    st.session_state.data_loaded = False
    st.session_state.loading_status = "Processing..."
    for key in ['current_district', 'current_province', 'current_zone']:
        if key in st.session_state:
            del st.session_state[key]
    # ── reset conversation memory so history doesn't bleed across options ──
    try:
        reset_location_memory()
    except Exception:
        pass
    # ── reset satellite results so a stale field map/briefing never lingers
    #    on top of an unrelated option ────────────────────────────────────
    try:
        reset_satellite_result_state()
    except Exception:
        pass

def main():
    st.set_page_config(page_title="🌱 Organic Farming Assistant", page_icon="🌿", layout="wide")
    inject_custom_css()

    for key, val in [("chat_history", []), ("data_loaded", False), ("current_option", "Location Based"),
                     ("location_coords", None), ("location_zone", None), ("loading_status", "Processing..."),
                     ("initial_load_done", False)]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── pre-load video embeddings once per session ──────────────────────────
    if not st.session_state.get("video_data_loaded"):
        load_video_data()
        st.session_state.video_data_loaded = True

    with st.sidebar:
        st.markdown("## 🌿 Select Your Interest 🌿")
        option = st.selectbox(
            "Info of Land Preparation by..:",
            ["Location Based", "Weather Forecast", "District Wise", "Agro Zone Wise",
             "All Pakistan Context", "Online Organic Store", "Crop Disease Detection",
             "Satellite Farm Insights"],
            index=0, on_change=handle_option_change, key="option_selector"
        )
        st.session_state.current_option = option

        if option == "Weather Forecast":
            st.markdown("### 🌤️ Weather Forecast for Farmers")
            st.info("Get detailed weather information including rain probability for better farming decisions!")

        elif option == "Location Based":
            coords = st_javascript("""await new Promise((resolve) => {
                navigator.geolocation.getCurrentPosition(
                    (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
                    (err) => resolve(null)
                );
            });""")
            if coords:
                lat, lon = coords
                st.session_state.location_coords = coords
                district_name, exact_location = get_location_details(lat, lon)
                detected_zone = find_agro_zone_from_location(lat, lon)
                st.session_state.location_zone = detected_zone
                st.session_state.location_district = district_name
                st.session_state.location_exact = exact_location
                if detected_zone and not st.session_state.data_loaded:
                    with st.spinner("🔄 Loading agricultural data for your location..."):
                        preload_success = preload_location_zone_data(detected_zone)
                        st.session_state.data_loaded = preload_success
                        if preload_success:
                            st.success(f"✅ Ready for {detected_zone} zone!")
                        else:
                            st.error("❌ Failed to load data for your zone.")
                if detected_zone:
                    render_sidebar_map(lat, lon, detected_zone, district_name, exact_location)
                else:
                    st.warning("⚠️ Unable to detect agro-ecological zone for your location")
            else:
                st.warning("⚠️ Location access denied or not available")

        elif option == "District Wise":
            province_districts = get_province_districts()
            st.markdown("### 🗼️ Location")
            if 'current_province' not in st.session_state:
                st.session_state.current_province = list(province_districts.keys())[0]
            current_province = st.selectbox("Select Province", list(province_districts.keys()),
                index=list(province_districts.keys()).index(st.session_state.current_province) if st.session_state.current_province in province_districts else 0,
                key="province_selector")
            st.session_state.current_province = current_province
            available_districts = province_districts[current_province]
            if 'current_district' not in st.session_state or st.session_state.current_district not in available_districts:
                st.session_state.current_district = available_districts[0]
            current_district = st.selectbox("Select District", available_districts,
                index=available_districts.index(st.session_state.current_district) if st.session_state.current_district in available_districts else 0,
                key="district_selector")
            st.session_state.current_district = current_district
            district_key = f"{current_province}_{current_district}"
            if not st.session_state.data_loaded or st.session_state.get('loaded_district_key') != district_key:
                with st.spinner("🔄 Loading agricultural data..."):
                    preload_success = preload_agro_data(current_province, current_district)
                    st.session_state.data_loaded = preload_success
                    st.session_state.loaded_district_key = district_key
                    if preload_success:
                        st.success(f"✅ Ready for {current_district}, {current_province}!")
                    else:
                        st.error(f"❌ Failed to load data for {current_district}, {current_province}")

        elif option == "Agro Zone Wise":
            agro_zones = get_agro_zones()
            st.markdown("### 🌍 Agro-Ecological Zone")
            if 'current_zone' not in st.session_state:
                st.session_state.current_zone = agro_zones[0]
            current_zone = st.selectbox("Select Zone", agro_zones,
                index=agro_zones.index(st.session_state.current_zone) if st.session_state.current_zone in agro_zones else 0,
                key="zone_selector")
            st.session_state.current_zone = current_zone
            if not st.session_state.data_loaded or st.session_state.get('loaded_zone') != current_zone:
                with st.spinner("🔄 Loading zone data..."):
                    preload_success = preload_zone_data(current_zone)
                    st.session_state.data_loaded = preload_success
                    st.session_state.loaded_zone = current_zone
                    if preload_success:
                        st.success(f"✅ Ready for {current_zone} zone!")
                    else:
                        st.error(f"❌ Failed to load data for {current_zone} zone")

        elif option == "All Pakistan Context":
            st.markdown("### 🇵🇰 Pakistan-Wide Agricultural Information")
            st.info("Ask questions about crops, climate, soil types across all districts and provinces of Pakistan!")
            if not st.session_state.data_loaded:
                with st.spinner("🔄 Loading Pakistan-wide data..."):
                    preload_success = preload_pakistan_context_data()
                    st.session_state.data_loaded = preload_success
                    if preload_success:
                        st.success("✅ Ready for Pakistan-wide queries!")

        elif option == "Online Organic Store":
            st.markdown("### 🛒 Online Organic Store Information")
            st.info("Ask questions about products, availability, and more.")
            if not st.session_state.data_loaded:
                with st.spinner("🔄 Loading store data..."):
                    preload_success = preload_web_store_data()
                    st.session_state.data_loaded = preload_success
                    if preload_success:
                        st.success("✅ Ready for Web Store queries!")

        elif option == "Crop Disease Detection":
            st.markdown("### 🌾 Crop Disease Detection")
            st.info("Supported crops: Rice · Wheat · Corn · Potato")
            st.file_uploader("Upload a crop leaf image", type=["jpg","jpeg","png"],
                             key="leaf_upload", label_visibility="collapsed")

        elif option == "Satellite Farm Insights":
            st.markdown("### 🛰️ Satellite Farm Insights")
            st.info("Draw your field boundary and pull live NDVI, soil moisture, temperature and terrain from satellite data.")
            # This renders all boundary/date/cloud controls in the sidebar and
            # returns True only on the run where "Fetch Satellite Data" was clicked.
            st.session_state["_sat_fetch_clicked"] = render_satellite_sidebar()

        example_queries = {
            "District Wise": ["What are the major crops?", "What are the soil types here?",
                "Which organic materials are most effective for improving soil health?",
                "What is the climate like?", "Best farming practices for this district"],
            "Agro Zone Wise": ["What are the major crops in this zone?", "What are the soil types in this zone?",
                "What is the climate in this agro-ecological zone?", "Best crops for this agro-ecological zone"],
            "Location Based": ["What are the major crops in my area?", "What are the soil types here?",
                "What is the climate in my zone?", "Which organic materials work best here?",
                "Best farming practices for my location"],
            "All Pakistan Context": ["Where can I grow mangoes in Pakistan?",
                "Which districts are best for wheat cultivation?", "Soil types in Lahore district",
                "Best districts for rice cultivation"],
            "Online Organic Store": ["What organic seeds are available?", "Is organic honey available?",
                "Please tell the products available."],
            "Weather Forecast": ["Weather is displayed automatically", "Location-based forecast",
                "Rain probability for farming decisions"],
            "Crop Disease Detection": ["Upload a leaf image to detect disease",
                "Ask about organic treatment after detection", "Get prevention tips for detected disease"],
            "Satellite Farm Insights": ["Fetch satellite data to see your field's health",
                "Why is my NDVI low?", "What does 'Dry' soil moisture status mean?",
                "Any organic fix for low canopy water content?"],
        }

        if option in example_queries:
            st.markdown("**💡 Example queries:**")
            for query in example_queries[option]:
                st.markdown(f"- {query}")

    render_fixed_header()

    if option == "Weather Forecast":
        st.markdown('<div class="main-container"><div class="main-content">', unsafe_allow_html=True)
        render_weather_forecast()
        st.markdown('</div></div>', unsafe_allow_html=True)
        return

    if option == "Crop Disease Detection":
        st.markdown('<div class="main-container"><div class="main-content">', unsafe_allow_html=True)
        render_leaf_detection()
        st.markdown('</div></div>', unsafe_allow_html=True)
        return

    if option == "Satellite Farm Insights":
        st.markdown('<div class="main-container"><div class="main-content">', unsafe_allow_html=True)
        render_satellite_main()
        st.markdown('</div></div>', unsafe_allow_html=True)
        return

    if (st.session_state.current_option == "Location Based" and
            (not st.session_state.location_coords or not st.session_state.location_zone)):
        show_location_detection_message()

    # ── render chat history ─────────────────────────────────────────────────
    st.markdown('<div class="main-container"><div class="main-content"><div class="main-chat-area">', unsafe_allow_html=True)

    for chat in st.session_state.chat_history:
        render_chat_message(chat["role"], chat["content"], chat["timestamp"])
        # Video sits inside a collapsed expander — answer is always visible first,
        # farmer opens the expander themselves when they want to watch the video.
        if chat.get("video_url"):
            col_vid, _ = st.columns([3, 1])
            with col_vid:
                with st.expander("🎥 Watch Related Farming Video", expanded=False):
                    st.video(chat["video_url"])

    st.markdown('</div></div></div>', unsafe_allow_html=True)

    user_question = st.chat_input("Ask your farming question...", key="chat_input_main")

    if user_question:
        user_timestamp = datetime.now().strftime("%I:%M %p")

        # ── Step 1: show user message immediately on screen ──────────────
        # Render it right here before any LLM call so the farmer sees their
        # question appear instantly without waiting for the answer.
        render_chat_message("user", user_question, user_timestamp)

        # ── Step 2: spinner + parallel LLM calls ─────────────────────────
        current_option   = st.session_state.current_option
        current_province = st.session_state.get("current_province")
        current_district = st.session_state.get("current_district")
        current_zone     = st.session_state.get("current_zone")
        location_zone    = st.session_state.location_zone
        district_name    = getattr(st.session_state, "location_district", "Unknown")

        def _get_answer():
            try:
                if current_option == "District Wise":
                    return get_land_prep_response(user_question, current_province, current_district)
                elif current_option == "Agro Zone Wise":
                    return get_zone_prep_response(user_question, current_zone)
                elif current_option == "Location Based":
                    return get_location_zone_response(user_question, location_zone, district_name)
                elif current_option == "All Pakistan Context":
                    return get_pakistan_context_response(user_question)
                else:
                    return get_web_scraper_response(user_question)
            except Exception as e:
                return f"❌ Error generating response: {str(e)}"

        executor      = ThreadPoolExecutor(max_workers=2)
        answer_future = executor.submit(_get_answer)
        video_future  = executor.submit(find_best_video, user_question)

        # ── Custom styled spinner while LLM generates ─────────────────────
        spinner_slot = st.empty()
        spinner_slot.markdown("""
            <div class="typing-bubble">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
                <span class="typing-label">Generating response...</span>
            </div>
        """, unsafe_allow_html=True)

        full_response = answer_future.result()
        spinner_slot.empty()   # remove spinner the moment answer is ready

        # ── Step 3: show full answer at once ─────────────────────────────
        bot_timestamp = datetime.now().strftime("%I:%M %p")
        render_chat_message("assistant", full_response, bot_timestamp)

        # ── Step 4: resolve video (likely already done during spinner) ────
        REFUSAL_PHRASE = "I can only help with questions about"
        if REFUSAL_PHRASE in full_response:
            video_url = None
        else:
            video_url = video_future.result()

        executor.shutdown(wait=False)

        # Show video expander immediately if there is one
        if video_url:
            col_vid, _ = st.columns([3, 1])
            with col_vid:
                with st.expander("🎥 Watch Related Farming Video", expanded=False):
                    st.video(video_url)

        # ── Step 5: persist both turns and rerun to rebuild full history ──
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question,
            "timestamp": user_timestamp,
        })
        st.session_state.chat_history.append({
            "role": "bot",
            "content": full_response,
            "timestamp": bot_timestamp,
            "video_url": video_url,
        })
        st.rerun()

if __name__ == '__main__':
    main()