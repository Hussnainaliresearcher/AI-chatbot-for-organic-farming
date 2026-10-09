"""
satellite_insights.py
──────────────────────
🛰️ Satellite Farm Vital Statistics — Google Earth Engine powered.
"""

import json
import math
from datetime import datetime, timedelta

import streamlit as st
import folium
import streamlit.components.v1 as components
import ee
from openai import OpenAI
from google.oauth2.service_account import Credentials

from floating_chat import render_floating_chat, reset_floating_chat_state


# ── styling (namespaced so it never collides with app.py's CSS) ─────────────
def _inject_satellite_css():
    st.markdown("""
        <style>
            .sat-info-box {
                text-align:center; padding:28px; margin:24px auto; max-width:640px;
                background:#e8f5e8; border-radius:10px; border-left:4px solid #4caf50;
            }
            .sat-info-box h3 { color:#1b5e20; margin:0 0 6px 0; }
            .sat-info-box p  { color:#2e7d32; margin:0; }

            .sat-capture-pill {
                display:inline-block; background:#ffcc80; color:#000; font-weight:600;
                font-size:14px; padding:6px 14px; border-radius:16px; margin-bottom:14px;
            }

            .sat-metric-grid { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:6px; }
            .sat-metric-card {
                flex:1; min-width:130px; background:linear-gradient(180deg,#bbf7d0 0%,#86efac 100%);
                border-radius:12px; padding:14px 10px; text-align:center;
                box-shadow:0 6px 18px rgba(4,120,87,0.12); border:1px solid rgba(0,0,0,0.06);
            }
            .sat-metric-card.warn { background:linear-gradient(180deg,#ffe0b2 0%,#ffcc80 100%); }
            .sat-metric-label { font-size:13px; font-weight:600; color:#1b5e20; margin-bottom:4px; }
            .sat-metric-card.warn .sat-metric-label { color:#8a4b00; }
            .sat-metric-value { font-size:1.5rem; font-weight:700; color:#064e3b; }
            .sat-metric-card.warn .sat-metric-value { color:#7a3c00; }
            .sat-metric-badge {
                display:inline-block; margin-top:4px; font-size:12px; font-weight:700;
                padding:2px 10px; border-radius:12px; background:#1b5e20; color:#fff;
            }
            .sat-metric-card.warn .sat-metric-badge { background:#e65100; }

            .sat-map-title {
                color:black; background:#ffcc80; font-weight:700; font-size:16px;
                padding:8px 10px; border-radius:6px 6px 0 0; border-left:4px solid #4caf50;
            }
            .sat-map-wrap { border-radius:0 0 8px 8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:16px; }

            .sat-brief-box {
                background:#fafffe; border-left:4px solid #66bb6a; border-radius:10px;
                padding:16px 18px; font-size:1.02rem; color:#2d4a2d; line-height:1.65;
                box-shadow:0 1px 3px rgba(0,0,0,0.06);
            }

            .sat-health-hero {
                display:flex; align-items:center; gap:18px; background:#fff;
                border-radius:14px; padding:16px 20px; margin-bottom:16px;
                box-shadow:0 4px 16px rgba(0,0,0,0.08); border:1px solid #e0e0e0;
            }
            .sat-health-score { font-size:2.4rem; font-weight:800; line-height:1; white-space:nowrap; }
            .sat-health-score span { font-size:1rem; font-weight:600; color:#888; }
            .sat-health-info { flex:1; }
            .sat-health-label { font-size:1.05rem; font-weight:700; margin-bottom:6px; }
            .sat-health-bar { background:#eee; border-radius:8px; height:10px; overflow:hidden; }
            .sat-health-fill { height:100%; border-radius:8px; }
            .sat-health-caption { font-size:0.82rem; color:#666; margin-top:6px; }

            .sat-legend { font-size:0.82rem; color:#556b55; margin:6px 2px 18px 2px; }

            @keyframes sat-dot-pulse {
                0%, 80%, 100% { transform: scale(0.6); opacity: 0.35; }
                40%            { transform: scale(1.2); opacity: 1;    }
            }
            .sat-typing-bubble {
                display:inline-flex; align-items:center; gap:6px;
                background:linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #66bb6a 100%);
                padding:12px 18px; border-radius:20px 20px 20px 4px;
                box-shadow:0 4px 14px rgba(27,94,32,0.35); margin:14px 0;
            }
            .sat-typing-bubble .dot { width:9px; height:9px; border-radius:50%; background:#fff; animation: sat-dot-pulse 1.4s ease-in-out infinite; }
            .sat-typing-bubble .dot:nth-child(2) { animation-delay:0.2s; }
            .sat-typing-bubble .dot:nth-child(3) { animation-delay:0.4s; }
            .sat-typing-label { color:#fff; font-style:italic; font-weight:600; font-size:14px; margin-left:4px; }
        </style>
    """, unsafe_allow_html=True)


# ── GEE + LLM setup ──────────────────────────────────────────────────────────
@st.cache_resource
def _init_gee(project_id: str):
    try:
        # Check if running on Streamlit Cloud with configured secrets
        if "gcp_service_account" in st.secrets:
            key_dict = dict(st.secrets["gcp_service_account"])
            credentials = Credentials.from_service_account_info(key_dict)
            
            # Apply the Earth Engine scope
            scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
            
            ee.Initialize(scoped_credentials, project=project_id)
            return True, "Initialized securely via Service Account."
        else:
            # Fallback for local testing
            ee.Initialize(project=project_id)
            return True, "Initialized successfully."
    except Exception as e:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            return True, "Authenticated and initialized."
        except Exception as auth_err:
            return False, f"Cloud auth error: {e}. Local auth error: {auth_err}"

def _get_llm_client() -> OpenAI:
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

_DEFAULT_GEE_PROJECT_ID = "auto-ml-copilot"

def _get_gee_project_id() -> str:
    try:
        return st.secrets.get("GEE_PROJECT_ID", _DEFAULT_GEE_PROJECT_ID)
    except Exception:
        return _DEFAULT_GEE_PROJECT_ID

def _typing_bubble_html(label: str) -> str:
    return ('<div style="display:flex;justify-content:center;align-items:center;padding:40px 0;">'
            '<div class="sat-typing-bubble" style="max-width:none;">'
            '<div class="dot"></div><div class="dot"></div><div class="dot"></div>'
            f'<span class="sat-typing-label">{label}</span>'
            '</div></div>')


# ── sidebar controls ─────────────────────────────────────────────────────────
def render_satellite_sidebar() -> bool:
    st.markdown("### 🛰️ Farm Boundary")
    st.session_state.sat_project_id = _get_gee_project_id()

    input_method = st.radio(
        "Define your field by:",
        ["Upload GeoJSON", "Corner + Dimensions", "Four Corners"],
        key="sat_input_method",
    )

    farm_geom = None 
    if input_method == "Upload GeoJSON":
        uploaded_file = st.file_uploader("Upload boundary (.geojson)", type=["geojson", "json"], key="sat_geojson")
        if uploaded_file is not None:
            try:
                geojson_data = json.load(uploaded_file)
                farm_geom = {"type": "geojson", "geometry": geojson_data["features"][0]["geometry"]}
                st.success("✅ GeoJSON loaded!")
            except Exception as e:
                st.error(f"Failed to parse GeoJSON: {e}")

    elif input_method == "Corner + Dimensions":
        st.markdown("**Top-Left (NW) Corner**")
        start_lat = st.number_input("Latitude", value=31.520400, format="%.6f", key="sat_lat_a")
        start_lon = st.number_input("Longitude", value=74.358700, format="%.6f", key="sat_lon_a")
        width_m = st.number_input("Width (E–W) in meters", min_value=10, value=200, step=10, key="sat_width")
        length_m = st.number_input("Length (N–S) in meters", min_value=10, value=250, step=10, key="sat_length")
        area_acres = (width_m * length_m) * 0.000247105
        st.info(f"📐 Calculated Area: {area_acres:.2f} Acres")
        farm_geom = {
            "type": "rect_corner",
            "start_lat": start_lat, "start_lon": start_lon,
            "width_m": width_m, "length_m": length_m,
        }

    elif input_method == "Four Corners":
        st.caption("Enter coordinates clockwise to avoid a twisted shape.")
        c1, c2 = st.columns(2)
        lat_a = c1.number_input("Lat A", value=31.521500, format="%.6f", step=0.0001, key="sat_lat_a4")
        lon_a = c2.number_input("Lon A", value=74.358700, format="%.6f", step=0.0001, key="sat_lon_a4")
        c3, c4 = st.columns(2)
        lat_b = c3.number_input("Lat B", value=31.521500, format="%.6f", step=0.0001, key="sat_lat_b4")
        lon_b = c4.number_input("Lon B", value=74.360000, format="%.6f", step=0.0001, key="sat_lon_b4")
        c5, c6 = st.columns(2)
        lat_c = c5.number_input("Lat C", value=31.520400, format="%.6f", step=0.0001, key="sat_lat_c4")
        lon_c = c6.number_input("Lon C", value=74.360000, format="%.6f", step=0.0001, key="sat_lon_c4")
        c7, c8 = st.columns(2)
        lat_d = c7.number_input("Lat D", value=31.520400, format="%.6f", step=0.0001, key="sat_lat_d4")
        lon_d = c8.number_input("Lon D", value=74.358700, format="%.6f", step=0.0001, key="sat_lon_d4")
        farm_geom = {
            "type": "quad",
            "coords": [[lon_a, lat_a], [lon_b, lat_b], [lon_c, lat_c], [lon_d, lat_d], [lon_a, lat_a]],
        }

    st.markdown("### 📅 Observation Window")
    default_end = datetime.today()
    default_start = default_end - timedelta(days=60)
    date_range = st.date_input("Date Range", [default_start, default_end], key="sat_date_range_widget")
    cloud_threshold = st.slider("Max Cloud Cover (%)", min_value=5, max_value=50, value=20, key="sat_cloud")

    st.session_state.sat_farm_geom = farm_geom
    st.session_state.sat_date_range = date_range
    st.session_state.sat_cloud_threshold = cloud_threshold

    st.markdown("<br>", unsafe_allow_html=True)
    return st.button("🚀 Fetch Satellite Data", use_container_width=True, key="sat_fetch_btn")


# ── geometry + GEE pipeline ──────────────────────────────────────────────────
def _build_ee_geometry(farm_geom: dict):
    if not farm_geom:
        raise ValueError("Please define your field boundary in the sidebar first.")

    if farm_geom["type"] == "geojson":
        return ee.Geometry(farm_geom["geometry"])

    if farm_geom["type"] == "rect_corner":
        start_lat = farm_geom["start_lat"]
        start_lon = farm_geom["start_lon"]
        width_m = farm_geom["width_m"]
        length_m = farm_geom["length_m"]
        delta_lat = length_m / 111320.0
        delta_lon = width_m / (111320.0 * math.cos(math.radians(start_lat)))
        min_lat, max_lat = start_lat - delta_lat, start_lat
        min_lon, max_lon = start_lon, start_lon + delta_lon
        return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    if farm_geom["type"] == "quad":
        return ee.Geometry.Polygon([farm_geom["coords"]])

    raise ValueError("Unrecognised field boundary input.")

def _get_mean_value(img, band_name, geometry):
    if img is None:
        return "N/A"
    try:
        val = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10, maxPixels=1e9).get(band_name)
        res = val.getInfo()
        return round(res, 2) if res is not None else "N/A"
    except Exception:
        return "N/A"

def _run_fetch_pipeline(project_id: str, farm_geom: dict, date_range, cloud_threshold: int) -> dict:
    if not project_id:
        return {"error": "⚠️ Please provide your Google Cloud Project ID in the sidebar."}

    success, msg = _init_gee(project_id)
    if not success:
        return {"error": f"GEE Initialization failed: {msg}"}

    try:
        farm_polygon = _build_ee_geometry(farm_geom)
    except Exception as e:
        return {"error": f"⚠️ {e}"}

    if not date_range or len(date_range) != 2:
        return {"error": "⚠️ Please select a full start and end date in the sidebar."}

    try:
        centroid = farm_polygon.centroid().getInfo()["coordinates"]
        center_lon, center_lat = centroid[0], centroid[1]
    except Exception:
        return {"error": "⚠️ Invalid boundary coordinates — make sure the shape is closed and numeric."}

    try:
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(farm_polygon)
            .filterDate(str(date_range[0]), str(date_range[1]))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        )
        s2_image = s2_collection.sort("system:time_start", False).first()
        if s2_image is None:
            return {"error": "No cloud-free Sentinel-2 images found. Try increasing the cloud cover tolerance."}

        timestamp_ms = s2_image.get("system:time_start").getInfo()
        capture_date = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")

        ndvi = s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI").clip(farm_polygon)
        ndmi = s2_image.normalizedDifference(["B8", "B11"]).rename("NDMI").clip(farm_polygon)

        smap_collection = ee.ImageCollection("NASA/SMAP/SPL4SMGP/008").filterBounds(farm_polygon)
        smap_image = smap_collection.sort("system:time_start", False).first()
        soil_moisture = (
            smap_image.select("sm_surface").multiply(100).rename("Soil_Moisture_Pct").clip(farm_polygon)
            if smap_image is not None else None
        )

        srtm = ee.Image("USGS/SRTMGL1_003").clip(farm_polygon)
        elevation = srtm.select("elevation").rename("Elevation")
        slope = ee.Terrain.slope(elevation).rename("Slope")

        modis_lst = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(farm_polygon)
            .filterDate(str(date_range[0]), str(date_range[1]))
            .select("LST_Day_1km")
            .median()
            .multiply(0.02)
            .subtract(273.15)
            .rename("LST_C")
            .clip(farm_polygon)
        )

        mean_ndvi = _get_mean_value(ndvi, "NDVI", farm_polygon)
        mean_ndmi = _get_mean_value(ndmi, "NDMI", farm_polygon)
        mean_elev = _get_mean_value(elevation, "Elevation", farm_polygon)
        mean_slope = _get_mean_value(slope, "Slope", farm_polygon)
        mean_lst = _get_mean_value(modis_lst, "LST_C", farm_polygon)
        mean_soil_moisture = _get_mean_value(soil_moisture, "Soil_Moisture_Pct", farm_polygon)

       # ── Added: Generate a GeoTIFF download URL for Multispectral bands ──
        try:
            # We pull Red, Green, Blue, and NIR, and rename them for QGIS clarity
            export_bands = s2_image.select(
                ["B4", "B3", "B2", "B8"],
                ["Red", "Green", "Blue", "NIR"]
            )
            geotiff_url = export_bands.getDownloadURL({
                'name': 'Farm_Multispectral_Image',
                'scale': 10,
                'region': farm_polygon,
                'format': 'GEO_TIFF'
            })
        except Exception:
            geotiff_url = None

    except Exception as e:
        return {"error": f"❌ Error while fetching satellite data: {e}"}

    ndvi_status = "Healthy" if (isinstance(mean_ndvi, float) and mean_ndvi > 0.5) else "Stressed"
    ndmi_status = "High" if (isinstance(mean_ndmi, float) and mean_ndmi > 0.2) else "Low"
    sm_status = "Optimal" if (isinstance(mean_soil_moisture, float) and mean_soil_moisture > 20) else "Dry"

    try:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")
        rgb_vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
        rgb_map_id = ee.Image(s2_image.clip(farm_polygon)).getMapId(rgb_vis)
        folium.TileLayer(tiles=rgb_map_id["tile_fetcher"].url_format, attr="Google Earth Engine",
                          name="True Color (RGB)", overlay=True, control=True).add_to(m)

        ndvi_vis = {"min": 0.0, "max": 0.8, "palette": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]}
        ndvi_map_id = ee.Image(ndvi).getMapId(ndvi_vis)
        folium.TileLayer(tiles=ndvi_map_id["tile_fetcher"].url_format, attr="Google Earth Engine",
                          name="NDVI (Vegetation Health)", overlay=True, control=True).add_to(m)

        boundary_data = farm_polygon.getInfo()
        folium.GeoJson(boundary_data, name="Farm Boundary",
                        style_function=lambda x: {"color": "#00ffff", "weight": 3, "fillOpacity": 0.05}).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        map_html = m._repr_html_()
    except Exception as e:
        map_html = f"<p>Map could not be rendered: {e}</p>"

    rag_summary = f"""[SPATIAL SATELLITE TELEMETRY - FARM OBSERVATION]
- Data Captured on: {capture_date}
- Location: Latitude {center_lat}, Longitude {center_lon}
- Vegetation Health (NDVI): {mean_ndvi} (Status: {ndvi_status})
- Canopy Hydration (NDMI): {mean_ndmi} (Status: {ndmi_status})
- Topsoil Moisture (SMAP L4): {mean_soil_moisture} % (Status: {sm_status})
- Land Surface Temperature: {mean_lst} °C
- Topography: Elevation {mean_elev} m, Slope {mean_slope} degrees
"""

    return {
        "error": None,
        "capture_date": capture_date,
        "center_lat": center_lat, "center_lon": center_lon,
        "mean_ndvi": mean_ndvi, "ndvi_status": ndvi_status,
        "mean_ndmi": mean_ndmi, "ndmi_status": ndmi_status,
        "mean_soil_moisture": mean_soil_moisture, "sm_status": sm_status,
        "mean_lst": mean_lst, "mean_elev": mean_elev, "mean_slope": mean_slope,
        "map_html": map_html,
        "rag_summary": rag_summary,
        "geotiff_url": geotiff_url,
    }


# ── LLM narration ─────────────────────────────────────────────────────────────
def _summarize_with_llm(rag_summary: str) -> str:
    client = _get_llm_client()
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": (
                    "You are an agronomist assistant writing for a farmer in Pakistan. "
                    "Convert the raw satellite telemetry below into a short, plain-language "
                    "field briefing, 4-6 sentences max. Explain NDVI as 'plant health', NDMI as "
                    "'leaf/canopy water content', LST as 'field surface temperature' — no raw "
                    "acronyms in the output. Say plainly what looks fine and what needs "
                    "attention, then end with 1-2 concrete organic-farming actions if anything "
                    "is stressed, low, or dry. Do not invent numbers not present below.\n\n"
                    f"Telemetry:\n{rag_summary}"
                ),
            }],
        )
        return r.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ Could not generate the plain-language summary: {exc}"

def _ask_satellite_chatbot(rag_summary: str, question: str) -> str:
    client = _get_llm_client()
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[{
                "role": "user",
                "content": (
                    f"You are an agriculture expert. Below is satellite telemetry for a farmer's field:\n"
                    f"{rag_summary}\n\n"
                    f"Farmer asks: {question}\n"
                    "RULES: Only answer if the question relates to this field's condition or to "
                    "organic farming generally. If unrelated, reply exactly: "
                    "'⚠️ I can only answer questions about your field's satellite data and organic farming.' "
                    "Otherwise answer in 4-5 lines max, be direct, practical, and reference the "
                    "relevant numbers where useful."
                ),
            }],
        )
        return r.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error: {exc}"


# ── main-area rendering ───────────────────────────────────────────────────────
def _metric_card(label: str, value: str, badge: str, warn: bool) -> str:
    cls = "sat-metric-card warn" if warn else "sat-metric-card"
    return (f'<div class="{cls}"><div class="sat-metric-label">{label}</div>'
            f'<div class="sat-metric-value">{value}</div>'
            f'<div class="sat-metric-badge">{badge}</div></div>')

def _compute_health_score(result: dict):
    def _norm(value, lo, hi):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.5 
        return max(0.0, min(1.0, (value - lo) / (hi - lo)))

    ndvi_n = _norm(result.get("mean_ndvi"), 0.0, 0.7)          
    ndmi_n = _norm(result.get("mean_ndmi"), 0.0, 0.4)          
    sm_n = _norm(result.get("mean_soil_moisture"), 5.0, 35.0)  

    score = round((ndvi_n * 0.45 + ndmi_n * 0.30 + sm_n * 0.25) * 100)

    if score >= 75:
        label, color = "Excellent", "#2e7d32"
    elif score >= 55:
        label, color = "Good", "#4caf50"
    elif score >= 35:
        label, color = "Needs Attention", "#e65100"
    else:
        label, color = "Critical", "#c62828"

    return score, label, color

def render_satellite_main():
    _inject_satellite_css()

    # ── Clean, optimized check directly off the button ──
    fetch_clicked = st.session_state.get("sat_fetch_btn", False)

    if fetch_clicked:
        spinner_slot = st.empty()
        spinner_slot.markdown(_typing_bubble_html("🔍 Connecting to satellites (Sentinel-2, MODIS, SMAP L4, SRTM)…"), unsafe_allow_html=True)
        result = _run_fetch_pipeline(
            st.session_state.get("sat_project_id", ""),
            st.session_state.get("sat_farm_geom"),
            st.session_state.get("sat_date_range"),
            st.session_state.get("sat_cloud_threshold", 20),
        )
        st.session_state.sat_result = result
        if not result.get("error"):
            spinner_slot.markdown(_typing_bubble_html("🧠 Writing your field briefing…"), unsafe_allow_html=True)
            st.session_state.sat_llm_summary = _summarize_with_llm(result["rag_summary"])
            reset_floating_chat_state("satchat")
        spinner_slot.empty()

    result = st.session_state.get("sat_result")

    # ── Seamlessly inject the download button into the sidebar ──
    if result and not result.get("error") and result.get("geotiff_url"):
        st.sidebar.link_button("💾 Download GeoTIFF Image", result["geotiff_url"], use_container_width=True)

    if not result:
        st.markdown(
            '<div class="sat-info-box"><h3>🛰️ Satellite Farm Insights</h3>'
            '<p>Define your field boundary in the sidebar and click '
            '<b>"🚀 Fetch Satellite Data"</b> to see your field\'s vegetation health, '
            'water content, soil moisture, temperature and terrain.</p></div>',
            unsafe_allow_html=True,
        )
        return

    if result.get("error"):
        st.error(result["error"])
        return

    st.markdown(f'<div class="sat-capture-pill">📸 Latest image captured: {result["capture_date"]}</div>', unsafe_allow_html=True)

    score, score_label, score_color = _compute_health_score(result)
    st.markdown(
        '<div class="sat-health-hero">'
        f'<div class="sat-health-score" style="color:{score_color};">{score}<span>/100</span></div>'
        '<div class="sat-health-info">'
        f'<div class="sat-health-label" style="color:{score_color};">{score_label}</div>'
        f'<div class="sat-health-bar"><div class="sat-health-fill" style="width:{score}%;background:{score_color};"></div></div>'
        '<div class="sat-health-caption">Combined field health score — weighted from plant health, canopy water and soil moisture.</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Farm Vital Statistics")
    cards_html = '<div class="sat-metric-grid">'
    cards_html += _metric_card("NDVI (Plant Health)", result["mean_ndvi"], result["ndvi_status"], result["ndvi_status"] != "Healthy")
    cards_html += _metric_card("NDMI (Canopy Water)", result["mean_ndmi"], result["ndmi_status"], result["ndmi_status"] != "High")
    cards_html += _metric_card("Soil Moisture", f'{result["mean_soil_moisture"]} %', result["sm_status"], result["sm_status"] != "Optimal")
    cards_html += _metric_card("Surface Temp", f'{result["mean_lst"]} °C', "—", False)
    cards_html += _metric_card("Elevation", f'{result["mean_elev"]} m', "—", False)
    cards_html += _metric_card("Slope", f'{result["mean_slope"]}°', "—", False)
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown(
        '<div class="sat-legend">NDVI above 0.5 → healthy canopy · NDMI above 0.2 → well-hydrated leaves '
        '· Soil moisture above 20% → optimal for most crops.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sat-map-title">🗺️ High-Resolution Field Map — spot dry patches &amp; stress early</div>', unsafe_allow_html=True)
    st.markdown('<div class="sat-map-wrap">', unsafe_allow_html=True)
    components.html(result["map_html"], height=480)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 🌾 Field Briefing")
    st.markdown(f'<div class="sat-brief-box">{st.session_state.get("sat_llm_summary", "")}</div>', unsafe_allow_html=True)

    with st.expander("🔧 Raw satellite telemetry (technical)", expanded=False):
        st.code(result["rag_summary"], language="markdown")

    st.markdown("<br>", unsafe_allow_html=True)

    render_floating_chat(
        namespace="satchat",
        title="🌾 Ask About Your Field",
        subtitle="Powered by your satellite data",
        on_ask=lambda q: _ask_satellite_chatbot(result["rag_summary"], q),
        suggested_questions=["Why is my NDVI low?", "Is my soil moisture okay?", "What should I do next?"],
        empty_message="Ask anything about your field's satellite readings…",
        placeholder="e.g. Why is my NDVI low?",
    )

def reset_satellite_result_state():
    for key in ["sat_result", "sat_llm_summary"]:
        if key in st.session_state:
            del st.session_state[key]
    reset_floating_chat_state("satchat")