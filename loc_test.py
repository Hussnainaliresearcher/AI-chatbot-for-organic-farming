import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
from streamlit_javascript import st_javascript

st.set_page_config(page_title="Agro Zone Locator", layout="centered")
st.title("📍 Agro-Ecological Zone Detection")

# Load agro-ecological zones
@st.cache_data
def load_agro_zones():
    return gpd.read_file("ali_try3_colors.geojson")

agro_zones = load_agro_zones()

# Ask for geolocation via JavaScript
coords = st_javascript("""await new Promise((resolve) => {
    navigator.geolocation.getCurrentPosition(
        (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
        (err) => resolve(null)
    );
});""")

if coords:
    lat, lon = coords
    st.success(f"📍 Your location: Latitude {lat}, Longitude {lon}")

    # Create point and check zone
    user_point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
    matched = gpd.sjoin(user_point, agro_zones, how="left", predicate="within")

    if not matched.empty:
        zone = matched.iloc[0]["zone_name"]
        st.success(f"✅ You are in the **{zone}** agro-ecological zone.")
    else:
        st.warning("⚠️ Your location does not match any known agro-ecological zone.")
else:
    st.info("📡 Please allow location access in your browser to detect your agro zone.")
