import streamlit as st
import requests
from datetime import datetime, timedelta
from streamlit_javascript import st_javascript

# ---------------------------
# CONFIG
# ---------------------------
WEATHER_API_KEY = "fdec585c9ae956ef0e6e8c6e88016663"  # <-- replace this
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

#st.set_page_config(page_title="Farmer Weather App", page_icon="🌾", layout="wide")

# ---------------------------
# HELPERS
# ---------------------------
def get_coords():
    """Get browser GPS coords via JS (returns [lat, lon] or None)."""
    coords = st_javascript(
        """await new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
                (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
                (err) => resolve(null)
            );
        });"""
    )
    if coords:
        try:
            return float(coords[0]), float(coords[1])
        except:
            return None
    return None

def fetch_current_and_forecast(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
    cur = requests.get(WEATHER_URL, params=params, timeout=8).json()
    fc = requests.get(FORECAST_URL, params=params, timeout=8).json()
    return cur, fc

def to_local(dt_utc, tz_offset_seconds):
    return datetime.utcfromtimestamp(dt_utc) + timedelta(seconds=tz_offset_seconds)

def emoji_for(cond):
    m = {
        "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️",
        "Thunderstorm": "⛈️", "Snow": "❄️", "Mist": "🌫️", "Fog": "🌫️", "Haze": "🌫️"
    }
    return m.get(cond, "🌍")

def bg_gradient(cond):
    g = {
        "Clear": "linear-gradient(180deg,#fff7b1 0%,#ffe680 100%)",
        "Clouds": "linear-gradient(180deg,#e8edf2 0%,#d1d6db 100%)",
        "Rain": "linear-gradient(180deg,#d9ecff 0%,#b8ddff 100%)",
        "Drizzle": "linear-gradient(180deg,#d9ecff 0%,#b8ddff 100%)",
        "Thunderstorm": "linear-gradient(180deg,#cbd5e1 0%,#9aa3ad 100%)",
        "Snow": "linear-gradient(180deg,#f7fbff 0%,#e6f2ff 100%)",
        "Mist": "linear-gradient(180deg,#f5f5f5 0%,#ececec 100%)",
    }
    return g.get(cond, "linear-gradient(180deg,#f5f9ff 0%,#eaf3ff 100%)")

def hourly_card_style(pop_percent):
    """Return inline CSS for hourly card based on precipitation chance."""
    if pop_percent >= 80:
        # Danger: dark with red accent
        return "background:#3b3b3b;color:white;border:2px solid #dc2626;"
    if pop_percent >= 70:
        return "background:#4b5563;color:white;border:1px solid #ef4444;"
    if pop_percent >= 40:
        return "background:#dff3ff;color:#083344;border:1px solid #8ecae6;"
    # safe
    return "background:#ffffff;color:#0f172a;border:1px solid rgba(0,0,0,0.06);"

def daily_card_style(cond, pop_percent):
    if pop_percent >= 70:
        return "background:#cfe7ff;"  # rainy tint
    if "rain" in cond.lower():
        return "background:#d0e7ff;"
    if "cloud" in cond.lower():
        return "background:#f1f5f9;"
    return "background:#fff7c2;"  # sunny tint

# ---------------------------
# APP UI & Flow
# ---------------------------
def main():
   # st.subheader("Weather Forecast for Farmers 🌾")
    
    coords = get_coords()
    if not coords:
        st.error("Could not access location. Please allow browser location (GPS) and refresh.")
        return

    lat, lon = coords
    try:
        current, forecast = fetch_current_and_forecast(lat, lon)
    except Exception as e:
        st.error("Network/API error: " + str(e))
        return

    # Basic response validation
    if not isinstance(current, dict) or "main" not in current or "weather" not in current:
        st.error("Weather API error: " + str(current.get("message", current)))
        return
    if not isinstance(forecast, dict) or "list" not in forecast:
        st.error("Forecast API error: " + str(forecast.get("message", forecast)))
        return

    # timezone offset (seconds) from forecast city if available, else from current
    tz_offset = forecast.get("city", {}).get("timezone", current.get("timezone", 0))

    # Current weather values
    city = current.get("name", "Unknown")
    temp_now = current["main"].get("temp")
    cond_main = current["weather"][0].get("main", "")
    cond_desc = current["weather"][0].get("description", "").title()
    emoji = emoji_for(cond_main)

    # Apply background theme
    gradient = bg_gradient(cond_main)
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {gradient};
            background-attachment: fixed;
        }}
        .current-box {{
            background: linear-gradient(180deg,#bbf7d0 0%,#86efac 100%);
            padding:14px;border-radius:16px;box-shadow:0 6px 18px rgba(4,120,87,0.12);
            text-align:center;border:1px solid rgba(0,0,0,0.06);
        }}
        .current-temp {{font-size:1.9rem;font-weight:700;color:#064e3b;margin:6px 0;}}
        .current-emoji {{font-size:54px;line-height:1;margin-bottom:4px;}}
        .hour-box, .day-box {{padding:10px;border-radius:12px;text-align:center;box-shadow:0 6px 18px rgba(2,6,23,0.04);}}
        .muted {{color:#334155;font-size:0.95rem;margin-top:6px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Current weather card (green)
    st.markdown(
        f"""
        <div class="current-box">
            <div class="current-emoji">{emoji}</div>
            <div class="current-temp">{temp_now:.1f}°C</div>
            <div class="muted">{cond_desc} — {city}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # small spacer

    # HOURLY FORECAST: next up to 6 entries (3-hr steps from OpenWeatherMap)
    st.subheader("🌤️ Next Hours (color = rain risk)")
    hourly_list = forecast.get("list", [])
    hours_to_show = min(6, len(hourly_list))
    cols = st.columns(hours_to_show)
    for i in range(hours_to_show):
        e = hourly_list[i]
        local_dt = to_local(e["dt"], tz_offset)
        hour_label = local_dt.strftime("%I %p")
        temp_h = e["main"].get("temp")
        pop = int(e.get("pop", 0) * 100)
        c = e["weather"][0].get("main", "")
        style = hourly_card_style(pop)
        warning = " ⚠️" if pop >= 80 else ""
        # render
        cols[i].markdown(
            f"""
            <div class="hour-box" style="{style}">
                <div style="font-size:20px;">{emoji_for(c)}</div>
                <div style="font-weight:700;margin-top:6px;">{hour_label}</div>
                <div style="margin-top:6px;">{temp_h:.0f}°C</div>
                <div style="margin-top:6px;">{pop}% 🌧️{warning}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # spacer

    # DAILY FORECAST: group forecast by local date and show up to next 5 days
    st.subheader("📅 5-Day Forecast")
    days = {}
    for e in hourly_list:
        d = to_local(e["dt"], tz_offset).date()
        if d not in days:
            days[d] = {"temps": [], "pops": [], "conds": []}
        days[d]["temps"].append(e["main"].get("temp"))
        days[d]["pops"].append(e.get("pop", 0))
        days[d]["conds"].append(e["weather"][0].get("main", ""))

    day_items = list(days.items())[:5]
    day_cols = st.columns(len(day_items))
    for idx, (d, vals) in enumerate(day_items):
        min_t = min(vals["temps"])
        max_t = max(vals["temps"])
        avg_pop = int(sum(vals["pops"]) / len(vals["pops"]) * 100) if vals["pops"] else 0
        # pick most common condition
        cond_day = max(set(vals["conds"]), key=vals["conds"].count) if vals["conds"] else ""
        style = daily_card_style(cond_day, avg_pop)
        warning = " ⚠️" if avg_pop >= 80 else ""
        day_cols[idx].markdown(
            f"""
            <div class="day-box" style="{style}; padding:12px; border-radius:12px;">
                <div style="font-weight:700;">{d.strftime('%a')}</div>
                <div style="font-size:20px;margin:6px 0;">{emoji_for(cond_day if avg_pop<=40 else 'Rain')}</div>
                <div>{int(max_t):d}° / {int(min_t):d}°C</div>
                <div style="margin-top:6px;">{avg_pop}% 🌧️{warning}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # optional legend to help farmers
    st.markdown(
        """
        <div style='margin-top:10px; text-align:center; color:#0f172a;'>
            <b>Legend:</b> &nbsp; <span style='padding:6px;border-radius:6px;background:#ffffff;border:1px solid rgba(0,0,0,0.06;'>Low</span>
            &nbsp; <span style='padding:6px;border-radius:6px;background:#d0e7ff;'>Medium</span>
            &nbsp; <span style='padding:6px;border-radius:6px;background:#4b5563;color:white;'>High</span>
            &nbsp; <span style='padding:6px;border-radius:6px;background:#3b3b3b;color:white;'>Very High ⚠️</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()