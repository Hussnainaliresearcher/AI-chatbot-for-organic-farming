import streamlit as st
from streamlit_javascript import st_javascript
import requests
from datetime import datetime, timedelta

WEATHER_API_KEY = "fdec585c9ae956ef0e6e8c6e88016663"
WEATHER_URL     = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL    = "https://api.openweathermap.org/data/2.5/forecast"


# ── helpers ───────────────────────────────────────────────────────────────────

def get_coords():
    coords = st_javascript("""await new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
            (err) => resolve(null)
        );
    });""")
    if coords:
        try:
            return float(coords[0]), float(coords[1])
        except Exception:
            return None
    return None


def fetch_weather(lat, lon):
    p = {"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"}
    session = requests.Session()
    for attempt in range(3):
        try:
            cur = session.get(WEATHER_URL,  params=p, timeout=15).json()
            fc  = session.get(FORECAST_URL, params=p, timeout=15).json()
            return cur, fc
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
        except Exception:
            raise


def to_local(dt_utc, tz_offset):
    return datetime.utcfromtimestamp(dt_utc) + timedelta(seconds=tz_offset)


def fmt_time(ts, tz):
    """Format a UTC unix timestamp to local 12-hour time without leading zero."""
    return to_local(ts, tz).strftime("%I:%M %p").lstrip("0")


def weather_emoji(cond):
    return {
        "Clear": "☀️", "Clouds": "⛅", "Rain": "🌧️",
        "Drizzle": "🌦️", "Thunderstorm": "⛈️", "Snow": "❄️",
        "Mist": "🌫️", "Fog": "🌫️", "Haze": "🌫️", "Dust": "💨",
    }.get(cond, "🌍")


def rain_badge(pop):
    """Return styled HTML badge for rain probability — larger, colour-coded."""
    if pop >= 80:
        bg, fg, label = "#dc2626", "#fff", "Very High"
    elif pop >= 60:
        bg, fg, label = "#ea580c", "#fff", "High"
    elif pop >= 40:
        bg, fg, label = "#2563eb", "#fff", "Medium"
    elif pop >= 10:
        bg, fg, label = "#16a34a", "#fff", "Low"
    else:
        bg, fg, label = "#d1fae5", "#065f46", "Dry"

    return (
        f'<div style="background:{bg};color:{fg};border-radius:10px;'
        f'padding:5px 0;margin-top:6px;font-size:1rem;font-weight:700;'
        f'min-width:100%;text-align:center;line-height:1.2;">'
        f'🌧️ {pop}%<br>'
        f'<span style="font-size:0.72rem;font-weight:500;opacity:0.9;">{label}</span>'
        f'</div>'
    )


def farmer_insight(cond, humidity, wind_kmh, pop_max):
    if pop_max >= 70:
        return "🌧️", "#1e3a5f", "#dbeafe", "Heavy rain expected — avoid spraying fertilizers or pesticides today."
    if wind_kmh > 30:
        return "💨", "#713f12", "#fef9c3", "Strong winds — postpone aerial spraying; risk of chemical drift."
    if humidity > 85:
        return "💧", "#134e4a", "#ccfbf1", "High humidity — monitor crops for fungal disease risk."
    if humidity < 30:
        return "🌵", "#7c2d12", "#ffedd5", "Very low humidity — irrigate crops if possible."
    if cond == "Clear" and wind_kmh < 15:
        return "✅", "#14532d", "#dcfce7", "Clear skies and calm winds — ideal day for field work and spraying."
    if cond == "Thunderstorm":
        return "⚡", "#1e1b4b", "#e0e7ff", "Thunderstorm risk — stay indoors and secure farm equipment."
    return "🌱", "#14532d", "#f0fdf4", "Conditions are moderate — a good day for routine farm activities."


# ── main render ───────────────────────────────────────────────────────────────

def render_weather_forecast():
    st.markdown("""
    <style>
    /* Remove extra top space so page fits under fixed header */
    .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

    /* ── Hero card ── */
    .hero-card {
        background: #a5d6a7;
        border-radius: 14px; padding: 8px 16px; color: #1b5e20;
        display: flex; align-items: center; justify-content: space-between;
        box-shadow: 0 3px 12px rgba(27,94,32,0.18); margin-bottom: 8px; margin-top: -70px;
        border: 1px solid #81c784;
    }
    .hero-temp   { font-size: 2.4rem; font-weight: 900; line-height: 1; color: #1b5e20; }
    .hero-feels  { font-size: 0.82rem; color: #2e7d32; margin-top: 1px; }
    .hero-cond   { font-size: 1rem;   font-weight: 700; margin-top: 3px; color: #1b5e20; }
    .hero-city   { font-size: 1rem;   font-weight: 800; margin-top: 4px; color: #ffffff;
                   background: #2e7d32; display: inline-block; padding: 2px 10px;
                   border-radius: 20px; }
    .hero-emoji  { font-size: 3.6rem; line-height: 1; }

    /* ── Metric pills ── */
    .metric-pill {
        background: white; border-radius: 14px; padding: 10px 6px;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb; height: 100%;
    }
    .metric-icon  { font-size: 1.6rem; line-height: 1; }
    .metric-value { font-size: 1.05rem; font-weight: 800; color: #1b5e20; margin-top: 4px; }
    .metric-label { font-size: 0.78rem; color: #6b7280; margin-top: 2px; font-weight: 500; }

    /* ── Section headings ── */
    .wx-heading {
        font-size: 0.85rem; font-weight: 800; letter-spacing: 1px;
        text-transform: uppercase; color: #4b5563; margin: 10px 0 6px 2px;
        border-left: 3px solid #4caf50; padding-left: 8px;
    }

    /* ── Hourly cards — blue/indigo theme ── */
    .hour-card {
        border-radius: 12px; padding: 6px 3px; text-align: center;
        background: linear-gradient(160deg,#eff6ff 0%,#dbeafe 100%);
        border: 1px solid #bfdbfe;
    }
    .hour-time { font-weight: 800; font-size: 0.82rem; color: #1e40af; }
    .hour-icon { font-size: 1.2rem; margin: 2px 0; }
    .hour-temp { font-weight: 800; font-size: 1.05rem; color: #1e3a8a; }

    /* ── Daily cards — warm amber theme ── */
    .day-card {
        border-radius: 14px; padding: 10px 4px; text-align: center;
        background: linear-gradient(160deg,#fffbeb 0%,#fef3c7 100%);
        border: 1px solid #fde68a;
    }
    .day-name  { font-weight: 800; font-size: 0.9rem; color: #92400e; }
    .day-icon  { font-size: 1.5rem; margin: 4px 0; }
    .day-temps { font-weight: 800; font-size: 1rem; color: #78350f; }

    /* ── Insight banner ── */
    .insight-banner {
        border-radius: 12px; padding: 11px 16px; margin-top: 8px;
        display: flex; align-items: center; gap: 12px;
        font-size: 0.95rem; font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .insight-icon { font-size: 1.6rem; flex-shrink: 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── location ──────────────────────────────────────────────────────────────
    coords = get_coords()
    if not coords:
        st.warning("📍 Allow browser location access to see your local weather forecast.")
        return

    lat, lon = coords
    try:
        current, forecast = fetch_weather(lat, lon)
    except Exception as e:
        st.error(f"Could not fetch weather data: {e}")
        return

    if "main" not in current or "list" not in forecast:
        st.error("Weather API returned unexpected data. Please try again.")
        return

    # ── parse current ─────────────────────────────────────────────────────────
    tz          = forecast.get("city", {}).get("timezone", current.get("timezone", 0))
    city        = current.get("name", "Your Location")
    country     = current.get("sys", {}).get("country", "")
    temp        = current["main"]["temp"]
    feels_like  = current["main"]["feels_like"]
    humidity    = current["main"]["humidity"]
    pressure    = current["main"]["pressure"]
    visibility  = current.get("visibility", 0) / 1000
    wind_ms     = current.get("wind", {}).get("speed", 0)
    wind_kmh    = round(wind_ms * 3.6)
    cond_main   = current["weather"][0]["main"]
    cond_desc   = current["weather"][0]["description"].title()
    sunrise_ts  = current.get("sys", {}).get("sunrise", 0)
    sunset_ts   = current.get("sys", {}).get("sunset", 0)
    sunrise_str = fmt_time(sunrise_ts, tz) if sunrise_ts else "—"
    sunset_str  = fmt_time(sunset_ts,  tz) if sunset_ts else "—"

    # ── parse hourly ──────────────────────────────────────────────────────────
    hourly = forecast["list"][:8]

    # ── parse daily ───────────────────────────────────────────────────────────
    daily_map = {}
    for slot in forecast["list"]:
        day_key = to_local(slot["dt"], tz).date()
        if day_key not in daily_map:
            daily_map[day_key] = []
        daily_map[day_key].append(slot)

    daily_items = []
    for day_key, slots in list(daily_map.items())[:6]:
        temps     = [s["main"]["temp"] for s in slots]
        pops      = [s.get("pop", 0) * 100 for s in slots]
        conds     = [s["weather"][0]["main"] for s in slots]
        avg_pop   = round(sum(pops) / len(pops))
        max_pop   = max(pops)
        cond_repr = max(set(conds), key=conds.count)
        daily_items.append({
            "date": day_key, "min_t": min(temps), "max_t": max(temps),
            "avg_pop": avg_pop, "max_pop": max_pop, "cond": cond_repr,
        })

    max_pop_today = daily_items[0]["max_pop"] if daily_items else 0

    # ── HERO CARD ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-card">
        <div>
            <div class="hero-temp">{temp:.0f}°C</div>
            <div class="hero-feels">Feels like {feels_like:.0f}°C</div>
            <div class="hero-cond">{cond_desc}</div>
            <div class="hero-city">📍 {city}, {country}</div>
        </div>
        <div style="text-align:right;">
            <div class="hero-emoji">{weather_emoji(cond_main)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── METRIC PILLS — 6 pills: 4 weather + sunrise + sunset ─────────────────
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    pills = [
        (p1, "💧", f"{humidity}%",          "Humidity"),
        (p2, "💨", f"{wind_kmh} km/h",      "Wind"),
        (p3, "🌡️", f"{pressure} hPa",       "Pressure"),
        (p4, "👁️", f"{visibility:.1f} km",  "Visibility"),
        (p5, "🌅", sunrise_str,             "Sunrise"),
        (p6, "🌇", sunset_str,              "Sunset"),
    ]
    for col, icon, val, lbl in pills:
        col.markdown(f"""<div class="metric-pill">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # ── HOURLY FORECAST ───────────────────────────────────────────────────────
    st.markdown('<div class="wx-heading">Next 24 Hours</div>', unsafe_allow_html=True)
    h_cols = st.columns(len(hourly))
    for i, slot in enumerate(hourly):
        local_dt = to_local(slot["dt"], tz)
        hour_lbl = local_dt.strftime("%I%p").lstrip("0")
        temp_h   = slot["main"]["temp"]
        pop      = round(slot.get("pop", 0) * 100)
        cond_h   = slot["weather"][0]["main"]
        h_cols[i].markdown(f"""
        <div class="hour-card">
            <div class="hour-time">{hour_lbl}</div>
            <div class="hour-icon">{weather_emoji(cond_h)}</div>
            <div class="hour-temp">{temp_h:.0f}°</div>
            {rain_badge(pop)}
        </div>""", unsafe_allow_html=True)

    # ── DAILY FORECAST ────────────────────────────────────────────────────────
    st.markdown('<div class="wx-heading">5-Day Outlook</div>', unsafe_allow_html=True)
    show_days = daily_items[1:6]
    d_cols    = st.columns(len(show_days))
    for i, day in enumerate(show_days):
        day_name  = day["date"].strftime("%a %d")
        emoji_day = weather_emoji("Rain" if day["avg_pop"] >= 60 else day["cond"])
        d_cols[i].markdown(f"""
        <div class="day-card">
            <div class="day-name">{day_name}</div>
            <div class="day-icon">{emoji_day}</div>
            <div class="day-temps">{day['max_t']:.0f}° / {day['min_t']:.0f}°</div>
            {rain_badge(day['avg_pop'])}
        </div>""", unsafe_allow_html=True)

    # ── FARMER INSIGHT BANNER ─────────────────────────────────────────────────
    icon, text_color, bg_color, tip = farmer_insight(
        cond_main, humidity, wind_kmh, max_pop_today
    )
    st.markdown(f"""
    <div class="insight-banner" style="background:{bg_color};color:{text_color};">
        <span class="insight-icon">{icon}</span>
        <span><b>Farming Tip:</b> {tip}</span>
    </div>""", unsafe_allow_html=True)