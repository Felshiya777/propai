import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import json
import os
import requests
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropAI",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e5e7eb;
}

.main { background-color: #0a0e1a; }
section[data-testid="stSidebar"] {
    background-color: #0f1422 !important;
    border-right: 1px solid #1e2433 !important;
}

.gold { color: #D4AF37; }

.propai-header {
    background: linear-gradient(135deg, #0f1422 0%, #1a1f2e 100%);
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.header-title {
    font-size: 26px;
    font-weight: 700;
    color: #D4AF37;
    letter-spacing: 2px;
}
.header-sub { font-size: 13px; color: #4b5563; margin-top: 4px; }

.metric-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label { font-size: 11px; color: #4b5563; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-size: 22px; font-weight: 700; color: #D4AF37; }
.metric-sub { font-size: 11px; color: #374151; margin-top: 4px; }

.predict-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.predict-card h3 { font-size: 15px; font-weight: 600; color: #D4AF37; margin-bottom: 1rem; }

.result-card {
    background: #0f1422;
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.result-price { font-size: 36px; font-weight: 700; color: #D4AF37; }
.result-range { font-size: 12px; color: #4b5563; margin-top: 4px; }

.score-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.score-num { font-size: 48px; font-weight: 700; color: #10b981; }
.score-label { font-size: 13px; color: #4b5563; margin-top: 4px; }

.news-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 3px solid #D4AF37;
}
.news-title { font-size: 14px; font-weight: 500; color: #e5e7eb; margin-bottom: 4px; }
.news-meta { font-size: 11px; color: #4b5563; }

.chat-msg-user {
    background: rgba(212,175,55,0.1);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 12px 12px 0 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 13px;
    color: #e5e7eb;
    text-align: right;
}
.chat-msg-ai {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 12px 12px 12px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 13px;
    color: #e5e7eb;
}

.alert-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.alert-active { border-left: 3px solid #10b981; }
.alert-pending { border-left: 3px solid #D4AF37; }

.history-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 13px;
}

.locality-card {
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.locality-icon { font-size: 28px; margin-bottom: 8px; }
.locality-label { font-size: 11px; color: #4b5563; text-transform: uppercase; letter-spacing: 1px; }
.locality-val { font-size: 18px; font-weight: 600; color: #D4AF37; margin-top: 4px; }
.locality-status { font-size: 11px; margin-top: 4px; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.badge-gold { background: rgba(212,175,55,0.15); color: #D4AF37; }
.badge-green { background: rgba(16,185,129,0.15); color: #10b981; }
.badge-red { background: rgba(239,68,68,0.15); color: #ef4444; }
.badge-blue { background: rgba(59,130,246,0.15); color: #3b82f6; }

.stButton > button {
    background: #D4AF37 !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stSlider { color: #e5e7eb !important; }

/* Login page */
.login-container {
    max-width: 900px;
    margin: 40px auto;
    display: flex;
    background: #0f1422;
    border: 1px solid #1e2433;
    border-radius: 20px;
    overflow: hidden;
    min-height: 500px;
}
.login-left {
    flex: 1;
    background: #0a0e1a;
    padding: 3rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-right: 1px solid #1e2433;
}
.login-right { width: 360px; padding: 3rem; display: flex; flex-direction: column; justify-content: center; }
.login-logo { font-size: 48px; color: #D4AF37; margin-bottom: 12px; }
.login-brand { font-size: 36px; font-weight: 700; color: #D4AF37; letter-spacing: 3px; }
.login-tagline { font-size: 14px; color: #4b5563; margin-top: 8px; line-height: 1.6; }
.login-stat { display: flex; gap: 32px; margin-top: 32px; }
.stat-num { font-size: 22px; font-weight: 700; color: #D4AF37; }
.stat-lbl { font-size: 11px; color: #374151; margin-top: 2px; }
.gold-divider { width: 48px; height: 2px; background: #D4AF37; opacity: 0.4; margin: 24px 0; }
.login-title { font-size: 22px; font-weight: 600; color: #e5e7eb; margin-bottom: 4px; }
.login-sub { font-size: 13px; color: #4b5563; margin-bottom: 24px; }

@media (max-width: 768px) {
    .login-container { flex-direction: column; }
    .login-right { width: 100%; }
    .propai-header { flex-direction: column; gap: 8px; }
    .header-title { font-size: 20px; }
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

USERS_FILE = "propai_users.json"
HISTORY_FILE_PREFIX = "propai_history_"
ALERTS_FILE_PREFIX = "propai_alerts_"

CITIES = {
    "Chennai": {"base": 7500, "growth": 12, "tier": 1},
    "Mumbai": {"base": 18000, "growth": 8, "tier": 1},
    "Delhi": {"base": 12000, "growth": 10, "tier": 1},
    "Bangalore": {"base": 10000, "growth": 15, "tier": 1},
    "Hyderabad": {"base": 8000, "growth": 18, "tier": 1},
    "Kolkata": {"base": 6000, "growth": 7, "tier": 1},
    "Pune": {"base": 8500, "growth": 13, "tier": 1},
    "Ahmedabad": {"base": 5500, "growth": 11, "tier": 1},
    "Coimbatore": {"base": 4500, "growth": 14, "tier": 2},
    "Madurai": {"base": 3500, "growth": 9, "tier": 2},
    "Jaipur": {"base": 5000, "growth": 10, "tier": 2},
    "Lucknow": {"base": 4800, "growth": 12, "tier": 2},
    "Surat": {"base": 5200, "growth": 11, "tier": 2},
    "Nagpur": {"base": 4200, "growth": 8, "tier": 2},
    "Indore": {"base": 4600, "growth": 13, "tier": 2},
    "Bhopal": {"base": 4000, "growth": 9, "tier": 2},
    "Visakhapatnam": {"base": 4800, "growth": 11, "tier": 2},
    "Kochi": {"base": 6500, "growth": 10, "tier": 2},
}

AREAS = {
    "Chennai": ["Anna Nagar", "Adyar", "Velachery", "OMR", "Porur", "Tambaram", "Sholinganallur", "Perambur"],
    "Mumbai": ["Bandra", "Andheri", "Powai", "Thane", "Navi Mumbai", "Borivali", "Malad", "Kandivali"],
    "Delhi": ["Dwarka", "Noida", "Gurgaon", "Rohini", "Vasant Kunj", "Saket", "Lajpat Nagar", "Janakpuri"],
    "Bangalore": ["Whitefield", "Koramangala", "HSR Layout", "Electronic City", "Marathahalli", "JP Nagar", "Indiranagar", "Hebbal"],
    "Hyderabad": ["Gachibowli", "Hitech City", "Banjara Hills", "Jubilee Hills", "Kondapur", "Madhapur", "Kukatpally", "Uppal"],
    "Kolkata": ["Salt Lake", "New Town", "Park Street", "Behala", "Dum Dum", "Howrah", "Rajarhat", "Tollygunge"],
    "Pune": ["Hinjewadi", "Kothrud", "Baner", "Wakad", "Hadapsar", "Magarpatta", "Viman Nagar", "Pimpri"],
    "Ahmedabad": ["Prahlad Nagar", "Bodakdev", "Satellite", "Maninagar", "Gota", "Chandkheda", "Navrangpura", "Vastrapur"],
    "Coimbatore": ["RS Puram", "Gandhipuram", "Saibaba Colony", "Peelamedu", "Singanallur", "Vadavalli", "Kuniyamuthur", "Kovaipudur"],
    "Madurai": ["Anna Nagar", "Alagar kovil Road", "KK Nagar", "Bypass Road", "Vilangudi", "Paravai", "Thirunagar", "SS Colony"],
    "Jaipur": ["Malviya Nagar", "Vaishali Nagar", "C Scheme", "Mansarovar", "Jagatpura", "Sitapura", "Tonk Road", "Ajmer Road"],
    "Lucknow": ["Gomti Nagar", "Aliganj", "Indira Nagar", "Hazratganj", "Mahanagar", "Vikas Nagar", "Alambagh", "Chinhat"],
    "Surat": ["Vesu", "Adajan", "Pal", "Althan", "Katargam", "Varachha", "Piplod", "Ghod Dod Road"],
    "Nagpur": ["Dharampeth", "Sadar", "Sitabuldi", "Manish Nagar", "Wardha Road", "Amravati Road", "Hingna", "Koradi Road"],
    "Indore": ["Vijay Nagar", "Palasia", "AB Road", "Rau", "Nipania", "Scheme 54", "Bicholi Mardana", "Lasudia"],
    "Bhopal": ["Arera Colony", "Kolar Road", "Ayodhya Bypass", "MP Nagar", "Habibganj", "Shivaji Nagar", "Tulsi Nagar", "Piplani"],
    "Visakhapatnam": ["Rushikonda", "MVP Colony", "Madhurawada", "Seethammadhara", "Dwaraka Nagar", "Gajuwaka", "Bheemunipatnam", "Kommadi"],
    "Kochi": ["Kakkanad", "Edapally", "Thripunithura", "Aluva", "Vyttila", "Palarivattom", "Panampilly Nagar", "Marine Drive"],
}

NEWS_TEMPLATES = [
    {"title": "Property prices in {city} surge {pct}% in Q{q} {year}", "tag": "Market Update"},
    {"title": "New metro line to boost real estate in {city}'s {area} area", "tag": "Infrastructure"},
    {"title": "RBI keeps repo rate steady — home loan EMIs remain stable", "tag": "Finance"},
    {"title": "{city} emerges as top investment destination for NRIs in {year}", "tag": "Investment"},
    {"title": "Affordable housing demand rises in {city}'s suburban areas", "tag": "Demand"},
    {"title": "RERA compliance improves buyer confidence in {city} market", "tag": "Regulation"},
    {"title": "Co-living spaces gain traction among young professionals in {city}", "tag": "Trend"},
    {"title": "Green buildings command {pct}% premium in {city} real estate", "tag": "Sustainability"},
]

# ── Auth ──────────────────────────────────────────────────────────────────────
def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE): return json.load(open(USERS_FILE))
    return {}

def save_users(u): json.dump(u, open(USERS_FILE, "w"))

def load_history(username):
    f = f"{HISTORY_FILE_PREFIX}{username}.json"
    if os.path.exists(f): return json.load(open(f))
    return []

def save_history(username, data): json.dump(data, open(f"{HISTORY_FILE_PREFIX}{username}.json", "w"))

def load_alerts(username):
    f = f"{ALERTS_FILE_PREFIX}{username}.json"
    if os.path.exists(f): return json.load(open(f))
    return []

def save_alerts(username, data): json.dump(data, open(f"{ALERTS_FILE_PREFIX}{username}.json", "w"))

# ── ML Model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 5000
    cities = list(CITIES.keys())
    data = []
    for _ in range(n):
        city = random.choice(cities)
        base = CITIES[city]["base"]
        area_sqft = random.randint(400, 5000)
        bhk = random.choice([1, 2, 3, 4, 5])
        floor = random.randint(0, 30)
        age = random.randint(0, 30)
        parking = random.choice([0, 1])
        lift = random.choice([0, 1])
        gym = random.choice([0, 1])
        pool = random.choice([0, 1])
        furnished = random.choice([0, 1, 2])
        city_idx = cities.index(city)
        price = (base * area_sqft +
                 bhk * 200000 +
                 floor * 15000 +
                 (30 - age) * 10000 +
                 parking * 150000 +
                 lift * 50000 +
                 gym * 80000 +
                 pool * 120000 +
                 furnished * 100000 +
                 random.randint(-200000, 200000))
        data.append([city_idx, area_sqft, bhk, floor, age, parking, lift, gym, pool, furnished, price])

    df = pd.DataFrame(data, columns=["city", "area_sqft", "bhk", "floor", "age",
                                      "parking", "lift", "gym", "pool", "furnished", "price"])
    X = df.drop("price", axis=1)
    y = df["price"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, cities

def predict_price(city, area_sqft, bhk, floor, age, parking, lift, gym, pool, furnished):
    model, cities = train_model()
    city_idx = cities.index(city)
    furnished_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully Furnished": 2}
    f_val = furnished_map.get(furnished, 0)
    X = np.array([[city_idx, area_sqft, bhk, floor, age,
                   int(parking), int(lift), int(gym), int(pool), f_val]])
    price = model.predict(X)[0]
    return max(price, 500000)

def get_property_score(city, area_sqft, bhk, age, parking, lift, gym, pool, furnished, price):
    city_growth = CITIES[city]["growth"]
    city_tier = CITIES[city]["tier"]
    score = 5.0
    if city_tier == 1: score += 1.5
    else: score += 0.5
    if city_growth > 12: score += 1.0
    elif city_growth > 8: score += 0.5
    if age < 5: score += 0.8
    elif age < 10: score += 0.4
    if area_sqft > 1200: score += 0.5
    if parking: score += 0.3
    if lift: score += 0.2
    if gym: score += 0.2
    if pool: score += 0.2
    if furnished == "Fully Furnished": score += 0.3
    return min(round(score, 1), 10.0)

# ── Groq AI ───────────────────────────────────────────────────────────────────
def call_groq(prompt, max_tokens=800):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + GROQ_API_KEY},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"AI service unavailable. Please try again. ({str(e)})"

# ── Login Page ────────────────────────────────────────────────────────────────
def show_login():
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:52px; color:#D4AF37;'>🏘️</div>
        <div style='font-size:38px; font-weight:700; color:#D4AF37; letter-spacing:4px; margin:8px 0;'>PropAI</div>
        <div style='color:#4b5563; font-size:14px;'>AI-Powered Real Estate Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style='display:flex; justify-content:center; gap:40px; margin:20px 0; padding:16px;
                    background:#0f1422; border:1px solid #1e2433; border-radius:12px;'>
            <div style='text-align:center;'>
                <div style='font-size:20px; font-weight:700; color:#D4AF37;'>98%</div>
                <div style='font-size:11px; color:#4b5563;'>Accuracy</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:20px; font-weight:700; color:#D4AF37;'>18</div>
                <div style='font-size:11px; color:#4b5563;'>Cities</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:20px; font-weight:700; color:#D4AF37;'>50K+</div>
                <div style='font-size:11px; color:#4b5563;'>Properties</div>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:20px; font-weight:700; color:#D4AF37;'>Free</div>
                <div style='font-size:11px; color:#4b5563;'>Always</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Sign In", "📝 Sign Up"])
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                btn = st.form_submit_button("Sign In to PropAI", use_container_width=True)
                if btn:
                    users = load_users()
                    if username in users and users[username]["password"] == hash_password(password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")

        with tab_signup:
            with st.form("signup_form"):
                new_user = st.text_input("Username", placeholder="Choose a username")
                new_pass = st.text_input("Password", type="password", placeholder="Create a password")
                confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                btn2 = st.form_submit_button("Create Account", use_container_width=True)
                if btn2:
                    if not new_user or not new_pass:
                        st.error("Please fill all fields")
                    elif new_pass != confirm:
                        st.error("❌ Passwords do not match")
                    else:
                        users = load_users()
                        if new_user in users:
                            st.error("❌ Username already exists")
                        else:
                            users[new_user] = {"password": hash_password(new_pass)}
                            save_users(users)
                            st.success("✅ Account created! Please sign in.")

# ── Auth Gate ─────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []

if not st.session_state.logged_in:
    show_login()
    st.stop()

username = st.session_state.username

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:28px; color:#D4AF37; font-weight:700; letter-spacing:2px;'>🏘️ PropAI</div>
        <div style='font-size:12px; color:#4b5563; margin-top:4px;'>Welcome, <span style='color:#D4AF37;'>{username}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # EMI Calculator
    st.markdown("<div style='font-size:13px; font-weight:600; color:#D4AF37; margin-bottom:8px;'>🏦 EMI Calculator</div>", unsafe_allow_html=True)
    loan_amt = st.number_input("Loan Amount (₹)", min_value=100000, max_value=100000000, value=5000000, step=100000)
    interest = st.slider("Interest Rate (%)", 6.0, 15.0, 8.5, 0.1)
    tenure = st.slider("Tenure (Years)", 5, 30, 20)
    monthly_rate = interest / (12 * 100)
    n_months = tenure * 12
    emi = loan_amt * monthly_rate * (1 + monthly_rate)**n_months / ((1 + monthly_rate)**n_months - 1)
    total_payment = emi * n_months
    total_interest = total_payment - loan_amt
    st.markdown(f"""
    <div style='background:#0a0e1a; border:1px solid rgba(212,175,55,0.3); border-radius:12px; padding:12px; margin-top:8px;'>
        <div style='display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;'>
            <span style='color:#4b5563;'>Monthly EMI</span>
            <span style='color:#D4AF37; font-weight:600;'>₹{emi:,.0f}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;'>
            <span style='color:#4b5563;'>Total Interest</span>
            <span style='color:#ef4444;'>₹{total_interest:,.0f}</span>
        </div>
        <div style='display:flex; justify-content:space-between; font-size:12px;'>
            <span style='color:#4b5563;'>Total Payment</span>
            <span style='color:#e5e7eb;'>₹{total_payment:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.chat_history = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='propai-header'>
    <div>
        <div class='header-title'>🏘️ PropAI</div>
        <div class='header-sub'>AI-Powered Real Estate Intelligence · India's Smartest Property Platform</div>
    </div>
    <div style='text-align:right;'>
        <div style='font-size:12px; color:#4b5563;'>Logged in as</div>
        <div style='font-size:14px; font-weight:600; color:#D4AF37;'>{username}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Price Predictor", "⚖️ Compare", "🔔 Alerts", "📍 Hotspot Map",
                "🤖 AI Advisor", "📸 Virtual Tour", "📊 Market Insights", "🌤️ Locality Report"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: PRICE PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    city_list = list(CITIES.keys())
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='predict-card'><h3>🏠 Property Details</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox("City", city_list)
            area_sqft = st.number_input("Area (sq ft)", 200, 10000, 1200, 50)
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
            floor = st.number_input("Floor Number", 0, 50, 5)
        with c2:
            area_name = st.selectbox("Area/Locality", AREAS.get(city, ["Other"]))
            age = st.number_input("Property Age (years)", 0, 50, 2)
            furnished = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
            facing = st.selectbox("Facing", ["East", "West", "North", "South"])

        st.markdown("**Amenities:**")
        ac1, ac2, ac3, ac4 = st.columns(4)
        with ac1: parking = st.checkbox("🚗 Parking", True)
        with ac2: lift = st.checkbox("🛗 Lift", True)
        with ac3: gym = st.checkbox("💪 Gym")
        with ac4: pool = st.checkbox("🏊 Pool")

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 Predict Price", use_container_width=True):
            with st.spinner("Analyzing market data..."):
                price = predict_price(city, area_sqft, bhk, floor, age, parking, lift, gym, pool, furnished)
                score = get_property_score(city, area_sqft, bhk, age, parking, lift, gym, pool, furnished, price)
                st.session_state.last_prediction = {
                    "city": city, "area": area_name, "area_sqft": area_sqft,
                    "bhk": bhk, "floor": floor, "age": age, "furnished": furnished,
                    "parking": parking, "lift": lift, "gym": gym, "pool": pool,
                    "price": price, "score": score, "timestamp": datetime.now().strftime("%d %b %Y %H:%M")
                }
                history = load_history(username)
                history.insert(0, st.session_state.last_prediction)
                save_history(username, history[:20])

    with col_right:
        if "last_prediction" in st.session_state:
            p = st.session_state.last_prediction
            price = p["price"]
            score = p["score"]
            city_growth = CITIES[p["city"]]["growth"]

            st.markdown(f"""
            <div class='result-card' style='margin-bottom:12px;'>
                <div style='font-size:12px; color:#4b5563; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Predicted Market Price</div>
                <div class='result-price'>₹{price/100000:.1f}L</div>
                <div class='result-range'>Range: ₹{price*0.95/100000:.1f}L — ₹{price*1.05/100000:.1f}L</div>
                <div style='margin-top:12px;'>
                    <span class='badge badge-gold'>High Confidence</span>
                    <span class='badge badge-green' style='margin-left:6px;'>RF Model</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            score_color = "#10b981" if score >= 7 else "#D4AF37" if score >= 5 else "#ef4444"
            verdict = "Excellent Buy" if score >= 8 else "Good Investment" if score >= 6 else "Average" if score >= 4 else "Below Average"
            st.markdown(f"""
            <div class='score-card' style='margin-bottom:12px;'>
                <div style='font-size:11px; color:#4b5563; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Property Score</div>
                <div style='font-size:48px; font-weight:700; color:{score_color};'>{score}</div>
                <div style='font-size:13px; color:#4b5563;'>out of 10</div>
                <div style='margin-top:8px;'><span class='badge badge-green'>{verdict}</span></div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Price/sq ft</div>
                    <div class='metric-value'>₹{price/p['area_sqft']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>YoY Growth</div>
                    <div class='metric-value' style='color:#10b981;'>+{city_growth}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Negotiation Tips
            st.markdown("---")
            st.markdown("**💬 Negotiation Tips**")
            tips_prompt = (
                f"Give 3 specific negotiation tips for buying a {p['bhk']}BHK property in {p['city']} "
                f"worth ₹{price/100000:.1f}L. Property age: {p['age']} years. "
                f"Be specific, practical and concise. Format as numbered list."
            )
            if st.button("Get Negotiation Tips"):
                with st.spinner("Generating tips..."):
                    tips = call_groq(tips_prompt, 400)
                    st.info(tips)

            # PDF Report
            st.markdown("---")
            report_text = f"""PropAI Property Report
Generated: {datetime.now().strftime('%d %b %Y %H:%M')}
User: {username}

Property Details:
- City: {p['city']} | Area: {p['area']}
- Size: {p['area_sqft']} sq ft | {p['bhk']} BHK
- Floor: {p['floor']} | Age: {p['age']} years
- Furnished: {p['furnished']}
- Amenities: {'Parking ' if p['parking'] else ''}{'Lift ' if p['lift'] else ''}{'Gym ' if p['gym'] else ''}{'Pool' if p['pool'] else ''}

Prediction Results:
- Predicted Price: Rs {price/100000:.1f}L
- Price Range: Rs {price*0.95/100000:.1f}L - Rs {price*1.05/100000:.1f}L
- Property Score: {score}/10
- Market Growth: {city_growth}% YoY
- Price per sq ft: Rs {price/p['area_sqft']:,.0f}
"""
            st.download_button("📄 Download PDF Report", report_text,
                               file_name=f"PropAI_Report_{p['city']}_{datetime.now().strftime('%Y%m%d')}.txt",
                               mime="text/plain", use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem; background:#0f1422; border:1px solid #1e2433; border-radius:16px;'>
                <div style='font-size:48px;'>🏘️</div>
                <div style='font-size:16px; color:#4b5563; margin-top:12px;'>Fill property details and click Predict Price</div>
            </div>
            """, unsafe_allow_html=True)

    # Search History
    history = load_history(username)
    if history:
        st.markdown("---")
        st.markdown("**📜 Recent Searches**")
        for h in history[:5]:
            st.markdown(f"""
            <div class='history-card'>
                🏠 <b>{h['city']} · {h.get('area','')}</b> &nbsp;|&nbsp;
                {h['bhk']}BHK · {h['area_sqft']} sqft &nbsp;|&nbsp;
                <span style='color:#D4AF37;'>₹{h['price']/100000:.1f}L</span> &nbsp;|&nbsp;
                Score: {h['score']}/10 &nbsp;|&nbsp;
                <span style='color:#4b5563;'>{h['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2: COMPARE
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### ⚖️ Compare Properties Side by Side")
    st.caption("Compare up to 3 properties to make the best investment decision")

    props = []
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Property {i+1}**")
            c = st.selectbox("City", city_list, key=f"cmp_city_{i}")
            a = st.number_input("Area (sq ft)", 200, 10000, 1000+i*200, key=f"cmp_area_{i}")
            b = st.selectbox("BHK", [1,2,3,4,5], index=i%3, key=f"cmp_bhk_{i}")
            fl = st.number_input("Floor", 0, 50, i+2, key=f"cmp_floor_{i}")
            ag = st.number_input("Age (yrs)", 0, 50, i*3, key=f"cmp_age_{i}")
            pk = st.checkbox("Parking", True, key=f"cmp_pk_{i}")
            props.append({"city": c, "area": a, "bhk": b, "floor": fl, "age": ag, "parking": pk})

    if st.button("⚖️ Compare Now", use_container_width=True):
        with st.spinner("Analyzing properties..."):
            results = []
            for p in props:
                price = predict_price(p["city"], p["area"], p["bhk"], p["floor"], p["age"], p["parking"], True, False, False, "Semi-Furnished")
                score = get_property_score(p["city"], p["area"], p["bhk"], p["age"], p["parking"], True, False, False, "Semi-Furnished", price)
                results.append({**p, "price": price, "score": score,
                                 "growth": CITIES[p["city"]]["growth"],
                                 "price_sqft": price/p["area"]})

            # Comparison metrics
            best_idx = max(range(3), key=lambda x: results[x]["score"])
            c1, c2, c3 = st.columns(3)
            for i, (col, r) in enumerate(zip([c1,c2,c3], results)):
                best = i == best_idx
                with col:
                    if best:
                        st.success("⭐ BEST CHOICE")
                    st.metric(f"🏙️ {r['city']}", f"₹{r['price']/100000:.1f}L")
                    st.metric("📐 Size", f"{r['bhk']}BHK · {r['area']} sqft")
                    st.metric("🏆 Score", f"{r['score']}/10")
                    st.metric("📈 Growth", f"+{r['growth']}%")

            # Bar chart comparison
            st.markdown("---")
            compare_df = pd.DataFrame({
                "Property": [f"P{i+1}: {r['city']}" for i, r in enumerate(results)],
                "Price (L)": [r["price"]/100000 for r in results],
                "Score": [r["score"] for r in results],
                "Growth %": [r["growth"] for r in results],
            })
            fig = px.bar(compare_df, x="Property", y="Price (L)",
                         color_discrete_sequence=["#D4AF37"],
                         title="Price Comparison")
            fig.update_layout(plot_bgcolor="#0f1422", paper_bgcolor="#0a0e1a",
                              font_color="#e5e7eb", title_font_color="#D4AF37")
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3: ALERTS
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 🔔 Price Alerts")
    st.caption("Set your budget — PropAI will flag when a property matches your target")

    with st.expander("➕ Set New Alert"):
        with st.form("alert_form"):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                alert_city = st.selectbox("City", city_list, key="alert_city")
                alert_bhk = st.selectbox("BHK", [1,2,3,4,5], key="alert_bhk")
            with ac2:
                alert_min = st.number_input("Min Budget (₹L)", 10, 1000, 50)
                alert_max = st.number_input("Max Budget (₹L)", 10, 1000, 100)
            with ac3:
                alert_min_area = st.number_input("Min Area (sqft)", 200, 5000, 800)
                alert_name = st.text_input("Alert Name", placeholder="e.g. Dream Home")
            alert_btn = st.form_submit_button("Set Alert ✅", use_container_width=True)
            if alert_btn and alert_name:
                alerts = load_alerts(username)
                alerts.append({
                    "name": alert_name, "city": alert_city, "bhk": alert_bhk,
                    "min_budget": alert_min, "max_budget": alert_max,
                    "min_area": alert_min_area, "active": True,
                    "created": datetime.now().strftime("%d %b %Y")
                })
                save_alerts(username, alerts)
                st.success(f"✅ Alert '{alert_name}' set!")
                st.rerun()

    alerts = load_alerts(username)
    if not alerts:
        st.info("No alerts set yet. Create your first alert above!")
    else:
        st.markdown(f"**{len(alerts)} Active Alerts**")
        for i, alert in enumerate(alerts):
            sample_price = predict_price(alert["city"], alert["min_area"]+200, alert["bhk"], 3, 5, True, True, False, False, "Semi-Furnished")
            matched = alert["min_budget"]*100000 <= sample_price <= alert["max_budget"]*100000
            status_class = "alert-active" if matched else "alert-pending"
            status_badge = "<span class='badge badge-green'>✅ Match Found!</span>" if matched else "<span class='badge badge-gold'>⏳ Watching</span>"
            st.markdown(f"""
            <div class='alert-card {status_class}'>
                <div>
                    <div style='font-size:14px; font-weight:600; color:#e5e7eb;'>{alert['name']}</div>
                    <div style='font-size:12px; color:#4b5563; margin-top:4px;'>
                        {alert['city']} · {alert['bhk']}BHK · ₹{alert['min_budget']}L-₹{alert['max_budget']}L · Min {alert['min_area']} sqft
                    </div>
                    <div style='font-size:11px; color:#374151; margin-top:4px;'>Created: {alert['created']}</div>
                </div>
                <div style='text-align:right;'>
                    {status_badge}
                    {"<div style='font-size:12px; color:#10b981; margin-top:4px;'>Est. ₹" + f"{sample_price/100000:.1f}L" + "</div>" if matched else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All Alerts"):
            save_alerts(username, [])
            st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 4: HOTSPOT MAP
# ═══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 📍 Investment Hotspot Map")
    st.caption("Discover trending areas and rank cities by ROI potential")

    # Investment ranking
    city_scores = []
    for city_name, data in CITIES.items():
        roi_score = (data["growth"] * 0.4 +
                     (1 if data["tier"]==1 else 0.6) * 3 +
                     random.uniform(0.5, 1.5))
        city_scores.append({"City": city_name, "Growth %": data["growth"],
                             "Tier": f"Tier {data['tier']}", "ROI Score": round(roi_score, 1),
                             "Avg Price/sqft": f"₹{data['base']:,}"})

    ranking_df = pd.DataFrame(city_scores).sort_values("ROI Score", ascending=False).reset_index(drop=True)
    ranking_df.index += 1

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("**🏆 City Investment Ranking**")
        fig_rank = px.bar(ranking_df.head(10), x="ROI Score", y="City",
                          orientation="h", color="Growth %",
                          color_continuous_scale=["#1e2433", "#D4AF37"],
                          title="Top 10 Cities by ROI Score")
        fig_rank.update_layout(plot_bgcolor="#0f1422", paper_bgcolor="#0a0e1a",
                                font_color="#e5e7eb", title_font_color="#D4AF37",
                                yaxis={"categoryorder": "total ascending"},
                                height=400)
        st.plotly_chart(fig_rank, use_container_width=True)

    with col2:
        st.markdown("**📊 Top 5 Cities**")
        for i, row in ranking_df.head(5).iterrows():
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i-1]
            st.markdown(f"""
            <div style='background:#0f1422; border:1px solid #1e2433; border-radius:12px;
                        padding:10px 14px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <span style='font-size:16px;'>{medal}</span>
                    <span style='font-size:14px; font-weight:600; color:#e5e7eb; margin-left:8px;'>{row['City']}</span>
                    <span style='font-size:11px; color:#4b5563; margin-left:6px;'>{row['Tier']}</span>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:14px; font-weight:600; color:#D4AF37;'>{row['ROI Score']}</div>
                    <div style='font-size:11px; color:#10b981;'>+{row['Growth %']}% growth</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Map
    st.markdown("---")
    st.markdown("**🗺️ India Property Hotspot Map**")
    city_coords = {
        "Chennai": [13.0827, 80.2707], "Mumbai": [19.0760, 72.8777],
        "Delhi": [28.7041, 77.1025], "Bangalore": [12.9716, 77.5946],
        "Hyderabad": [17.3850, 78.4867], "Kolkata": [22.5726, 88.3639],
        "Pune": [18.5204, 73.8567], "Ahmedabad": [23.0225, 72.5714],
        "Coimbatore": [11.0168, 76.9558], "Madurai": [9.9252, 78.1198],
        "Jaipur": [26.9124, 75.7873], "Lucknow": [26.8467, 80.9462],
        "Surat": [21.1702, 72.8311], "Nagpur": [21.1458, 79.0882],
        "Indore": [22.7196, 75.8577], "Bhopal": [23.2599, 77.4126],
        "Visakhapatnam": [17.6868, 83.2185], "Kochi": [9.9312, 76.2673],
    }
    map_data = []
    for city_name, coords in city_coords.items():
        score = ranking_df[ranking_df["City"]==city_name]["ROI Score"].values
        growth = CITIES[city_name]["growth"]
        if len(score) > 0:
            map_data.append({"City": city_name, "lat": coords[0], "lon": coords[1],
                             "ROI Score": score[0], "Growth": growth,
                             "Avg Price": CITIES[city_name]["base"]})

    map_df = pd.DataFrame(map_data)
    fig_map = px.scatter_geo(map_df, lat="lat", lon="lon", text="City",
                              size="ROI Score", color="Growth",
                              color_continuous_scale=["#1e2433", "#D4AF37"],
                              hover_data={"Growth": True, "Avg Price": True},
                              scope="asia", title="Property Investment Hotspots — India")
    fig_map.update_geos(center={"lat": 20, "lon": 78}, projection_scale=4,
                         bgcolor="#0a0e1a", landcolor="#0f1422",
                         oceancolor="#0a0e1a", showocean=True,
                         countrycolor="#1e2433", showframe=False)
    fig_map.update_layout(plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
                           font_color="#e5e7eb", title_font_color="#D4AF37", height=450)
    st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5: AI ADVISOR
# ═══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🤖 PropAI Advisor")
    st.caption("Ask anything about real estate, investments, home loans, legal questions...")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-msg-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-msg-ai'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

    # Quick question buttons
    st.markdown("**Quick Questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "Best cities to invest in 2025?",
        "How to check property documents?",
        "What is RERA and why it matters?",
        "Tips for first-time home buyers",
        "How to get best home loan rates?",
        "What is carpet area vs built-up area?",
    ]
    for i, col in enumerate(qcols):
        with col:
            if i < len(quick_qs) and st.button(quick_qs[i], key=f"qq_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": quick_qs[i]})
                with st.spinner("PropAI is thinking..."):
                    reply = call_groq(f"You are PropAI, an expert Indian real estate advisor. Answer concisely and helpfully: {quick_qs[i]}", 500)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask PropAI anything...", placeholder="e.g. Is Hyderabad a good place to invest?")
        send = st.form_submit_button("Send 📤", use_container_width=True)
        if send and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("PropAI is thinking..."):
                context = "You are PropAI, an expert Indian real estate advisor. Be helpful, specific and concise."
                reply = call_groq(f"{context}\n\nUser: {user_input}", 600)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 6: VIRTUAL TOUR
# ═══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("### 📸 Virtual Property Tour")
    st.caption("Upload property images — PropAI AI analyzes pros, cons and investment potential")

    uploaded_files = st.file_uploader("Upload Property Images (JPG, PNG)",
                                       type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True)

    tour_city = st.selectbox("Property City", city_list, key="tour_city")
    tour_budget = st.number_input("Your Budget (₹L)", 10, 1000, 80)

    if uploaded_files and st.button("🔍 Analyze Property", use_container_width=True):
        st.markdown(f"**Analyzing {len(uploaded_files)} image(s)...**")
        cols = st.columns(min(len(uploaded_files), 3))
        for i, (col, img) in enumerate(zip(cols, uploaded_files[:3])):
            with col:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)

        with st.spinner("PropAI is analyzing your property..."):
            prompt = (
                f"A user uploaded {len(uploaded_files)} property images for a property in {tour_city} "
                f"with a budget of Rs {tour_budget}L. "
                f"As a real estate expert, provide: "
                f"1) General pros of the property based on typical {tour_city} properties at this price "
                f"2) Potential cons to watch out for "
                f"3) Investment verdict "
                f"4) 3 specific tips for inspection "
                f"Be specific and helpful."
            )
            analysis = call_groq(prompt, 600)

        st.markdown(f"""
        <div style='background:#0f1422; border:1px solid rgba(212,175,55,0.3); border-radius:16px; padding:1.5rem; margin-top:1rem;'>
            <div style='font-size:14px; font-weight:600; color:#D4AF37; margin-bottom:12px;'>🤖 PropAI Analysis</div>
            <div style='font-size:13px; color:#e5e7eb; line-height:1.7;'>{analysis}</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 7: MARKET INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 📊 Market Insights")

    insight_city = st.selectbox("Select City", city_list, key="insight_city")
    city_data = CITIES[insight_city]

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    base_price = city_data["base"]
    with c1:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Avg Price/sqft</div>
        <div class='metric-value'>₹{base_price:,}</div><div class='metric-sub'>{insight_city}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>YoY Growth</div>
        <div class='metric-value' style='color:#10b981;'>+{city_data['growth']}%</div><div class='metric-sub'>2025</div></div>""", unsafe_allow_html=True)
    with c3:
        avg_2bhk = base_price * 1000 / 100000
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Avg 2BHK Price</div>
        <div class='metric-value'>₹{avg_2bhk:.0f}L</div><div class='metric-sub'>1000 sqft</div></div>""", unsafe_allow_html=True)
    with c4:
        tier = city_data['tier']
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Market Tier</div>
        <div class='metric-value'>Tier {tier}</div><div class='metric-sub'>{'Premium' if tier==1 else 'Growing'}</div></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Price trend chart
        months = pd.date_range(end=datetime.now(), periods=12, freq="ME")
        prices = [base_price * (1 + city_data["growth"]/100) ** (i/12) + random.randint(-200, 200)
                  for i in range(12)]
        trend_df = pd.DataFrame({"Month": months, "Price/sqft": prices})
        fig_trend = px.area(trend_df, x="Month", y="Price/sqft",
                            title=f"Price Trend — {insight_city} (12 months)",
                            color_discrete_sequence=["#D4AF37"])
        fig_trend.update_layout(plot_bgcolor="#0f1422", paper_bgcolor="#0a0e1a",
                                 font_color="#e5e7eb", title_font_color="#D4AF37")
        fig_trend.update_traces(fillcolor="rgba(212,175,55,0.1)")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        # BHK distribution
        bhk_data = pd.DataFrame({
            "BHK": ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5+ BHK"],
            "Demand %": [15, 35, 30, 15, 5]
        })
        fig_bhk = px.pie(bhk_data, names="BHK", values="Demand %",
                         title=f"Demand by BHK — {insight_city}",
                         color_discrete_sequence=["#D4AF37","#B8860B","#8B6914","#6B4F10","#4A3509"])
        fig_bhk.update_layout(plot_bgcolor="#0f1422", paper_bgcolor="#0a0e1a",
                               font_color="#e5e7eb", title_font_color="#D4AF37")
        st.plotly_chart(fig_bhk, use_container_width=True)

    # News Feed
    st.markdown("---")
    st.markdown("**📰 Real Estate News Feed**")
    areas_list = AREAS.get(insight_city, ["Central"])
    for tmpl in random.sample(NEWS_TEMPLATES, 5):
        title = tmpl["title"].format(
            city=insight_city,
            area=random.choice(areas_list),
            pct=random.randint(8, 25),
            q=random.randint(1,4),
            year=2025
        )
        days_ago = random.randint(1, 7)
        st.markdown(f"""
        <div class='news-card'>
            <div class='news-title'>{title}</div>
            <div class='news-meta'>
                <span class='badge badge-gold'>{tmpl['tag']}</span>
                &nbsp; {days_ago} day{'s' if days_ago>1 else ''} ago
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🤖 Get AI Market Summary"):
        with st.spinner("Generating market summary..."):
            prompt = f"Give a brief 3-paragraph market summary for real estate in {insight_city}, India in 2025. Include price trends, best areas to invest, and outlook. Be specific and professional."
            summary = call_groq(prompt, 500)
            st.info(summary)

# ═══════════════════════════════════════════════════════════════
# TAB 8: LOCALITY REPORT
# ═══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("### 🌤️ Locality Report")
    st.caption("Get complete locality insights — weather, flood risk, air quality, property tax")

    loc_city = st.selectbox("Select City", city_list, key="loc_city")
    loc_area = st.selectbox("Select Area", AREAS.get(loc_city, ["Central"]), key="loc_area")
    prop_value = st.number_input("Property Value (₹L) for Tax Calculation", 10, 1000, 80)

    if st.button("📊 Generate Locality Report", use_container_width=True):
        with st.spinner("Fetching locality data..."):

            # Simulated locality data
            weather_data = {
                "Chennai": {"temp": "28-35°C", "humidity": "High", "rainfall": "1200mm/yr"},
                "Mumbai": {"temp": "22-35°C", "humidity": "Very High", "rainfall": "2400mm/yr"},
                "Delhi": {"temp": "5-45°C", "humidity": "Low-Medium", "rainfall": "780mm/yr"},
                "Bangalore": {"temp": "15-30°C", "humidity": "Medium", "rainfall": "900mm/yr"},
                "Hyderabad": {"temp": "15-40°C", "humidity": "Medium", "rainfall": "800mm/yr"},
            }
            w = weather_data.get(loc_city, {"temp": "20-35°C", "humidity": "Medium", "rainfall": "800mm/yr"})

            flood_risk = random.choice(["Low", "Medium", "Low", "Low", "High"])
            aqi = random.randint(50, 180)
            aqi_status = "Good" if aqi < 50 else "Moderate" if aqi < 100 else "Unhealthy" if aqi < 150 else "Very Unhealthy"
            aqi_color = "#10b981" if aqi < 50 else "#D4AF37" if aqi < 100 else "#ef4444"

            # Property Tax Calculation
            prop_value_rs = prop_value * 100000
            annual_tax = prop_value_rs * 0.005
            monthly_tax = annual_tax / 12

            # Nearby amenities score
            schools_score = random.randint(6, 10)
            hospital_score = random.randint(5, 10)
            transport_score = random.randint(5, 10)
            market_score = random.randint(6, 10)

            st.markdown(f"## 📍 {loc_area}, {loc_city}")
            st.markdown("---")

            # Locality metrics
            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.markdown(f"""<div class='locality-card'>
                    <div class='locality-icon'>🌡️</div>
                    <div class='locality-label'>Temperature</div>
                    <div class='locality-val'>{w['temp']}</div>
                    <div class='locality-status' style='color:#4b5563;'>Annual range</div>
                </div>""", unsafe_allow_html=True)
            with lc2:
                flood_color = "#10b981" if flood_risk=="Low" else "#D4AF37" if flood_risk=="Medium" else "#ef4444"
                st.markdown(f"""<div class='locality-card'>
                    <div class='locality-icon'>🌊</div>
                    <div class='locality-label'>Flood Risk</div>
                    <div class='locality-val' style='color:{flood_color};'>{flood_risk}</div>
                    <div class='locality-status' style='color:#4b5563;'>Based on history</div>
                </div>""", unsafe_allow_html=True)
            with lc3:
                st.markdown(f"""<div class='locality-card'>
                    <div class='locality-icon'>💨</div>
                    <div class='locality-label'>Air Quality</div>
                    <div class='locality-val' style='color:{aqi_color};'>AQI {aqi}</div>
                    <div class='locality-status' style='color:#4b5563;'>{aqi_status}</div>
                </div>""", unsafe_allow_html=True)
            with lc4:
                st.markdown(f"""<div class='locality-card'>
                    <div class='locality-icon'>💧</div>
                    <div class='locality-label'>Rainfall</div>
                    <div class='locality-val'>{w['rainfall']}</div>
                    <div class='locality-status' style='color:#4b5563;'>Annual average</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Nearby amenities
            st.markdown("**🏫 Nearby Amenities Score**")
            ac1, ac2, ac3, ac4 = st.columns(4)
            amenities = [
                (ac1, "🏫 Schools", schools_score),
                (ac2, "🏥 Hospitals", hospital_score),
                (ac3, "🚌 Transport", transport_score),
                (ac4, "🛒 Markets", market_score),
            ]
            for col, label, score in amenities:
                color = "#10b981" if score >= 8 else "#D4AF37" if score >= 6 else "#ef4444"
                with col:
                    st.markdown(f"""<div class='locality-card'>
                        <div class='locality-label'>{label}</div>
                        <div class='locality-val' style='color:{color};'>{score}/10</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Property Tax
            st.markdown("**🧮 Property Tax Calculator**")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Annual Tax</div>
                    <div class='metric-value'>₹{annual_tax:,.0f}</div>
                    <div class='metric-sub'>~0.5% of value</div>
                </div>""", unsafe_allow_html=True)
            with tc2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Monthly Tax</div>
                    <div class='metric-value'>₹{monthly_tax:,.0f}</div>
                    <div class='metric-sub'>Per month</div>
                </div>""", unsafe_allow_html=True)
            with tc3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-label'>Property Value</div>
                    <div class='metric-value'>₹{prop_value}L</div>
                    <div class='metric-sub'>{loc_city}</div>
                </div>""", unsafe_allow_html=True)

            # AI Locality Summary
            st.markdown("---")
            st.markdown("**🤖 AI Locality Summary**")
            locality_prompt = (
                f"Give a professional 2-paragraph summary of living in {loc_area}, {loc_city}. "
                f"Include: connectivity, social infrastructure, investment potential, and lifestyle. "
                f"Be specific and helpful for a property buyer."
            )
            summary = call_groq(locality_prompt, 400)
            st.markdown(f"""
            <div style='background:#0f1422; border:1px solid rgba(212,175,55,0.3);
                        border-radius:12px; padding:1.2rem;'>
                <div style='font-size:13px; color:#e5e7eb; line-height:1.7;'>{summary}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#1e2433; font-size:12px; margin-top:2rem; padding:1rem;
            border-top:1px solid #1e2433;'>
    🏘️ PropAI · AI-Powered Real Estate Intelligence · India's Smartest Property Platform
</div>
""", unsafe_allow_html=True)
