# ============================================================
#  utils/styles.py  —  CSS global de l'interface DataMiner Pro
# ============================================================
import streamlit as st


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

:root {
    --bg: #050a15;
    --surface: #0f1823;
    --surface2: #1a2332;
    --surface3: #252f41;
    --accent: #00e5ff;
    --accent2: #00ff91;
    --accent3: #ff5252;
    --accent4: #c77dff;
    --accent5: #ff9e40;
    --text: #ffffff;
    --text-muted: #94a3b8;
    --border: #334155;
    --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-2: linear-gradient(135deg, #00e5ff 0%, #00ff91 100%);
    --gradient-3: linear-gradient(135deg, #ff5252 0%, #ff9e40 100%);
    --shadow-glow: 0 0 25px rgba(0, 229, 255, 0.4);
    --shadow-hover: 0 10px 30px rgba(0, 229, 255, 0.3);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    animation: fadeIn 0.6s ease-out;
}

.stApp { 
    background: var(--bg);
    background-image: 
        radial-gradient(circle at 20% 80%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(189, 52, 254, 0.05) 0%, transparent 50%);
}

.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
    animation: slideUp 0.8s ease-out;
}

/* Animations globales */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { 
        opacity: 0;
        transform: translateY(20px);
    }
    to { 
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

@keyframes glow {
    0%, 100% { box-shadow: var(--shadow-glow); }
    50% { box-shadow: 0 0 30px rgba(0, 212, 255, 0.5); }
}

@keyframes slideInLeft {
    from { 
        opacity: 0;
        transform: translateX(-30px);
    }
    to { 
        opacity: 1;
        transform: translateX(0);
    }
}

/* Sidebar moderne */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface) 0%, var(--surface2) 100%);
    border-right: 1px solid var(--border);
    backdrop-filter: blur(10px);
    animation: slideInLeft 0.6s ease-out;
}

section[data-testid="stSidebar"] .stRadio label {
    color: var(--text);
    font-size: 0.95rem;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 0.2rem 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid transparent;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--surface3);
    border-color: var(--accent);
    transform: translateX(5px);
    box-shadow: var(--shadow-hover);
}

section[data-testid="stSidebar"] .stRadio label[data-testid="stMarkdownContainer"] {
    background: var(--gradient-2);
    color: var(--bg);
    font-weight: 600;
    animation: glow 2s infinite;
}

/* ── Titres ── */
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: var(--text);
}

/* ── Cartes modernes avec effets */
.card {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--gradient-2);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.card:hover::before {
    transform: scaleX(1);
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-hover);
    border-color: var(--accent);
}

.metric-card {
    background: linear-gradient(135deg, var(--surface2) 0%, var(--surface3) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
}

.metric-card:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    border-color: var(--accent);
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: var(--gradient-2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: pulse 2s infinite;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* Badge moderne */
.badge {
    background: var(--gradient-2);
    border: none;
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--bg);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    animation: glow 3s infinite;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
}

/* Boutons modernes */
.stButton > button {
    background: linear-gradient(135deg, var(--surface2) 0%, var(--surface3) 100%);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    position: relative;
    overflow: hidden;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--gradient-2);
    transition: left 0.3s ease;
    z-index: -1;
}

.stButton > button:hover::before {
    left: 0;
}

.stButton > button:hover {
    border-color: var(--accent);
    color: var(--bg);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ── Dataframe moderne */
.stDataFrame {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.stDataFrame table {
    background: var(--surface);
}

.stDataFrame th {
    background: var(--surface2);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stDataFrame td {
    border-color: var(--border);
    transition: background 0.2s ease;
}

.stDataFrame tr:hover td {
    background: var(--surface3);
}

/* Inputs modernes */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div,
.stTextArea > div > div {
    background: linear-gradient(135deg, var(--surface2) 0%, var(--surface3) 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    transition: all 0.3s ease;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.stSelectbox:focus-within > div > div,
.stNumberInput:focus-within > div > div,
.stTextInput:focus-within > div > div,
.stTextArea:focus-within > div > div {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1), inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* File uploader moderne */
.stFileUploader {
    background: linear-gradient(135deg, var(--surface2) 0%, var(--surface3) 100%);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stFileUploader::before {
    content: 'Drop your files here';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    font-weight: 600;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.stFileUploader:hover::before {
    opacity: 1;
}

.stFileUploader:hover {
    border-color: var(--accent);
    background: linear-gradient(135deg, var(--surface3) 0%, var(--surface2) 100%);
    transform: scale(1.02);
}

/* Alertes modernes */
.stAlert {
    border-radius: 12px;
    border-left-width: 4px;
    padding: 1rem 1.5rem;
    font-weight: 500;
    animation: slideUp 0.4s ease-out;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* Onglets modernes */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border-radius: 12px;
    padding: 6px;
    gap: 6px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    transition: all 0.3s ease;
    position: relative;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--surface3);
    color: var(--text);
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-2);
    color: var(--bg);
    box-shadow: var(--shadow-hover);
    transform: translateY(-1px);
}

/* Métriques modernes */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 4px;
    height: 100%;
    background: var(--gradient-2);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
}

/* Progress bar moderne */
.stProgress > div > div {
    background: var(--gradient-2);
    border-radius: 8px;
}

/* Slider moderne */
.stSlider > div > div {
    background: var(--surface2);
    border-radius: 8px;
}

.stSlider [role="slider"] {
    background: var(--accent);
    box-shadow: var(--shadow-glow);
}

/* Effets de chargement */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.loading {
    background: linear-gradient(90deg, var(--surface2) 0%, var(--surface3) 50%, var(--surface2) 100%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 1.5rem;
        max-width: 100%;
    }
    
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.5rem; }
    h3 { font-size: 1.2rem; }
    
    .card {
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    
    .metric-card {
        padding: 0.8rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
    }
    
    .stButton > button {
        padding: 0.8rem;
        font-size: 0.9rem;
    }
    
    section[data-testid="stSidebar"] {
        width: 280px !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
    }
}

@media (max-width: 480px) {
    .main .block-container {
        padding: 0.8rem 1rem;
    }
    
    h1 { font-size: 1.5rem; }
    h2 { font-size: 1.3rem; }
    h3 { font-size: 1.1rem; }
    
    .card {
        padding: 0.8rem;
        border-radius: 12px;
    }
    
    .metric-card {
        padding: 0.6rem;
        border-radius: 8px;
    }
    
    .metric-value {
        font-size: 1.2rem;
    }
    
    .metric-label {
        font-size: 0.7rem;
    }
    
    .stButton > button {
        padding: 0.6rem;
        font-size: 0.8rem;
        border-radius: 8px;
    }
    
    .badge {
        padding: 0.2rem 0.6rem;
        font-size: 0.65rem;
    }
    
    section[data-testid="stSidebar"] {
        width: 250px !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 0.6rem;
        font-size: 0.8rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem;
        padding: 0.4rem 0.8rem;
    }
}

/* Animation pour l'entrée des éléments */
@keyframes staggerIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.card:nth-child(1) { animation: staggerIn 0.6s ease-out 0.1s both; }
.card:nth-child(2) { animation: staggerIn 0.6s ease-out 0.2s both; }
.card:nth-child(3) { animation: staggerIn 0.6s ease-out 0.3s both; }
.card:nth-child(4) { animation: staggerIn 0.6s ease-out 0.4s both; }

/* Effet de particules de fond optionnel */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: -1;
}

.particle {
    position: absolute;
    width: 2px;
    height: 2px;
    background: var(--accent);
    border-radius: 50%;
    opacity: 0.3;
    animation: float 20s infinite linear;
}

@keyframes float {
    from {
        transform: translateY(100vh) translateX(0);
    }
    to {
        transform: translateY(-100px) translateX(100px);
    }
}
</style>
"""


def inject_css():
    """Injecte le CSS global dans l'application Streamlit."""
    st.markdown(CSS, unsafe_allow_html=True)