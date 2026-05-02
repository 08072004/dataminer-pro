# ============================================================
#  app.py  —  Point d'entrée principal de DataMiner Pro
#
#  Lancement :
#      streamlit run app.py
#
#  Structure du projet :
#      app.py                   ← ce fichier (routeur principal)
#      sidebar.py               ← barre latérale de navigation
#      utils/
#          __init__.py
#          styles.py            ← CSS global
#          plots.py             ← helpers matplotlib thème sombre
#          session.py           ← initialisation du session_state
#      volets/
#          __init__.py
#          preprocessing.py     ← Volet 1 : Prétraitement
#          clustering.py        ← Volet 2 : Clustering
#          classification.py    ← Volet 3 : Classification
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

# ── Configuration de la page (doit être le 1er appel Streamlit) ─────────────
st.set_page_config(
    page_title="DataMiner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modules internes ─────────────────────────────────────────────────────────
from utils.styles import inject_css
from utils.session import init_session
import sidebar
from volets import preprocessing, clustering, classification

# ── Initialisation ───────────────────────────────────────────────────────────
inject_css()       # Injecte le CSS global (thème sombre)
init_session()     # Initialise les clés du session_state

# ── Navigation ───────────────────────────────────────────────────────────────
volet = sidebar.render()

# ── Routage vers le volet sélectionné ────────────────────────────────────────
if "Volet 1" in volet:
    preprocessing.render()
elif "Volet 2" in volet:
    clustering.render()
elif "Volet 3" in volet:
    classification.render()