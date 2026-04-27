# ============================================================
#  sidebar.py  —  Barre latérale de navigation DataMiner Pro
# ============================================================
import streamlit as st


VOLETS = [
    "📦  Volet 1 — Prétraitement",
    "🔵  Volet 2 — Clustering",
    "🤖  Volet 3 — Classification",
]


def render() -> str:
    """
    Affiche la sidebar (logo, navigation, infos dataset).
    Retourne la clé du volet sélectionné (str).
    """
    with st.sidebar:
        st.markdown("## 🔬 DataMiner Pro")
        st.markdown(
            "<span class='badge'>v1.0 · TP Fouille de Données</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        volet = st.radio("Navigation", VOLETS, label_visibility="collapsed")

        st.markdown("---")
        _dataset_info()

        st.markdown("---")
        st.markdown(
            "<small style='color:#8b949e'>Fouille de Données 1 · 2025-2026</small>",
            unsafe_allow_html=True,
        )

    return volet


def _dataset_info():
    """Affiche un résumé du dataset chargé dans la sidebar."""
    from utils.session import get_active_df

    df = get_active_df()
    if df is not None:
        st.markdown("**Dataset actuel**")
        st.markdown(f"- Lignes : `{df.shape[0]}`")
        st.markdown(f"- Colonnes : `{df.shape[1]}`")
        st.markdown(f"- Valeurs manquantes : `{df.isnull().sum().sum()}`")
    else:
        st.info("Aucun dataset chargé")