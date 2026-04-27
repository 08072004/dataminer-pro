# ============================================================
#  utils/session.py  —  Initialisation du session_state Streamlit
# ============================================================
import streamlit as st


# Clés et valeurs par défaut pour le session_state
SESSION_DEFAULTS = {
    # Volet 1 — Données brutes et transformées
    "df":            None,   # DataFrame original chargé
    "df_cleaned":    None,   # DataFrame après nettoyage
    "df_normalized": None,   # DataFrame après normalisation
    "scaler_type":   "Min-Max",

    # Volet 2 — Clustering
    "cluster_labels": None,

    # Volet 3 — Classification
    "target_col":  None,
    "X_train":     None,
    "X_test":      None,
    "y_train":     None,
    "y_test":      None,
    "feat_names":  None,
    "model":       None,
    "y_pred":      None,
    "model_name":  None,
}


def init_session():
    """
    Initialise toutes les clés du session_state avec leurs valeurs
    par défaut si elles n'existent pas encore.
    Doit être appelée une seule fois au démarrage de l'application.
    """
    for key, default in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_active_df():
    """
    Retourne le DataFrame le plus avancé dans le pipeline :
    normalisé > nettoyé > brut.
    """
    if st.session_state.df_normalized is not None:
        return st.session_state.df_normalized
    if st.session_state.df_cleaned is not None:
        return st.session_state.df_cleaned
    return st.session_state.df