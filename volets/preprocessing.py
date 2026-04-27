# ============================================================
#  volets/preprocessing.py  —  Volet 1 : Prétraitement
#
#  Contient 5 onglets :
#    1. Importation
#    2. Exploration  (statistiques descriptives, types)
#    3. Nettoyage    (valeurs manquantes, doublons)
#    4. Normalisation (Min-Max / Z-score)
#    5. Visualisation (Boxplot, Scatter Plot)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.plots import dark_fig


# ─────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE DU VOLET
# ─────────────────────────────────────────────────────────────
def render():
    st.markdown("# 📦 Volet 1 — Prétraitement")
    st.markdown("Pipeline complet de préparation des données avant analyse.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Importation",
        "🔍 Exploration",
        "🧹 Nettoyage",
        "📐 Normalisation",
        "📊 Visualisation",
    ])

    with tab1: _tab_importation()
    with tab2: _tab_exploration()
    with tab3: _tab_nettoyage()
    with tab4: _tab_normalisation()
    with tab5: _tab_visualisation()


# ─────────────────────────────────────────────────────────────
#  ONGLET 1 — IMPORTATION
# ─────────────────────────────────────────────────────────────
def _tab_importation():
    st.markdown("### 📂 Chargement du Dataset")

    upload = st.file_uploader(
        "Importer un fichier CSV ou Excel",
        type=["csv", "xlsx", "xls"],
    )

    col_sep, col_enc = st.columns(2)
    sep = col_sep.selectbox("Séparateur CSV", [",", ";", "\\t", "|"])
    enc = col_enc.selectbox("Encodage", ["utf-8", "latin-1", "iso-8859-1"])

    if upload:
        try:
            if upload.name.endswith(".csv"):
                df = pd.read_csv(upload, sep=sep, encoding=enc)
            else:
                df = pd.read_excel(upload)

            # Réinitialiser les étapes suivantes quand un nouveau fichier est chargé
            st.session_state.df = df
            st.session_state.df_cleaned = None
            st.session_state.df_normalized = None

            st.success(
                f"✅ Dataset chargé : **{df.shape[0]}** lignes × **{df.shape[1]}** colonnes"
            )
        except Exception as e:
            st.error(f"Erreur de chargement : {e}")

    if st.session_state.df is not None:
        st.markdown("**Aperçu des données :**")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  ONGLET 2 — EXPLORATION
# ─────────────────────────────────────────────────────────────
def _tab_exploration():
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset.")
        return

    df = st.session_state.df
    st.markdown("### 🔍 Exploration du Dataset")

    # ── Infos générales
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes",           df.shape[0])
    c2.metric("Colonnes",         df.shape[1])
    c3.metric("Val. manquantes",  int(df.isnull().sum().sum()))
    c4.metric("Doublons",         int(df.duplicated().sum()))

    st.markdown("---")

    # ── Types des attributs
    st.markdown("#### Types des attributs")
    type_df = pd.DataFrame({
        "Colonne":        df.columns,
        "Type":           df.dtypes.values,
        "Valeurs uniques": [df[c].nunique() for c in df.columns],
        "Manquants":      df.isnull().sum().values,
        "% Manquants":    (df.isnull().sum().values / len(df) * 100).round(2),
    })
    st.dataframe(type_df, use_container_width=True)

    st.markdown("---")

    # ── Statistiques descriptives (les 5 nombres + moyenne + mode)
    st.markdown("#### Statistiques descriptives (5 nombres + moyenne / mode)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if num_cols:
        desc = df[num_cols].describe().T
        desc["mode"] = [
            df[c].mode()[0] if not df[c].mode().empty else np.nan
            for c in num_cols
        ]
        desc = desc[["min", "25%", "50%", "75%", "max", "mean", "mode"]]
        desc.columns = ["Min", "Q1", "Médiane", "Q3", "Max", "Moyenne", "Mode"]
        st.dataframe(desc.round(4), use_container_width=True)
    else:
        st.info("Aucune colonne numérique détectée.")


# ─────────────────────────────────────────────────────────────
#  ONGLET 3 — NETTOYAGE
# ─────────────────────────────────────────────────────────────
def _tab_nettoyage():
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset.")
        return

    df_work = (
        st.session_state.df_cleaned.copy()
        if st.session_state.df_cleaned is not None
        else st.session_state.df.copy()
    )

    st.markdown("### 🧹 Gestion des Valeurs Manquantes")

    # ── Résumé des valeurs manquantes
    missing = df_work.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("✅ Aucune valeur manquante détectée !")
    else:
        st.markdown(f"**{len(missing)} colonnes** contiennent des valeurs manquantes :")
        miss_df = pd.DataFrame({
            "Colonne":      missing.index,
            "Manquants":    missing.values,
            "Pourcentage":  (missing.values / len(df_work) * 100).round(2),
        })
        st.dataframe(miss_df, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # ── Imputation
    with col_a:
        st.markdown("**Imputation des valeurs manquantes**")
        strategy_num = st.selectbox(
            "Stratégie (colonnes numériques)",
            ["Moyenne", "Médiane", "Mode", "Valeur constante"],
        )
        fill_val = 0.0
        if strategy_num == "Valeur constante":
            fill_val = st.number_input("Valeur de remplacement", value=0.0)

        if st.button("🔧 Appliquer l'imputation"):
            df_c = df_work.copy()
            num_cols = df_c.select_dtypes(include=np.number).columns
            cat_cols = df_c.select_dtypes(exclude=np.number).columns

            # Numériques
            for col in num_cols:
                if df_c[col].isnull().any():
                    if strategy_num == "Moyenne":
                        df_c[col].fillna(df_c[col].mean(), inplace=True)
                    elif strategy_num == "Médiane":
                        df_c[col].fillna(df_c[col].median(), inplace=True)
                    elif strategy_num == "Mode":
                        df_c[col].fillna(df_c[col].mode()[0], inplace=True)
                    else:
                        df_c[col].fillna(fill_val, inplace=True)

            # Catégorielles → mode
            for col in cat_cols:
                if df_c[col].isnull().any():
                    fill = df_c[col].mode()[0] if not df_c[col].mode().empty else "inconnu"
                    df_c[col].fillna(fill, inplace=True)

            st.session_state.df_cleaned = df_c
            st.session_state.df_normalized = None
            st.success(
                f"✅ Imputation appliquée. Valeurs manquantes restantes : "
                f"{df_c.isnull().sum().sum()}"
            )

    # ── Doublons
    with col_b:
        st.markdown("**Suppression des doublons**")
        dupl = df_work.duplicated().sum()
        st.info(f"Doublons détectés : **{dupl}**")

        if st.button("🗑️ Supprimer les doublons"):
            df_c = (
                st.session_state.df_cleaned.copy()
                if st.session_state.df_cleaned is not None
                else st.session_state.df.copy()
            )
            before = len(df_c)
            df_c.drop_duplicates(inplace=True)
            st.session_state.df_cleaned = df_c
            st.success(f"✅ {before - len(df_c)} doublon(s) supprimé(s).")


# ─────────────────────────────────────────────────────────────
#  ONGLET 4 — NORMALISATION
# ─────────────────────────────────────────────────────────────
def _tab_normalisation():
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset.")
        return

    df_work = (
        st.session_state.df_cleaned
        if st.session_state.df_cleaned is not None
        else st.session_state.df.copy()
    )
    num_cols = df_work.select_dtypes(include=np.number).columns.tolist()

    st.markdown("### 📐 Normalisation des Données")

    if not num_cols:
        st.warning("Aucune colonne numérique à normaliser.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        method = st.radio("Méthode", ["Min-Max Scaling", "Standardisation (Z-score)"])
        cols_to_norm = st.multiselect(
            "Colonnes à normaliser", num_cols, default=num_cols
        )

        if st.button("⚡ Normaliser"):
            if not cols_to_norm:
                st.warning("Sélectionnez au moins une colonne.")
            else:
                df_n = df_work.copy()
                scaler = (
                    MinMaxScaler()
                    if method == "Min-Max Scaling"
                    else StandardScaler()
                )
                df_n[cols_to_norm] = scaler.fit_transform(df_n[cols_to_norm])
                st.session_state.df_normalized = df_n
                st.session_state.scaler_type = (
                    "Min-Max" if method == "Min-Max Scaling" else "Z-score"
                )
                st.success("✅ Normalisation appliquée.")

    with col2:
        st.markdown("**Formules de référence**")
        st.markdown("""
        **Min-Max Scaling :**
        `X' = (X − X_min) / (X_max − X_min)`
        → Valeurs comprises entre **0 et 1**

        ---

        **Standardisation (Z-score) :**
        `X' = (X − μ) / σ`
        → Moyenne = **0**, Écart-type = **1**
        """)

    # Aperçu après normalisation
    if st.session_state.df_normalized is not None:
        st.markdown("**Aperçu après normalisation :**")
        st.dataframe(
            st.session_state.df_normalized[num_cols].head(8),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
#  ONGLET 5 — VISUALISATION
# ─────────────────────────────────────────────────────────────
def _tab_visualisation():
    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset.")
        return

    # Utilise le DataFrame le plus avancé dans le pipeline
    df_v = (
        st.session_state.df_normalized
        if st.session_state.df_normalized is not None
        else st.session_state.df_cleaned
        if st.session_state.df_cleaned is not None
        else st.session_state.df
    )

    num_cols = df_v.select_dtypes(include=np.number).columns.tolist()
    st.markdown("### 📊 Visualisation")

    col_bp, col_sc = st.columns(2)

    # ── Boxplot
    with col_bp:
        st.markdown("#### Boxplot")
        bp_col = st.selectbox("Variable", num_cols, key="bp")
        fig, ax = dark_fig(figsize=(6, 4))
        ax.boxplot(
            df_v[bp_col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#1c2f4a", color="#58a6ff"),
            whiskerprops=dict(color="#8b949e"),
            capprops=dict(color="#8b949e"),
            medianprops=dict(color="#7ee787", linewidth=2),
            flierprops=dict(marker="o", color="#f78166", alpha=0.7),
        )
        ax.set_title(f"Boxplot — {bp_col}", color="#e6edf3", fontsize=11)
        ax.set_xlabel(bp_col)
        st.pyplot(fig)
        plt.close()

    # ── Scatter Plot
    with col_sc:
        st.markdown("#### Scatter Plot")
        if len(num_cols) >= 2:
            sc1 = st.selectbox("Axe X", num_cols, index=0, key="sc1")
            sc2 = st.selectbox("Axe Y", num_cols, index=1, key="sc2")
            fig2, ax2 = dark_fig(figsize=(6, 4))
            ax2.scatter(
                df_v[sc1], df_v[sc2],
                alpha=0.6, s=20, c="#58a6ff", edgecolors="none",
            )
            ax2.set_xlabel(sc1)
            ax2.set_ylabel(sc2)
            ax2.set_title(f"{sc1} vs {sc2}", color="#e6edf3", fontsize=11)
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("Minimum 2 colonnes numériques requises pour le Scatter Plot.")