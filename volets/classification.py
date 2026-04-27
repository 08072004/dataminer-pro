# ============================================================
#  volets/classification.py  —  Volet 3 : Page d'Attente Classification
#  
#  Page d'attente professionnelle indiquant que la classification
#  sera étudiée dans les prochains modules
# ============================================================
import streamlit as st
import time
from utils.session import get_active_df


def render():
    """Affiche une page d'attente moderne pour le volet classification."""
    
    # Container principal avec animation
    with st.container():
        # En-tête avec animation
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0; animation: fadeIn 1s ease-out;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                🤖 Volet 3 — Classification Supervisée
            </h1>
            <div class="badge" style="margin: 0 auto; display: inline-block;">
                Bientôt disponible
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte d'information principale
        st.markdown("""
        <div class="card" style="margin: 2rem 0; text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1.5rem;">🚧</div>
            <h2 style="color: var(--text); margin-bottom: 1rem;">Module en Construction</h2>
            <p style="color: var(--text-muted); font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;">
                L'apprentissage supervisé et la classification seront étudiés dans les prochains modules de votre formation.
            </p>
            <div style="background: linear-gradient(135deg, var(--surface2) 0%, var(--surface3) 100%); border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;">
                <h3 style="color: var(--accent); margin-bottom: 1rem;">📋 Ce que vous apprendrez :</h3>
                <ul style="text-align: left; color: var(--text); line-height: 1.8;">
                    <li>✂️ Partitionnement des données (Train/Test)</li>
                    <li>🤖 Algorithmes de classification (KNN, Decision Tree, Naive Bayes, SVM)</li>
                    <li>📊 Évaluation des performances (Matrice de confusion, métriques)</li>
                    <li>🔍 Comparaison des modèles</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progression temporelle
        st.markdown("""
        <div class="card" style="text-align: center;">
            <h3 style="color: var(--accent); margin-bottom: 1.5rem;">📅 Progression du cours</h3>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
                <div class="metric-card" style="flex: 1; min-width: 200px;">
                    <div class="metric-value">✅</div>
                    <div class="metric-label">Volet 1 - Prétraitement</div>
                </div>
                <div class="metric-card" style="flex: 1; min-width: 200px;">
                    <div class="metric-value">✅</div>
                    <div class="metric-label">Volet 2 - Clustering</div>
                </div>
                <div class="metric-card" style="flex: 1; min-width: 200px; opacity: 0.6;">
                    <div class="metric-value">🔒</div>
                    <div class="metric-label">Volet 3 - Classification</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Message interactif
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0; padding: 1.5rem; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%); border-radius: 12px; border: 1px solid var(--accent);">
            <h4 style="color: var(--accent); margin-bottom: 0.5rem;">💡 Conseil</h4>
            <p style="color: var(--text); margin: 0;">
                Continuez d'explorer les volets 1 et 2 pour maîtriser les bases avant d'aborder la classification !
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Animation de chargement subtile
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <div style="display: inline-block; padding: 1rem 2rem; background: var(--surface2); border-radius: 20px; border: 1px solid var(--border);">
                <span style="color: var(--text-muted); font-family: 'Space Mono', monospace;">
                    Patience... La machine apprend 🧠
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)