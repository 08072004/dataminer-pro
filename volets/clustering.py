# ============================================================
#  volets/clustering.py  —  Volet 2 : Clustering
#
#  Contient 4 onglets :
#    1. Courbe d'Elbow        (choix de k)
#    2. K-Means               (algorithme + visualisation PCA)
#    3. K-Medoids             (implémentation manuelle)
#    4. Évaluation            (silhouette + comparaison)
# ============================================================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances

from utils.plots import dark_fig, dark_fig_multi
from utils.session import get_active_df


# ─────────────────────────────────────────────────────────────
#  ALGORITHME K-MEDOIDS (implémentation manuelle)
# ─────────────────────────────────────────────────────────────
def kmedoids(X: np.ndarray, k: int, max_iter: int = 100, random_state: int = 42):
    """
    K-Medoids par la méthode PAM simplifiée.
    Utilise un point réel (médoïde) comme centre du cluster.

    Paramètres
    ----------
    X            : array (n_samples, n_features)
    k            : nombre de clusters
    max_iter     : nombre maximal d'itérations
    random_state : graine aléatoire

    Retourne
    --------
    labels       : array (n_samples,)  — étiquette de cluster pour chaque point
    medoid_idx   : array (k,)          — indices des médoïdes dans X
    inertia      : float               — somme des distances point→médoïde
    """
    np.random.seed(random_state)
    n = len(X)
    medoid_idx = np.random.choice(n, k, replace=False)
    dist_matrix = pairwise_distances(X)

    for _ in range(max_iter):
        # Assignation : chaque point → médoïde le plus proche
        labels = np.argmin(dist_matrix[:, medoid_idx], axis=1)

        # Mise à jour : chercher le meilleur médoïde dans chaque cluster
        new_medoids = []
        for j in range(k):
            cluster_pts = np.where(labels == j)[0]
            if len(cluster_pts) == 0:
                new_medoids.append(medoid_idx[j])   # cluster vide : on garde l'ancien
                continue
            sub_dist = dist_matrix[np.ix_(cluster_pts, cluster_pts)]
            best_local = np.argmin(sub_dist.sum(axis=1))
            new_medoids.append(cluster_pts[best_local])

        new_medoids = np.array(new_medoids)
        if np.array_equal(np.sort(medoid_idx), np.sort(new_medoids)):
            break   # convergence
        medoid_idx = new_medoids

    labels = np.argmin(dist_matrix[:, medoid_idx], axis=1)
    inertia = float(
        sum(dist_matrix[i, medoid_idx[labels[i]]] for i in range(n))
    )
    return labels, medoid_idx, inertia


# ─────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE DU VOLET
# ─────────────────────────────────────────────────────────────
def render():
    st.markdown("# 🔵 Volet 2 — Clustering")
    st.markdown("Découverte de structures cachées dans les données.")

    df_c = get_active_df()
    if df_c is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset dans le Volet 1.")
        st.stop()

    num_cols = df_c.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("Aucune colonne numérique disponible pour le clustering.")
        st.stop()

    st.markdown("**Sélectionner les features à utiliser :**")
    selected_cols = st.multiselect("Colonnes", num_cols, default=num_cols)
    if not selected_cols:
        st.warning("Sélectionnez au moins 2 colonnes.")
        st.stop()

    X = df_c[selected_cols].dropna().values

    tab_elbow, tab_kmeans, tab_kmedoids, tab_eval = st.tabs([
        "📈 Courbe d'Elbow",
        "🔵 K-Means",
        "🟠 K-Medoids",
        "📏 Évaluation",
    ])

    with tab_elbow:  _tab_elbow(X)
    with tab_kmeans: _tab_kmeans(X)
    with tab_kmedoids: _tab_kmedoids(X)
    with tab_eval:   _tab_evaluation(X)


# ─────────────────────────────────────────────────────────────
#  ONGLET 1 — COURBE D'ELBOW
# ─────────────────────────────────────────────────────────────
def _tab_elbow(X: np.ndarray):
    st.markdown("### 📈 Méthode du Coude (Elbow)")
    st.markdown(
        "On trace l'**inertie** (somme des distances²  points → centre) "
        "pour différentes valeurs de k. Le 'coude' indique le k optimal."
    )

    max_k = st.slider("Nombre maximum de clusters à tester", 2, 15, 10)
    auto_detect = st.checkbox("🤖 Détecter automatiquement le k optimal", value=True)

    if st.button("🔍 Calculer la courbe d'Elbow"):
        inertias = []
        k_range = range(2, max_k + 1)

        with st.spinner("Calcul en cours..."):
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)

        # Détection automatique du coude
        optimal_k = None
        if auto_detect and len(inertias) >= 3:
            optimal_k = _detect_elbow_point(list(k_range), inertias)
            st.session_state.auto_detected_k = optimal_k

        fig, ax = dark_fig(figsize=(9, 4))
        ax.plot(
            list(k_range), inertias,
            marker="o", color="#58a6ff", linewidth=2,
            markersize=7, markerfacecolor="#7ee787", markeredgecolor="#58a6ff",
        )
        ax.fill_between(list(k_range), inertias, alpha=0.08, color="#58a6ff")
        
        # Marquer le coude détecté
        if optimal_k:
            idx = list(k_range).index(optimal_k)
            ax.plot(optimal_k, inertias[idx], 'r*', markersize=15, 
                   markeredgecolor='white', markeredgewidth=2,
                   label=f'k optimal = {optimal_k}')
            ax.legend(framealpha=0.1, labelcolor="#e6edf3")
            
            # Ligne verticale pour le coude
            ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5, linewidth=2)
            
        ax.set_xlabel("Nombre de clusters k")
        ax.set_ylabel("Inertie")
        ax.set_title("Courbe d'Elbow — Inertie vs k", color="#e6edf3")
        ax.grid(axis="y", alpha=0.15, color="#8b949e")
        st.pyplot(fig)
        plt.close()

        # Afficher les résultats
        if optimal_k:
            st.success(f"🎯 **k optimal détecté automatiquement : {optimal_k}**")
            st.info("💡 Le coude a été détecté automatiquement. Cette valeur sera utilisée par défaut dans les autres onglets.")
        else:
            st.info("💡 Choisissez le **k** où la courbe commence à s'aplatir (le 'coude').")


def _detect_elbow_point(k_range, inertias):
    """
    Détecte automatiquement le point de coude (elbow) en utilisant 
    la méthode de la "deuxième dérivée" et de la "distance maximale".
    """
    if len(inertias) < 3:
        return None
    
    # Méthode 1: Deuxième dérivée (maximum de courbure)
    try:
        # Calculer les dérivées secondes discrètes
        first_deriv = np.diff(inertias)
        second_deriv = np.diff(first_deriv)
        
        # Normaliser
        if len(second_deriv) > 0:
            # Le coude est où la deuxième dérivée est maximale (en valeur absolue)
            elbow_idx = np.argmax(np.abs(second_deriv)) + 2  # +2 car diff réduit la taille
            k_optimal = k_range[elbow_idx]
            
            # Vérifier que c'est un coude raisonnable (pas le premier ou dernier point)
            if 2 < elbow_idx < len(k_range) - 1:
                return k_optimal
    except:
        pass
    
    # Méthode 2: Distance maximale à la ligne (méthode de la "distance")
    try:
        # Point de départ (k=2) et point final
        p1 = np.array([k_range[0], inertias[0]])
        p2 = np.array([k_range[-1], inertias[-1]])
        
        max_distance = 0
        best_k = k_range[1]  # défaut
        
        for i in range(1, len(k_range) - 1):
            point = np.array([k_range[i], inertias[i]])
            
            # Distance perpendiculaire du point à la ligne p1-p2
            distance = np.abs((p2[1] - p1[1]) * point[0] - 
                           (p2[0] - p1[0]) * point[1] + 
                           p2[0] * p1[1] - p2[1] * p1[0]) / \
                         np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
            
            if distance > max_distance:
                max_distance = distance
                best_k = k_range[i]
        
        return best_k
    except:
        pass
    
    # Méthode 3: Simple heuristique - plus grand changement relatif
    try:
        max_change = 0
        best_k = k_range[1]
        
        for i in range(1, len(inertias)):
            change = abs(inertias[i-1] - inertias[i]) / inertias[i-1]
            if change > max_change and i < len(inertias) - 1:
                max_change = change
                best_k = k_range[i]
        
        return best_k
    except:
        pass
    
    # Fallback: retourner une valeur raisonnable
    return min(5, len(k_range) // 2 + 2)
#  ONGLET 2 — K-MEANS
# ─────────────────────────────────────────────────────────────
def _tab_kmeans(X: np.ndarray):
    st.markdown("### 🔵 K-Means Clustering")
    st.markdown(
        "K-Means utilise la **moyenne** des points comme centre de cluster. "
        "Objectif : minimiser l'inertie intra-cluster."
    )

    # Utiliser le k détecté automatiquement si disponible
    default_k = st.session_state.get('auto_detected_k', 3)
    k = int(st.number_input("Nombre de clusters k", min_value=2, max_value=20, value=default_k, key="km_k"))
    
    # Afficher si k vient de la détection automatique
    if 'auto_detected_k' in st.session_state and k == st.session_state.auto_detected_k:
        st.info(f"🤖 Utilisation du k optimal détecté automatiquement : {k}")

    if st.button("▶️ Lancer K-Means"):
        with st.spinner("Clustering K-Means en cours..."):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            st.session_state.cluster_labels = labels

        st.success(f"✅ K-Means terminé. Inertie : **{km.inertia_:.2f}**")

        col1, col2 = st.columns(2)

        # Distribution des clusters
        with col1:
            fig, ax = dark_fig(figsize=(6, 4))
            unique, counts = np.unique(labels, return_counts=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
            bars = ax.bar(
                [f"C{u}" for u in unique], counts,
                color=colors, edgecolor="#30363d", linewidth=0.5,
            )
            for bar, cnt in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom",
                    color="#e6edf3", fontsize=10,
                )
            ax.set_title("Distribution des clusters", color="#e6edf3")
            ax.set_ylabel("Nombre de points")
            st.pyplot(fig)
            plt.close()

        # Projection PCA 2D
        with col2:
            _plot_pca_2d(X, labels, km.cluster_centers_, centers_are_indices=False,
                         title="Projection PCA 2D — K-Means",
                         marker_style="*", marker_color="white", edge_color="#58a6ff")


# ─────────────────────────────────────────────────────────────
#  ONGLET 3 — K-MEDOIDS
# ─────────────────────────────────────────────────────────────
def _tab_kmedoids(X: np.ndarray):
    st.markdown("### 🟠 K-Medoids Clustering")
    st.info(
        "K-Medoids utilise un **point réel** (médoïde) comme centre — "
        "plus robuste aux valeurs aberrantes que K-Means."
    )

    # Utiliser le k détecté automatiquement si disponible
    default_k = st.session_state.get('auto_detected_k', 3)
    k = int(st.number_input("Nombre de clusters k", min_value=2, max_value=20, value=default_k, key="kmed_k"))
    
    # Afficher si k vient de la détection automatique
    if 'auto_detected_k' in st.session_state and k == st.session_state.auto_detected_k:
        st.info(f"🤖 Utilisation du k optimal détecté automatiquement : {k}")

    if st.button("▶️ Lancer K-Medoids"):
        with st.spinner("K-Medoids en cours..."):
            labels, medoid_idx, inertia = kmedoids(X, k)

        st.success(f"✅ K-Medoids terminé. Inertie (somme distances) : **{inertia:.2f}**")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = dark_fig(figsize=(6, 4))
            unique, counts = np.unique(labels, return_counts=True)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique)))
            bars = ax.bar(
                [f"C{u}" for u in unique], counts,
                color=colors, edgecolor="#30363d",
            )
            for bar, cnt in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(cnt), ha="center", va="bottom", color="#e6edf3",
                )
            ax.set_title("Distribution des clusters (K-Medoids)", color="#e6edf3")
            st.pyplot(fig)
            plt.close()

        with col2:
            _plot_pca_2d(X, labels, medoid_idx, centers_are_indices=True,
                         title="Projection PCA 2D — K-Medoids",
                         marker_style="D", marker_color="white", edge_color="#f78166",
                         cmap="Set1")


# ─────────────────────────────────────────────────────────────
#  ONGLET 4 — ÉVALUATION
# ─────────────────────────────────────────────────────────────
def _tab_evaluation(X: np.ndarray):
    st.markdown("### 📏 Évaluation — Silhouette & Comparaison")
    st.markdown("""
    **Coefficient de Silhouette** mesure :
    - la **compacité** : distance entre un point et les autres points de son cluster
    - la **séparation** : distance entre un point et les points du cluster le plus proche

    → **Valeur proche de 1** = bon clustering
    """)

    k_max = st.slider("Comparer de k=2 à k=", 3, 15, 8)

    if st.button("📊 Comparer K-Means vs K-Medoids"):
        k_range = range(2, k_max + 1)
        sil_km, sil_kmed = [], []
        iner_km, iner_kmed = [], []

        with st.spinner("Comparaison en cours (patience pour K-Medoids)..."):
            for k in k_range:
                # K-Means
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lkm = km.fit_predict(X)
                sil_km.append(silhouette_score(X, lkm))
                iner_km.append(km.inertia_)

                # K-Medoids
                lkmed, med_idx, _ = kmedoids(X, k)
                sil_kmed.append(silhouette_score(X, lkmed))
                dm = pairwise_distances(X)
                iner_kmed.append(
                    float(sum(dm[i, med_idx[lkmed[i]]] for i in range(len(X))))
                )

        fig, axes = dark_fig_multi(1, 2, figsize=(14, 5))

        # Silhouette
        axes[0].plot(list(k_range), sil_km,  marker="o", label="K-Means",
                     color="#58a6ff", linewidth=2)
        axes[0].plot(list(k_range), sil_kmed, marker="s", label="K-Medoids",
                     color="#f78166", linewidth=2, linestyle="--")
        axes[0].set_title("Score de Silhouette", color="#e6edf3")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Silhouette")
        axes[0].legend(framealpha=0.1, labelcolor="#e6edf3")
        axes[0].grid(alpha=0.1, color="#8b949e")

        # Inertie (Elbow)
        axes[1].plot(list(k_range), iner_km,  marker="o", label="K-Means",
                     color="#58a6ff", linewidth=2)
        axes[1].plot(list(k_range), iner_kmed, marker="s", label="K-Medoids",
                     color="#f78166", linewidth=2, linestyle="--")
        axes[1].set_title("Inertie (Elbow)", color="#e6edf3")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Inertie")
        axes[1].legend(framealpha=0.1, labelcolor="#e6edf3")
        axes[1].grid(alpha=0.1, color="#8b949e")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Résumé
        best_k_km   = list(k_range)[int(np.argmax(sil_km))]
        best_k_kmed = list(k_range)[int(np.argmax(sil_kmed))]
        c1, c2 = st.columns(2)
        c1.metric("Meilleur k — K-Means",   best_k_km,   f"Silhouette = {max(sil_km):.3f}")
        c2.metric("Meilleur k — K-Medoids", best_k_kmed, f"Silhouette = {max(sil_kmed):.3f}")

        st.markdown("""
        | Algorithme | Avantages | Inconvénients |
        |---|---|---|
        | **K-Means** | Rapide, simple | Sensible aux valeurs aberrantes |
        | **K-Medoids** | Robuste au bruit | Plus lent (O(n²)) |
        """)


# ─────────────────────────────────────────────────────────────
#  HELPER INTERNE — Projection PCA 2D
# ─────────────────────────────────────────────────────────────
def _plot_pca_2d(
    X, labels, centers, centers_are_indices: bool,
    title: str, marker_style: str, marker_color: str, edge_color: str,
    cmap: str = "Set2",
):
    """
    Projette X en 2D via PCA et affiche les clusters + centres.

    centers_are_indices=True  → centers est un tableau d'indices dans X
    centers_are_indices=False → centers est un tableau de coordonnées
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = dark_fig(figsize=(6, 4))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap=cmap, alpha=0.7, s=25, edgecolors="none",
    )

    if centers_are_indices:
        c_2d = X_2d[centers]
    else:
        c_2d = pca.transform(centers)

    ax.scatter(
        c_2d[:, 0], c_2d[:, 1],
        marker=marker_style, s=200 if marker_style == "*" else 120,
        c=marker_color, edgecolors=edge_color, linewidth=1.5, zorder=5,
        label="Centres",
    )
    ax.legend(framealpha=0.1, labelcolor="#e6edf3")
    ax.set_title(title, color="#e6edf3")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)
    plt.close()