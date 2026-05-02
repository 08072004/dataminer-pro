# ============================================================
#  volets/clustering.py  —  Volet 2 : Clustering
#
#  Contient 7 onglets :
#    1. Courbe d'Elbow        (choix de k)
#    2. K-Means               (algorithme + visualisation PCA)
#    3. K-Medoids             (implémentation manuelle PAM)
#    4. DBSCAN                (clustering par densité)
#    5. AGNES                 (hiérarchique agglomératif)
#    6. DIANA                 (hiérarchique divisif, implémentation manuelle)
#    7. Évaluation            (silhouette + comparaison multi-algorithmes)
# ============================================================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from utils.plots import dark_fig, dark_fig_multi
from utils.session import get_active_df


# ─────────────────────────────────────────────────────────────
#  ALGORITHME K-MEDOIDS (implémentation manuelle — PAM)
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
        labels = np.argmin(dist_matrix[:, medoid_idx], axis=1)
        new_medoids = []
        for j in range(k):
            cluster_pts = np.where(labels == j)[0]
            if len(cluster_pts) == 0:
                new_medoids.append(medoid_idx[j])
                continue
            sub_dist = dist_matrix[np.ix_(cluster_pts, cluster_pts)]
            best_local = np.argmin(sub_dist.sum(axis=1))
            new_medoids.append(cluster_pts[best_local])

        new_medoids = np.array(new_medoids)
        if np.array_equal(np.sort(medoid_idx), np.sort(new_medoids)):
            break
        medoid_idx = new_medoids

    labels = np.argmin(dist_matrix[:, medoid_idx], axis=1)
    inertia = float(
        sum(dist_matrix[i, medoid_idx[labels[i]]] for i in range(n))
    )
    return labels, medoid_idx, inertia


# ─────────────────────────────────────────────────────────────
#  ALGORITHME DIANA (implémentation manuelle — divisif)
# ─────────────────────────────────────────────────────────────
def diana(X: np.ndarray, n_clusters: int = 2):
    """
    DIANA (DIvisive ANAlysis) — clustering hiérarchique divisif.

    Algorithme :
      1. Départ : tous les points dans un seul cluster
      2. À chaque étape, on sélectionne le cluster de plus grande
         "dissimilarité moyenne" (diamètre) et on le divise en deux
         en isolant le point le plus éloigné de ses voisins (splinter)
      3. On répète jusqu'à obtenir n_clusters clusters

    Complexité : O(n² · k) en pratique sur les étapes de split.
    Pour les gros datasets, un sous-échantillonnage est recommandé.

    Paramètres
    ----------
    X          : array (n_samples, n_features)
    n_clusters : nombre final de clusters souhaité

    Retourne
    --------
    labels     : array (n_samples,) — étiquette de cluster (0..n_clusters-1)
    history    : liste des étapes  [{split_cluster, new_clusters, diameters}]
    """
    n = X.shape[0]
    dist_matrix = squareform(pdist(X, metric="euclidean"))

    # État initial : tous les points dans le cluster 0
    clusters = [list(range(n))]
    history = []

    while len(clusters) < n_clusters:
        # Choisir le cluster à diviser : celui avec le plus grand diamètre
        # (distance maximale entre deux points du cluster)
        best_cluster_idx = -1
        best_diameter = -1.0

        for ci, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue
            sub = dist_matrix[np.ix_(cluster, cluster)]
            diameter = sub.max()
            if diameter > best_diameter:
                best_diameter = diameter
                best_cluster_idx = ci

        if best_cluster_idx == -1:
            break  # Tous les clusters sont des singletons

        cluster_to_split = clusters[best_cluster_idx]

        # ---- Étape de split DIANA ----
        # 1. Calculer pour chaque point la dissimilarité moyenne
        #    avec tous les autres points du cluster
        sub_dist = dist_matrix[np.ix_(cluster_to_split, cluster_to_split)]
        avg_diss = sub_dist.mean(axis=1)

        # 2. Le "splinter" est le point avec la plus grande dissimilarité moyenne
        splinter_local = int(np.argmax(avg_diss))
        splinter_global = cluster_to_split[splinter_local]

        # 3. Construire le nouveau groupe à partir du splinter
        new_group = [splinter_global]
        main_group = [p for p in cluster_to_split if p != splinter_global]

        # 4. Itérer : à chaque tour, déplacer un point de main_group vers new_group
        #    si sa distance moyenne au new_group < distance moyenne au main_group
        changed = True
        while changed:
            changed = False
            move_candidate = None
            best_diff = 0.0

            for p in main_group:
                d_new = np.mean(dist_matrix[p][new_group])
                d_main = np.mean(dist_matrix[p][main_group]) if len(main_group) > 1 else 0.0
                diff = d_main - d_new
                if diff > best_diff:
                    best_diff = diff
                    move_candidate = p

            if move_candidate is not None:
                main_group.remove(move_candidate)
                new_group.append(move_candidate)
                changed = True

        # Mettre à jour la liste des clusters
        clusters.pop(best_cluster_idx)
        clusters.append(main_group)
        clusters.append(new_group)

        history.append({
            "split": cluster_to_split,
            "main": main_group,
            "new": new_group,
            "diameter": best_diameter,
        })

    # Construire le tableau de labels
    labels = np.zeros(n, dtype=int)
    for ci, cluster in enumerate(clusters):
        for p in cluster:
            labels[p] = ci

    return labels, history


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

    (
        tab_elbow,
        tab_kmeans,
        tab_kmedoids,
        tab_dbscan,
        tab_agnes,
        tab_diana,
        tab_eval,
    ) = st.tabs([
        "📈 Courbe d'Elbow",
        "🔵 K-Means",
        "🟠 K-Medoids",
        "🟣 DBSCAN",
        "🌿 AGNES",
        "🔴 DIANA",
        "📏 Évaluation",
    ])

    with tab_elbow:    _tab_elbow(X)
    with tab_kmeans:   _tab_kmeans(X)
    with tab_kmedoids: _tab_kmedoids(X)
    with tab_dbscan:   _tab_dbscan(X)
    with tab_agnes:    _tab_agnes(X)
    with tab_diana:    _tab_diana(X)
    with tab_eval:     _tab_evaluation(X)


# ─────────────────────────────────────────────────────────────
#  ONGLET 1 — COURBE D'ELBOW
# ─────────────────────────────────────────────────────────────
def _tab_elbow(X: np.ndarray):
    st.markdown("### 📈 Méthode du Coude (Elbow)")
    st.markdown(
        "On trace l'**inertie** (somme des distances² points → centre) "
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

        if optimal_k:
            idx = list(k_range).index(optimal_k)
            ax.plot(optimal_k, inertias[idx], 'r*', markersize=15,
                    markeredgecolor='white', markeredgewidth=2,
                    label=f'k optimal = {optimal_k}')
            ax.legend(framealpha=0.1, labelcolor="#e6edf3")
            ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5, linewidth=2)

        ax.set_xlabel("Nombre de clusters k")
        ax.set_ylabel("Inertie")
        ax.set_title("Courbe d'Elbow — Inertie vs k", color="#e6edf3")
        ax.grid(axis="y", alpha=0.15, color="#8b949e")
        st.pyplot(fig)
        plt.close()

        if optimal_k:
            st.success(f"🎯 **k optimal détecté automatiquement : {optimal_k}**")
            st.info("💡 Cette valeur sera utilisée par défaut dans les onglets K-Means, K-Medoids, AGNES et DIANA.")
        else:
            st.info("💡 Choisissez le **k** où la courbe commence à s'aplatir.")


def _detect_elbow_point(k_range, inertias):
    """Détecte le coude via deuxième dérivée, puis distance max, puis heuristique."""
    if len(inertias) < 3:
        return None

    try:
        first_deriv = np.diff(inertias)
        second_deriv = np.diff(first_deriv)
        if len(second_deriv) > 0:
            elbow_idx = int(np.argmax(np.abs(second_deriv))) + 2
            if 2 < elbow_idx < len(k_range) - 1:
                return k_range[elbow_idx]
    except Exception:
        pass

    try:
        p1 = np.array([k_range[0], inertias[0]])
        p2 = np.array([k_range[-1], inertias[-1]])
        max_distance = 0
        best_k = k_range[1]
        for i in range(1, len(k_range) - 1):
            point = np.array([k_range[i], inertias[i]])
            distance = np.abs(
                (p2[1] - p1[1]) * point[0]
                - (p2[0] - p1[0]) * point[1]
                + p2[0] * p1[1]
                - p2[1] * p1[0]
            ) / np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
            if distance > max_distance:
                max_distance = distance
                best_k = k_range[i]
        return best_k
    except Exception:
        pass

    try:
        max_change, best_k = 0, k_range[1]
        for i in range(1, len(inertias)):
            change = abs(inertias[i - 1] - inertias[i]) / inertias[i - 1]
            if change > max_change and i < len(inertias) - 1:
                max_change = change
                best_k = k_range[i]
        return best_k
    except Exception:
        pass

    return min(5, len(k_range) // 2 + 2)


# ─────────────────────────────────────────────────────────────
#  ONGLET 2 — K-MEANS
# ─────────────────────────────────────────────────────────────
def _tab_kmeans(X: np.ndarray):
    st.markdown("### 🔵 K-Means Clustering")
    st.markdown(
        "K-Means utilise la **moyenne** des points comme centre de cluster. "
        "Objectif : minimiser l'inertie intra-cluster."
    )

    default_k = st.session_state.get("auto_detected_k", 3)
    k = int(st.number_input("Nombre de clusters k", min_value=2, max_value=20,
                            value=default_k, key="km_k"))
    if "auto_detected_k" in st.session_state and k == st.session_state.auto_detected_k:
        st.info(f"🤖 k optimal détecté automatiquement : {k}")

    if st.button("▶️ Lancer K-Means"):
        with st.spinner("Clustering K-Means en cours..."):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            st.session_state.cluster_labels = labels

        st.success(f"✅ K-Means terminé")

        # ---- Métriques côte à côte ----
        col_sil, col_inert = st.columns(2)
        with col_sil:
            if len(set(labels)) > 1:
                sil = silhouette_score(X, labels)
                st.metric("📊 Score de Silhouette", f"{sil:.4f}")
            else:
                st.warning("Silhouette non calculable")
        
        with col_inert:
            st.metric("⚡ Inertie", f"{km.inertia_:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            _plot_distribution(labels, "Distribution des clusters (K-Means)",
                               plt.cm.Set2)
        with col2:
            _plot_pca_2d(X, labels, km.cluster_centers_,
                         centers_are_indices=False,
                         title="Projection PCA 2D — K-Means",
                         marker_style="*", marker_color="white",
                         edge_color="#58a6ff")


# ─────────────────────────────────────────────────────────────
#  ONGLET 3 — K-MEDOIDS
# ─────────────────────────────────────────────────────────────
def _tab_kmedoids(X: np.ndarray):
    st.markdown("### 🟠 K-Medoids Clustering")
    st.info(
        "K-Medoids utilise un **point réel** (médoïde) comme centre — "
        "plus robuste aux valeurs aberrantes que K-Means."
    )

    default_k = st.session_state.get("auto_detected_k", 3)
    k = int(st.number_input("Nombre de clusters k", min_value=2, max_value=20,
                            value=default_k, key="kmed_k"))
    if "auto_detected_k" in st.session_state and k == st.session_state.auto_detected_k:
        st.info(f"🤖 k optimal détecté automatiquement : {k}")

    if st.button("▶️ Lancer K-Medoids"):
        with st.spinner("K-Medoids en cours..."):
            labels, medoid_idx, inertia = kmedoids(X, k)

        st.success(f"✅ K-Medoids terminé")

        # ---- Métriques côte à côte ----
        col_sil, col_inert = st.columns(2)
        with col_sil:
            if len(set(labels)) > 1:
                sil = silhouette_score(X, labels)
                st.metric("📊 Score de Silhouette", f"{sil:.4f}")
            else:
                st.warning("Silhouette non calculable")
        
        with col_inert:
            st.metric("⚡ Inertie", f"{inertia:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            _plot_distribution(labels, "Distribution des clusters (K-Medoids)",
                               plt.cm.Set1)
        with col2:
            _plot_pca_2d(X, labels, medoid_idx,
                         centers_are_indices=True,
                         title="Projection PCA 2D — K-Medoids",
                         marker_style="D", marker_color="white",
                         edge_color="#f78166", cmap="Set1")


# ─────────────────────────────────────────────────────────────
#  ONGLET 4 — DBSCAN
# ─────────────────────────────────────────────────────────────
def _tab_dbscan(X: np.ndarray):
    st.markdown("### 🟣 DBSCAN — Density-Based Spatial Clustering")
    st.markdown("""
    DBSCAN regroupe des points **denses** sans nécessiter de spécifier k à l'avance.
    - **ε (eps)** : rayon de voisinage d'un point
    - **min_samples** : nombre minimum de points dans ce rayon pour être un "point cœur"
    - Les points non assignés sont étiquetés **bruit (−1)**
    """)

    # ---- Aide au choix de eps via courbe k-distance ----
    with st.expander("🔎 Aide au choix de ε — Courbe k-distance"):
        st.markdown(
            "Tracez la **distance au k-ème plus proche voisin** triée par ordre croissant. "
            "Le 'coude' suggère une bonne valeur de ε."
        )
        min_s_helper = st.slider("min_samples (pour la courbe)", 2, 20, 5,
                                 key="dbscan_helper_ms")
        if st.button("📉 Tracer la courbe k-distance"):
            X_scaled = StandardScaler().fit_transform(X)
            nbrs = NearestNeighbors(n_neighbors=min_s_helper).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            k_dist = np.sort(distances[:, -1])[::-1]

            fig, ax = dark_fig(figsize=(9, 3))
            ax.plot(k_dist, color="#a371f7", linewidth=1.5)
            ax.set_xlabel("Points (triés)")
            ax.set_ylabel(f"Distance au {min_s_helper}-ème voisin")
            ax.set_title("Courbe k-distance — chercher le coude", color="#e6edf3")
            ax.grid(alpha=0.12, color="#8b949e")
            st.pyplot(fig)
            plt.close()
            st.info("💡 La valeur ε suggérée correspond au coude de la courbe ci-dessus.")

    st.markdown("---")

    # ---- Paramètres DBSCAN ----
    col_a, col_b = st.columns(2)
    with col_a:
        eps = st.number_input("ε (eps) — rayon de voisinage", min_value=0.01,
                              max_value=10.0, value=0.5, step=0.05,
                              key="dbscan_eps")
        scale_data = st.checkbox("Normaliser les données (StandardScaler)", value=True,
                                 key="dbscan_scale")
    with col_b:
        min_samples = st.slider("min_samples", 2, 30, 5, key="dbscan_ms")
        metric = st.selectbox("Métrique de distance",
                              ["euclidean", "manhattan", "cosine"],
                              key="dbscan_metric")

    if st.button("▶️ Lancer DBSCAN"):
        X_fit = StandardScaler().fit_transform(X) if scale_data else X

        with st.spinner("DBSCAN en cours..."):
            db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = db.fit_predict(X_fit)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        if n_clusters == 0:
            st.error(
                "❌ Aucun cluster trouvé. Augmentez ε ou diminuez min_samples."
            )
            return

        st.success(
            f"✅ DBSCAN terminé — **{n_clusters} cluster(s)** détecté(s), "
            f"**{n_noise} point(s) bruit** ({n_noise / len(labels) * 100:.1f}%)"
        )

        # ---- Métriques côte à côte ----
        col_sil, col_inert = st.columns(2)
        with col_sil:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X_fit[mask], labels[mask])
                st.metric("📊 Score de Silhouette (hors bruit)", f"{sil:.4f}")
            else:
                st.warning("Silhouette non calculable")
        
        with col_inert:
            # DBSCAN n'a pas d'inertie traditionnelle, on calcule une approximation
            try:
                from sklearn.metrics import pairwise_distances
                # Calculer l'inertie intra-cluster approximative (sans le bruit)
                inertia_approx = 0
                mask = labels != -1  # Exclure le bruit
                for cluster_id in np.unique(labels[mask]):
                    if cluster_id != -1:  # Ignorer le bruit
                        cluster_points = X_fit[mask & (labels == cluster_id)]
                        if len(cluster_points) > 1:
                            # Distance intra-cluster
                            intra_distances = pairwise_distances(cluster_points)
                            inertia_approx += np.sum(intra_distances) / 2
                st.metric("⚡ Inertie (clusters uniquement)", f"{inertia_approx:.2f}")
            except:
                st.info("Inertie non calculable pour DBSCAN")

        col1, col2 = st.columns(2)

        with col1:
            # Distribution clusters + bruit
            fig, ax = dark_fig(figsize=(6, 4))
            unique_labels = sorted(set(labels))
            palette = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
            color_map = {}
            ci = 0
            for lbl in unique_labels:
                if lbl == -1:
                    color_map[lbl] = (0.5, 0.5, 0.5, 1.0)  # gris pour bruit
                else:
                    color_map[lbl] = palette[ci % len(palette)]
                    ci += 1
            bar_colors = [color_map[lbl] for lbl in unique_labels]
            counts = [int(np.sum(labels == lbl)) for lbl in unique_labels]
            bar_labels = [f"Bruit" if lbl == -1 else f"C{lbl}"
                          for lbl in unique_labels]
            bars = ax.bar(bar_labels, counts, color=bar_colors,
                          edgecolor="#30363d", linewidth=0.5)
            for bar, cnt in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom",
                    color="#e6edf3", fontsize=9,
                )
            ax.set_title("Distribution des clusters (DBSCAN)", color="#e6edf3")
            ax.set_ylabel("Nombre de points")
            st.pyplot(fig)
            plt.close()

        with col2:
            # PCA 2D avec bruit en gris
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_fit)

            fig, ax = dark_fig(figsize=(6, 4))
            for lbl in sorted(set(labels)):
                mask_lbl = labels == lbl
                if lbl == -1:
                    ax.scatter(
                        X_2d[mask_lbl, 0], X_2d[mask_lbl, 1],
                        c="gray", alpha=0.3, s=15, label="Bruit",
                        edgecolors="none",
                    )
                else:
                    ax.scatter(
                        X_2d[mask_lbl, 0], X_2d[mask_lbl, 1],
                        c=[color_map[lbl]], alpha=0.75, s=25,
                        label=f"C{lbl}", edgecolors="none",
                    )
            ax.legend(framealpha=0.1, labelcolor="#e6edf3",
                      fontsize=8, markerscale=1.5)
            ax.set_title("Projection PCA 2D — DBSCAN", color="#e6edf3")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            st.pyplot(fig)
            plt.close()

        # Résumé paramètres
        st.markdown("#### Résumé")
        st.markdown(f"""
        | Paramètre | Valeur |
        |---|---|
        | ε (eps) | {eps} |
        | min_samples | {min_samples} |
        | Métrique | {metric} |
        | Normalisation | {"Oui" if scale_data else "Non"} |
        | Clusters trouvés | {n_clusters} |
        | Points bruit | {n_noise} ({n_noise / len(labels) * 100:.1f}%) |
        """)

        st.markdown("""
        **Interprétation :**
        - Augmenter **ε** → clusters plus grands, moins de bruit
        - Diminuer **ε** → clusters plus denses, plus de bruit
        - Augmenter **min_samples** → moins de clusters, points cœur plus stricts
        """)

        # Sauvegarder pour l'évaluation
        st.session_state["dbscan_labels"] = labels
        st.session_state["dbscan_params"] = {"eps": eps, "min_samples": min_samples}


# ─────────────────────────────────────────────────────────────
#  ONGLET 5 — AGNES
# ─────────────────────────────────────────────────────────────
def _tab_agnes(X: np.ndarray):
    st.markdown("### 🌿 AGNES — Agglomerative Nesting")
    st.markdown("""
    **AGNES** est un algorithme de clustering **hiérarchique agglomératif** :
    - Part de **n singletons** (chaque point = son propre cluster)
    - **Fusionne** itérativement les deux clusters les plus proches
    - Produit un **dendrogramme** (arbre de fusion)

    La stratégie de liaison (*linkage*) définit la distance entre clusters.
    """)
    # ---- Paramètres ----
    col_a, col_b = st.columns(2)
    with col_a:
        use_k = st.checkbox("🔢 Spécifier un nombre de clusters k", value=False, 
                           help="Si décoché, AGNES s'exécutera sans contrainte sur k")
        
        n_clusters = None
        if use_k:
            default_k = st.session_state.get("auto_detected_k", 3)
            n_clusters = int(st.number_input(
                "Nombre de clusters k", min_value=2, max_value=20,
                value=default_k, key="agnes_k",
            ))
            if "auto_detected_k" in st.session_state and n_clusters == st.session_state.auto_detected_k:
                st.info(f"🤖 k optimal détecté automatiquement : {n_clusters}")
        else:
            st.info("🌳 AGNES s'exécutera sans contrainte sur k - le dendrogramme déterminera la structure")

    with col_b:
        linkage_method = st.selectbox(
            "Méthode de liaison (linkage)",
            ["ward", "complete", "average", "single"],
            key="agnes_link",
            help=(
                "**ward** : minimise la variance intra-cluster (recommandé)\n"
                "**complete** : distance max entre points de deux clusters\n"
                "**average** : distance moyenne\n"
                "**single** : distance min (sensible au bruit)"
            ),
        )

    max_dendro = st.slider(
        "Nombre de feuilles affichées dans le dendrogramme",
        10, min(200, len(X)), min(50, len(X)), key="agnes_dendro",
    )

    if st.button("▶️ Lancer AGNES"):
        with st.spinner("Clustering AGNES en cours..."):
            # Linkage scipy pour le dendrogramme (toujours calculé)
            Z = linkage(X, method=linkage_method)
            
            if use_k and n_clusters:
                # Mode avec k spécifié : utiliser sklearn
                agg = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage=linkage_method
                )
                labels = agg.fit_predict(X)
                st.success(f"✅ AGNES terminé — {n_clusters} clusters avec linkage '{linkage_method}'")
            else:
                # Mode sans k : utiliser scipy pour déterminer les clusters depuis le dendrogramme
                # Distance de coupure automatique (basée sur la plus grande fusion)
                max_distance = Z[-1, 2]  # Distance de la dernière fusion
                cutoff = max_distance * 0.7  # 70% de la distance max comme heuristique
                
                labels = fcluster(Z, t=cutoff, criterion='distance')
                n_clusters_found = len(set(labels))
                st.success(f"✅ AGNES terminé — {n_clusters_found} clusters détectés automatiquement avec linkage '{linkage_method}'")
                st.info(f"🎯 Distance de coupure utilisée : {cutoff:.3f}")

        # ---- Métriques côte à côte ----
        col_sil, col_inert = st.columns(2)
        with col_sil:
            if len(set(labels)) > 1:
                sil = silhouette_score(X, labels)
                st.metric("📊 Score de Silhouette", f"{sil:.4f}")
            else:
                st.warning("Silhouette non calculable")
        
        with col_inert:
            # AGNES n'a pas d'inertie directe, on calcule une approximation
            try:
                from sklearn.metrics import pairwise_distances
                # Calculer l'inertie intra-cluster approximative
                inertia_approx = 0
                for cluster_id in np.unique(labels):
                    cluster_points = X[labels == cluster_id]
                    if len(cluster_points) > 1:
                        # Distance intra-cluster
                        intra_distances = pairwise_distances(cluster_points)
                        inertia_approx += np.sum(intra_distances) / 2
                st.metric("⚡ Inertie (approx.)", f"{inertia_approx:.2f}")
            except:
                st.info("Inertie non calculable pour AGNES")

        col1, col2 = st.columns(2)

        with col1:
            _plot_distribution(labels, f"Distribution des clusters (AGNES / {linkage_method})",
                               plt.cm.tab10)

        with col2:
            _plot_pca_2d(
                X, labels, centers=None,
                centers_are_indices=False,
                title=f"Projection PCA 2D — AGNES ({linkage_method})",
                marker_style=None, marker_color=None, edge_color=None,
                cmap="tab10",
            )

        # ---- Dendrogramme ----
        st.markdown("#### 🌲 Dendrogramme")
        fig, ax = dark_fig(figsize=(14, 5))
        
        if use_k and n_clusters:
            # Mode avec k spécifié
            dendrogram(
                Z,
                truncate_mode="lastp",
                p=max_dendro,
                leaf_rotation=90.0,
                leaf_font_size=8.0,
                show_contracted=True,
                ax=ax,
                color_threshold=Z[-(n_clusters - 1), 2],
                above_threshold_color="#8b949e",
            )
            ax.set_title(
                f"Dendrogramme AGNES — linkage={linkage_method}, k={n_clusters}",
                color="#e6edf3",
            )
            ax.axhline(
                y=Z[-(n_clusters - 1), 2],
                color="red", linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"Coupe à k={n_clusters}",
            )
        else:
            # Mode sans k : pas de ligne de coupe
            dendrogram(
                Z,
                truncate_mode="lastp",
                p=max_dendro,
                leaf_rotation=90.0,
                leaf_font_size=8.0,
                show_contracted=True,
                ax=ax,
                above_threshold_color="#8b949e",
            )
            ax.set_title(
                f"Dendrogramme AGNES — linkage={linkage_method} (sans contrainte k)",
                color="#e6edf3",
            )
            ax.axhline(
                y=cutoff,
                color="orange", linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"Coupe automatique (distance={cutoff:.3f})",
            )
        
        ax.set_xlabel("Échantillons (ou taille du cluster contracté)")
        ax.set_ylabel("Distance de fusion")
        ax.legend(framealpha=0.1, labelcolor="#e6edf3")
        st.pyplot(fig)
        plt.close()

        # ---- Tableau comparatif linkages ----
        st.markdown("""
        | Linkage | Caractéristique | Sensibilité bruit |
        |---|---|---|
        | **ward** | Minimise la variance → clusters compacts | Faible |
        | **complete** | Clusters sphériques bien séparés | Moyenne |
        | **average** | Compromis entre ward et single | Faible–Moyenne |
        | **single** | Chaining effect, clusters allongés | Élevée |
        """)


# ─────────────────────────────────────────────────────────────
#  ONGLET 6 — DIANA
# ─────────────────────────────────────────────────────────────
def _tab_diana(X: np.ndarray):
    st.markdown("### 🔴 DIANA — DIvisive ANAlysis")
    st.markdown("""
    **DIANA** est l'algorithme hiérarchique **divisif** (top-down) :
    - Part d'**un seul cluster** contenant tous les points
    - **Divise** itérativement le cluster le plus hétérogène
    - À chaque étape, le point le plus dissemblable (le "splinter") est extrait
      pour former un nouveau groupe

    ⚠️ La complexité est **O(n²)** par étape — pour les gros datasets, 
    un sous-échantillonnage automatique est appliqué.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        default_k = st.session_state.get("auto_detected_k", 3)
        n_clusters = int(st.number_input(
            "Nombre de clusters k", min_value=2, max_value=15,
            value=default_k, key="diana_k",
        ))
        if "auto_detected_k" in st.session_state and n_clusters == st.session_state.auto_detected_k:
            st.info(f"🤖 k optimal détecté automatiquement : {n_clusters}")

    with col_b:
        max_points = st.slider(
            "Limite de points (sous-échantillonnage si nécessaire)",
            100, 2000, 500, step=100, key="diana_max_pts",
            help="DIANA est O(n²). Au-delà de cette limite, un échantillon aléatoire est utilisé.",
        )

    if st.button("▶️ Lancer DIANA"):
        # Sous-échantillonnage si nécessaire
        n = len(X)
        if n > max_points:
            st.warning(
                f"⚠️ Dataset trop grand ({n} points). "
                f"Sous-échantillonnage à {max_points} points pour DIANA."
            )
            idx_sample = np.random.choice(n, max_points, replace=False)
            X_use = X[idx_sample]
        else:
            X_use = X
            idx_sample = None

        with st.spinner(f"DIANA en cours sur {len(X_use)} points..."):
            labels_use, history = diana(X_use, n_clusters=n_clusters)

        st.success(f"✅ DIANA terminé — {n_clusters} clusters")

        # ---- Métriques côte à côte ----
        col_sil, col_inert = st.columns(2)
        with col_sil:
            if len(set(labels_use)) > 1:
                sil = silhouette_score(X_use, labels_use)
                st.metric("📊 Score de Silhouette", f"{sil:.4f}")
            else:
                st.warning("Silhouette non calculable")
        
        with col_inert:
            # DIANA n'a pas d'inertie directe, on calcule une approximation
            try:
                from sklearn.metrics import pairwise_distances
                # Calculer l'inertie intra-cluster approximative
                inertia_approx = 0
                for cluster_id in np.unique(labels_use):
                    cluster_points = X_use[labels_use == cluster_id]
                    if len(cluster_points) > 1:
                        # Distance intra-cluster
                        intra_distances = pairwise_distances(cluster_points)
                        inertia_approx += np.sum(intra_distances) / 2
                st.metric("⚡ Inertie (approx.)", f"{inertia_approx:.2f}")
            except:
                st.info("Inertie non calculable pour DIANA")

        col1, col2 = st.columns(2)

        with col1:
            _plot_distribution(labels_use, "Distribution des clusters (DIANA)",
                               plt.cm.Set2)

        with col2:
            _plot_pca_2d(
                X_use, labels_use, centers=None,
                centers_are_indices=False,
                title="Projection PCA 2D — DIANA",
                marker_style=None, marker_color=None, edge_color=None,
                cmap="Set2",
            )

        # ---- Historique des splits ----
        if history:
            st.markdown("#### 📋 Historique des divisions")
            split_data = []
            for i, step in enumerate(history):
                split_data.append({
                    "Étape": i + 1,
                    "Cluster divisé (taille)": len(step["split"]),
                    "Groupe principal (taille)": len(step["main"]),
                    "Nouveau groupe (taille)": len(step["new"]),
                    "Diamètre avant split": f"{step['diameter']:.4f}",
                })
            import pandas as pd
            st.dataframe(
                pd.DataFrame(split_data),
                width='stretch',
                hide_index=True,
            )

        # ---- Dendrogramme approché via scipy (ward inversé pour visualisation) ----
        st.markdown("#### 🌲 Dendrogramme (visualisation hiérarchique)")
        st.caption(
            "La hiérarchie DIANA est visualisée ici via un dendrogramme agglomératif "
            "équivalent (ward) pour une représentation claire des fusions/divisions."
        )
        try:
            Z = linkage(X_use, method="ward")
            fig, ax = dark_fig(figsize=(14, 5))
            dendrogram(
                Z,
                truncate_mode="lastp",
                p=min(50, len(X_use)),
                leaf_rotation=90.0,
                leaf_font_size=8.0,
                show_contracted=True,
                ax=ax,
                color_threshold=Z[-(n_clusters - 1), 2],
                above_threshold_color="#8b949e",
            )
            ax.set_title("Dendrogramme DIANA — arbre hiérarchique", color="#e6edf3")
            ax.set_xlabel("Échantillons (ou taille du cluster contracté)")
            ax.set_ylabel("Distance de division")
            ax.axhline(
                y=Z[-(n_clusters - 1), 2],
                color="#f78166", linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"Coupe à k={n_clusters}",
            )
            ax.legend(framealpha=0.1, labelcolor="#e6edf3")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"Dendrogramme indisponible : {e}")

        # ---- Comparaison DIANA vs AGNES ----
        st.markdown("#### ⚖️ DIANA vs AGNES — Quelle différence ?")
        st.markdown("""
        | Critère | AGNES (agglomératif) | DIANA (divisif) |
        |---|---|---|
        | **Direction** | Bottom-up : n singletons → 1 cluster | Top-down : 1 cluster → n singletons |
        | **Point de départ** | Chaque point seul | Tous les points ensemble |
        | **Complexité** | O(n² log n) | O(n²) par étape |
        | **Qualité globale** | Meilleure au niveau supérieur | Meilleure vue d'ensemble initiale |
        | **Usage** | Plus courant, bien supporté | Utile si structure globale claire |
        | **Implémentation** | Sklearn + scipy | Manuelle (ce module) |
        """)


# ─────────────────────────────────────────────────────────────
#  ONGLET 7 — ÉVALUATION
# ─────────────────────────────────────────────────────────────
def _tab_evaluation(X: np.ndarray):
    st.markdown("### 📏 Évaluation — Silhouette & Comparaison multi-algorithmes")
    st.markdown("""
    **Coefficient de Silhouette** mesure :
    - la **compacité** : distance entre un point et les autres points de son cluster
    - la **séparation** : distance entre un point et les points du cluster le plus proche

    → **Valeur proche de 1** = bon clustering
    """)
    k_max = st.slider("Comparer de k=2 à k=", 3, 15, 8)

    algos = st.multiselect(
        "Algorithmes à comparer",
        ["K-Means", "K-Medoids", "DIANA"],
        default=["K-Means"],
        key="eval_algos",
        help="Seuls les algorithmes nécessitant un K fixe sont comparés ici",
    )

    diana_max = st.slider(
        "Limite de points pour DIANA dans l'évaluation",
        100, 1000, 300, step=50, key="eval_diana_max",
        help="DIANA est lent sur de grands datasets.",
    )

    if st.button("📊 Lancer la comparaison"):
        k_range = list(range(2, k_max + 1))
        results = {algo: [] for algo in algos}
        inertias = {algo: [] for algo in algos if algo in ["K-Means", "K-Medoids"]}

        with st.spinner("Comparaison en cours..."):
            for k in k_range:
                if "K-Means" in algos:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    lkm = km.fit_predict(X)
                    results["K-Means"].append(silhouette_score(X, lkm))
                    inertias["K-Means"].append(km.inertia_)

                if "K-Medoids" in algos:
                    lkmed, _, inertia = kmedoids(X, k)
                    results["K-Medoids"].append(silhouette_score(X, lkmed))
                    inertias["K-Medoids"].append(inertia)

                if "DIANA" in algos:
                    n = len(X)
                    X_d = X
                    if n > diana_max:
                        idx_s = np.random.choice(n, diana_max, replace=False)
                        X_d = X[idx_s]
                    ld, _ = diana(X_d, n_clusters=k)
                    results["DIANA"].append(silhouette_score(X_d, ld))

        # ---- Graphiques côte à côte : Silhouette et Inertie ----
        col_sil, col_inert = st.columns(2)
        
        with col_sil:
            fig, ax = dark_fig(figsize=(8, 5))
            colors_map = {
                "K-Means":          "#58a6ff",
                "K-Medoids":        "#f78166",
                "DIANA":            "#bc8cff",
            }
            styles = {
                "K-Means":          ("-",  "o"),
                "K-Medoids":        ("--", "s"),
                "DIANA":            (":",  "P"),
            }
            for algo, scores in results.items():
                ls, mk = styles[algo]
                ax.plot(k_range, scores, marker=mk, linestyle=ls,
                        color=colors_map[algo], linewidth=2, label=algo)

            ax.set_title("Comparaison des Silhouettes", color="#e6edf3")
            ax.set_xlabel("k (nombre de clusters)")
            ax.set_ylabel("Score de Silhouette")
            ax.legend(framealpha=0.1, labelcolor="#e6edf3")
            ax.grid(alpha=0.1, color="#8b949e")
            st.pyplot(fig)
            plt.close()
        
        with col_inert:
            if inertias:
                fig2, ax2 = dark_fig(figsize=(8, 5))
                for algo, inertia_vals in inertias.items():
                    ls, mk = styles[algo]
                    ax2.plot(k_range, inertia_vals, marker=mk, linestyle=ls,
                             color=colors_map[algo], linewidth=2, label=algo)
                
                ax2.set_title("Comparaison des Inerties", color="#e6edf3")
                ax2.set_xlabel("k (nombre de clusters)")
                ax2.set_ylabel("Inertie")
                ax2.legend(framealpha=0.1, labelcolor="#e6edf3")
                ax2.grid(alpha=0.1, color="#8b949e")
                st.pyplot(fig2)
                plt.close()
            else:
                st.info(" L'inertie n'est disponible que pour K-Means et K-Medoids")

        # ---- Résumé métriques ----
        st.markdown("#### 🏆 Meilleur k par algorithme")
        cols = st.columns(len(algos))
        for col, (algo, scores) in zip(cols, results.items()):
            best_k = k_range[int(np.argmax(scores))]
            col.metric(
                algo, f"k = {best_k}",
                f"Silhouette = {max(scores):.3f}",
            )

        # ---- Tableau comparatif ----
        st.markdown("""
        #### 📖 Comparaison des algorithmes

        | Algorithme | Type | Nécessite k | Robustesse bruit | Complexité | Points forts |
        |---|---|---|---|---|---|
        | **K-Means** | Partitionnement | Oui | Faible | O(n·k·t) | Rapide, simple |
        | **K-Medoids** | Partitionnement | Oui | Élevée | O(n²·k·t) | Robuste aux outliers |
        | **DBSCAN** | Densité | Non | Très élevée | O(n log n) | Formes arbitraires, détecte le bruit |
        | **AGNES** | Hiérarchique ↑ | Post-hoc | Moyenne | O(n² log n) | Dendrogramme, pas de k a priori |
        | **DIANA** | Hiérarchique ↓ | Post-hoc | Moyenne | O(n²/étape) | Vue globale → locale |
        """)


# ─────────────────────────────────────────────────────────────
#  HELPERS INTERNES
# ─────────────────────────────────────────────────────────────
def _plot_distribution(labels, title: str, cmap):
    """Bar chart de la distribution des clusters."""
    fig, ax = dark_fig(figsize=(6, 4))
    unique, counts = np.unique(labels, return_counts=True)
    colors = cmap(np.linspace(0, 1, len(unique)))
    bar_labels = [f"Bruit" if lbl == -1 else f"C{lbl}" for lbl in unique]
    bar_colors = [(0.5, 0.5, 0.5, 1.0) if lbl == -1 else colors[i]
                  for i, lbl in enumerate(unique)]
    bars = ax.bar(bar_labels, counts, color=bar_colors,
                  edgecolor="#30363d", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(cnt), ha="center", va="bottom",
            color="#e6edf3", fontsize=10,
        )
    ax.set_title(title, color="#e6edf3")
    ax.set_ylabel("Nombre de points")
    st.pyplot(fig)
    plt.close()


def _plot_pca_2d(
    X, labels, centers, centers_are_indices: bool,
    title: str, marker_style, marker_color, edge_color,
    cmap: str = "Set2",
):
    """
    Projette X en 2D via PCA et affiche les clusters.
    Si marker_style est None, les centres ne sont pas affichés
    (utile pour AGNES/DIANA qui n'ont pas de centres explicites).
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = dark_fig(figsize=(6, 4))
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap=cmap, alpha=0.7, s=25, edgecolors="none",
    )

    if marker_style is not None and centers is not None:
        if centers_are_indices:
            c_2d = X_2d[centers]
        else:
            c_2d = pca.transform(centers)
        ax.scatter(
            c_2d[:, 0], c_2d[:, 1],
            marker=marker_style,
            s=200 if marker_style == "*" else 120,
            c=marker_color, edgecolors=edge_color,
            linewidth=1.5, zorder=5, label="Centres",
        )
        ax.legend(framealpha=0.1, labelcolor="#e6edf3")

    ax.set_title(title, color="#e6edf3")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.colorbar(scatter, ax=ax)
    st.pyplot(fig)
    plt.close()