import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _kmeans_plus_plus_init(data_norm, k, weights):
    n = len(data_norm)

    # Prvi centroid — nasumicno
    centroids = [data_norm.iloc[np.random.randint(n)]]

    for centroid_idx in range(k - 1):
        centroid_matrix = pd.DataFrame(centroids) # tabela koja raste svakom iteracijom, 1x13 -> 2x13..

        # Za svaku tacku: min kvadratno rastojanje do najblizeg vec izabranog centroida
        # np niz duzine 506
        # uzimamo min od vec postojecih, samo nas najbliza zanima
        distances = np.array([
            ((data_norm.iloc[i] - centroid_matrix) ** 2 * weights).sum(axis=1).min()
            for i in range(n)
        ])
        # Verovatnoce proporcionalne rastojanju — sto dalje, veca sansa
        probs = distances / distances.sum()
        chosen = np.random.choice(n, p=probs)
        centroids.append(data_norm.iloc[chosen])

    # vraca tabelu 3x13, ako je K=3, a svaki centar ima 13 koordinata
    return pd.DataFrame(centroids).reset_index(drop=True)


def _single_run(data_norm, k, weights):
    n, m = data_norm.shape
    centroids = _kmeans_plus_plus_init(data_norm, k, weights)
    assign = np.zeros(n, dtype=int)
    old_quality = float('inf')
    total_quality = float('inf')

    for iteration in range(50):
        quality = np.zeros(k)

        # 1. dodela tacaka klasterima
        # petlja kroz svih 506 kuca
        for i in range(n):
            slucaj = data_norm.iloc[i]
            dist = ((slucaj - centroids) ** 2 * weights).sum(axis=1)
            assign[i] = np.argmin(dist)

        # 2. preracunavanje centroida
        for c in range(k):
            # subset sadrzi samo kuce dodeljene trenutnom klasteru, npr 200, onda je 200x13
            subset = data_norm[assign == c]
            centroids.loc[c] = subset.mean()
            quality[c] = subset.var().sum() * len(subset)

        total_quality = quality.sum()
        if old_quality == total_quality:
            break
        old_quality = total_quality

    return assign, centroids, total_quality


def _warn_cluster_quality(data_norm, assign, centroids):
    k = len(centroids)

    # Prosecna distanca po klasteru
    # izoluje kuce za taj klaster
    # oduzima od svake kuce centroid -> 200x13, to su razlike
    # razlike se kvadriraju
    # sumira se 13 kvadratnih razlika u jedan broj za jednu kucu
    # spljostilo se u niz od 200 brojeva (axis=1)
    # avg dist je na kraju niz od 3 broja, za svaki K
    avg_dists = np.array([
        ((data_norm[assign == c] - centroids.iloc[c]) ** 2).sum(axis=1).mean()
        for c in range(k)
    ])

    # Threshold rep — 1.5x prosek svih klastera
    # da li je previse razvucen
    # mnozi prosek prosecnih distanci pa uporedjuje sa svakom prosecnom distancom svakog klastera
    threshold_rep = avg_dists.mean() * 1.5
    for c in range(k):
        if avg_dists[c] > threshold_rep:
            print(f" Klaster {c}: lose reprezentovan (avg distanca={avg_dists[c]:.3f} > {threshold_rep:.3f})")

    # Rastojanja izmedju svih parova centroida
    # oduzimanje svih obelezja medju parovima, znaci za 13 obelezja
    # onda se kvadrira razlika
    # onda se sumira
    pair_dists = np.array([
        ((centroids.iloc[c1] - centroids.iloc[c2]) ** 2).sum()
        for c1 in range(k)
        for c2 in range(c1 + 1, k)
    ])

    # Threshold sim — 0.5x prosek svih parova

    # racuna se prosecna udaljenost izmedju svih parova centara
    # prosek taj se mnozi sa 0.5
    # svaki par centroida cija je medjusovna udaljenost manja od praga smatra se previse slicnim/bliskim
    threshold_sim = pair_dists.mean() * 0.5
    pair_idx = 0
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
            if pair_dists[pair_idx] < threshold_sim:
                print(f"  [!] Klasteri {c1} i {c2} su previse slicni (distanca={pair_dists[pair_idx]:.3f} < {threshold_sim:.3f})")
            pair_idx += 1


def _silhouette_score(data_norm, assign):
    n = len(data_norm)
    clusters = np.unique(assign)
    scores = np.zeros(n)

    for i in range(n):
        point = data_norm.iloc[i]
        own_cluster = assign[i]

        # a(i) — prosecna distanca do ostalih tacaka u istom klasteru
        own_subset = data_norm[assign == own_cluster].drop(data_norm.index[i], errors='ignore')
        if len(own_subset) == 0:
            scores[i] = 0
            continue
        a = ((own_subset - point) ** 2).sum(axis=1).mean()

        # b(i) — prosecna distanca do najblizeg susednog klastera
        b = float('inf')
        for c in clusters:
            if c == own_cluster:
                continue
            neighbor_subset = data_norm[assign == c]
            avg_dist = ((neighbor_subset - point) ** 2).sum(axis=1).mean()
            if avg_dist < b:
                b = avg_dist

        scores[i] = (b - a) / max(a, b)

    return scores.mean()


def find_optimal_k(data, k_min=2, k_max=8, weights=None, n_restarts=10):
    n, m = data.shape

    if weights is None:
        weights = np.ones(m)
    weights = np.array(weights, dtype=float)

    data_mean = data.mean()
    data_std = data.std()
    data_norm = (data - data_mean) / data_std

    k_values = range(k_min, k_max + 1)
    scores = []

    for k in k_values:
        print(f"\nTestiram k={k}...")
        best_quality = float('inf')
        best_assign = None

        for restart in range(n_restarts):
            assign, centroids, total_quality = _single_run(data_norm, k, weights)
            if total_quality < best_quality:
                best_quality = total_quality
                best_assign = assign

        score = _silhouette_score(data_norm, best_assign)
        scores.append(score)
        print(f"  silhouette score za k={k}: {score:.4f}")

    # Vizuelizacija
    plt.figure()
    plt.plot(list(k_values), scores, marker='o')
    plt.xlabel('Broj klastera k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette analiza — optimalno k')
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    optimal_k = k_min + np.argmax(scores)
    print(f"\nOptimalno k = {optimal_k} (silhouette={max(scores):.4f})")
    return optimal_k


def learn(data, k, weights=None, n_restarts=10):
    n, m = data.shape

    # Tezine atributa — default: sve jednako
    if weights is None:
        weights = np.ones(m)
    weights = np.array(weights, dtype=float)

    # Normalizacija
    data_mean = data.mean()
    data_std = data.std()
    data_norm = (data - data_mean) / data_std

    # N restartova — uzimamo najbolji rezultat
    best_quality = float('inf')
    best_assign = None
    best_centroids = None

    for restart in range(n_restarts):
        print(f"\nRestart {restart + 1}/{n_restarts}:")
        assign, centroids, total_quality = _single_run(data_norm, k, weights)
        if total_quality < best_quality:
            best_quality = total_quality
            best_assign = assign
            best_centroids = centroids.copy()

    # Upozorenja na najboljem modelu
    print("\nProvera kvaliteta klastera:")
    _warn_cluster_quality(data_norm, best_assign, best_centroids)

    # Denormalizacija centroida
    centroids_orig = best_centroids * data_std.values + data_mean.values

    return best_assign, centroids_orig, data_mean, data_std


def transform(data, centroids_orig, data_mean, data_std, weights=None):
    n, m = data.shape

    if weights is None:
        weights = np.ones(m)
    weights = np.array(weights, dtype=float)

    data_norm = (data - data_mean) / data_std
    centroids_norm = (centroids_orig - data_mean) / data_std

    assign = np.zeros(n, dtype=int)
    for i in range(n):
        slucaj = data_norm.iloc[i]
        dist = ((slucaj - centroids_norm) ** 2 * weights).sum(axis=1)
        assign[i] = np.argmin(dist)

    return assign


if __name__ == '__main__':
    data = pd.read_csv('../data/BOSTON.CSV')

    # Bonus: nadji optimalno k pa onda ucenje
    optimal_k = find_optimal_k(data, k_min=2, k_max=8)

    assign, centroids, data_mean, data_std = learn(data, k=optimal_k, n_restarts=5)

    print("\nCentroidi:")
    print(centroids)

    for c in range(optimal_k):
        print(f"Klaster {c}: {(assign == c).sum()} tacaka")