import numpy as np
import seaborn as sns
from time import time
from sklearn.cluster import KMeans


def kmeans(X, k):
    """Perform k-means clustering on a numerical NumPy array X.

    Returns
    -------
    centroids : ndarray of shape (k, n_features)
    labels : ndarray of shape (n_samples,)
    """
    model = KMeans(n_clusters=k, n_init="auto", random_state=0)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    return centroids, labels


# Load diamonds and keep only the numerical columns (7 total)
_diamonds = sns.load_dataset("diamonds")
diamonds_numeric = _diamonds.select_dtypes(include="number")


def kmeans_diamonds(n, k):
    """Run k-means on the first n rows of the numeric diamonds data."""
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """Average runtime (seconds) of kmeans_diamonds(n, k) over n_iter runs."""
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    return float(np.mean(times))
