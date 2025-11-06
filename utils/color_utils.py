import numpy as np
from sklearn.cluster import KMeans

def get_dominant_colors(img_array, n_colors, initial_clusters=80):
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    return np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)