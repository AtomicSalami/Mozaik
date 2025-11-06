import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

def extract_top_colors(img_array, n_colors=25, initial_clusters=80):
    pixels = img_array.reshape(-1, 3)

    # Step 1: Extract a lot of initial colors
    kmeans = KMeans(n_clusters=initial_clusters, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    colors = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

    # Step 2: Merge them down to final n_colors using KMeans again
    while len(colors) > n_colors:
        clusters_now = max(n_colors, len(colors) // 2)  # Gradually shrink
        kmeans_merge = KMeans(n_clusters=clusters_now, random_state=42, n_init='auto')
        kmeans_merge.fit(colors)
        colors = np.clip(kmeans_merge.cluster_centers_, 0, 255).astype(np.uint8)

    return colors

def map_preset_rgb(value):
    rgb_colors = {
        0: (0, 0, 0), 1: (128, 128, 128), 2: (192, 192, 192), 3: (255, 255, 255),
        4: (255, 0, 0), 5: (0, 255, 0), 6: (0, 0, 255), 7: (255, 255, 0),
        8: (255, 0, 255), 9: (0, 255, 255), 10: (128, 0, 0), 11: (0, 128, 0),
        12: (0, 0, 128), 13: (128, 128, 0), 14: (128, 0, 128), 15: (0, 128, 128),
        16: (192, 0, 0), 17: (0, 192, 0), 18: (0, 0, 192), 19: (192, 192, 0),
        20: (192, 0, 192), 21: (0, 192, 192), 22: (64, 0, 0), 23: (0, 64, 0),
        24: (0, 0, 64), 25: (64, 64, 0), 26: (64, 0, 64)
    }
    closest_id = min(rgb_colors, key=lambda k: abs(rgb_colors[k][0] - value))
    return rgb_colors[closest_id]

def map_preset_grayscale(value):
    grayscale_colors = {
        0: (0, 0, 0), 1: (64, 64, 64), 2: (105, 105, 105),
        3: (169, 169, 169), 4: (220, 220, 220), 5: (245, 245, 245),
        6: (255, 255, 255),
    }
    return min(grayscale_colors.values(), key=lambda x: abs(x[0] - value))
