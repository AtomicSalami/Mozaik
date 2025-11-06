from PIL import Image, ImageFilter
import numpy as np
from sklearn.cluster import KMeans
from config import TILE_SIZE_MM, DPI, WEBP_LIMIT_PX

def bricks_to_px(bricks):
    return int((bricks * TILE_SIZE_MM / 10) * DPI)

def preprocess_image(image, n_colors=25, blur_strength=1.5):
    width_bricks, height_bricks = 100, 100
    img_original = Image.fromarray(image)
    orig_w, orig_h = img_original.size
    aspect_ratio = orig_w / orig_h

    if width_bricks / height_bricks > aspect_ratio:
        height_bricks = max(height_bricks, int(width_bricks / aspect_ratio))
    else:
        width_bricks = max(width_bricks, int(height_bricks * aspect_ratio))

    brick_size = bricks_to_px(1)
    width_px = min(width_bricks * brick_size, WEBP_LIMIT_PX)
    height_px = min(height_bricks * brick_size, WEBP_LIMIT_PX)

    img_resized = img_original.resize((width_px, height_px))
    img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=blur_strength))
    arr_blurred = np.array(img_blurred.convert("RGB"))

    # Use strict k-means for exact number of colors
    pixels = arr_blurred.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    color_centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

    return arr_blurred, color_centers, width_bricks, height_bricks, brick_size
