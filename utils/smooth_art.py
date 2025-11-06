from PIL import Image, ImageDraw
import numpy as np
import cv2
from core.draw_utils import draw_numbers
from core.ui_helpers import create_color_legend
from core.tile_utils import create_label_map

def generate_smooth_art_output(
    arr_blurred,
    label_map_pixel,
    color_to_id,
    color_centers,
    brick_size,
    width_px,
    height_px,
    canny_low=10,
    canny_high=40
):
    # Step 1: create smooth label map (per pixel)
    label_map_smooth, _ = create_label_map(
        arr_blurred, color_centers, 1, "rgb", width_px, height_px
    )

    # Step 2: prepare reverse mapping: label -> RGB
    id_to_color = {v: k for k, v in color_to_id.items()}

    # Step 3: draw image using vectorized mask
    smoothed_img = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    for label_id, rgb in id_to_color.items():
        smoothed_img[label_map_smooth == label_id] = rgb

    # Step 4: Canny edge detection
    smoothed_gray = cv2.cvtColor(smoothed_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(smoothed_gray, threshold1=canny_low, threshold2=canny_high)
    smoothed_img[edges > 0] = (0, 0, 0)

    # Step 5: convert to PIL and draw numbers using pixel-art label map
    smoothed_pil = Image.fromarray(smoothed_img)
    draw_smooth = ImageDraw.Draw(smoothed_pil)
    draw_numbers(label_map_pixel, draw_smooth, brick_size)

    # Step 6: color legend
    legend = create_color_legend({k: v for k, v in color_to_id.items()})
    return smoothed_pil, legend