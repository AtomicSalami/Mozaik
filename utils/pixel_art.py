from PIL import ImageDraw, Image
import numpy as np
from core.tile_utils import create_label_map, smooth_label_map
from core.draw_utils import draw_tiles, draw_outlines, draw_numbers
from core.save_processed_image import save_processed_image
from core.ui_helpers import create_color_legend

def generate_pixel_art_output(arr_blurred, color_centers, brick_size, width_px, height_px, mode_choice="Color-Book"):
    label_map_pixel, color_to_id = create_label_map(arr_blurred, color_centers, brick_size, "rgb", width_px, height_px)
    id_to_color = {v: k for k, v in color_to_id.items()}

    label_map_pixel = smooth_label_map(label_map_pixel)

    output, draw = draw_tiles(label_map_pixel, id_to_color, brick_size, width_px, height_px)

    if mode_choice == "Color-Book":
        draw_outlines(label_map_pixel, draw, brick_size)
        draw_numbers(label_map_pixel, draw, brick_size)
    elif mode_choice == "Mozaik":
        for ty in range(label_map_pixel.shape[0]):
            for tx in range(label_map_pixel.shape[1]):
                x = tx * brick_size
                y = ty * brick_size
                draw.rectangle([x, y, x + brick_size, y + brick_size], outline=(0, 0, 0))

    save_processed_image(output, 'processed_')

    outlines_only = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    outlines_draw = ImageDraw.Draw(outlines_only)
    draw_outlines(label_map_pixel, outlines_draw, brick_size)

    legend = create_color_legend({k: v for k, v in color_to_id.items()})
    return output, outlines_only, label_map_pixel, legend, color_to_id
