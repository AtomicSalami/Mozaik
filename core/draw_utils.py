from PIL import Image, ImageDraw, ImageFont
import numpy as np
import scipy.ndimage
from config import DIRECTIONS
import numpy as np
import os

def save_number_centers(label_map, filename="label_numbers_processed.txt"):
    os.makedirs("output", exist_ok=True)

    label_matrix = np.full_like(label_map, fill_value=0)
    unique_labels = np.unique(label_map)

    directions = DIRECTIONS

    for label in unique_labels:
        if label == 0:
            continue
        mask = (label_map == label).astype(np.uint8)
        labeled_area, ncomponents = scipy.ndimage.label(mask)

        for i in range(1, ncomponents + 1):
            positions = np.argwhere(labeled_area == i)
            if len(positions) == 0:
                continue

            if len(positions) == 1:
                center_y, center_x = positions[0]
            else:
                center = np.mean(positions, axis=0)
                candidate_tiles = []
                for (ty_, tx_) in positions:
                    score = 0
                    for dx, dy in directions:
                        nx, ny = tx_ + dx, ty_ + dy
                        if 0 <= nx < label_map.shape[1] and 0 <= ny < label_map.shape[0]:
                            if label_map[ny, nx] != label_map[ty_, tx_]:
                                score += 1
                        else:
                            score += 1
                    dist = np.sum((np.array([ty_, tx_]) - center) ** 2)
                    candidate_tiles.append((score, dist, ty_, tx_))

                surrounded_tiles = [tile for tile in candidate_tiles if tile[0] == 0]
                if surrounded_tiles:
                    _, _, center_y, center_x = min(surrounded_tiles, key=lambda t: t[1])
                else:
                    best_tile = min(candidate_tiles, key=lambda t: (t[0], t[1]))
                    center_y, center_x = best_tile[2], best_tile[3]

            label_matrix[center_y, center_x] = label  # Place label number

    # Save result as text
    np.savetxt(os.path.join("output", filename), label_matrix, fmt="%d")
    print(f"Saved labeled numbers to: output/{filename}")


def draw_tiles(label_map, id_to_color, brick_size, width_px, height_px):
    output = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(output)

    for ty in range(label_map.shape[0]):
        for tx in range(label_map.shape[1]):
            color_id = label_map[ty, tx]
            color = id_to_color.get(color_id, (255, 255, 255))
            x = tx * brick_size
            y = ty * brick_size
            draw.rectangle([x, y, x + brick_size, y + brick_size], fill=color, outline=None)

    return output, draw

def draw_outlines(label_map, draw, brick_size):
    for ty in range(label_map.shape[0]):
        for tx in range(label_map.shape[1]):
            color = label_map[ty, tx]
            right_color = label_map[ty, tx+1] if (tx+1) < label_map.shape[1] else None
            down_color = label_map[ty+1, tx] if (ty+1) < label_map.shape[0] else None

            x = tx * brick_size
            y = ty * brick_size

            if right_color and right_color != color:
                draw.line([(x + brick_size, y), (x + brick_size, y + brick_size)], fill=(50, 50, 50), width=1)
            if down_color and down_color != color:
                draw.line([(x, y + brick_size), (x + brick_size, y + brick_size)], fill=(50, 50, 50), width=1)

def draw_numbers(label_map, draw, brick_size):
    save_number_centers(label_map)
    unique_labels = np.unique(label_map)
    font = ImageFont.load_default()

    directions = DIRECTIONS

    for label in unique_labels:
        if label == 0:
            continue
        mask = (label_map == label).astype(np.uint8)
        labeled_area, ncomponents = scipy.ndimage.label(mask)

        for i in range(1, ncomponents + 1):
            positions = np.argwhere(labeled_area == i)
            if len(positions) == 0:
                continue

            if len(positions) == 1:
                center_y, center_x = positions[0]
            else:
                center = np.mean(positions, axis=0)
                candidate_tiles = []

                for (ty_, tx_) in positions:
                    score = 0
                    for dx, dy in directions:
                        nx, ny = tx_ + dx, ty_ + dy
                        if 0 <= nx < label_map.shape[1] and 0 <= ny < label_map.shape[0]:
                            if label_map[ny, nx] != label_map[ty_, tx_]:
                                score += 1
                        else:
                            score += 1
                    dist = np.sum((np.array([ty_, tx_]) - center) ** 2)
                    candidate_tiles.append((score, dist, ty_, tx_))

                surrounded_tiles = [tile for tile in candidate_tiles if tile[0] == 0]

                if surrounded_tiles:
                    _, _, center_y, center_x = min(surrounded_tiles, key=lambda t: t[1])
                else:
                    best_tile = min(candidate_tiles, key=lambda t: (t[0], t[1]))
                    center_y, center_x = best_tile[2], best_tile[3]

            center_px = int(center_x * brick_size + brick_size / 2)
            center_py = int(center_y * brick_size + brick_size / 2)
            draw.text((center_px-4, center_py-4), str(label), fill=(0, 0, 0), font=font)

def draw_numbers_for_smooth(label_map, draw, brick_size):
    """New version: merge diagonally touching same-color regions before writing number."""
    import numpy as np
    import scipy.ndimage
    from PIL import ImageFont

    unique_labels = np.unique(label_map)
    font = ImageFont.load_default()

    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]], dtype=int)  # 8-connectivity

    for label in unique_labels:
        if label == 0:
            continue

        mask = (label_map == label).astype(np.uint8)
        labeled_area, num_features = scipy.ndimage.label(mask, structure=structure)

        largest_area = 0
        center_x, center_y = 0, 0

        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_area == i)
            if len(coords) > largest_area:
                largest_area = len(coords)
                center = np.mean(coords, axis=0).astype(int)
                center_y, center_x = center

        if largest_area > 0:
            center_px = int(center_x * brick_size + brick_size / 2)
            center_py = int(center_y * brick_size + brick_size / 2)
            draw.text((center_px-4, center_py-4), str(label), fill=(0, 0, 0), font=font)
