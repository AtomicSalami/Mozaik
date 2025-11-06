import numpy as np
import scipy.ndimage
import os
from config import DIRECTIONS

def create_label_map(arr, palette, brick_size, color_mode, width_px, height_px):
    label_map = np.zeros((height_px // brick_size, width_px // brick_size), dtype=np.int32)
    palette_rounded = [tuple(map(int, c)) for c in palette]
    color_to_id = {c: idx + 1 for idx, c in enumerate(palette_rounded)}

    for y in range(0, height_px, brick_size):
        for x in range(0, width_px, brick_size):
            block = arr[y:y+brick_size, x:x+brick_size]
            if color_mode == "rgb":
                mean_color = np.mean(block.reshape(-1, 3), axis=0)
            else:
                mean_value = np.mean(block)
                mean_color = [mean_value]*3

            color = closest_color(mean_color, palette)
            tx = x // brick_size
            ty = y // brick_size
            label_map[ty, tx] = color_to_id[tuple(color)]
    return label_map, color_to_id

def closest_color(color, palette):
    diffs = np.linalg.norm(palette - color, axis=1)
    idx = np.argmin(diffs)
    return tuple(palette[idx])

def save_label_map(label_map, filename="label_map_output.txt"):
        os.makedirs("output", exist_ok=True)
        np.savetxt(os.path.join("output", filename), label_map, fmt="%d")
        print(f"Label map saved to output/{filename}")

def smooth_label_map(label_map, smoothing_choice="none", merge_size_threshold=1):
    smoothing_times = {"none": 1, "x2": 2, "x3": 3}
    merge_passes = smoothing_times.get(smoothing_choice, 1)

    directions = DIRECTIONS

    for _ in range(merge_passes):
        labeled, ncomponents = scipy.ndimage.label(label_map > 0)

        for i in range(1, ncomponents + 1):
            positions = np.argwhere(labeled == i)
            if len(positions) <= merge_size_threshold:
                for ty, tx in positions:
                    neighbor_counts = {}
                    for dx, dy in directions:
                        nx, ny = tx + dx, ty + dy
                        if 0 <= nx < label_map.shape[1] and 0 <= ny < label_map.shape[0]:
                            neighbor_label = label_map[ny, nx]
                            if neighbor_label != label_map[ty, tx] and neighbor_label != 0:
                                neighbor_counts[neighbor_label] = neighbor_counts.get(neighbor_label, 0) + 1

                    if neighbor_counts:
                        new_label = max(neighbor_counts.items(), key=lambda item: item[1])[0]
                        label_map[ty, tx] = new_label

    save_label_map(label_map)

    return label_map
