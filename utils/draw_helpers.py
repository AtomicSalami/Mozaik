import numpy as np

def get_label_center(positions, label_map, directions):
    if len(positions) == 1:
        return positions[0]
    center = np.mean(positions, axis=0)
    candidate_tiles = [(np.sum((np.array([ty_, tx_]) - center)**2), ty_, tx_)
                       for ty_, tx_ in positions]
    return min(candidate_tiles, key=lambda t: t[0])[1:]