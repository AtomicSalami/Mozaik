TILE_SIZE_MM = 1
DPI = 150
WEBP_LIMIT_PX = 16000
SECTION_SIZE = 32 
GAP_SIZE = 30
RADIUS = 5
DIRECTIONS = [
    (dx, dy)
    for dx in range(-RADIUS, RADIUS + 1)
    for dy in range(-RADIUS, RADIUS + 1)
    if not (dx == 0 and dy == 0)
]