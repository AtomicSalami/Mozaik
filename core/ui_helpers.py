from PIL import Image, ImageDraw

def create_color_legend(color_counts):
    square_size = 20
    legend_width = 200
    legend_height = (len(color_counts) + 1) * (square_size + 5)
    legend = Image.new("RGB", (legend_width, min(legend_height, 16000)), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    y_offset = 5
    for color, count in color_counts.items():
        if isinstance(color, tuple):
            rgb_color = color
        else:
            rgb_color = (color, color, color)
        draw.rectangle([5, y_offset, 5 + square_size, y_offset + square_size], fill=rgb_color, outline=(0, 0, 0))
        draw.text((35, y_offset + 5), f"RGB: {rgb_color} - {count} bricks", fill=(0, 0, 0))
        y_offset += square_size + 5
    return legend
