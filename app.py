import gradio as gr
import numpy as np
from utils.preprocessing import preprocess_image
from utils.pixel_art import generate_pixel_art_output
from utils.smooth_art import generate_smooth_art_output

# Cache session state
session_state = {}

def mozaik_interface(image, n_colors, blur_strength):
    arr_blurred, color_centers, width_bricks, height_bricks, brick_size = preprocess_image(
        image, n_colors, blur_strength
    )

    width_px = width_bricks * brick_size
    height_px = height_bricks * brick_size

    pixel_art_output, outlines_only, label_map_pixel, legend_img, color_to_id = generate_pixel_art_output(
        arr_blurred, color_centers, brick_size, width_px, height_px, mode_choice="Mozaik"
    )

    # Cache for possible use in Color-Book
    session_state["blurred"] = arr_blurred
    session_state["clusters"] = color_centers
    session_state["label_map_pixel"] = label_map_pixel
    session_state["brick_size"] = brick_size
    session_state["width_px"] = width_px
    session_state["height_px"] = height_px
    session_state["color_to_id"] = color_to_id

    total_bricks = width_bricks * height_bricks
    brick_count_text = f"Total bricks needed: {total_bricks} ({width_bricks}x{height_bricks} bricks)"

    return pixel_art_output, brick_count_text, legend_img

def colorbook_interface(image, n_colors, blur_strength, mode, canny_low, canny_high, downscale_natural):
    if "blurred" not in session_state:
        arr_blurred, color_centers, width_bricks, height_bricks, brick_size = preprocess_image(
            image, n_colors, blur_strength
        )
        width_px = width_bricks * brick_size
        height_px = height_bricks * brick_size

        pixel_art_output, outlines_only, label_map_pixel, legend_img, color_to_id = generate_pixel_art_output(
            arr_blurred, color_centers, brick_size, width_px, height_px, mode_choice="Color-Book"
        )

        session_state["blurred"] = arr_blurred
        session_state["clusters"] = color_centers
        session_state["label_map_pixel"] = label_map_pixel
        session_state["brick_size"] = brick_size
        session_state["width_px"] = width_px
        session_state["height_px"] = height_px
        session_state["color_to_id"] = color_to_id
    else:
        arr_blurred = session_state["blurred"]
        color_centers = session_state["clusters"]
        label_map_pixel = session_state["label_map_pixel"]
        brick_size = session_state["brick_size"]
        width_px = session_state["width_px"]
        height_px = session_state["height_px"]
        color_to_id = session_state["color_to_id"]

    if mode == "Pixel-Art":
        pixel_art_output, outlines_only, label_map_pixel, legend_img, color_to_id = generate_pixel_art_output(
            arr_blurred, color_centers, brick_size, width_px, height_px, mode_choice="Color-Book"
        )
        total_bricks = (width_px // brick_size) * (height_px // brick_size)
        brick_count_text = f"Total bricks needed: {total_bricks} ({width_px//brick_size}x{height_px//brick_size} bricks)"
        return pixel_art_output, brick_count_text, legend_img

    elif mode == "Natural":
        smoothed_img, legend_img = generate_smooth_art_output(
            arr_blurred,
            label_map_pixel,
            color_to_id,
            color_centers,
            brick_size,
            width_px,
            height_px,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        total_bricks = (width_px // brick_size) * (height_px // brick_size)
        brick_count_text = f"Total bricks needed: {total_bricks} ({width_px//brick_size}x{height_px//brick_size} bricks)"
        return smoothed_img, brick_count_text, legend_img

def clear_session_state():
    session_state.clear()

with gr.Blocks() as app:
    gr.Markdown("### Mozabrik Image Processor")

    # Image Upload (always at top)
    image_input = gr.Image(type="numpy", label="Upload Image")

    # Attach clear event
    image_input.clear(clear_session_state)

    with gr.Tabs():
        with gr.Tab("Mozaik"):
            with gr.Row():
                n_colors_mozaik = gr.Slider(maximum=25, minimum=5, label="Color palette size", step=5, value=25)
                blur_strength_mozaik = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, label="Blur Strength", value=1.5)
            process_button_mozaik = gr.Button("Generate Mozaik")

        with gr.Tab("Color Book"):
            with gr.Row():
                colorbook_mode = gr.Radio(["Pixel-Art", "Natural"], label="Color Book Mode", value="Natural")
            with gr.Row():
                n_colors_cb = gr.Slider(maximum=30, minimum=10, label="Color palette size", step=5, value=15)
                blur_strength_cb = gr.Slider(minimum=0.5, maximum=5.0, step=0.1, label="Blur Strength", value=1.5)
            with gr.Row():
                canny_low = gr.Slider(minimum=1, maximum=100, label="Canny Low Threshold", step=1, value=10)
                canny_high = gr.Slider(minimum=50, maximum=200, label="Canny High Threshold", step=1, value=40)
            process_button_cb = gr.Button("Generate Color Book")

    with gr.Row():
        output_img = gr.Image(label="Generated Output")
        brick_count = gr.Textbox(label="Brick Count")
        legend_img = gr.Image(label="Color Breakdown")

    # Button actions
    process_button_mozaik.click(
        mozaik_interface,
        [image_input, n_colors_mozaik, blur_strength_mozaik],
        [output_img, brick_count, legend_img]
    )

    process_button_cb.click(
        colorbook_interface,
        [image_input, n_colors_cb, blur_strength_cb, colorbook_mode, canny_low, canny_high, gr.Checkbox(visible=False)],
        [output_img, brick_count, legend_img]
    )

app.launch(share=False, inbrowser=True)
