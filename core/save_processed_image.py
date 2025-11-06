import os
from datetime import datetime

def save_processed_image(image, original_filename):
    # Ensure the output directory exists
    os.makedirs("output", exist_ok=True)

    # Extract just the base name (without folder or extension)
    base_name = os.path.splitext(os.path.basename(original_filename))[0]

    # Format the current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create filename
    output_filename = f"{base_name}_{timestamp}.png"

    # Create full path
    output_path = os.path.join("output", output_filename)

    # Save image
    image.save(output_path)

    print(f"Saved processed image to: {output_path}")
