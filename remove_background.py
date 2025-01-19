import os
import argparse
from PIL import Image
from rembg import remove
import io
from tqdm import tqdm

def remove_background(input_image_path, output_image_path):
    """
    Removes the background from a single image and saves the result.

    :param input_image_path: Path to the input image.
    :param output_image_path: Path to save the background-removed image.
    """
    try:
        with Image.open(input_image_path) as img:
            # Convert image to bytes in PNG format for rembg
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            # Remove background
            result_bytes = remove(img_bytes)

            # Reconstruct PIL image from result
            result_image = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

            # Save the new image as PNG to preserve transparency
            result_image.save(output_image_path, format='PNG')

    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")

def process_directory(input_dir, output_dir, file_extensions):
    """
    Removes backgrounds from all images in a directory and saves them to the output directory.

    :param input_dir: Path to the input directory containing images.
    :param output_dir: Path to the output directory where processed images will be saved.
    :param file_extensions: Set of allowed image file extensions.
    """
    # List all files in the input directory
    files = os.listdir(input_dir)

    # Filter out non-image files based on extensions
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in file_extensions]

    if not image_files:
        print(f"No images found in {input_dir}. Exiting.")
        return

    print(f"Found {len(image_files)} image(s) in '{input_dir}'. Processing...")

    # Process each image with a progress bar
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"  # Save as PNG to preserve transparency
        output_path = os.path.join(output_dir, output_filename)

        remove_background(input_path, output_path)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Remove backgrounds from images.")
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Path to the input image file or directory.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory where processed images will be saved.'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='*',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
        help='List of image file extensions to process (e.g., .jpg .png). Default: common image formats.'
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    file_extensions = set(ext.lower() for ext in args.extensions)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input path is a directory or a single file
    if os.path.isdir(input_path):
        print(f"Input is a directory: '{input_path}'")
        process_directory(input_path, output_dir, file_extensions)
    elif os.path.isfile(input_path):
        print(f"Input is a single file: '{input_path}'")
        filename = os.path.basename(input_path)
        output_filename = os.path.splitext(filename)[0] + ".png"  # Save as PNG
        output_path = os.path.join(output_dir, output_filename)
        remove_background(input_path, output_path)
        print(f"Processed image saved to '{output_path}'")
    else:
        print(f"The input path '{input_path}' is neither a file nor a directory.")

    print("\nBackground removal completed.")

if __name__ == '__main__':
    main()
