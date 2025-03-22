import os
import platform
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from rembg import remove

if platform.system() == "Windows":
    BASE_DRIVE_PATH = Path("G:/My Drive/ChessAIProject")
else:  # macOS & Linux
    BASE_DRIVE_PATH = Path("~/Google Drive/ChessAIProject").expanduser()

INPUT_DIR = BASE_DRIVE_PATH / "ChessPieceImages/TrainingImages"
OUTPUT_DIR = BASE_DRIVE_PATH / "ChessPieceImages/TrainingImagesPreprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

def preprocess_and_save_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Recursively find all image files (supports jpg, jpeg, png)
    image_paths = list(input_dir.rglob("*.jp*g")) + list(input_dir.rglob("*.pn*g"))

    for img_path in tqdm(image_paths, desc="Processing Images"):
        try:
            relative_path = img_path.relative_to(input_dir)
            output_image_path = output_dir / relative_path

            output_image_path.parent.mkdir(parents=True, exist_ok=True)

            image = Image.open(img_path).convert("RGBA")
            image_no_bg = remove(image)
            image_no_bg = image_no_bg.convert("RGB")

            tensor = transform(image_no_bg)

            # Convert back to image for saving
            unnormalize = transforms.Compose([
                transforms.Normalize(mean=[-1], std=[2]),  # Undo normalization
                transforms.ToPILImage()
            ])
            processed_image = unnormalize(tensor)

            processed_image.save(output_image_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Preprocessed images saved to {output_dir}")

# Run preprocessing
preprocess_and_save_images(INPUT_DIR, OUTPUT_DIR)
