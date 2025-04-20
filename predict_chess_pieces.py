import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from chess_piece_model import ChessPieceModel
import configparser
from pathlib import Path
import os
import zipfile
import requests
from io import BytesIO


def load_config(path="config.properties"):
    config = configparser.ConfigParser()
    config.read(path)
    return config


def download_and_extract_if_missing(directory: Path, drive_url: str):
    if directory.exists() and any(directory.iterdir()):
        return

    print("Downloading and extracting test data...")
    zip_path = directory.with_suffix(".zip")

    response = requests.get(drive_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download from {drive_url}")

    with open(zip_path, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(directory)

    print(f"Extracted test data to {directory}")
    zip_path.unlink()  # Delete ZIP file after extraction


def main():
    config = load_config()
    model_path = Path(config["MODEL"]["SavePath"])
    test_data_dir = Path(config["DATA"]["TestDataDirectory"])
    test_drive_url = config["DATA"]["TestDriveURL"]
    config_path = "config.properties"

    # Download and extract test images if missing
    download_and_extract_if_missing(test_data_dir, test_drive_url)

    # Account for nested "TestImages/TestImages/<class>/" structure
    nested_test_dir = test_data_dir / "TestImages"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_wrapper = ChessPieceModel(drive_url=None, config_path=config_path, device=device)
    model = model_wrapper.model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    test_dataset = datasets.ImageFolder(root=str(nested_test_dir), transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    class_names = test_dataset.classes

    # Print header
    print("\nPrediction Results with Class Probabilities")
    print("=" * 120)

    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image = image.to(device)
            outputs = model(image)

            # Convert to probabilities
            probs = F.softmax(outputs, dim=1).squeeze()
            _, predicted = torch.max(outputs, 1)

            true_label = class_names[label.item()]
            predicted_label = class_names[predicted.item()]
            correct = "Correct" if true_label == predicted_label else "Incorrect"
            image_path = test_dataset.samples[idx][0]
            image_name = os.path.basename(image_path)

            print(f"\nImage: {image_name}")
            print(f"True Label:      {true_label}")
            print(f"Predicted Label: {predicted_label} {correct}")
            print("Class Probabilities:")
            for i, class_name in enumerate(class_names):
                print(f"  {class_name:<15}: {probs[i].item():.4f}")

    print("=" * 120)


if __name__ == "__main__":
    main()
