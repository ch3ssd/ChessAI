import pytest
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from rembg import remove
from preprocess_training_images import preprocess_and_save_images


@pytest.fixture
def temp_image_dirs(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a dummy image
    dummy_image_path = input_dir / "dummy.png"
    image = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    image.save(dummy_image_path)

    return input_dir, output_dir, dummy_image_path

def test_preprocess_and_save_images(temp_image_dirs):
    input_dir, output_dir, dummy_image_path = temp_image_dirs

    # Run the function
    preprocess_and_save_images(input_dir, output_dir)

    # Check if output image exists
    output_image_path = output_dir / "dummy.png"
    assert output_image_path.exists(), "Processed image was not saved."

    # Check if output image is a valid image
    try:
        processed_image = Image.open(output_image_path)
        processed_image.verify()  # Ensure it's a valid image
    except Exception as e:
        pytest.fail(f"Output image is invalid: {e}")

    print("Test passed: Image preprocessing function works correctly!")

def test_preprocess_image_error_handling(monkeypatch, temp_image_dirs):
    input_dir, output_dir, dummy_image_path = temp_image_dirs

    # Monkeypatch Image.open to raise an error
    def broken_open(*args, **kwargs):
        raise OSError("Simulated image open error")

    monkeypatch.setattr("PIL.Image.open", broken_open)

    # Should not crash even if an image fails to open
    preprocess_and_save_images(input_dir, output_dir)

    # Output dir should still be created, but image won't be saved
    output_image_path = output_dir / "dummy.png"
    assert not output_image_path.exists()

def test_transformation_output_shape(temp_image_dirs):
    input_dir, _, dummy_image_path = temp_image_dirs

    image = Image.open(dummy_image_path).convert("RGBA")
    image_no_bg = remove(image).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor = transform(image_no_bg)
    assert tensor.shape == (3, 128, 128), "Unexpected tensor shape after transform"
