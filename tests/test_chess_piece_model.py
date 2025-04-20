import os
import torch
import pytest
import shutil
import configparser
from pathlib import Path
from PIL import Image
from chess_piece_model import ChessPieceModel

@pytest.fixture
def temp_config_and_data(tmp_path):
    # Setup dummy config
    config_path = tmp_path / "config.properties"
    model_path = tmp_path / "dummy_model.pth"
    data_dir = tmp_path / "TrainingImages"
    (data_dir / "Pawn").mkdir(parents=True)
    (data_dir / "Knight").mkdir(parents=True)

    # Create dummy images
    dummy_image = Image.new("RGB", (224, 224), color="white")
    for i in range(5):
        dummy_image.save(data_dir / "Pawn" / f"pawn_{i}.jpg")
    for i in range(5):
        dummy_image.save(data_dir / "Knight" / f"knight_{i}.jpg")

    # Save config
    config = configparser.ConfigParser()
    config["DATA"] = {
        "DataDirectory": str(data_dir),
        "DriveURL": "https://dummyurl.com/fake.zip"
    }
    config["MODEL"] = {
        "BatchSize": "2",
        "SavePath": str(model_path)
    }
    with open(config_path, "w") as f:
        config.write(f)

    return config_path

@pytest.fixture
def model_and_loader(temp_config_and_data, monkeypatch):
    monkeypatch.setattr(ChessPieceModel, "download_and_extract_data", lambda self, url: None)
    return ChessPieceModel(drive_url=None, config_path=str(temp_config_and_data))

def test_config_loading(model_and_loader):
    assert model_and_loader.batch_size == 2

def test_data_loading(model_and_loader):
    assert len(model_and_loader.original_dataset) == 10

def test_data_transforms(model_and_loader):
    sample, _ = model_and_loader.original_dataset[0]
    assert sample.shape == (3, 224, 224)

def test_initialize_data_loader(model_and_loader):
    assert len(model_and_loader.train_loader) > 0
    assert len(model_and_loader.val_loader) > 0

def test_train_val_split(model_and_loader):
    total = len(model_and_loader.original_dataset)
    assert total == len(model_and_loader.train_loader.dataset) + len(model_and_loader.val_loader.dataset)

def test_model_fc_layer(model_and_loader):
    assert model_and_loader.model.fc.out_features == len(model_and_loader.original_dataset.classes)

def test_forward_pass(model_and_loader):
    sample_batch = next(iter(model_and_loader.train_loader))
    images, _ = sample_batch
    out = model_and_loader.model(images.to(model_and_loader.device))
    assert out.shape[0] == images.shape[0]

def test_load_data_into_model(model_and_loader):
    model_and_loader.load_data_into_model()
    assert model_and_loader.model is not None
