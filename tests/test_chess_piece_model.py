import pytest
import torch
import os
import configparser
from pathlib import Path
from chess_piece_model import ChessPieceModel

CONFIG_PATH = "config.properties"

# Load DriveURL from the config file
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
DRIVE_URL = config["DATA"]["DriveURL"]

@pytest.fixture(scope="session")
def model_and_loader():
    """Initialize the ChessPieceModel with Google Drive dataset and configuration file."""
    model = ChessPieceModel(drive_url=DRIVE_URL, config_path=CONFIG_PATH)
    return model

@pytest.fixture
def sample_batch(model_and_loader):
    """Fetch a single batch from the train_loader."""
    return next(iter(model_and_loader.train_loader))

def test_config_loading():
    """Ensure the config is properly loaded."""
    model = ChessPieceModel(drive_url=DRIVE_URL, config_path=CONFIG_PATH)
    assert "DATA" in model.config
    assert "MODEL" in model.config
    assert "DataDirectory" in model.config["DATA"]
    assert "BatchSize" in model.config["MODEL"]

def test_data_loading(sample_batch, model_and_loader):
    """Check if data loading works properly."""
    images, labels = sample_batch
    assert images.shape[0] > 0
    assert len(model_and_loader.original_dataset.classes) > 0

def test_data_transforms(model_and_loader):
    """Test that data transforms return callable Compose objects."""
    train_t, val_t = model_and_loader.get_data_transforms()
    assert callable(train_t)
    assert callable(val_t)

def test_initialize_data_loader(model_and_loader):
    """Ensure loaders split the dataset correctly."""
    dataset, train_loader, val_loader = model_and_loader.initialize_data_loader()
    assert len(dataset) == len(train_loader.dataset) + len(val_loader.dataset)

def test_train_val_split(model_and_loader):
    train_size = len(model_and_loader.train_loader.dataset)
    val_size = len(model_and_loader.val_loader.dataset)
    total_size = train_size + val_size
    assert total_size == len(model_and_loader.original_dataset)
    assert abs(train_size - 0.8 * total_size) < 1
    assert abs(val_size - 0.2 * total_size) < 1

def test_model_fc_layer(model_and_loader):
    num_classes = len(model_and_loader.original_dataset.classes)
    assert model_and_loader.model.fc.out_features == num_classes

def test_forward_pass(sample_batch, model_and_loader):
    images, _ = sample_batch
    images = images.to(model_and_loader.device)
    with torch.no_grad():
        output = model_and_loader.model(images)
    assert output.shape[0] == images.shape[0]
    assert output.shape[1] == len(model_and_loader.original_dataset.classes)

def test_load_data_into_model(model_and_loader):
    model_and_loader.load_data_into_model()  # Also exercises print/debug logic

def test_skip_download_if_data_exists(monkeypatch, tmp_path):
    """Test download logic is skipped if data directory already exists."""
    # Setup fake config
    data_dir = tmp_path / "TrainingImagesPreprocessed"
    data_dir.mkdir(parents=True, exist_ok=True)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    config["DATA"]["DataDirectory"] = str(data_dir)

    # Monkeypatch config loader to use our temp config
    def fake_load_config(self, path): return config
    monkeypatch.setattr(ChessPieceModel, "load_config", fake_load_config)

    model = ChessPieceModel(drive_url="http://fake-url.com/skip.zip", config_path="fake-path")
    # Should not raise or attempt download

def test_download_failure(monkeypatch, tmp_path):
    """Test that download raises error when status code is not 200."""
    data_dir = tmp_path / "TrainingImagesPreprocessed"

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    config["DATA"]["DataDirectory"] = str(data_dir)

    # Monkeypatch config loader
    def fake_load_config(self, path): return config
    monkeypatch.setattr(ChessPieceModel, "load_config", fake_load_config)

    # Monkeypatch requests.get to return mock failure
    class FakeResponse:
        status_code = 404
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(RuntimeError, match="Failed to download dataset"):
        ChessPieceModel(drive_url="http://fake-url.com/fail.zip", config_path="fake-path")

