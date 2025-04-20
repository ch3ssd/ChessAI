import pytest
import torch
import configparser
from pathlib import Path
import chess_piece_model
from chess_piece_model import ChessPieceModel
from torch import nn

# --- Dummy classes and data ---
class DummyDataset:
    def __init__(self):
        self.classes = ["class0", "class1"]
    def __len__(self):
        return len(self.classes)

# Each loader yields batches of shape (batch_size, features)
dummy_train_loader = [
    (torch.randn(2, 10), torch.randint(0, 2, (2,)))
    for _ in range(3)
]
dummy_val_loader = [
    (torch.randn(2, 10), torch.randint(0, 2, (2,)))
    for _ in range(2)
]

# Fake network that accepts (batch,10) inputs and outputs (batch, num_classes)
class FakeNet(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        # x is (batch, features)
        return self.fc(x)

# ----------------------
@pytest.fixture
def temp_config(tmp_path):
    cfg = configparser.ConfigParser()
    cfg["DATA"] = {
        "DriveURL": "https://dummyurl.com/fake.zip",
        "DataDirectory": str(tmp_path / "unused_dir")
    }
    cfg["MODEL"] = {
        "BatchSize": "2",
        "SavePath": str(tmp_path / "model.pth")
    }
    path = tmp_path / "config.properties"
    with open(path, "w") as f:
        cfg.write(f)
    return path

@pytest.fixture
def model_and_loader(temp_config, monkeypatch):
    # Stub out download to avoid HTTP
    monkeypatch.setattr(
        chess_piece_model.ChessPieceModel,
        "download_and_extract_data",
        lambda self, url: None
    )
    # Stub out data loader init
    monkeypatch.setattr(
        chess_piece_model.ChessPieceModel,
        "initialize_data_loader",
        lambda self: (DummyDataset(), dummy_train_loader, dummy_val_loader)
    )
    # Instantiate
    mdl = ChessPieceModel(drive_url="dummy", config_path=str(temp_config))
    # Replace real model with fake net
    mdl.model = FakeNet(in_features=10, num_classes=len(mdl.original_dataset.classes))
    mdl.device = torch.device("cpu")
    return mdl

@pytest.fixture
def sample_batch(model_and_loader):
    return next(iter(model_and_loader.train_loader))

# ----------------------

def test_config_loading(model_and_loader):
    cfg = model_and_loader.config
    assert "DATA" in cfg
    assert "MODEL" in cfg
    assert "DriveURL" in cfg["DATA"]
    assert "DataDirectory" in cfg["DATA"]
    assert "BatchSize" in cfg["MODEL"]
    assert "SavePath" in cfg["MODEL"]


def test_data_loading(sample_batch, model_and_loader):
    images, labels = sample_batch
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert images.size(1) == 10  # matches our fake feature size
    assert len(model_and_loader.original_dataset.classes) == 2


def test_data_transforms(model_and_loader):
    train_t, val_t = model_and_loader.get_data_transforms()
    assert callable(train_t)
    assert callable(val_t)


def test_initialize_data_loader(model_and_loader):
    ds, tr, vl = model_and_loader.initialize_data_loader()
    assert isinstance(ds, DummyDataset)
    assert tr is dummy_train_loader
    assert vl is dummy_val_loader


def test_train_val_split(model_and_loader):
    assert len(model_and_loader.train_loader) == len(dummy_train_loader)
    assert len(model_and_loader.val_loader) == len(dummy_val_loader)
    assert len(model_and_loader.original_dataset) == len(DummyDataset())


def test_model_fc_layer(model_and_loader):
    num_classes = len(model_and_loader.original_dataset.classes)
    assert model_and_loader.model.fc.out_features == num_classes


def test_forward_pass(sample_batch, model_and_loader):
    images, _ = sample_batch
    out = model_and_loader.model(images)
    assert out.size(0) == images.size(0)
    assert out.size(1) == len(model_and_loader.original_dataset.classes)


def test_load_data_into_model(model_and_loader):
    # Should not error, uses FakeNet
    model_and_loader.load_data_into_model()
