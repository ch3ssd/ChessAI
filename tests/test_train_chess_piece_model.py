import os
import torch
import shutil
import configparser
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from train_chess_piece_model import ChessTrainer, should_skip_training

@pytest.fixture
def dummy_config(tmp_path):
    config = configparser.ConfigParser()
    config["DATA"] = {
        "DriveURL": "https://example.com/dummy.zip"
    }
    config["MODEL"] = {
        "SavePath": str(tmp_path / "saved_model" / "chess_model.pth")
    }
    config_path = tmp_path / "config.properties"
    with open(config_path, "w") as f:
        config.write(f)
    return config_path


@pytest.fixture
def dummy_model_wrapper():
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

    class DummyWrapper:
        def __init__(self):
            self.model = DummyNet()
            self.device = torch.device("cpu")
            self.train_loader = [
                (torch.randn(2, 10), torch.randint(0, 2, (2,)))
                for _ in range(3)
            ]
            self.val_loader = [
                (torch.randn(2, 10), torch.randint(0, 2, (2,)))
                for _ in range(2)
            ]

    return DummyWrapper()

def test_should_skip_training_true(tmp_path):
    model_file = tmp_path / "model.pth"
    model_file.touch()
    assert should_skip_training(model_file) is True


def test_should_skip_training_false(tmp_path):
    model_file = tmp_path / "model.pth"
    assert should_skip_training(model_file) is False


def test_train_one_epoch(dummy_model_wrapper, dummy_config):
    trainer = ChessTrainer(dummy_model_wrapper, config_path=str(dummy_config))
    loss, acc = trainer.train_one_epoch()
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100


def test_validate(dummy_model_wrapper, dummy_config):
    trainer = ChessTrainer(dummy_model_wrapper, config_path=str(dummy_config))
    loss, acc = trainer.validate()
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0 <= acc <= 100


def test_train_and_save_model(tmp_path, dummy_model_wrapper, dummy_config):
    trainer = ChessTrainer(dummy_model_wrapper, config_path=str(dummy_config))
    trainer.train(epochs=1)
    model_file = Path(trainer.save_path)
    assert model_file.exists()
    assert model_file.suffix == ".pth"
