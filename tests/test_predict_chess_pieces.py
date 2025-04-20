import pytest
import torch
import torch.nn as nn
from unittest import mock
from pathlib import Path
import shutil
import configparser
import os


@pytest.fixture
def dummy_config(tmp_path):
    model_path = tmp_path / "model.pth"
    data_dir = tmp_path / "TestImages"
    nested_dir = data_dir / "TestImages"
    nested_dir.mkdir(parents=True)

    config = configparser.ConfigParser()
    config["MODEL"] = {"SavePath": str(model_path)}
    config["DATA"] = {
        "TestDataDirectory": str(data_dir),
        "TestDriveURL": "https://dummyurl.com/fake.zip"
    }

    config_path = tmp_path / "config.properties"
    with open(config_path, "w") as f:
        config.write(f)

    # Save dummy model weights
    dummy_model = nn.Sequential(nn.Linear(10, 2))
    torch.save(dummy_model.state_dict(), model_path)

    return config_path, data_dir


@mock.patch("predict_chess_pieces.download_and_extract_if_missing")
@mock.patch("predict_chess_pieces.torch.load")
@mock.patch("predict_chess_pieces.ChessPieceModel")
@mock.patch("predict_chess_pieces.datasets.ImageFolder")
@mock.patch("predict_chess_pieces.DataLoader")
def test_main_flow(mock_loader, mock_folder, mock_model_cls, mock_torch_load, mock_download, dummy_config, capsys):
    from predict_chess_pieces import main

    config_path, test_dir = dummy_config
    shutil.copy(config_path, "config.properties")  # simulate main's config load

    dummy_model = mock.Mock()
    dummy_model.eval = mock.Mock(return_value=dummy_model)
    dummy_model.return_value = torch.tensor([[2.0, 1.0]])  # mock model(image)
    mock_model_wrapper = mock.Mock()
    mock_model_wrapper.model = dummy_model
    mock_model_cls.return_value = mock_model_wrapper

    mock_folder.return_value.classes = ["Knight", "Bishop"]
    mock_folder.return_value.samples = [
        ("/some/path/Knight/img1.jpg", 0),
        ("/some/path/Bishop/img2.jpg", 1)
    ]

    mock_loader.return_value = [
        (torch.randn(1, 3, 224, 224), torch.tensor([0])),
        (torch.randn(1, 3, 224, 224), torch.tensor([1]))
    ]

    main()

    captured = capsys.readouterr()
    assert "Prediction Results with Class Probabilities" in captured.out
    assert "Image:" in captured.out
    assert "True Label:" in captured.out
    assert "Predicted Label:" in captured.out
    assert "Class Probabilities:" in captured.out
