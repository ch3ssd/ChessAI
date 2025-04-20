import os
from io import BytesIO
import pytest
import torch
import zipfile
import shutil
import configparser
from pathlib import Path
from unittest import mock
from PIL import Image

from predict_chess_pieces import (
    load_config,
    download_and_extract_if_missing,
    main as predict_main
)

@pytest.fixture
def dummy_config(tmp_path):
    config = configparser.ConfigParser()
    test_data_dir = tmp_path / "TestImages"
    (test_data_dir / "TestImages" / "Knight").mkdir(parents=True, exist_ok=True)
    (test_data_dir / "TestImages" / "Bishop").mkdir(parents=True, exist_ok=True)

    # Create dummy image files
    knight_img = test_data_dir / "TestImages" / "Knight" / "knight.jpg"
    bishop_img = test_data_dir / "TestImages" / "Bishop" / "bishop.jpg"
    Image.new("RGB", (224, 224)).save(knight_img)
    Image.new("RGB", (224, 224)).save(bishop_img)

    config["MODEL"] = {
        "SavePath": str(tmp_path / "model.pth")
    }
    config["DATA"] = {
        "TestDataDirectory": str(test_data_dir),
        "TestDriveURL": "https://dummyurl.com/data.zip"
    }

    config_path = tmp_path / "config.properties"
    with open(config_path, "w") as f:
        config.write(f)

    return config_path, test_data_dir


@pytest.fixture
def dummy_model_file(tmp_path):
    model = torch.nn.Linear(10, 2)
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    return path

def test_load_config(dummy_config):
    config_path, _ = dummy_config
    config = load_config(config_path)
    assert "MODEL" in config and "SavePath" in config["MODEL"]


@mock.patch("predict_chess_pieces.requests.get")
def test_download_and_extract_if_missing(mock_get, tmp_path):
    dummy_dir = tmp_path / "TestImages"
    dummy_zip = tmp_path / "TestImages.zip"

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode='w') as zf:
        zf.writestr("TestImages/Knight/dummy.jpg", Image.new("RGB", (224, 224)).tobytes())
    zip_buf.seek(0)

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.content = zip_buf.getvalue()
    mock_get.return_value = mock_response

    download_and_extract_if_missing(dummy_dir, "https://fake.url")
    assert dummy_dir.exists()


@mock.patch("predict_chess_pieces.download_and_extract_if_missing")
@mock.patch("predict_chess_pieces.torch.load")
@mock.patch("predict_chess_pieces.ChessPieceModel")
@mock.patch("predict_chess_pieces.datasets.ImageFolder")
@mock.patch("predict_chess_pieces.DataLoader")
def test_main_flow(mock_loader, mock_folder, mock_model_cls, mock_torch_load, mock_download,
                   dummy_config, dummy_model_file, capsys):
    config_path, test_dir = dummy_config

    # Mock model that returns logits
    dummy_model = mock.Mock()
    dummy_model.eval = mock.Mock()
    dummy_model.__call__ = mock.Mock(return_value=torch.tensor([[2.0, 1.0]]))  # predicts class 0

    mock_model_cls.return_value.model = dummy_model

    # Mock dataset
    mock_folder.return_value.classes = ["Knight", "Bishop"]
    mock_folder.return_value.samples = [
        ("/some/path/Knight/img1.jpg", 0),
        ("/some/path/Bishop/img2.jpg", 1)
    ]

    mock_loader.return_value = [
        (torch.randn(1, 3, 224, 224), torch.tensor([0])),  # correct prediction
        (torch.randn(1, 3, 224, 224), torch.tensor([1]))   # incorrect prediction
    ]

    # Ensure model file exists
    model_path = Path(config_path).parent / "model.pth"
    torch.save(torch.nn.Linear(10, 2).state_dict(), model_path)

    # Replace config.properties in working directory
    shutil.copy(config_path, "config.properties")

    try:
        predict_main()
        captured = capsys.readouterr()
        assert "Prediction Results with Class Probabilities" in captured.out
        assert "Correct" in captured.out or "Incorrect" in captured.out
        assert "Class Probabilities:" in captured.out
    finally:
        if os.path.exists("config.properties"):
            os.remove("config.properties")
