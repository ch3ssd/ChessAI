import torch
import shutil
import pytest
import configparser
from pathlib import Path
from unittest import mock

@pytest.fixture
def dummy_config(tmp_path):
    config = configparser.ConfigParser()
    model_path = tmp_path / "model.pth"
    test_data_dir = tmp_path / "TestImages"
    (test_data_dir / "TestImages" / "Knight").mkdir(parents=True)

    config["MODEL"] = {"SavePath": str(model_path)}
    config["DATA"] = {
        "TestDataDirectory": str(test_data_dir),
        "TestDriveURL": "https://example.com/fake.zip"
    }

    config_path = tmp_path / "config.properties"
    with open(config_path, "w") as f:
        config.write(f)

    return config_path, test_data_dir

@mock.patch("predict_chess_pieces.download_and_extract_if_missing")
@mock.patch("predict_chess_pieces.torch.load")
@mock.patch("predict_chess_pieces.ChessPieceModel")
@mock.patch("predict_chess_pieces.datasets.ImageFolder")
@mock.patch("predict_chess_pieces.DataLoader")
def test_main_flow(mock_loader, mock_folder, mock_model_cls, mock_torch_load, mock_download,
                   dummy_config, capsys):
    from predict_chess_pieces import main

    config_path, test_dir = dummy_config
    shutil.copy(config_path, "config.properties")

    # ✅ Properly mock the model and its return value
    dummy_model = mock.Mock()
    dummy_model.eval = mock.Mock()
    dummy_model.return_value = torch.tensor([[2.0, 1.0]])  # simulate output from model(image)

    # This will be returned by ChessPieceModel().model
    model_wrapper_mock = mock.Mock()
    model_wrapper_mock.model = dummy_model
    mock_model_cls.return_value = model_wrapper_mock

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
    assert "Knight" in captured.out or "Bishop" in captured.out
