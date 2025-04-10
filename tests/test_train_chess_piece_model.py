import pytest
import torch
from train_chess_piece_model import ChessTrainer
from chess_piece_model import ChessPieceModel

# Replace with your actual Google Drive link for testing
DRIVE_URL = "https://drive.google.com/uc?id=1o50VIu51M11jbHXe5LFSVDfuQ-VNiwoS"
CONFIG_PATH = "config.properties"


@pytest.fixture(scope="session")
def trainer():
    model = ChessPieceModel(drive_url=DRIVE_URL, config_path=CONFIG_PATH)
    return ChessTrainer(model)


def test_initialization(trainer):
    """Test initialization of the ChessTrainer"""
    assert trainer.model is not None
    assert trainer.criterion is not None
    assert trainer.optimizer is not None


def test_train_one_epoch(trainer):
    """Test that the train_one_epoch function returns a valid loss and accuracy"""
    train_loss, train_acc = trainer.train_one_epoch()
    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert 0.0 <= train_acc <= 100.0


def test_validate(trainer):
    """Test that the validate function returns a valid validation loss and accuracy"""
    val_loss, val_acc = trainer.validate()
    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert 0.0 <= val_acc <= 100.0
