import pytest
import torch
from train_chess_piece_model import ChessTrainer
from chess_piece_model import ChessPieceModel
import configparser
import os

CONFIG_PATH = "config.properties"

@pytest.fixture(scope="session")
def trainer():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    drive_url = config["DATA"]["DriveURL"]
    model = ChessPieceModel(drive_url=drive_url, config_path=CONFIG_PATH)
    return ChessTrainer(model)


def test_initialization(trainer):
    """Test initialization of the ChessTrainer"""
    assert trainer.model is not None
    assert trainer.criterion is not None
    assert trainer.optimizer is not None
    assert trainer.device is not None
    assert trainer.train_loader is not None
    assert trainer.val_loader is not None


def test_train_one_epoch(trainer):
    """Test a single training epoch"""
    train_loss, train_acc = trainer.train_one_epoch()
    assert isinstance(train_loss, float)
    assert isinstance(train_acc, float)
    assert 0.0 <= train_acc <= 100.0


def test_validate(trainer):
    """Test the validation method"""
    val_loss, val_acc = trainer.validate()
    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert 0.0 <= val_acc <= 100.0


def test_train_full_run(trainer):
    """Test the train method end-to-end with a small number of epochs"""
    trainer.train(epochs=1)  # Run one epoch just to hit the method
