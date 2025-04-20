import torch
import torch.nn as nn
import torch.optim as optim
from chess_piece_model import ChessPieceModel
import configparser
from pathlib import Path
import os


class ChessTrainer:
    def __init__(self, model_wrapper: ChessPieceModel, config_path: str, learning_rate: float = 0.001):
        self.config = self.load_config(config_path)
        self.save_path = Path(self.config["MODEL"]["SavePath"])
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.train_loader = model_wrapper.train_loader
        self.val_loader = model_wrapper.val_loader
        self.device = model_wrapper.device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_config(self, config_path: str):
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def train(self, epochs: int):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(self.train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.validate()

        self.save_model()
        print(f"Training complete. Model saved to {self.save_path.resolve()}")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy

    def save_model(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)


def should_skip_training(model_path: Path):
    return model_path.exists() and model_path.is_file()


if __name__ == "__main__":
    CONFIG_PATH = "config.properties"
    EPOCHS = 10

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    drive_url = config["DATA"]["DriveURL"]
    model_path = Path(config["MODEL"]["SavePath"])

    if should_skip_training(model_path):
        print(f"Model already exists at {model_path}. Skipping training.")
    else:
        model_wrapper = ChessPieceModel(drive_url=drive_url, config_path=CONFIG_PATH)
        trainer = ChessTrainer(model_wrapper, config_path=CONFIG_PATH)
        trainer.train(epochs=EPOCHS)
