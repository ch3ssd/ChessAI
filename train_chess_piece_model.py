import torch
import torch.nn as nn
import torch.optim as optim
from chess_piece_model import ChessPieceModel
import configparser


class ChessTrainer:
    def __init__(self, model_wrapper: ChessPieceModel, learning_rate: float = 0.001):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.train_loader = model_wrapper.train_loader
        self.val_loader = model_wrapper.val_loader
        self.device = model_wrapper.device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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

        print("Training complete!")

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


if __name__ == "__main__":
    CONFIG_PATH = "config.properties"
    EPOCHS = 10

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    DRIVE_URL = config["DATA"]["DriveURL"]

    model_wrapper = ChessPieceModel(drive_url=DRIVE_URL, config_path=CONFIG_PATH)
    trainer = ChessTrainer(model_wrapper)
    trainer.train(epochs=EPOCHS)
