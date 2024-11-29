import torch
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    def __init__(self, model, spectrogram_data, batch_size=32, learning_rate=0.001):
        self.model = model
        self.dataloader = self._prepare_data(spectrogram_data, batch_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def _prepare_data(self, spectrogram_data, batch_size):
        flattened_data = spectrogram_data.reshape(spectrogram_data.shape[0], -1)
        dataset = TensorDataset(torch.tensor(flattened_data, dtype=torch.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                loss = self.model.training_step(batch[0])
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    def save_model(self, file_path):
        """
        Save the model and optimizer state to a file.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, file_path)
        print(f"Model and optimizer saved to {file_path}")

    def load_model(file_path, model, optimizer=None):
        """
        Load the model and optimizer state from a file.
        """
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model and optimizer loaded from {file_path}")
        return model, optimizer
