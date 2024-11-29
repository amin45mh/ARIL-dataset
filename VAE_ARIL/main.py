from spectrogram_generator import SpectrogramGenerator
from vae_architecture import VAE
from trainer import Trainer
import torch

# Generate Spectrogram Data
spectrogram_generator = SpectrogramGenerator('data/train_data_split_amp.mat')
spectrogram_data = spectrogram_generator.generate_spectrograms()

# Limit the data for testing purposes
limited_spectrogram_data = spectrogram_data[:10]  # Use only the first 10 samples

flattened_spectrograms = [tensor.flatten() for tensor in spectrogram_data]  # flatten
spectrogram_tensor = torch.stack(flattened_spectrograms)

# Initialize VAE Model
vae_model = VAE(input_dim=spectrogram_tensor.shape[1], latent_dim=10)

# Train the VAE
trainer = Trainer(vae_model, spectrogram_data, batch_size=32, learning_rate=0.001)
trainer.train(epochs=10)    # change the epochs

# Save the model
trainer.save_model('vae_model.pth')

# Load the model later
# Trainer.load_model('vae_model.pth')
