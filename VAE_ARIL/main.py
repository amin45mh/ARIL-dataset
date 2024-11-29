from spectrogram_generator import SpectrogramGenerator
from vae_architecture import VAE
from trainer import Trainer

# Step 1: Generate Spectrogram Data
spectrogram_generator = SpectrogramGenerator('train_data_split_amp.mat')
spectrogram_data = spectrogram_generator.generate_spectrograms()

# Step 2: Initialize VAE Model
vae_model = VAE(input_dim=spectrogram_data.shape[1] * spectrogram_data.shape[2], latent_dim=10)

# Step 3: Train the VAE
trainer = Trainer(vae_model, spectrogram_data, batch_size=16, learning_rate=0.001)
trainer.train(epochs=10)

# Step 4: Save the model
trainer.save_model('vae_model.pth')

# Step 5: Load the model later
new_vae_model = VAE(input_dim=spectrogram_data.shape[1] * spectrogram_data.shape[2], latent_dim=10)
Trainer.load_model('vae_model.pth', new_vae_model)
