import os
import scipy.io
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class SpectrogramGenerator:
    def __init__(self, data_file, output_dir, sr=1000, n_fft=64, hop_length=32):
        self.data_file = data_file
        self.output_dir = output_dir
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.csi_data = self._load_data()
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_data(self):
        mat_data = scipy.io.loadmat(self.data_file)
        return mat_data['train_data']

    def generate_spectrograms(self):
        for i in range(self.csi_data.shape[0]):
            for j in range(self.csi_data.shape[1]):
                csi_data_packet = self.csi_data[i, j, :]
                stft_result = librosa.stft(csi_data_packet, n_fft=self.n_fft, hop_length=self.hop_length)
                spectrogram = np.abs(stft_result)
                spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

                # Save as image
                plt.figure(figsize=(8, 4))
                librosa.display.specshow(spectrogram_db, sr=self.sr, hop_length=self.hop_length, 
                                         x_axis='time', y_axis='hz')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram - Sample {i+1}, Subcarrier {j+1}')
                filename = f'spectrogram_sample{i+1}_subcarrier{j+1}.png'
                plt.savefig(os.path.join(self.output_dir, filename))
                plt.close()

        print(f"Spectrograms saved in {self.output_dir}")
