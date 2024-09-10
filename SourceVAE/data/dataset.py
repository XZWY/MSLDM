import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset
import torch

def adjust_rms(audio, target_rms):
    current_rms = torch.sqrt(torch.mean(audio**2))
    return audio * (target_rms / current_rms)

def calculate_snr(signal, noise):
    signal_power = torch.mean(signal**2)
    noise_power = torch.mean(noise**2)
    if noise_power == 0:
        return float('inf')  # To handle cases where noise might be zero
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

class dataset_slakh2100(Dataset):
    def __init__(self, meta_data_path, sample_rate, segment_length, shuffle=True):
        self.data = pd.read_csv(meta_data_path)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.sample_rate = sample_rate
        self.segment_length = segment_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]


        start_sample = int(row['start'] * self.sample_rate)
        end_sample = int(row['end'] * self.sample_rate)
        num_samples = end_sample - start_sample

        # Load the audio segment directly
        audio, _ = torchaudio.load(
            row['filename'],
            frame_offset=start_sample,
            num_frames=num_samples,
            # normalize=False
        )
        # Ensure audio is mono and <= segment length
        if len(audio.shape) > 1:
            audio = audio[0, :self.segment_length]
        else:
            audio = audio[:self.segment_length]

        # Handle cases where the audio segment is shorter than needed
        if audio.shape[0] < self.segment_length:
            padding = self.segment_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))

        if audio.isnan().any():
            return self.__getitem__((idx+1)%self.__len__())

        return audio


if __name__=='__main__':
    # Usage example
    dataset = dataset_slakh2100(
        meta_data_path='/data2/romit/alan/MusicDacVAE/data/metadata/slakh_validation_fs1_hs0.5_not_augmented_as0_active_chunks.csv', 
        sample_rate=22050, 
        segment_length=22050)
    data = dataset[3]
    print(data.dtype)

    # batched_data = audio_collate_fn([dataset[3], dataset[4]])
    # for key in batched_data.keys():
    #     if type(batched_data[key]) is torch.Tensor:
    #         print(key, batched_data[key].shape)
    #     else:
    #         print(key, batched_data[key])

    # print(data)