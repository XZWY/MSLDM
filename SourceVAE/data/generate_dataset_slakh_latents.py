import abc
import functools
import itertools
import math
import os
import warnings
from abc import ABC
from pathlib import Path
from typing import *

import av
import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

def _identity(x):
    return x

def get_duration_sec(file, cache=False):
    try:
        with open(file + ".dur", "r") as f:
            duration = float(f.readline().strip("\n"))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + ".dur", "w") as f:
                f.write(str(duration) + "\n")
        return duration

def load_audio(file, sr, offset, duration, resample=True, approx=False, time_base="samples", check_duration=True):
    resampler = None
    if time_base == "sec":
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    if not os.path.exists(file):
        return np.zeros((2, duration), dtype=np.float32), sr
    container = av.open(file)
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    if approx:

        if offset + duration > audio_duration * sr:
            # Move back one window. Cap at audio_duration
            offset = min(audio_duration * sr - duration, offset - duration)
    else:
        if check_duration:
            assert (
                    offset + duration <= audio_duration * sr
            ), f"End {offset + duration} beyond duration {audio_duration*sr}"
    if resample:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sr)
    else:
        assert sr == audio.sample_rate
    offset = int(
        offset / sr / float(audio.time_base)
    )  # int(offset / float(audio.time_base)) # Use units of time_base for seeking
    duration = int(duration)  # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0):  # Only first audio stream
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)
        # print(frame[0], frame[0].to_ndarray(format="fltp").shape, len(frame), '-----------------------------------------------------------------------------')
        frame = frame[0].to_ndarray(format="fltp")  # Convert to floats and not int16
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read : total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f"Expected {duration} frames, got {total_read}"
    return sig, sr

def _identity(x):
    return x

class MultiSourceDataset(Dataset):
    def __init__(self, sr, channels, min_duration, max_duration, aug_shift, aug_size, sample_length, audio_files_dir, stems, transform=None):
        super().__init__()
        self.sr = sr
        self.channels = channels
        self.min_duration = min_duration or math.ceil(sample_length / sr) # 12
        self.max_duration = max_duration or math.inf # 640
        self.sample_length = sample_length # 262144
        self.audio_files_dir = audio_files_dir # /data2/romit/alan/multi-source-diffusion-models/data/slakh2100/train
        self.stems = stems
        assert (
                sample_length / sr < self.min_duration
        ), f"Sample length {sample_length} per sr {sr} ({sample_length / sr:.2f}) should be shorter than min duration {self.min_duration}"
        self.aug_shift = aug_shift
        self.aug_size = aug_size
        self.sub_interval_length = int(self.sample_length // aug_size)
        self.transform = transform if transform is not None else _identity
        self.init_dataset()

    def filter(self, tracks):
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(self.audio_files_dir, track)
            files = librosa.util.find_files(f"{track_dir}", ext=["mp3", "opus", "m4a", "aac", "wav"])
            
            # skip if there are no sources per track
            if not files:
                continue
            
            # 4 corresponding to 4 tracks
            durations_track = np.array([get_duration_sec(file, cache=True) * self.sr for file in files]) # Could be approximate
            
            # skip if there is a source that is shorter than minimum track length
            if (durations_track / self.sr < self.min_duration).any():
                continue
            
            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.sr >= self.max_duration).any():
                continue
            
            # skip if in the track the different sources have different lengths
            if not (durations_track == durations_track[0]).all():
                print(f"{track} skipped because sources are not aligned!")
                print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])
        
        print(f"self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        self.tracks = keep
        self.durations = durations
        self.cumsum = np.cumsum(np.array(self.durations))

    def init_dataset(self):
        # Load list of tracks and starts/durations
        tracks = os.listdir(self.audio_files_dir)
        print(f"Found {len(tracks)} tracks.")
        self.filter(tracks)

    def get_index_offset(self, item, shift_idx=0):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length // 2
        # shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
        shift = int(shift_idx * self.sub_interval_length) - half_interval
        # shift = 0
        offset = item * self.sample_length + shift  # Note we centred shifts, so adding now 
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f"Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}"
        
        index = np.searchsorted(self.cumsum, midpoint)  # index <-> midpoint of interval lies in this song
        start, end = self.cumsum[index - 1] if index > 0 else 0.0, self.cumsum[index]  # start and end of current song
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        
        if offset > end - self.sample_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start:  # Going under song
            offset = min(end - self.sample_length, offset + half_interval)  # Now should fit
        assert (
                start <= offset <= end - self.sample_length
        ), f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        
        offset = offset - start
        # print('sssss', item, shift_idx, shift, offset)
        return index, offset

    def get_song_chunk(self, index, offset):
        track_name, total_length = self.tracks[index], self.durations[index]
        data_list = []
        for stem in self.stems:
            data, sr = load_audio(os.path.join(self.audio_files_dir, track_name, f'{stem}.wav'),
                                  sr=self.sr, offset=offset, duration=self.sample_length, approx=True)
            data = 0.5 * data[0:1, :] + 0.5 * data[1:, :]
            assert data.shape == (
                self.channels,
                self.sample_length,
            ), f"Expected {(self.channels, self.sample_length)}, got {data.shape}"
            data_list.append(data)
        return np.concatenate(data_list, axis=0) # 4, n_samples

    def get_item(self, item):
        if self.aug_shift:
            index, offset = self.get_index_offset(int(item/self.aug_size), item%self.aug_size)
        else:
            index, offset = self.get_index_offset(item) # original index, 
        # print('final', item, index, offset)
        wav = self.get_song_chunk(index, offset)
        return self.transform(torch.from_numpy(wav)), item, index, offset

    def __len__(self):
        if self.aug_shift:
            return int(np.floor(self.cumsum[-1] / self.sample_length))*self.aug_size
        else:
            return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)

import sys
import torch
# sys.path.append('/data2/romit/alan/MusicDacVAE')

from models.model.dac_vae import DACVAE

import soundfile as sf
from tqdm import tqdm

import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="SourceVAE Argument Parser")

# Add arguments to the parser
parser.add_argument('--ckpt_path', type=str,
                    help='Path to the checkpoint file.')
parser.add_argument('--save_dir', type=str,
                    help='Directory to save the results.')
parser.add_argument('--mode', type=str, default='validation', choices=['validation', 'train', 'test'],
                    help='Mode for the dataset (validation, train, test).')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation.')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for the dataloader.')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for data loading.')

# Parse the arguments
args = parser.parse_args()

# Load all the values from the parser
ckpt_path = args.ckpt_path
save_dir = args.save_dir
mode = args.mode
device = args.device
batch_size = args.batch_size
n_workers = args.n_workers

save_dir = os.path.join(save_dir, mode)
# Print values (optional, just for verification)
print(f"Checkpoint Path: {ckpt_path}")
print(f"Save Directory: {save_dir}")
print(f"Mode: {mode}")
print(f"Device: {device}")
print(f"Batch Size: {batch_size}")
print(f"Number of Workers: {n_workers}")


os.makedirs(save_dir, exist_ok=True)

# instantiate model
vae = DACVAE(
    encoder_dim = 64,
    encoder_rates = [2, 4, 5, 8],
    latent_dim = 80,
    decoder_dim = 1536,
    decoder_rates = [8, 5, 4, 2],
    sample_rate = 22050).to(device)

# load checkpoints
model_ckpt = torch.load(ckpt_path, map_location=device)
vae.load_state_dict(model_ckpt['generator'])
vae.eval()
print('finish loading ckpts from: ', ckpt_path)


# init dataset
aug_shift = (mode=='train') # only augment the dataset for training set
dataset = MultiSourceDataset(
    sr=22050, 
    channels=1, 
    min_duration=15.0, 
    max_duration=640., 
    aug_shift=False,
    aug_size=5,
    sample_length=327672, 
    audio_files_dir=os.path.join('../../msldm/data/slakh2100', mode)
    # audio_files_dir='/data2/romit/alan/multi-source-diffusion-models/data/slakh2100/' + mode,
    stems=['bass', 'drums', 'guitar', 'piano'], 
    transform=None)

print(len(dataset))

# dataset = torch.utils.data.Subset(
#     dataset, range(106511, len(dataset))
#     )

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=n_workers,
    shuffle=False,
    batch_size=batch_size,
)

# generating dataset
with torch.no_grad():
    for i, batch in tqdm(enumerate(dataloader)):
        audio, item, index, offset = batch

        # audio = audio.reshape(batch_size*4, -1).unsqueeze(1).to(device)
        # latents = vae.encode(audio).mode() # bs,*4, n_latent, n_frames
        # n_latents, n_frames = latents.shape[-2], latents.shape[-1]
        # latents = latents.reshape(batch_size, 4, n_latents, n_frames).cpu().numpy()

        bs = audio.shape[0]
        audio = audio.sum(1).unsqueeze(1).to(device)
        latents = vae.encode(audio).mode() # bs, n_latent, n_frames
        n_latents, n_frames = latents.shape[-2], latents.shape[-1]
        latents = latents.reshape(bs, n_latents, n_frames).cpu().numpy()        
        for j in range(batch_size):
            curr_item = item[j]
            curr_index = index[j]
            curr_offset = offset[j]

            filename = f'idx{curr_item}_track{curr_index}_offset{curr_offset}.npy'
            save_path = os.path.join(save_dir, filename)
            if latents[j].shape[-1] != 1024:
                continue
            np.save(save_path, latents[j])
            print(filename, ' saved successfully')


