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
    def __init__(self, sr, channels, min_duration, max_duration, aug_shift, sample_length, audio_files_dir, stems, transform=None):
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

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.aug_shift else 0
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
        index, offset = self.get_index_offset(item)
        wav = self.get_song_chunk(index, offset)
        return self.transform(torch.from_numpy(wav))

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)

# class MultiSourceLatentDataset(Dataset):
#     def __init__(self, latent_files_dir):
#         self.latent_files_dir = latent_files_dir

#         # List all .npy files in the directory
#         self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
#         # Sort the files by the idx (numeric part at the beginning of the filename)
#         self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

#         self.large_indices = [ 1,  3,  5,  6,  7,  8,  9, 10, 12, 19, 20, 25, 27, 28, 30, 31, 35, 40, 41, 47, 48, 52, 60, 61, 68, 74, 77]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         if idx >= 106518 or idx==4428:
#             return self.__getitem__(0)

#         # Get the file name corresponding to the index
#         file_name = self.files[idx]
#         # Load the .npy file
#         file_path = os.path.join(self.latent_files_dir, file_name)
#         data = np.load(file_path)
        
#         # Convert to PyTorch tensor
#         data_tensor = torch.tensor(data)[:, self.large_indices, :]
#         n_latent = data_tensor.shape[1]
#         data_tensor = data_tensor.reshape(4*n_latent, -1)
        
#         return data_tensor # 4, 80, 1024
    
# class MultiSourceLatentDataset(Dataset):
#     def __init__(self, latent_files_dir):
#         self.latent_files_dir = latent_files_dir

#         # List all .npy files in the directory
#         self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
#         # Sort the files by the idx (numeric part at the beginning of the filename)
#         self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

#         self.large_indices = [ 1,  3,  5,  6,  7,  8,  9, 10, 12, 19, 20, 25, 27, 28, 30, 31, 35, 40, 41, 47, 48, 52, 60, 61, 68, 74, 77]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         if idx >= 106518:
#             return self.__getitem__(0)

#         # Get the file name corresponding to the index
#         file_name = self.files[idx]
#         # Load the .npy file
#         file_path = os.path.join(self.latent_files_dir, file_name)
#         data = np.load(file_path)
        
#         # Convert to PyTorch tensor
#         data_tensor = torch.tensor(data)#[:, self.large_indices, :]
#         n_latent = data_tensor.shape[1]
#         data_tensor = data_tensor.reshape(4*n_latent, -1)
        
#         return data_tensor # 4, 80, 1024

class MultiSourceLatentDatasetFiltered(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

        active_train = np.load('/data2/romit/alan/MusicDacVAE/data/active_train.npy')
        mask = list(active_train[:,2:].sum(1) == 2)
        self.files = [file for file, m in zip(self.files, mask) if m]

        print(self.files[:10])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # if idx >= 106518 or idx==4428:
        #     return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)
        n_latent = data_tensor.shape[1] # 4, 27, 1024

        data_tensor = data_tensor.reshape(4*n_latent, -1)

        if data_tensor.shape[-1] != 1024:
            # print(idx)
            return self.__getitem__(0)

        return data_tensor # 4, 80, 1024

class MultiSourceLatentDatasetSingleLatent(Dataset):
    def __init__(self, latent_files_dir, stem='bass'):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

        self.stem_idx = -1
        if stem == 'bass':
            self.stem_idx = 0
        elif stem == 'drums':
            self.stem_idx = 1
        elif stem == 'guitar':
            self.stem_idx = 2
        elif stem == 'piano':
            self.stem_idx = 3
        else:
            pass
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)
        n_latent = data_tensor.shape[1] # 4, 27, 1024

        return data_tensor[self.stem_idx]
        data_tensor = data_tensor.reshape(4*n_latent, -1)

        return data_tensor # 4, 80, 1024

class MultiSourceLatentDatasetMix(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)
        # n_latent = data_tensor.shape[1] # 4, 27, 1024
        if data_tensor.shape[-1] != 1024:
            # print(idx)
            return self.__getitem__(0)
        # data_tensor = data_tensor.reshape(4*n_latent, -1)

        return data_tensor # 4, 80, 1024

class MultiSourceLatentDatasetOld(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)
        n_latent = data_tensor.shape[1] # 4, 27, 1024

        data_tensor = data_tensor.reshape(4*n_latent, -1)

        return data_tensor # 4, 80, 1024

class MultiSourceLatentDataset(Dataset):
    def __init__(self, latent_files_dir, normalize=False, sigma=0.4):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

        self.normalize = normalize
        self.sigma = sigma

        self.large_indices = [ 1,  2,  3,  5,  6,  7,  8,  9, 10, 12, 19, 20, 23, 25, 27, 28, 30, 31, 32, 35, 40, 41, 47, 48, 52, 60, 61, 68, 71, 74, 76, 77]
        self.stds = torch.tensor([0.5378, 0.2285, 0.6424, 0.6244, 0.6127, 0.4941, 0.6091, 0.5833, 0.4497,
        0.5987, 0.5537, 0.4010, 0.2263, 0.3802, 0.6368, 0.4803, 0.5001, 0.5807,
        0.1905, 0.6330, 0.5768, 0.3699, 0.7884, 0.5132, 0.5958, 0.6393, 0.3409,
        0.3833, 0.1764, 0.6393, 0.1912, 0.4383]).unsqueeze(0).unsqueeze(-1) # 1, 27, 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)[:, self.large_indices, :]
        n_latent = data_tensor.shape[1] # 4, 27, 1024

        if self.normalize:
            data_tensor = data_tensor / self.stds * self.sigma

        data_tensor = data_tensor.reshape(4*n_latent, -1)

        
        return data_tensor # 4, 80, 1024
        

class MultiSourceLatentDatasetWaveNorm(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)

        n_latent = data_tensor.shape[1]
        data_tensor = data_tensor.reshape(4*n_latent, -1)

        data_tensor = data_tensor / 0.4257 * 0.0265
        
        return data_tensor # 4, 80, 1024
    
class MultiSourceLatentDatasetBatch(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data)
        n_latent = data_tensor.shape[1]
        data_tensor = data_tensor.reshape(4*n_latent, -1)
        
        return data_tensor # 4, 80, 1024

class SingleSourceLatentDataset(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)

        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data) # 4, n_latent, -1
        n_latent = data_tensor.shape[1]
        data_tensor = data_tensor[2]#.unsqueeze(0)
        
        return data_tensor # 4, 80, 1024

class SingleSourceLatentDatasetTiny(Dataset):
    def __init__(self, latent_files_dir):
        self.latent_files_dir = latent_files_dir

        # List all .npy files in the directory
        self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
        # Sort the files by the idx (numeric part at the beginning of the filename)
        self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))
        self.files = self.files[5000:5000+16]

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        if idx >= 106518 or idx==4428:
            return self.__getitem__(0)
        # Get the file name corresponding to the index
        file_name = self.files[idx]
        # Load the .npy file
        file_path = os.path.join(self.latent_files_dir, file_name)
        data = np.load(file_path)
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data) # 4, n_latent, -1
        n_latent = data_tensor.shape[1]
        data_tensor = data_tensor[2]#.unsqueeze(0)
        
        return data_tensor # 4, 80, 1024

# class SingleSourceLatentDatasetTiny(Dataset):
#     def __init__(self, latent_files_dir):
#         self.latent_files_dir = latent_files_dir

#         # List all .npy files in the directory
#         self.files = [f for f in os.listdir(latent_files_dir) if f.endswith('.npy')]
        
#         # Sort the files by the idx (numeric part at the beginning of the filename)
#         self.files.sort(key=lambda x: int(x.split('_')[0].replace('idx', '')))
#         self.files = self.files[:16]

#     def __len__(self):
#         return 16

#     def __getitem__(self, idx):
#         if idx >= 106518:
#             return self.__getitem__(0)

#         # Get the file name corresponding to the index
#         file_name = self.files[idx]
#         # Load the .npy file
#         file_path = os.path.join(self.latent_files_dir, file_name)
#         data = np.load(file_path)
        
#         # Convert to PyTorch tensor
#         data_tensor = torch.tensor(data) # 4, n_latent, -1
#         n_latent = data_tensor.shape[1]
#         data_tensor = data_tensor[3]#.unsqueeze(0)
#         # data_tensor = data_tensor# * 0.0556 + 0.0201
        
#         return data_tensor # 4, 80, 1024

# Datasets for evaluation --------------------------------------------------------------

def load_audio_tracks(paths: List[Union[str, Path]], sample_rate: int) -> Tuple[torch.Tensor, ...]:
    signals, sample_rates = zip(*[torchaudio.load(path) for path in paths])
    for sr in sample_rates:
        assert sr == sample_rate, f"sample rate {sr} is different from target sample rate {sample_rate}"
    return tuple(signals)


def assert_is_audio(*signal: torch.Tensor):
    for s in signal:
        assert len(s.shape) == 2
        assert s.shape[0] == 1 or s.shape[0] == 2


def is_silent(signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
    assert_is_audio(signal)
    num_samples = signal.shape[-1]
    return torch.linalg.norm(signal) / num_samples < silence_threshold


def is_multi_source(signal: torch.Tensor, silence_threshold: float = 1.5e-5) -> bool:
    num_silent_signals = 0
    for source in signal:
        if is_silent(source.unsqueeze(0), silence_threshold):
            num_silent_signals += 1
        if num_silent_signals > 2:
            return False
    return True  
    

def get_nonsilent_and_multi_instr_chunks(
    separated_track: Tuple[torch.Tensor],
    max_chunk_size: int,
    min_chunk_size: int,
    silence_threshold: Union[float,None],
    keep_only_multisource: bool ,
):
    for source in separated_track:
        assert_is_audio(source)
    
    separated_track = torch.cat(separated_track)
    _, num_samples = separated_track.shape
    num_chunks = num_samples // max_chunk_size + int(num_samples % max_chunk_size != 0)

    available_chunks = []
    for i in range(num_chunks):
        chunk = separated_track[:, i * max_chunk_size : (i + 1) * max_chunk_size]
        _, chunk_samples = chunk.shape

        # Remove if silent
        if silence_threshold is not None and is_silent(chunk.sum(0, keepdims=True), silence_threshold):
            continue
        
        # Remove if it contains only one source
        if keep_only_multisource and not is_multi_source(chunk):
            continue
        
        # Remove if it contains less than the minimum chunk size
        if chunk_samples < min_chunk_size:
            continue
        
        available_chunks.append(i)
    return available_chunks


class SeparationDataset(Dataset, ABC):
    @abc.abstractmethod
    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        ...


class SupervisedDataset(SeparationDataset):
    def __init__(
        self,
        audio_dir: Union[str, Path],
        stems: List[str],
        sample_rate: int,
        sample_eps_in_sec: int = 0.1
    ):
        super().__init__()
        self.sr = sample_rate
        self.sample_eps = round(sample_eps_in_sec * sample_rate)

        # Load list of files and starts/durations
        self.audio_dir = Path(audio_dir)
        self.tracks = sorted(os.listdir(self.audio_dir))
        self.stems = stems
        
        #TODO: add check if stem is never present in any track

    def __len__(self):
        return len(self.filenames)

    @functools.lru_cache(1)
    def get_tracks(self, track: str) -> Tuple[torch.Tensor, ...]:
        assert track in self.tracks
        stem_paths = {stem: self.audio_dir / track / f"{stem}.wav" for stem in self.stems}
        stem_paths = {stem: stem_path for stem, stem_path in stem_paths.items() if stem_path.exists()}
        assert len(stem_paths) >= 1, track
        
        stems_tracks = {}
        for stem, stem_path in stem_paths.items():
            audio_track, sr = torchaudio.load(stem_path)
            assert sr == self.sample_rate, f"sample rate {sr} is different from target sample rate {self.sample_rate}"
            stems_tracks[stem] = audio_track
                        
        channels, samples = zip(*[t.shape for t in stems_tracks.values()])
        
        for s1, s2 in itertools.product(samples, samples):
            assert abs(s1 - s2) <= self.sample_eps, f"{track}: {abs(s1 - s2)}"
            if s1 != s2:
                warnings.warn(
                    f"The tracks with name {track} have a different number of samples ({s1}, {s2})"
                )

        n_samples = min(samples)
        n_channels = channels[0]
        stems_tracks = {s:t[:, :n_samples] for s,t in stems_tracks.items()}
        
        for stem in self.stems:
            if not stem in stems_tracks:
                stems_tracks[stem] = torch.zeros(n_channels, n_samples)
        
        return tuple([stems_tracks[stem] for stem in self.stems])

    @property
    def sample_rate(self) -> int:
        return self.sr

    def __getitem__(self, item):
        return self.get_tracks(self.tracks[item])


class ChunkedSupervisedDataset(SupervisedDataset):
    def __init__(
        self,
        audio_dir: Union[Path, str],
        stems: List[str],
        sample_rate: int,
        max_chunk_size: int,
        min_chunk_size: int,
        silence_threshold: Optional[float]= None,
        only_multisource: bool = False,
    ):
        super().__init__(audio_dir=audio_dir, stems=stems, sample_rate=sample_rate)

        self.max_chunk_size ,self.min_chunk_size= max_chunk_size, min_chunk_size
        self.available_chunk = {}
        self.index_to_track, self.index_to_chunk = [], []
        self.silence_threshold = silence_threshold
        self.only_multisource = only_multisource

    
        #with mp.Pool() as pool:
        for track in self.tracks:
            _, available_chunks = self._get_available_chunks(track)
            self.available_chunk[track] = available_chunks
            self.index_to_track.extend([track] * len(available_chunks))
            self.index_to_chunk.extend(available_chunks)

        assert len(self.index_to_chunk) == len(self.index_to_track)

    def __len__(self):
        return len(self.index_to_track)

    def get_chunk_track(self, item: int) -> str:
        return self.index_to_track[item]

    def get_chunk_indices(self, item: int) -> Tuple[int, int]:
        ci = self.index_to_chunk[item]
        return ci * self.max_chunk_size, (ci + 1) * self.max_chunk_size

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        chunk_start, chunk_stop = self.get_chunk_indices(item)
        tracks = self.get_tracks(self.get_chunk_track(item))
        tracks = tuple([t[:, chunk_start:chunk_stop] for t in tracks])
        return tracks
    
    def _get_available_chunks(self, track: str):
        tracks = self.get_tracks(track) # (num_stems, [1, num_samples])
        available_chunks = get_nonsilent_and_multi_instr_chunks(
            tracks, 
            self.max_chunk_size, 
            self.min_chunk_size,
            self.silence_threshold,
            self.only_multisource,
            )
        return track, available_chunks 