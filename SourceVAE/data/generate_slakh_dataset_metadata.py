import numpy as np
import pandas as pd
import random as random
import os
import librosa
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def get_slakh_dataset_filenames(dataset_dir):
    filelists = {}
    for source_type in ['piano', 'guitar', 'bass', 'drums']:
        filelists[source_type] = {}
        for mode in ['train', 'validation', 'test']:
            path = os.path.join(dataset_dir, source_type+'_22050', mode)
            filelists[source_type][mode] = librosa.util.find_files(path, ext=["mp3", "opus", "m4a", "aac", "wav"])
    return filelists


def is_silent(signal: np.ndarray, silence_threshold: float = 1.5e-5) -> bool:
    num_samples = signal.shape[-1]
    return np.linalg.norm(signal) / num_samples < silence_threshold



def get_active_chunks(filelist, frame_seconds=4, hop_seconds=1, shift_augment=False, augment_size=100):
    # Prepare the dataframe to store results
    columns = ['filename', 'start', 'end']
    data = []

    # Process each file in the filelist
    for filename in tqdm(filelist):
        # Load the audio file
        signal, sr = librosa.load(filename, sr=None)
        frame_length = int(frame_seconds * sr)
        hop_length = int(hop_seconds * sr)

        # Iterate over the signal with a sliding window
        for start in range(0, len(signal) - frame_length + 1, hop_length):
            end = start + frame_length
            chunk = signal[start:end]

            # Check if the chunk is not silent
            if not is_silent(chunk):
                if shift_augment:
                    # Perform augmentation with random shifts within the range [-0.5 * hop_length, 0.5 * hop_length]
                    shifts = np.random.randint(-hop_length // 2, hop_length // 2 + 1, size=augment_size)
                    for shift in shifts:
                        new_start = max(0, start + shift)
                        new_end = new_start + frame_length
                        if new_end <= len(signal):
                            shifted_chunk = signal[new_start:new_end]
                            if not is_silent(shifted_chunk):
                                data.append([filename, new_start / sr, new_end / sr])
                else:
                    # No augmentation, just append the original chunk info
                    data.append([filename, start / sr, end / sr])

    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=columns)
    return df

def get_slakh_vae_dataset_config(
        slakh_dataset_dir, 
        save_dir, 
        mode='train', 
        source_types=['piano', 'drums', 'guitar', 'bass'], 
        frame_seconds=4, 
        hop_seconds=1, 
        shift_augment=False, 
        augment_size=100):
    # Step 1: Get the filenames for each source type and mode
    all_files = get_slakh_dataset_filenames(slakh_dataset_dir)

    # Prepare a DataFrame to hold all active chunks across all source types
    all_active_chunks = pd.DataFrame()

    # Step 2: Process each source type
    for source in source_types:
        # Get the file list for the current source type and mode
        file_list = all_files[source][mode]
        
        # Get active chunks for this file list
        active_chunks = get_active_chunks(file_list, frame_seconds, hop_seconds, shift_augment, augment_size)
        
        # Add a column to indicate the source type
        active_chunks['source_type'] = source
        
        # Append the results to the main DataFrame
        all_active_chunks = pd.concat([all_active_chunks, active_chunks], ignore_index=True)
    
    # Step 3: Save the concatenated DataFrame to a CSV file
    augment_status = 'augmented' if shift_augment else 'not_augmented'
    output_csv_path = os.path.join(save_dir, f'slakh_{mode}_fs{frame_seconds}_hs{hop_seconds}_{augment_status}_as{augment_size}_active_chunks.csv')
    all_active_chunks.to_csv(output_csv_path, index=False)
    
    # Return the path to the saved CSV file
    return output_csv_path


if __name__=='__main__':
    slakh_dataset_dir = '../../msldm/data'
    import argparse

    # Define the argument parser
    parser = argparse.ArgumentParser(description="MusicDacVAE Argument Parser")

    parser.add_argument('--mode', type=str, default='validation', choices=['validation', 'train', 'test'],
                        help='Mode for the dataset (validation, train, test).')

    # Parse the arguments
    args = parser.parse_args()
    mode = args.mode

    if mode=='train':
        frame_seconds=1
        hop_seconds=0.5
        shift_augment = True
    else:
        frame_seconds=5
        hop_seconds=1
        shift_augment = False

    get_slakh_vae_dataset_config(
        slakh_dataset_dir, 
        './metadata', 
        mode='train', 
        source_types=['piano', 'drums', 'guitar', 'bass'], 
        frame_seconds=frame_seconds, 
        hop_seconds=hop_seconds, 
        shift_augment=shift_augment, 
        augment_size=5)