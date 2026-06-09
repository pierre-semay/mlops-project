import os
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
import argparse
from librosa.feature import rms

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_folder', type=str)
args = parser.parse_args()

df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
df_test  = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))

# Helper functions
def pad_audio(y, target_length):
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), 'constant')
    return y

def truncate_audio(y, target_length):
    if len(y) > target_length:
        y = y[:target_length]
    return y

def adjust_audio(y, target_length=32000):
    if len(y) < target_length:
        return pad_audio(y, target_length)
    else:
        return truncate_audio(y, target_length)

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                  original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return desired_sample_rate, waveform

def segment_cough_sound(signal, sr, cough_threshold=0.05, min_cough_duration=0.1, padding=0.05):
    hop_length = int(min_cough_duration * sr)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    energy = rms(y=signal, hop_length=hop_length)[0]
    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    cough_threshold = np.max(normalized_energy) * cough_threshold
    min_cough_samples = round(sr * min_cough_duration)

    cough_segments = []
    event_start = None

    for i, value in enumerate(normalized_energy):
        if value >= cough_threshold:
            if event_start is None:
                event_start = i * hop_length
        else:
            if event_start is not None:
                cough_duration = i * hop_length - event_start
                if cough_duration >= min_cough_samples:
                    event_end = i * hop_length + int(padding * sr)
                    event_start -= int(padding * sr)
                    event_start = max(event_start, 0)
                    cough_segments.append(signal[event_start: event_end + 1])
                event_start = None

    return cough_segments

def process_cough_file(file_path, save_segments=True):
    new_path = file_path.replace(".wav", "_segmented")

    if os.path.exists(new_path) and os.listdir(new_path):
        print(f"Skipping (already segmented): {file_path}")
        return

    try:
        signal, sr = sf.read(file_path)
        if len(signal) == 0 or sr == 0:
            print(f"Skipping (invalid file): {file_path}")
            return

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        segments = segment_cough_sound(signal, sr)
        print(f"{file_path} → {len(segments)} segments found")

        if save_segments:
            os.makedirs(new_path, exist_ok=True)
            for i, seg in enumerate(segments):
                seg_path = os.path.join(new_path, f"seg{i}.wav")
                sf.write(seg_path, seg, sr)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_audio_file(file_path, target_length=32000, desired_sr=16000):
    new_path = file_path.replace(".wav", "_processed.wav")

    if os.path.exists(new_path):
        print(f"Skipping (already processed): {new_path}")
        return

    try:
        waveform, sr = sf.read(file_path)
        if len(waveform) == 0 or sr == 0:
            print(f"Skipping (invalid file): {file_path}")
            return

        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)

        sr, waveform = ensure_sample_rate(sr, waveform, desired_sr)
        waveform = adjust_audio(waveform, target_length)
        sf.write(new_path, waveform, sr)
        print(f"Saved: {new_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_dataset(root_dir):
    for user_id in os.listdir(root_dir):
        user_path = os.path.join(root_dir, user_id)
        if not os.path.isdir(user_path):
            continue
        file_path = os.path.join(user_path, "cough.wav")
        if os.path.exists(file_path):
            process_cough_file(file_path)
        else:
            print(f"Missing: {file_path}")

def process_dataset_segments(root_dir):
    for user_id in os.listdir(root_dir):
        user_path = os.path.join(root_dir, user_id)
        if not os.path.isdir(user_path):
            continue
        segmented_folder = os.path.join(user_path, "cough_segmented")
        if not os.path.exists(segmented_folder):
            print(f"No segmented folder: {segmented_folder}")
            continue
        for filename in os.listdir(segmented_folder):
            if filename.endswith(".wav") and not filename.endswith("_processed.wav"):
                process_audio_file(os.path.join(segmented_folder, filename))

# Run preprocessing
sounds_dir = os.path.join(args.data_dir, "sounds", "sounds")
process_dataset(sounds_dir)

process_dataset_segments(sounds_dir)

# Save processed data as a new data asset
import shutil

output_sounds = os.path.join(args.output_folder, "sounds", "sounds")
os.makedirs(output_sounds, exist_ok=True)

for user_id in os.listdir(sounds_dir):
    user_path = os.path.join(sounds_dir, user_id)
    seg_folder = os.path.join(user_path, "cough_segmented")
    if not os.path.exists(seg_folder):
        continue
    out_seg = os.path.join(output_sounds, user_id, "cough_segmented")
    os.makedirs(out_seg, exist_ok=True)
    for fname in os.listdir(seg_folder):
        if fname.endswith("_processed.wav"):
            shutil.copy(os.path.join(seg_folder, fname), out_seg)

shutil.copy(os.path.join(args.data_dir, "train.csv"), args.output_folder)
shutil.copy(os.path.join(args.data_dir, "test.csv"), args.output_folder)

print(f"Processed data saved to {args.output_folder}")