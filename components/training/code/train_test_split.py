import os
import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_folder', type=str)

args = parser.parse_args()

df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))

df_train = df_train.drop(
    df_train[df_train['candidateID'] == '5ee582f2832c2'].index
)

df_train['coldPresent_missing'] = (
    df_train['coldPresent'].isna().astype(int)
)

df_train['coldPresent'] = (
    df_train['coldPresent'].fillna(0)
)

print(f"train samples: {len(df_train)}")

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_embeddings_sequence(waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

META_COLS = [
    'age',
    'gender',
    'tbContactHistory',
    'wheezingHistory',
    'phlegmCough',
    'familyAsthmaHistory',
    'feverHistory',
    'coldPresent',
    'packYears',
    'coldPresent_missing'
]

df_filled = df_train.fillna(0)

X_audio = []
X_meta = []
y_list = []

skipped = 0

for _, row in df_filled.iterrows():

    cid = row['candidateID']
    disease = row['disease']
    meta = row[META_COLS].values.astype(np.float32)

    seg_folder = os.path.join(
        args.data_dir,
        "sounds/sounds",
        cid,
        "cough_segmented"
    )

    if os.path.exists(seg_folder):

        for fname in os.listdir(seg_folder):

            if fname.endswith("_processed.wav"):

                fp = os.path.join(seg_folder, fname)

                try:
                    waveform, sr = librosa.load(
                        fp,
                        sr=16000
                    )

                    emb_seq = extract_embeddings_sequence(
                        waveform
                    )

                    X_audio.append(emb_seq)
                    X_meta.append(meta)
                    y_list.append(disease)

                except Exception as e:
                    print(f"Error in {fp}: {e}")
                    skipped += 1

X_audio = np.array(X_audio, dtype=np.float32)
X_meta = np.array(X_meta, dtype=np.float32)
y_arr = np.array(y_list)

print(f"X_audio shape : {X_audio.shape}")
print(f"X_meta shape : {X_meta.shape}")
print(f"Labels shape : {y_arr.shape}")

scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)

le = LabelEncoder()
y_encoded = le.fit_transform(y_arr)

split = train_test_split(
    X_audio,
    X_meta_scaled,
    y_encoded,
    test_size=0.2,
    random_state=1234,
    stratify=y_encoded
)

(
    X_audio_train,
    X_audio_val,
    X_meta_train,
    X_meta_val,
    y_train,
    y_val
) = split

os.makedirs(args.output_folder, exist_ok=True)

np.save(os.path.join(args.output_folder, "X_audio_train.npy"), X_audio_train)
np.save(os.path.join(args.output_folder, "X_audio_val.npy"), X_audio_val)

np.save(os.path.join(args.output_folder, "X_meta_train.npy"), X_meta_train)
np.save(os.path.join(args.output_folder, "X_meta_val.npy"), X_meta_val)

np.save(os.path.join(args.output_folder, "y_train.npy"), y_train)
np.save(os.path.join(args.output_folder, "y_val.npy"), y_val)

joblib.dump(
    scaler,
    os.path.join(args.output_folder, "scaler.pkl")
)

joblib.dump(
    le,
    os.path.join(args.output_folder, "label_encoder.pkl")
)

print("Train/validation split saved.")

#test 3
#test 5
#test 7
#test 9