import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import scipy.signal
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import joblib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--output_folder', type=str)  
args = parser.parse_args()

df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
df_test  = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))

df_train = df_train.drop(df_train[df_train['candidateID'] == '5ee582f2832c2'].index)

df_train['coldPresent_missing'] = df_train['coldPresent'].isna().astype(int)
df_train['coldPresent'] = df_train['coldPresent'].fillna(0)

print(f"train samples: {len(df_train)}")
print(df_train['disease'].value_counts())

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet geladen.")

def extract_embeddings_sequence(waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()


META_COLS = ['age', 'gender', 'tbContactHistory', 'wheezingHistory',
             'phlegmCough', 'familyAsthmaHistory', 'feverHistory',
             'coldPresent', 'packYears', 'coldPresent_missing']

df_filled = df_train.fillna(0)

X_audio  = []  # tijdreeks embeddings    
X_meta = []  # metadata
y_list = []  # labels

skipped = 0

for _, row in df_filled.iterrows():
    cid     = row['candidateID']
    disease = row['disease']
    meta    = row[META_COLS].values.astype(np.float32)
    
    # --- Cough segmenten ---
    seg_folder = os.path.join(args.data_dir, "sounds/sounds", cid, "cough_segmented")
    if os.path.exists(seg_folder):
        for fname in os.listdir(seg_folder):
            if fname.endswith("_processed.wav"):
                fp = os.path.join(seg_folder, fname)
                try:
                    waveform, sr = librosa.load(fp, sr=16000)
                    emb_seq = extract_embeddings_sequence(waveform)
                    X_audio.append(emb_seq)
                    X_meta.append(meta)
                    y_list.append(disease)
                except Exception as e:
                    print(f"Fout bij {fp}: {e}")
                    skipped += 1

X_audio  = np.array(X_audio,  dtype=np.float32)   # (samples, MAX_FRAMES, 1024)
X_meta = np.array(X_meta, dtype=np.float32)   # (samples, n_meta)
y_arr  = np.array(y_list)

print(f"X_audio shape : {X_audio.shape}")
print(f"X_meta shape: {X_meta.shape}")
print(f"Labels shape: {y_arr.shape}")
print(f"Overgeslagen : {skipped}")

scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)


le = LabelEncoder()
y_encoded = le.fit_transform(y_arr)
num_classes = len(le.classes_)
print(f"Klassen: {le.classes_}  ({num_classes} klassen)")

# Hier splitsen we de data maar houden we onze x_audio en x_meta apart, zodat we ze later als aparte inputs kunnen gebruiken in het model.
split = train_test_split(
    X_audio,
    X_meta_scaled,
    y_encoded,
    test_size=0.2,
    random_state=1234,
    stratify=y_encoded
)

X_audio_train, X_audio_val, X_meta_train, X_meta_val, y_train, y_val = split

y_train_onehot = to_categorical(y_train, num_classes)
y_val_onehot   = to_categorical(y_val,   num_classes)

print(f"Training  : {len(X_audio_train)} samples")
print(f"Validatie : {len(X_audio_val)} samples")

def fit_model_LSTM(window_size_audio=4, window_size_meta=10):
    # audio tak
    audio_input = layers.Input(shape=(window_size_audio, 1024)) #onze audio files zijn steeds 4 frames van 1024 features (YAMNet embeddings)
    
    audio_features = layers.LSTM(128, return_sequences=True)(audio_input) 
    audio_features = layers.Dropout(0.3)(audio_features) 
    audio_features = layers.LSTM(64)(audio_features)
    audio_features = layers.Dropout(0.3)(audio_features)
    
    # metadata tak
    meta_input = layers.Input(shape=(window_size_meta,)) # Er zijn 10 metadata features
    
    meta_features = layers.Dense(32, activation='relu')(meta_input)
    meta_features = layers.Dropout(0.2)(meta_features)
    
    # beide takken combineren
    combined = layers.concatenate([audio_features, meta_features])
    out = layers.Dense(64, activation='relu')(combined)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(3, activation='softmax')(out) # We hebben 3 mogelijke resultaten.
    
    model = Model(inputs=[audio_input, meta_input], outputs=out)
    model.compile(
        optimizer="adam", 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # training
    early_stop = EarlyStopping(
        monitor='val_loss', #We monitoren hier op val_loss omdat dit precieser is.
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        [X_audio_train, X_meta_train], y_train_onehot,
        validation_data=([X_audio_val, X_meta_val], y_val_onehot),
        epochs= 60,
        batch_size= 16,
        callbacks=[early_stop],
        verbose=1
    )

    val_loss, val_acc = model.evaluate([X_audio_val, X_meta_val], y_val_onehot, verbose=0) 
    y_pred = np.argmax(model.predict([X_audio_val, X_meta_val], verbose=0), axis=1)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"-LSTM - Val Accuracy: {val_acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=le.classes_.astype(str)))
    
    return model, history, val_acc, macro_f1, y_pred

lstm_model, hist_lstm, acc_lstm, f1_lstm, pred_lstm = fit_model_LSTM()

os.makedirs(args.output_folder, exist_ok=True)
lstm_model.save(os.path.join(args.output_folder, "lstm_model.keras"))
joblib.dump(scaler, os.path.join(args.output_folder, "scaler.pkl"))
joblib.dump(le, os.path.join(args.output_folder, "label_encoder.pkl"))
print("Model saved.")

#hallo dit is een test
#hallo dit is ook een test
#wow zo toevallig, dit is ook een test!
#ik begin dit wel een beetje te veel te vinden.
