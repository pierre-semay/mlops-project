import os
import numpy as np
import joblib
import argparse

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model

from sklearn.metrics import (
    classification_report,
    f1_score
)

parser = argparse.ArgumentParser()

parser.add_argument('--split_data', type=str)
parser.add_argument('--output_folder', type=str)

args = parser.parse_args()

X_audio_train = np.load(os.path.join(args.split_data, "X_audio_train.npy"))

X_audio_val = np.load(os.path.join(args.split_data, "X_audio_val.npy"))

X_meta_train = np.load(os.path.join(args.split_data, "X_meta_train.npy"))

X_meta_val = np.load(os.path.join(args.split_data, "X_meta_val.npy"))

y_train = np.load(os.path.join(args.split_data, "y_train.npy"))

y_val = np.load(os.path.join(args.split_data, "y_val.npy"))

le = joblib.load(os.path.join(args.split_data, "label_encoder.pkl"))

num_classes = len(le.classes_)

y_train_onehot = to_categorical(y_train, num_classes)

y_val_onehot = to_categorical(y_val, num_classes)

def fit_model_LSTM(window_size_audio=4, window_size_meta=10):

    audio_input = layers.Input(shape=(window_size_audio, 1024))

    audio_features = layers.LSTM(128, return_sequences=True)(audio_input)
    audio_features = layers.Dropout(0.3)(audio_features)
    audio_features = layers.LSTM(64)(audio_features)
    audio_features = layers.Dropout(0.3)(audio_features)

    meta_input = layers.Input(shape=(window_size_meta,))

    meta_features = layers.Dense(32, activation='relu')(meta_input)
    meta_features = layers.Dropout(0.2)(meta_features)

    combined = layers.concatenate([audio_features, meta_features])
    out = layers.Dense(64,activation='relu')(combined)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(num_classes, activation='softmax')(out)

    model = Model(inputs=[audio_input, meta_input], outputs=out)

    model.compile(
        optimizer="adam",
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        [X_audio_train, X_meta_train], y_train_onehot,
        validation_data=([X_audio_val, X_meta_val], y_val_onehot),
        epochs=60,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    val_loss, val_acc = model.evaluate([X_audio_val, X_meta_val], y_val_onehot, verbose=0)
    y_pred = np.argmax(model.predict([X_audio_val, X_meta_val], verbose=0), axis=1)
    macro_f1 = f1_score( y_val, y_pred, average='macro')

    print(classification_report(y_val, y_pred, target_names=le.classes_.astype(str)))
    
    return model, history, val_acc, macro_f1, y_pred #TODO is dat hier allemaal nodig

model, history, val_acc, macro_f1, y_pred = fit_model_LSTM()

os.makedirs(args.output_folder, exist_ok=True)
model.save(os.path.join(args.output_folder, "lstm_model.keras"))
print("Model saved.")

#please wees juist

