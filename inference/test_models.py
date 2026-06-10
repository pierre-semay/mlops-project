import os
import numpy as np
import joblib
import tensorflow as tf

# Load both models
print("Loading models...")
lstm_model = tf.keras.models.load_model("models/cough-classification-lstm/INPUT_model_path/lstm_model.keras")
cnn_model  = tf.keras.models.load_model("models/cough-classification-cnn/INPUT_model_path/1dcnn_model.keras")
print("Models loaded successfully")

# Load label encoder
le = joblib.load("../components/training/code/label_encoder.pkl")  # adjust path if needed

# Create dummy input to test the models
dummy_audio = np.random.rand(1, 4, 1024).astype(np.float32)   # (batch, frames, features)
dummy_meta  = np.random.rand(1, 10).astype(np.float32)         # (batch, meta_features)

# Test LSTM
print("\nTesting LSTM...")
lstm_pred = lstm_model.predict([dummy_audio, dummy_meta])
print(f"LSTM output shape: {lstm_pred.shape}")
print(f"LSTM predicted class: {le.classes_[np.argmax(lstm_pred)]}")

# Test CNN
print("\nTesting CNN...")
cnn_pred = cnn_model.predict([dummy_audio, dummy_meta])
print(f"CNN output shape: {cnn_pred.shape}")
print(f"CNN predicted class: {le.classes_[np.argmax(cnn_pred)]}")

print("\nBoth models work correctly!")