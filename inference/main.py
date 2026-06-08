import os
import joblib
import librosa
from fastapi.responses import HTMLResponse
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, UploadFile, File, Form
import keras
import psycopg2

@keras.saving.register_keras_serializable(package="Custom")
class CustomDense(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

keras.saving.get_custom_objects()['Dense'] = CustomDense

app = FastAPI(title="Medical Sound Classification API")

MODEL_PATH = "best_model_lstm.keras"
SCALER_PATH = "scaler_lstm_experiment.pkl"

print("Bezig met het laden van het Keras LSTM model en de Scaler...")
scaler = joblib.load(SCALER_PATH)
model = keras.models.load_model(MODEL_PATH, custom_objects={"Dense": CustomDense})
print("Model succesvol geladen!")

print("Bezig met het laden van YAMNet van TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

CLASS_MAPPING = {
    0: "Gezond",
    1: "Asthma",
    2: "Copd",
}

DB_HOST = "postgres-service"
DB_NAME = "sound_classification"
DB_USER = "mlops_user"
DB_PASS = "mlops_password"

def init_db():
    """Maakt de tabel aan voor het loggen van vragen en antwoorden als deze nog niet bestaat."""
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_logs (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                age REAL,
                gender REAL,
                tb_contact_history REAL,
                wheezing_history REAL,
                phlegm_cough REAL,
                family_asthma_history REAL,
                fever_history REAL,
                cold_present REAL,
                pack_years REAL,
                filename TEXT,
                predicted_index INT,
                predicted_label TEXT,
                confidence_scores TEXT
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Database succesvol geïnitialiseerd (Tabel is gereed)!")
    except Exception as e:
        print(f"WAARSCHUWING: Kon niet verbinden met de database tijdens opstarten: {e}")

init_db()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serveert de HTML frontend startpagina."""
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_embeddings_sequence(waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

@app.post("/predict")
async def predict(
    age: float = Form(...),
    gender: float = Form(...),
    tbContactHistory: float = Form(...),
    wheezingHistory: float = Form(...),
    phlegmCough: float = Form(...),
    familyAsthmaHistory: float = Form(...),
    feverHistory: float = Form(...),
    coldPresent: float = Form(None),
    packYears: float = Form(...),
    file: UploadFile = File(...)
):
    coldPresent_missing = 1.0 if coldPresent is None else 0.0
    coldPresent_value = 0.0 if coldPresent is None else coldPresent

    mijn_meta = np.array([[
        age, gender, tbContactHistory, wheezingHistory, phlegmCough, 
        familyAsthmaHistory, feverHistory, coldPresent_value, packYears, coldPresent_missing
    ]], dtype=np.float32)

    mijn_meta_scaled = scaler.transform(mijn_meta)

    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        waveform, _ = librosa.load(temp_file_path, sr=16000)
        emb = extract_embeddings_sequence(waveform)
        emb = emb[np.newaxis, ...]

        TARGET_TIMESTEPS = 4 
        if emb.shape[1] != TARGET_TIMESTEPS:
            if emb.shape[1] > TARGET_TIMESTEPS:
                emb = emb[:, :TARGET_TIMESTEPS, :]
            else:
                padded = np.zeros((1, TARGET_TIMESTEPS, 1024), dtype=np.float32)
                padded[:, :emb.shape[1], :] = emb
                emb = padded
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    predictions = model.predict([emb, mijn_meta_scaled], verbose=0)
    predicted_class_idx = int(np.argmax(predictions[0]))
    confidence_scores = predictions[0].tolist()

    predicted_label = CLASS_MAPPING.get(predicted_class_idx, f"Onbekende Klasse ({predicted_class_idx})")

    kansen_per_klasse = {
        CLASS_MAPPING.get(i, f"Klasse {i}"): round(prob, 3) 
        for i, prob in enumerate(confidence_scores)
    }

    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inference_logs (
                age, gender, tb_contact_history, wheezing_history, phlegm_cough,
                family_asthma_history, fever_history, cold_present, pack_years,
                predicted_index, predicted_label, confidence_scores
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            age, gender, tbContactHistory, wheezingHistory, phlegmCough,
            familyAsthmaHistory, feverHistory, coldPresent_value, packYears,
            predicted_class_idx, predicted_label, str(kansen_per_klasse)
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print("Inference succesvol opgeslagen in de database (zonder bestandsnaam).")
    except Exception as db_error:
        print(f"FOUT: Kon inference niet opslaan in database: {db_error}")

    return {
        "voorspelling_index": predicted_class_idx,
        "voorspelling_label": predicted_label,
        "kansen_per_klasse": kansen_per_klasse
    }