# main.py
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
import sys

# --- MLOPS TRICK: BYPASS KERAS ARCHITECTURE VERSION MISMATCHES ---
# We maken een virtuele module aan om Keras te foppen als hij zoekt naar de oude 'keras.src.engine'
import keras

# Maak een dummy klasse aan voor de missende 'Functional' import
class DummyModule:
    pass

# Koppel de moderne Functional klasse aan de oude verwachting van Model B
sys.modules['keras.src.engine'] = DummyModule
sys.modules['keras.src.engine.functional'] = DummyModule
DummyModule.Functional = keras.src.models.Functional


@keras.saving.register_keras_serializable(package="Custom")
class CustomDense(keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            del config['quantization_config']
        return super().from_config(config)

keras.saving.get_custom_objects()['Dense'] = CustomDense
# ------------------------------------------------------------------

app = FastAPI(title="Medical Sound Classification API")

# Paden naar de twee verschillende modellen
MODEL_A_PATH = "best_model_lstm.keras"
MODEL_B_PATH = "models/cough-classification/INPUT_model_path/lstm_model.keras"  # <-- VUL HIER DE NAAM VAN MODEL B IN
SCALER_PATH = "scaler_lstm_experiment.pkl"

# main.py

print("Bezig met het laden van de Keras Modellen en de Scaler...")
scaler = joblib.load(SCALER_PATH)

# Model A (LSTM) veilig laden
try:
    model_a = keras.models.load_model(MODEL_A_PATH, custom_objects={"Dense": CustomDense})
    print("Model A (LSTM) succesvol geladen!")
except Exception as e:
    print(f"CRITIEKE FOUT: Kon Model A niet laden: {e}")
    model_a = None

# Model B veilig laden met fallback om crashes te voorkomen
try:
    # Verwijder eventueel de sys.modules aliasing die we hiervoor hebben geprobeerd, 
    # Keras 3 heeft een ingebouwde legacy compile modus:
    model_b = keras.models.load_model(MODEL_B_PATH, compile=False, custom_objects={"Dense": CustomDense})
    print("Model B succesvol geladen (uncompiled fallback)!")
except Exception as e:
    print(f"WAARSCHUWING: Model B is corrupt of incompatibel met deze Keras-versie: {e}")
    print("Systeem start door ZONDER Model B om cluster-downtime te voorkomen.")
    model_b = None

print("Bezig met het laden van YAMNet van TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

CLASS_MAPPING = {
    0: "Gezond",
    1: "Ziekte 1",
    2: "Ziekte 2",
}

# --- DATABASE CONFIGURATIE ---
DB_HOST = "postgres-service"  
DB_NAME = "sound_classification"  
DB_USER = "mlops_user"            
DB_PASS = "mlops_password"        

# Helper functie voor database logging
def log_to_db(age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years, idx, label, scores, model_name):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
        # We voegen een extra kolom 'model_used' toe in de logica om te zien welk endpoint is aangeroepen
        cursor.execute("""
            ALTER TABLE inference_logs ADD COLUMN IF NOT EXISTS model_used TEXT;
            INSERT INTO inference_logs (
                age, gender, tb_contact_history, wheezing_history, phlegm_cough,
                family_asthma_history, fever_history, cold_present, pack_years,
                predicted_index, predicted_label, confidence_scores, model_used
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years,
            idx, label, str(scores), model_name
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Inference succesvol opgeslagen voor {model_name}")
    except Exception as e:
        print(f"FOUT bij database logging: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_embeddings_sequence(waveform):
    scores, embeddings, spectrogram = yamnet_model(waveform)
    return embeddings.numpy()

# --- GEMEENSCHAPPELIJKE PREPROCESSING (HELPER) ---
async def preprocess_inputs(age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years, file):
    cold_missing = 1.0 if cold is None else 0.0
    cold_val = 0.0 if cold is None else cold

    meta = np.array([[age, gender, tb, wheezing, phlegm, asthma, fever, cold_val, pack_years, cold_missing]], dtype=np.float32)
    meta_scaled = scaler.transform(meta)

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as b:
        b.write(await file.read())
    try:
        waveform, _ = librosa.load(temp_path, sr=16000)
        emb = extract_embeddings_sequence(waveform)[np.newaxis, ...]
        TARGET_TIMESTEPS = 4 
        if emb.shape[1] != TARGET_TIMESTEPS:
            if emb.shape[1] > TARGET_TIMESTEPS:
                emb = emb[:, :TARGET_TIMESTEPS, :]
            else:
                padded = np.zeros((1, TARGET_TIMESTEPS, 1024), dtype=np.float32)
                padded[:, :emb.shape[1], :] = emb
                emb = padded
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return emb, meta_scaled

# --- ENDPOINT 1: MODEL A (LSTM) ---
@app.post("/predict/model-a")
async def predict_model_a(
    age: float = Form(...), gender: float = Form(...), tbContactHistory: float = Form(...),
    wheezingHistory: float = Form(...), phlegmCough: float = Form(...), familyAsthmaHistory: float = Form(...),
    feverHistory: float = Form(...), coldPresent: float = Form(None), packYears: float = Form(...),
    file: UploadFile = File(...)
):
    emb, meta_scaled = await preprocess_inputs(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, file)
    
    # Voorspelling met Model A
    predictions = model_a.predict([emb, meta_scaled], verbose=0)
    idx = int(np.argmax(predictions[0]))
    scores = predictions[0].tolist()
    label = CLASS_MAPPING.get(idx, f"Onbekend ({idx})")
    kansen = {CLASS_MAPPING.get(i, f"Klasse {i}"): round(p, 3) for i, p in enumerate(scores)}
    
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model A (LSTM)")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model A (LSTM)"}

# --- ENDPOINT 2: MODEL B ---
@app.post("/predict/model-b")
async def predict_model_b(
    age: float = Form(...), gender: float = Form(...), tbContactHistory: float = Form(...),
    wheezingHistory: float = Form(...), phlegmCough: float = Form(...), familyAsthmaHistory: float = Form(...),
    feverHistory: float = Form(...), coldPresent: float = Form(None), packYears: float = Form(...),
    file: UploadFile = File(...)
):
    emb, meta_scaled = await preprocess_inputs(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, file)
    
    # Voorspelling met Model B
    predictions = model_b.predict([emb, meta_scaled], verbose=0)
    idx = int(np.argmax(predictions[0]))
    scores = predictions[0].tolist()
    label = CLASS_MAPPING.get(idx, f"Onbekend ({idx})")
    kansen = {CLASS_MAPPING.get(i, f"Klasse {i}"): round(p, 3) for i, p in enumerate(scores)}
    
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model B")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model B"}