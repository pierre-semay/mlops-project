import sys
import numpy as np

# --- MLOPS MONKEY PATCH FOR NUMPY 2.0 SCALER COMPATIBILITY ---
# We importeren de daadwerkelijke 1.x multiarray module
import numpy.core.multiarray as multiarray

# We mappen zowel de hoofdmodule als de specifieke multiarray submodule naar de 2.x verwachting
sys.modules['numpy._core'] = np
sys.modules['numpy._core.multiarray'] = multiarray
# --------------------------------------------------------------

import os
import joblib
import librosa
from fastapi.responses import HTMLResponse
import numpy as np
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, UploadFile, File, Form
import psycopg2

# --- MLOPS MONKEY PATCH FOR NUMPY 2.0 SCALER COMPATIBILITY ---
sys.modules['numpy._core'] = np

app = FastAPI(title="Medical Sound Classification API")

MODEL_A_PATH = "models/cough-classification-lstm/INPUT_model_path"
MODEL_B_PATH = "models/cough-classification-cnn/INPUT_model_path"  
SCALER_PATH = "scaler_lstm_experiment.pkl"

print("Bezig met het laden van de Modellen en de Scaler...")
scaler = joblib.load(SCALER_PATH)

# Model A (LSTM) laden via de SavedModel map
try:
    model_a = tf.keras.models.load_model(MODEL_A_PATH)
    print("Model A (LSTM) succesvol geladen!")
except Exception as e:
    print(f"CRITIEKE FOUT: Kon Model A niet laden van {MODEL_A_PATH}: {e}")
    model_a = None

# Model B (CNN) laden via de SavedModel map
try:
    model_b = tf.keras.models.load_model(MODEL_B_PATH)
    print("Model B (CNN) succesvol geladen!")
except Exception as e:
    print(f"WAARSCHUWING: Kon Model B niet laden van {MODEL_B_PATH}: {e}")
    print("Systeem start door ZONDER Model B om cluster-downtime te voorkomen.")
    model_b = None

print("Bezig met het laden van YAMNet van TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

CLASS_MAPPING = {
    0: "Gezond",
    1: "Asthma",
    2: "COPD",
}

# --- DATABASE CONFIGURATIE ---
DB_HOST = "postgres-service"  
DB_NAME = "sound_classification"  
DB_USER = "mlops_user"            
DB_PASS = "mlops_password"        

def log_to_db(age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years, idx, label, scores, model_name):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()
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
    if model_a is None:
        return {"error": "Model A is momenteel niet beschikbaar op deze cluster node."}

    emb, meta_scaled = await preprocess_inputs(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, file)
    
    predictions = model_a.predict([emb, meta_scaled], verbose=0)
    idx = int(np.argmax(predictions[0]))
    scores = predictions[0].tolist()
    label = CLASS_MAPPING.get(idx, f"Onbekend ({idx})")
    kansen = {CLASS_MAPPING.get(i, f"Klasse {i}"): round(p, 3) for i, p in enumerate(scores)}
    
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model A (LSTM)")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model A (LSTM)"}

# --- ENDPOINT 2: MODEL B (CNN) ---
@app.post("/predict/model-b")
async def predict_model_b(
    age: float = Form(...), gender: float = Form(...), tbContactHistory: float = Form(...),
    wheezingHistory: float = Form(...), phlegmCough: float = Form(...), familyAsthmaHistory: float = Form(...),
    feverHistory: float = Form(...), coldPresent: float = Form(None), packYears: float = Form(...),
    file: UploadFile = File(...)
):
    if model_b is None:
        return {"error": "Model B is momenteel niet beschikbaar op deze cluster node."}

    emb, meta_scaled = await preprocess_inputs(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, file)
    
    predictions = model_b.predict([emb, meta_scaled], verbose=0)
    idx = int(np.argmax(predictions[0]))
    scores = predictions[0].tolist()
    label = CLASS_MAPPING.get(idx, f"Onbekend ({idx})")
    kansen = {CLASS_MAPPING.get(i, f"Klasse {i}"): round(p, 3) for i, p in enumerate(scores)}
    
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model B (CNN)")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model B (CNN)"}