import os
import contextlib
import sys
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import joblib
import librosa
import psycopg2

# Global placeholders for the ML components
model_a = None
model_b = None
yamnet_model = None
scaler = None

MODEL_A_PATH = os.getenv("MODEL_A_PATH", "models/cough-classification-cnn/INPUT_model_path/1dcnn_model.keras")
MODEL_B_PATH = os.getenv("MODEL_B_PATH", "models/cough-classification-lstm/INPUT_model_path/lstm_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_lstm_experiment.pkl")

CLASS_MAPPING = {
    0: "Gezond",
    1: "Ziekte 1",
    2: "Ziekte 2",
}

# --- DATABASE CONFIGURATIE (Matches database.yaml variables) ---
DB_HOST = os.getenv("DB_HOST", "postgres-service")  
DB_NAME = os.getenv("DB_NAME", "sound_classification")  
DB_USER = os.getenv("DB_USER", "mlops_user")            
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password") # Matches deployment.yaml / database.yaml

# --- ASYNC LIFESPAN WORKER (Fixes the Uvicorn Crash) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model_a, model_b, yamnet_model, scaler
    
    print("Bezig met het laden van de Keras Modellen en de Scaler...")
    keras.backend.clear_session()
    
    # Load components safely after Uvicorn has bound to the port
    scaler = joblib.load(SCALER_PATH)
    model_a = keras.models.load_model(MODEL_A_PATH, compile=False)
    model_b = keras.models.load_model(MODEL_B_PATH, compile=False)
    
    print("Bezig met het laden van YAMNet van TensorFlow Hub...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    print("Alle ML-modellen en de scaler zijn succesvol geladen!")
    yield
    keras.backend.clear_session()

# Pass lifespan to FastAPI
app = FastAPI(title="Medical Sound Classification API", lifespan=lifespan)

# --- UNCHANGED DATABASE LOGGING LOGIC ---
def log_to_db(age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years, idx, label, scores, model_name):
    try:
        # Using DB_PASSWORD here to match environment variables
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
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

# --- ENDPOINT 1: MODEL A (1D CNN) ---
@app.post("/predict/model-a")
async def predict_model_a(
    age: float = Form(...), gender: float = Form(...), tbContactHistory: float = Form(...),
    wheezingHistory: float = Form(...), phlegmCough: float = Form(...), familyAsthmaHistory: float = Form(...),
    feverHistory: float = Form(...), coldPresent: float = Form(None), packYears: float = Form(...),
    file: UploadFile = File(...)
):
    emb, meta_scaled = await preprocess_inputs(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, file)
    
    predictions = model_a.predict([emb, meta_scaled], verbose=0)
    idx = int(np.argmax(predictions[0]))
    scores = predictions[0].tolist()
    label = CLASS_MAPPING.get(idx, f"Onbekend ({idx})")
    kansen = {CLASS_MAPPING.get(i, f"Klasse {i}"): round(p, 3) for i, p in enumerate(scores)}
    
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model A (1D CNN)")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model A (1D CNN)"}

# --- ENDPOINT 2: MODEL B (Azure LSTM) ---
@app.post("/predict/model-b")
async def predict_model_b(
    age: float = Form(...), gender: float = Form(...), tbContactHistory: float = Form(...),
    wheezingHistory: float = Form(...), phlegmCough: float = Form(...), familyAsthmaHistory: float = Form(...),
    feverHistory: float = Form(...), coldPresent: float = Form(None), packYears: float = Form(...),
    file: UploadFile = File(...)
):
    emb, meta_scaled = await preprocess_inputs(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, file)
    
    predictions = model_b.predict([emb, meta_scaled], verbose=0)
    idx = int(np.argmax(predictions[0]))
    scores = predictions[0].tolist()
    label = CLASS_MAPPING.get(idx, f"Onbekend ({idx})")
    kansen = {CLASS_MAPPING.get(i, f"Klasse {i}"): round(p, 3) for i, p in enumerate(scores)}
    
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model B (Azure LSTM)")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model B (Azure LSTM)"}