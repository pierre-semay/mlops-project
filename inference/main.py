# main.py
import sys
import os
import contextlib

# --- 1. TRANSCOMPILATION NAMESPACE SHIM ---
class LegacyModuleMock:
    pass

keras_src_mock = LegacyModuleMock()
sys.modules['keras.src.engine'] = keras_src_mock
import keras.src.models.functional as modern_functional
sys.modules['keras.src.engine.functional'] = modern_functional

# --- 2. GLOBAL VARIABLE NAME TRANSLATION PATCHER ---
import keras.src.saving.serialization_lib as serialization
original_deserialize = serialization.deserialize_keras_object

def structural_axis_patcher(config, *args, **kwargs):
    if isinstance(config, dict):
        class_name = config.get("class_name")
        inner_cfg = config.get("config", {})
        
        # Force strict suffix alignments for the Keras 3 runtime layer map
        if class_name == "LSTM" or config.get("name") == "lstm_cell":
            config["config"]["name"] = "lstm_cell_1"
        if class_name == "Dense" or config.get("name") == "dense_1":
            config["config"]["name"] = "dense_1"
        if class_name == "Dense" or config.get("name") == "dense_2":
            config["config"]["name"] = "dense_2"

        if class_name == "BatchNormalization" and isinstance(inner_cfg.get("axis"), list):
            inner_cfg["axis"] = inner_cfg["axis"][0] if inner_cfg["axis"] else 2
            
        if class_name == "LSTM" and "time_major" in inner_cfg:
            inner_cfg.pop("time_major", None)

        if isinstance(inner_cfg, dict) and "layers" in inner_cfg:
            for layer in inner_cfg["layers"]:
                l_name = layer.get("class_name")
                l_cfg = layer.get("config", {})
                
                # Align nested layers
                if l_name == "LSTM" or layer.get("name") == "lstm_cell":
                    layer["config"]["name"] = "lstm_cell_1"
                if l_name == "Dense" or layer.get("name") == "dense_1":
                    layer["config"]["name"] = "dense_1"
                if l_name == "Dense" or layer.get("name") == "dense_2":
                    layer["config"]["name"] = "dense_2"

                if l_name == "BatchNormalization" and isinstance(l_cfg.get("axis"), list):
                    l_cfg["axis"] = l_cfg["axis"][0] if l_cfg["axis"] else 2
                    
                if l_name == "LSTM" and "time_major" in l_cfg:
                    l_cfg.pop("time_major", None)
                        
    return original_deserialize(config, *args, **kwargs)

serialization.deserialize_keras_object = structural_axis_patcher

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
import keras
import tensorflow_hub as hub
import joblib
import librosa
import psycopg2

# --- 3. GLOBAL CONFIGURATIONS & PATHS ---
MODEL_A_PATH = os.getenv("MODEL_A_PATH", "models/cough-classification-lstm/INPUT_model_path/lstm_model.keras")
MODEL_B_PATH = os.getenv("MODEL_B_PATH", "models/cough-classification-cnn/INPUT_model_path/1dcnn_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_lstm_experiment.pkl")

CLASS_MAPPING = {0: "Gezond", 1: "Ziekte 1", 2: "Ziekte 2"}

DB_HOST = os.getenv("DB_HOST", "postgres-service")  
DB_NAME = os.getenv("DB_NAME", "sound_classification")  
DB_USER = os.getenv("DB_USER", "mlops_user")            
DB_PASSWORD = os.getenv("DB_PASSWORD", "mlops_password")

model_a = None
model_b = None
yamnet_model = None
scaler = None

# --- 4. ASYNC LIFESPAN WORKER ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model_a, model_b, yamnet_model, scaler
    
    print("Bezig met het laden van de Keras Modellen en de Scaler...")
    keras.backend.clear_session()
    
    scaler = joblib.load(SCALER_PATH)
    
    # Native loading works flawlessly now because our patch renames variables on the fly
    model_a = keras.models.load_model(MODEL_A_PATH, compile=False)
    model_b = keras.models.load_model(MODEL_B_PATH, compile=False)

    print("Bezig met het laden van YAMNet van TensorFlow Hub...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    print("Alle ML-modellen en de scaler zijn succesvol geladen!")
    yield
    keras.backend.clear_session()

# --- 5. INITIALIZE FASTAPI ---
app = FastAPI(title="Medical Sound Classification API", lifespan=lifespan)

# --- 6. ROUTES & HELPER LOGIC ---
def log_to_db(age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years, idx, label, scores, model_name):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cursor = conn.cursor()
        cursor.execute("""
            ALTER TABLE inference_logs ADD COLUMN IF NOT EXISTS model_used TEXT;
            INSERT INTO inference_logs (
                age, gender, tb_contact_history, wheezing_history, phlegm_cough,
                family_asthma_history, fever_history, cold_present, pack_years,
                predicted_index, predicted_label, confidence_scores, model_used
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (age, gender, tb, wheezing, phlegm, asthma, fever, cold, pack_years, idx, label, str(scores), model_name))
        conn.commit()
        cursor.close()
        conn.close()
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
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model A")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model A"}

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
    log_to_db(age, gender, tbContactHistory, wheezingHistory, phlegmCough, familyAsthmaHistory, feverHistory, coldPresent, packYears, idx, label, kansen, "Model B")
    return {"voorspelling_index": idx, "voorspelling_label": label, "kansen_per_klasse": kansen, "model_gebruikt": "Model B"}