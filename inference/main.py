# main.py
import sys
import os
import contextlib
import json
import zipfile
import io

# --- TRANSCOMPILATION NAMESPACE SHIM ---
class LegacyModuleMock:
    pass

keras_src_mock = LegacyModuleMock()
sys.modules['keras.src.engine'] = keras_src_mock
import keras.src.models.functional as modern_functional
sys.modules['keras.src.engine.functional'] = modern_functional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
import keras
import tensorflow_hub as hub
import joblib
import librosa
import psycopg2

# --- CORE UTILITY: RUNTIME ZIP CONFIG REWRITER ---
def load_and_patch_keras_model(filepath):
    """
    Opens a .keras zip archive, intercepts both structural config files,
    recursively normalizes array-wrapped BatchNormalization 'axis' values
    to integers, and reconstructs the functional model instance.
    """
    if not os.path.exists(filepath):
        raise IOError(f"Model file not found at: {filepath}")

    with open(filepath, 'rb') as f:
        zip_data = f.read()

    modified_zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as z_in:
        with zipfile.ZipFile(modified_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as z_out:
            for item in z_in.infolist():
                file_bytes = z_in.read(item.filename)
                
                # Intercept both possible structural configuration targets
                if item.filename in ["config.json", "model.json"]:
                    try:
                        config_dict = json.loads(file_bytes.decode('utf-8'))
                        
                        def recursive_axis_unwrap(item_node):
                            if isinstance(item_node, dict):
                                if item_node.get("class_name") == "BatchNormalization":
                                    inner_cfg = item_node.get("config", {})
                                    if isinstance(inner_cfg.get("axis"), list):
                                        inner_cfg["axis"] = inner_cfg["axis"][0] if inner_cfg["axis"] else 2
                                for key, val in item_node.items():
                                    recursive_axis_unwrap(val)
                            elif isinstance(item_node, list):
                                for element in item_node:
                                    recursive_axis_unwrap(element)
                        
                        recursive_axis_unwrap(config_dict)
                        file_bytes = json.dumps(config_dict).encode('utf-8')
                    except Exception as json_err:
                        print(f"Warning: Could not parse or patch {item.filename}: {json_err}")
                
                z_out.writestr(item, file_bytes)

    modified_zip_buffer.seek(0)
    
    # FIX: Wrap the memory stream in an active zipfile handle for Keras 3 to read natively
    with zipfile.ZipFile(modified_zip_buffer, 'r') as patched_zip:
        # Pass the open handle to load_model, completely bypassing the disk path string requirement
        return keras.models.load_model(patched_zip, compile=False)


# --- ASYNC LIFESPAN WORKER ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model_a, model_b, yamnet_model, scaler
    
    print("Bezig met het laden van de Keras Modellen en de Scaler...")
    keras.backend.clear_session()
    
    scaler = joblib.load(SCALER_PATH)
    
    # Run structural cleanup utility
    model_a = load_and_patch_keras_model(MODEL_A_PATH)
    model_b = load_and_patch_keras_model(MODEL_B_PATH)

    print("Bezig met het laden van YAMNet van TensorFlow Hub...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    
    print("Alle ML-modellen en de scaler zijn succesvol geladen!")
    yield
    keras.backend.clear_session()

app = FastAPI(title="Medical Sound Classification API", lifespan=lifespan)

# --- DATABASE LOGGING LOGIC ---
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

# --- ENDPOINT 1: MODEL A ---
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

# --- ENDPOINT 2: MODEL B ---
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