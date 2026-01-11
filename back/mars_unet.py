import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64

MODEL_PATH = "mars_unet_final.h5"  # nombre de modelo
IMG_SIZE = (256, 256) 

CLASSES = {
    0: "Suelo",
    1: "Roca",
    2: "Arena",
    3: "Roca Grande",
    4: "Fondo"
}

# Colores (R, G, B) para la visualización
COLORS = np.array([
    [128, 128, 128], # 0: Suelo (Gris)
    [200, 50, 50],   # 1: Roca (Rojo)
    [230, 200, 0],   # 2: Arena (Amarillo)
    [0, 200, 0],     # 3: Roca Grande (Verde)
    [0, 0, 0]        # 4: Fondo (Negro)
], dtype=np.uint8)

app = FastAPI()

# no recomendable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelo... esto puede tardar unos segundos...")
try:
    # compile=False es importante para evitar errores de versiones de optimizadores
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Modelo cargado exitosamente. Listo para recibir fotos de Marte.")
except Exception as e:
    print(f"Error fatal cargando el modelo: {e}")

def process_image(image_bytes):
    """Convierte bytes -> Tensor (1, 256, 256, 3)"""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image_resized = cv2.resize(image, IMG_SIZE)
    # Normalizar a 0-1
    input_tensor = np.expand_dims(image_resized / 255.0, axis=0)
    return input_tensor

def tensor_to_base64(mask_indices):
    """Pinta la máscara y la convierte a string base64"""
    mask_colored = COLORS[mask_indices]
    img_pil = Image.fromarray(mask_colored)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def analyze_viability(mask_indices):
    """Calcula porcentajes y determina riesgo"""
    total_pixels = mask_indices.size
    counts = np.bincount(mask_indices.flatten(), minlength=5)
    
    pct_suelo = (counts[0] / total_pixels) * 100
    pct_roca = (counts[1] / total_pixels) * 100
    pct_arena = (counts[2] / total_pixels) * 100
    pct_roca_grande = (counts[3] / total_pixels) * 100
    
    # Lógica de Viabilidad
    if pct_roca_grande > 5:
        status = "PELIGRO"
        msg = "Detectadas rocas grandes que impiden el paso."
    elif pct_arena > 35:
        status = "PRECAUCIÓN"
        msg = "Alto nivel de arena. Riesgo de atascamiento."
    elif pct_suelo > 50:
        status = "VIABLE"
        msg = "Terreno mayormente firme. Tránsito permitido."
    else:
        status = "INCIERTO"
        msg = "Terreno complejo, se requiere supervisión manual."
        
    return status, msg, {
        "suelo": round(pct_suelo, 1),
        "arena": round(pct_arena, 1),
        "rocas": round(pct_roca_grande + pct_roca, 1)
    }

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"message": "API de Marte activa y escuchando..."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # pre-procesar
    input_tensor = process_image(image_bytes)
    
    # predecir
    prediction = model.predict(input_tensor)
    mask_indices = np.argmax(prediction, axis=-1)[0]
    
    # formatear respuesta
    mask_b64 = tensor_to_base64(mask_indices)
    status, msg, stats = analyze_viability(mask_indices)
    
    return {
        "filename": file.filename,
        "segmentation_map": mask_b64,
        "viability": {
            "status": status,
            "message": msg,
            "composition": stats
        }
    }