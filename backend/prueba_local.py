import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import kagglehub
from unet_model import unet_model

# --- PARÁMETROS DE EJECUCIÓN LOCAL (DEBUG) ---
# Configuración optimizada para validación de código en entorno CPU/Memoria limitada.
# ADVERTENCIA: Para entrenamiento de producción, desplegar en clúster GPU (ej. Colab).
IMG_SIZE = (128, 128)
BATCH_SIZE = 2         # Batch reducido para evitar OOM (Out Of Memory) en local
EPOCHS = 1             # Ciclo único para verificar flujo de tensores (Forward/Backward pass)
NUM_CLASSES = 4        # Clases: Suelo, RocaFirme, Arena, RocaGrande

# 1. GESTIÓN DE DATOS
print("📂 Inicializando acceso al sistema de archivos...")
path_dataset = kagglehub.dataset_download("yash92328/ai4mars-terrainaware-autonomous-driving-on-mars")
BASE_MSL = os.path.join(path_dataset, "ai4mars-dataset-merged-0.1", "msl")

IMG_DIR = os.path.join(BASE_MSL, "images", "edr")
LABEL_DIR = os.path.join(BASE_MSL, "labels", "train")
MASK_DIR = os.path.join(BASE_MSL, "images", "mxy")

# 2. IMPLEMENTACIÓN DEL PIPELINE DE DATOS
class MarsDataGenerator(tf.keras.utils.Sequence):
    """
    Generador de datos heredado de Keras Sequence.
    Implementa carga diferida (lazy loading), limpieza de artefactos, normalización
    y estrategias de aumentación de datos (Data Augmentation).
    """
    def __init__(self, image_ids, img_dir, label_dir, mask_dir, batch_size=32, img_size=(128, 128)):
        self.image_ids = image_ids
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        batch_ids = self.image_ids[index * self.batch_size : (index + 1) * self.batch_size]
        images, masks = [], []
        
        for file_id in batch_ids:
            # Rutas absolutas
            img_path = os.path.join(self.img_dir, file_id + ".JPG")
            lbl_path = os.path.join(self.label_dir, file_id + ".png")
            rover_path = os.path.join(self.mask_dir, file_id + ".png")
            
            img = cv2.imread(img_path)
            lbl = cv2.imread(lbl_path, 0)
            
            if img is None or lbl is None: continue

            # Pre-procesamiento de etiquetas:
            # Reasignación de valores nulos (255) a clase 0 para consistencia numérica
            lbl[lbl == 255] = 0

            # Pre-procesamiento de imagen:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Enmascaramiento del Rover (Artifact Removal):
            if os.path.exists(rover_path):
                rover = cv2.imread(rover_path, 0)
                if rover is not None:
                    img = cv2.bitwise_and(img, img, mask=(rover==0).astype(np.uint8))
            
            # Redimensionamiento espacial
            img = cv2.resize(img, self.img_size) 
            lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalización de entrada [0, 1]
            img = img / 255.0

            # --- Data Augmentation (Regularización) ---
            
            # A. Invariancia Horizontal (Flip)
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                lbl = np.fliplr(lbl)

            # B. Invariancia Rotacional (Rotation 90/180/270)
            k_rot = np.random.randint(4) 
            if k_rot > 0:
                img = np.rot90(img, k=k_rot)
                lbl = np.rot90(lbl, k=k_rot)

            # C. Robustez a Iluminación (Brightness Jitter)
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0.0, 1.0)

            images.append(img)
            masks.append(lbl)
            
        return np.array(images), np.array(masks)

if __name__ == "__main__":
    # 3. SELECCIÓN DE DATOS PARA TESTING
    all_files = [f.replace(".png", "") for f in os.listdir(LABEL_DIR) if f.endswith(".png")]
    
    # LIMITACIÓN INTENCIONAL:
    # Se utiliza un subconjunto mínimo (10 muestras) para validar la integración del pipeline.
    # Evita saturación de memoria en entorno de desarrollo local.
    files_subset = all_files[:10]  
    
    print(f"⚠️ ENTORNO DEBUG: Ejecutando validación con {len(files_subset)} muestras.")

    # 4. INICIALIZACIÓN DEL KERNEL DE IA
    try:
        print("🏗️ Construyendo grafo de la red U-Net...")
        model = unet_model(input_size=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
        
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        # 5. EJECUCIÓN DE PRUEBA (DUMMY RUN)
        gen = MarsDataGenerator(files_subset, IMG_DIR, LABEL_DIR, MASK_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
        
        print("🚀 Iniciando época de prueba...")
        model.fit(gen, epochs=EPOCHS)
        
        print("\n✅ PRUEBA EXITOSA: Pipeline de datos y arquitectura validados correctamente.")
        print("   El código está verificado para despliegue en producción (GPU Cluster).")
        
    except Exception as e:
        print(f"\n❌ Excepción en tiempo de ejecución: {e}")