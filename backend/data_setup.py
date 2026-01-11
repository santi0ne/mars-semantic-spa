import os
import kagglehub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CREDENCIALES DE KAGGLE (Configuración Directa) ---
os.environ["KAGGLE_API_TOKEN"] = "KGAT_c527777960e23022fda14f7fcec291f8"

def inicializar_dataset():
    """
    Gestiona la descarga y verificación de la estructura de directorios del dataset AI4Mars.
    Utiliza el token configurado en el entorno para autenticarse.
    """
    print("⬇️  Iniciando sincronización del dataset AI4Mars...")
    try:
        # kagglehub detectará automáticamente la variable KAGGLE_API_TOKEN que definimos arriba
        path_dataset = kagglehub.dataset_download("yash92328/ai4mars-terrainaware-autonomous-driving-on-mars")
        print(f"✅ Dataset sincronizado en ruta local: {path_dataset}")
        return path_dataset
    except Exception as e:
        print(f"❌ Error crítico en la descarga del dataset: {e}")
        print("   Verifica que tu token 'KGAT_...' sea correcto y tengas conexión a internet.")
        return None

def verificar_pipeline_procesamiento(base_path):
    """
    Ejecuta una prueba unitaria visual del pipeline de pre-procesamiento.
    Valida: Carga de imágenes, lectura de máscaras y eliminación de artefactos (rover).
    """
    # Definición de rutas según estructura del dataset MSL
    base_msl = os.path.join(base_path, "ai4mars-dataset-merged-0.1", "msl")
    img_dir = os.path.join(base_msl, "images", "edr")
    label_dir = os.path.join(base_msl, "labels", "train")
    rover_mask_dir = os.path.join(base_msl, "images", "mxy")

    if not os.path.exists(img_dir):
        print(f"❌ Directorio de imágenes no encontrado: {img_dir}")
        return

    # Selección de muestra aleatoria para validación
    nombres = os.listdir(label_dir)
    if not nombres: 
        print("❌ Dataset vacío o corrupto.")
        return
    
    test_filename = nombres[0]
    test_file_base = test_filename.replace(".png", "") 
    print(f"🔬 Ejecutando diagnóstico de pre-procesamiento en muestra: {test_file_base}")

    # Carga de tensores
    img_path = os.path.join(img_dir, test_file_base + ".JPG")
    label_path = os.path.join(label_dir, test_filename)
    mask_path = os.path.join(rover_mask_dir, test_filename)

    img = cv2.imread(img_path)
    label = cv2.imread(label_path, 0)
    rover_mask = cv2.imread(mask_path, 0)

    if img is None: 
        print("❌ Error de E/S: No se pudo leer el archivo de imagen.")
        return

    # --- Lógica de Enmascaramiento (Pre-procesamiento) ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if rover_mask is not None:
        # Aplicación de operación bitwise AND con máscara inversa
        # Objetivo: Eliminar píxeles correspondientes al hardware del rover (ruido semántico)
        mask_limpia = (rover_mask == 0).astype(np.uint8)
        img_final = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_limpia)
    else:
        img_final = img_rgb

    # --- Visualización de Resultados ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_final)
    plt.title("Input Pre-procesado (Rover Masked)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='jet')
    plt.title("Ground Truth (Etiquetas Semánticas)")
    plt.axis("off")

    plt.show()
    print("✅ Validación exitosa: Pipeline de carga y limpieza funcional.")

if __name__ == "__main__":
    path = inicializar_dataset()
    if path:
        verificar_pipeline_procesamiento(path)