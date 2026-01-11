import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(128, 128, 3), num_classes=4):
    """
    Implementación de la arquitectura U-Net para segmentación semántica.
    
    Diseño basado en arquitectura 'Fully Convolutional Network' (FCN) con estructura 
    simétrica de Encoder-Decoder y conexiones residuales (Skip Connections) para 
    recuperación de información espacial.

    Args:
        input_size (tuple): Dimensiones del tensor de entrada (H, W, C).
        num_classes (int): Número de canales de salida (categorías semánticas).

    Returns:
        model (tf.keras.Model): Modelo compilado listo para entrenamiento.
    """
    inputs = layers.Input(input_size)

    # --- ENCODER (Contracting Path) ---
    # Extracción de características jerárquicas y reducción de dimensionalidad espacial.
    
    # Bloque 1
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Bloque 2
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bloque 3
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- BOTTLENECK (Latent Space) ---
    # Representación comprimida de alto nivel semántico.
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)

    # --- DECODER (Expansive Path) ---
    # Reconstrucción de dimensiones espaciales y fusión con mapas de características del Encoder.

    # Bloque de Subida 1
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3]) # Skip Connection: Recupera detalles espaciales del Bloque 3
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)

    # Bloque de Subida 2
    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2]) # Skip Connection: Recupera detalles espaciales del Bloque 2
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    # Bloque de Subida 3
    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1]) # Skip Connection: Recupera detalles espaciales del Bloque 1
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    # --- CAPA DE SALIDA (Classification Head) ---
    # Clasificación píxel a píxel (Pixel-wise Classification) mediante función Softmax.
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs], name="Mars_UNet")
    return model