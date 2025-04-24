"""Gesture recognition demo with Teachable Machine model.
Based on WebRTC streaming for real-time processing.
"""

import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow import keras

HERE = Path(__file__).parent
ROOT = HERE

logger = logging.getLogger(__name__)

# Definición de las clases para los gestos
GESTURE_CLASSES = [
    "Palma",
    "Ok",
    "JCBG",
    "Vacío"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float


@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(GESTURE_CLASSES), 3))


COLORS = generate_label_colors()

# Cargar el modelo al inicio
@st.cache_resource  # type: ignore
def load_keras_model():
    return keras.models.load_model('keras_model.h5')

st.title("Reconocimiento de Gestos en tiempo real")

with st.sidebar:
    st.subheader("Usa un modelo entrenado en Teachable Machine para identificar gestos en tiempo real")
    score_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5, 0.05)
    filtro = st.radio("Aplicar Filtro", ('Con Filtro', 'Sin Filtro'))

# Cargar modelo
try:
    # Session-specific caching
    cache_key = "gesture_recognition_model"
    if cache_key in st.session_state:
        model = st.session_state[cache_key]
    else:
        model = load_keras_model()
        st.session_state[cache_key] = model
    
    model_loaded = True
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.info("Asegúrate de que el archivo 'keras_model.h5' esté en el mismo directorio que este script.")
    model_loaded = False

# NOTE: La callback se ejecutará en otro hilo,
#       así que utilizamos una cola para garantizar la seguridad entre hilos
#       al pasar datos desde el interior hacia el exterior de la callback.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def normalize_image(img_array):
    """Normaliza la imagen al formato requerido por el modelo."""
    # Redimensionar a 224x224
    resized = cv2.resize(img_array, (224, 224))
    # Normalizar de 0-255 a -1 - 1
    normalized = (resized.astype(np.float32) / 127.0) - 1
    return normalized


def apply_filter(image, filtro_option):
    """Aplica un filtro a la imagen si está seleccionado."""
    if filtro_option == 'Con Filtro':
        return cv2.bitwise_not(image)
    return image


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # Aplicar filtro si está seleccionado
    filtered_image = apply_filter(image.copy(), filtro)
    
    if model_loaded:
        # Preparar la imagen para el modelo
        normalized_image = normalize_image(filtered_image)
        
        # Preparar la entrada para el modelo
        input_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        input_data[0] = normalized_image
        
        # Realizar predicción
        prediction = model.predict(input_data, verbose=0)
        
        # Procesar resultados
        detections = []
        for i, score in enumerate(prediction[0]):
            if score >= score_threshold:
                detections.append(
                    Detection(
                        class_id=i,
                        label=GESTURE_CLASSES[i],
                        score=float(score),
                    )
                )
        
        # Mostrar resultado en la imagen
        h, w = image.shape[:2]
        # Zona para mostrar texto (rectángulo semi-transparente en la parte superior)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        
        # Mostrar resultados en la imagen
        if detections:
            text = " | ".join([f"{det.label}: {det.score:.2f}" for det in detections])
            cv2.putText(
                image,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                image,
                "No se detectaron gestos",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        
        result_queue.put(detections)
    else:
        # Si el modelo no se cargó correctamente, muestra un mensaje
        cv2.putText(
            image,
            "Error: Modelo no cargado",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        result_queue.put([])
        
    return av.VideoFrame.from_ndarray(image, format="bgr24")


# Solo mostrar el streamer WebRTC si el modelo se cargó correctamente o si queremos mostrar el error en el stream
webrtc_ctx = webrtc_streamer(
    key="gesture-recognition",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if model_loaded and st.checkbox("Mostrar detecciones", value=True):
    if webrtc_ctx.state.playing:
        detection_placeholder = st.empty()
        # NOTA: La transformación de video con detección de gestos y
        # este bucle que muestra las etiquetas resultantes se ejecutan
        # en diferentes hilos de forma asíncrona.
        # Por lo tanto, los frames de video renderizados y las etiquetas mostradas aquí
        # no están estrictamente sincronizados.
        while True:
            try:
                result = result_queue.get(timeout=1.0)
                if result:
                    detection_placeholder.table(result)
                else:
                    detection_placeholder.text("No se detectaron gestos con suficiente confianza")
            except queue.Empty:
                detection_placeholder.warning("No se reciben datos. Asegúrate de que la cámara esté funcionando.")
                break

st.markdown(
    "Esta aplicación utiliza un modelo de reconocimiento de gestos "
    "entrenado en Teachable Machine y se ejecuta en tiempo real con WebRTC."
)

# Mostrar información sobre posibles problemas
st.sidebar.subheader("Solución de problemas")
st.sidebar.markdown("""
- Si la cámara no funciona, asegúrate de que tu navegador tenga permiso para acceder a ella.
- Si la transmisión se detiene, actualiza la página.
- Asegúrate de que el archivo 'keras_model.h5' esté en el mismo directorio que este script.
""")
