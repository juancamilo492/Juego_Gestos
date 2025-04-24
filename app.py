"""Detección de gestos con modelo Keras.
Este código está adaptado de 
https://github.com/robmarkcole/object-detection-app
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
from keras.models import load_model

HERE = Path(__file__).parent
ROOT = HERE

logger = logging.getLogger(__name__)

# Las clases de gestos que reconoce tu modelo
CLASSES = ["Palma", "Ok", "JCBG", "Vacío"]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray  # Para el área de detección del gesto


@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))


COLORS = generate_label_colors()

# Session-specific caching
cache_key = "gesture_detection_model"
if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = load_model('./keras_model.h5')
    st.session_state[cache_key] = model

score_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5, 0.05)

# Cola para pasar datos entre threads
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# Función para aplicar filtros (como en tu app original)
def apply_filter(image, use_filter):
    if use_filter:
        return cv2.bitwise_not(image)
    return image

# Checkbox para aplicar filtro
use_filter = st.checkbox("Aplicar filtro invertido", value=False)

def normalize_image(image):
    # Redimensionar a 224x224 (tamaño que requiere tu modelo)
    resized_image = cv2.resize(image, (224, 224))
    # Normalizar valores de píxeles
    normalized_image_array = (resized_image.astype(np.float32) / 127.0) - 1
    # Asegurar que los datos tienen la forma correcta para el modelo
    return normalized_image_array

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    
    # Aplicar filtro si está activado
    if use_filter:
        image = apply_filter(image, use_filter)
    
    # Guardar las dimensiones originales para visualización
    h, w = image.shape[:2]
    
    # Preparar la imagen para la predicción
    input_image = normalize_image(image)
    
    # Redimensionar para el modelo como lo hacías en tu app original
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = input_image
    
    # Ejecutar la predicción
    prediction = model.predict(data)
    
    # Crear una lista de detecciones donde los valores superan el umbral
    detections = []
    for i, score in enumerate(prediction[0]):
        if score >= score_threshold:
            # Creamos un cuadro que abarca toda la pantalla para mostrar el gesto
            box = np.array([0.1, 0.1, 0.9, 0.9]) * np.array([w, h, w, h])
            detections.append(
                Detection(
                    class_id=i,
                    label=CLASSES[i],
                    score=float(score),
                    box=box
                )
            )
    
    # Renderizar cuadros delimitadores y etiquetas
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.class_id]
        xmin, ymin, xmax, ymax = detection.box.astype("int")

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    
    # Poner los resultados en la cola para mostrarlos
    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Reconocimiento de Gestos con WebRTC")
st.markdown("Esta aplicación usa tu modelo de Keras entrenado en Teachable Machine para reconocer gestos en tiempo real.")

# Opciones de la aplicación
with st.sidebar:
    st.subheader("Opciones")
    st.write("Ajusta el umbral de confianza y activa/desactiva el filtro para mejorar el reconocimiento.")

# Configurar el streamer WebRTC
webrtc_ctx = webrtc_streamer(
    key="gesture-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Mostrar los resultados
if st.checkbox("Mostrar resultados de detección", value=True):
    if webrtc_ctx.state.playing:
        results_placeholder = st.empty()
        
        while True:
            detections = result_queue.get()
            
            # Mostrar las detecciones
            if detections:
                results_placeholder.table(detections)
            else:
                results_placeholder.text("No se detectaron gestos con suficiente confianza.")

st.markdown(
    "Esta aplicación está adaptada para utilizar un modelo Keras de reconocimiento de gestos."
)
