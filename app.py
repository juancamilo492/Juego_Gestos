import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp
import paho.mqtt.client as mqtt

# Configurar MQTT
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "streamlit/gesto"

# Conectar MQTT y mostrar feedback
client = mqtt.Client()
mqtt_conectado = False

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_conectado = True
except Exception as e:
    st.error(f"Error al conectar al broker MQTT: {e}")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Función para detectar gesto
def detectar_gesto(landmarks):
    dedos_estirados = []
    tips_ids = [4, 8, 12, 16, 20]

    # Pulgar (eje X)
    dedos_estirados.append(landmarks[4][0] > landmarks[3][0])

    # Otros dedos (eje Y)
    for i in [8, 12, 16, 20]:
        dedos_estirados.append(landmarks[i][1] < landmarks[i - 2][1])

    if dedos_estirados == [False, False, False, False, False]:
        return "Puño cerrado ✊"
    elif dedos_estirados == [True, True, True, True, True]:
        return "Palma abierta 🖐"
    elif (
        np.linalg.norm(np.array(landmarks[4][:2]) - np.array(landmarks[8][:2])) < 0.05
        and all(dedos_estirados[i] for i in [2, 3, 4])
    ):
        return "Gesto OK 👌"
    else:
        return None

# Procesar el video frame por frame
gesture_placeholder = st.empty()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            gesto = detectar_gesto(landmarks)
            if gesto:
                gesture_text = gesto
                if mqtt_conectado:
                    client.publish(MQTT_TOPIC, gesto)
                gesture_placeholder.markdown(f"### Gesto detectado: **{gesto}**")

    if gesture_text:
        cv2.putText(image, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Interfaz de usuario
st.title("Detector de Gestos con MQTT 🖐✊👌")

if mqtt_conectado:
    st.success("Conectado a MQTT correctamente ✅")
else:
    st.warning("No conectado a MQTT ❌")

webrtc_streamer(
    key="gesture",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
