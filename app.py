import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import mediapipe as mp
import paho.mqtt.client as mqtt

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# MQTT
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "streamlit/gesto"

# Inicializar estado de la conexión MQTT
if "mqtt_status" not in st.session_state:
    st.session_state.mqtt_status = "Conectando al broker MQTT..."

# Callback para conexión MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.mqtt_status = "✅ Conexión MQTT exitosa con test.mosquitto.org"
    else:
        st.session_state.mqtt_status = f"❌ Error de conexión MQTT (código {rc})"

# Cliente MQTT
client = mqtt.Client()
client.on_connect = on_connect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Mostrar estado de conexión en la barra lateral
st.sidebar.markdown("### Estado de MQTT:")
st.sidebar.info(st.session_state.mqtt_status)

# Función para detectar gesto
def detectar_gesto(landmarks):
    dedos_estirados = []

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

# Área de visualización del gesto detectado
gesture_display = st.empty()

# Procesamiento del video
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image = cv2.resize(image, (320, 240))  # Resolución reducida
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
                client.publish(MQTT_TOPIC, gesto)

    if gesture_text:
        gesture_display.markdown(f"### 👉 Gesto detectado: **{gesture_text}**")

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Interfaz principal de Streamlit
st.title("Detector de Gestos con MQTT 🖐✊👌")
webrtc_streamer(
    key="gesture",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
