import streamlit as st
import numpy as np
import cv2
import pytesseract
from PIL import Image
from keras.models import load_model
import streamlit.components.v1 as components
import time
import base64
from io import BytesIO  # Added missing import

# Carga del modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Gestos en Tiempo Real")

# Barra lateral con información
with st.sidebar:
    st.subheader("Usa un modelo entrenado en Teachable Machine para identificar gestos en tiempo real, e inclusive reconoce texto si lo hay")
    filtro = st.radio("Aplicar Filtro", ('Sin Filtro', 'Con Filtro'))
    fps = st.slider("Frames por segundo", min_value=1, max_value=30, value=10)
    confidence_threshold = st.slider("Umbral de confianza", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    show_ocr = st.checkbox("Activar reconocimiento de texto (OCR)", value=False)
    
    # Botones para iniciar y detener la transmisión
    start_button = st.button("Iniciar Cámara")
    stop_button = st.button("Detener Cámara")

# Contenedores para mostrar los resultados
video_container = st.empty()
result_container = st.empty()
text_container = st.empty()

# Función para aplicar el filtro a la imagen
def apply_filter(image, filtro):
    if filtro == 'Con Filtro':
        return cv2.bitwise_not(image)
    return image

# Función para normalizar la imagen
def normalize_image(img_array):
    # Redimensionar a 224x224
    img_array = cv2.resize(img_array, (224, 224))
    
    # Asegurarse de que la imagen es RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # Si tiene canal alpha
        img_array = img_array[:, :, :3]
        
    # Normalizar valores
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    return normalized_image_array

# Función para realizar la predicción
def predict_image(image_array):
    data[0] = image_array
    prediction = model.predict(data)
    return prediction

# Función para extraer texto de la imagen
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# Función para crear un componente HTML que accede a la cámara
def get_camera_component():
    html_code = """
    <div style="text-align: center;">
        <video id="webcam" autoplay playsinline width="640" height="480" style="transform: scaleX(-1);"></video>
        <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
    </div>
    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        
        // Acceder a la cámara
        async function setupCamera() {
            const stream = await navigator.mediaDevices.getUserMedia({
                'audio': false,
                'video': {
                    facingMode: 'user',
                    width: {ideal: 640},
                    height: {ideal: 480}
                }
            });
            video.srcObject = stream;
            
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    resolve(video);
                };
            });
        }
        
        // Capturar frame y enviarlo al backend
        function captureFrame() {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Enviar datos al backend de Streamlit
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: imageData
            }, '*');
        }
        
        // Configurar y comenzar
        setupCamera();
        
        // Enviar mensaje cuando se carga el componente
        window.parent.postMessage({
            type: 'streamlit:componentReady',
            value: true
        }, '*');
    </script>
    """
    return components.html(html_code, height=500)

# Estado de la aplicación para controlar la cámara
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Controles para iniciar/detener la cámara
if start_button:
    st.session_state.camera_active = True
if stop_button:
    st.session_state.camera_active = False

# Mostrar la cámara si está activa
if st.session_state.camera_active:
    # Crear el componente de cámara HTML
    frame_data = get_camera_component()
    
    # Procesar el frame si se recibe datos
    if frame_data and frame_data.startswith('data:image'):
        try:
            # Decodificar la imagen base64
            _, encoded = frame_data.split(",", 1)
            binary = base64.b64decode(encoded)
            
            # Convertir a imagen numpy
            img_array = np.array(Image.open(BytesIO(binary)))
            
            # Aplicar filtro si está seleccionado
            img_filtered = apply_filter(img_array, filtro)
            
            # Mostrar la imagen procesada
            video_container.image(img_filtered, channels="RGB", use_column_width=True)
            
            # Normalizar para la predicción
            normalized_image = normalize_image(img_array)
            
            # Realizar predicción
            prediction = predict_image(normalized_image)
            
            # Mostrar resultados de la predicción
            result_text = ""
            max_index = np.argmax(prediction[0])
            classes = ["Palma", "Ok", "JCBG", "Vacío"]
            
            if prediction[0][max_index] >= confidence_threshold:
                result_text = f"{classes[max_index]}: {prediction[0][max_index]:.2f}"
            else:
                result_text = "Ningún gesto detectado con suficiente confianza."
                
            result_container.markdown(f"### {result_text}")
            
            # Extraer texto si está activado el OCR
            if show_ocr:
                extracted_text = extract_text_from_image(img_filtered)
                if extracted_text:
                    text_container.markdown(f"**Texto detectado:** {extracted_text}")
                else:
                    text_container.markdown("No se detectó texto en la imagen.")
                    
            # Controlar FPS
            time.sleep(1/fps)
            
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
else:
    video_container.info("Presiona 'Iniciar Cámara' para comenzar la detección en tiempo real.")

# Añadir código JavaScript para capturar frames continuamente
if st.session_state.camera_active:
    js_code = f"""
    <script>
        function captureFrames() {{
            const canvas = document.getElementById('canvas');
            if (canvas) {{
                const context = canvas.getContext('2d');
                const video = document.getElementById('webcam');
                
                if (video && video.readyState === 4) {{
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);
                    
                    window.parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: imageData
                    }}, '*');
                }}
            }}
            setTimeout(captureFrames, {1000/fps});
        }}
        
        // Iniciar la captura después de un breve retraso
        setTimeout(captureFrames, 1000);
    </script>
    """
    components.html(js_code, height=0)

# Instrucciones para el usuario
st.markdown("""
### Instrucciones
1. Presiona "Iniciar Cámara" para comenzar el reconocimiento en tiempo real
2. Muestra tu gesto frente a la cámara
3. Ajusta la sensibilidad usando el deslizador "Umbral de confianza"
4. Activa el reconocimiento de texto si necesitas detectar texto en la imagen
5. Presiona "Detener Cámara" cuando hayas terminado
""")
