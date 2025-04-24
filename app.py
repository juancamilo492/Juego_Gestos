import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import mediapipe as mp

logger = logging.getLogger(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

class HandLandmark(NamedTuple):
    hand_index: int
    landmarks: List[tuple]

result_queue: "queue.Queue[List[HandLandmark]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_data = []

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            hand_data.append(HandLandmark(idx, landmarks))

    result_queue.put(hand_data)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="gesture-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Mostrar landmarks detectados"):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            hand_results = result_queue.get()
            labels_placeholder.write(f"Manos detectadas: {len(hand_results)}")
