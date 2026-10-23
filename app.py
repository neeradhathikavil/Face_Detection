import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Face Detection", page_icon="🙂", layout="centered")

mp_fd = mp.solutions.face_detection
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def annotate(bgr, detector):
    """Draw boxes, confidence and keypoints on a BGR image. Returns (image, face_count)."""
    h, w = bgr.shape[:2]
    result = detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    count = 0
    if result.detections:
        for d in result.detections:
            count += 1
            bb = d.location_data.relative_bounding_box
            x, y = max(int(bb.xmin * w), 0), max(int(bb.ymin * h), 0)
            bw, bh = int(bb.width * w), int(bb.height * h)
            cv2.rectangle(bgr, (x, y), (x + bw, y + bh), (0, 255, 120), 2)
            cv2.putText(bgr, f"{d.score[0] * 100:.0f}%", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            for kp in d.location_data.relative_keypoints:
                cv2.circle(bgr, (int(kp.x * w), int(kp.y * h)), 3, (0, 0, 255), -1)
    return bgr, count


class Processor(VideoProcessorBase):
    def __init__(self):
        self.detector = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, self.count = annotate(img, self.detector)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def detect_pil(image, conf):
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=conf) as det:
        out, n = annotate(bgr, det)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), n


st.title("🙂 Face Detection")
st.caption("Built with Streamlit, MediaPipe and OpenCV")

conf = st.sidebar.slider("Minimum confidence (photo modes)", 0.1, 0.99, 0.5, 0.05)
st.sidebar.info("Live mode uses a fixed confidence of 0.5.")

tab_live, tab_snap, tab_up = st.tabs(["🎥 Live camera", "📸 Take photo", "🖼️ Upload image"])

with tab_live:
    st.write("Click **START** and allow camera access.")
    ctx = webrtc_streamer(
        key="live-face",
        video_processor_factory=Processor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with tab_snap:
    shot = st.camera_input("Take a picture")
    if shot:
        img, n = detect_pil(Image.open(shot), conf)
        st.image(img, caption=f"Faces detected: {n}", use_container_width=True, output_format="PNG")

with tab_up:
    f = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if f:
        img, n = detect_pil(Image.open(f), conf)
        st.image(img, caption=f"Faces detected: {n}", use_container_width=True, output_format="PNG")
