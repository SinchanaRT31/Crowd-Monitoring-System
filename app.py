import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(layout="wide")
st.title("🚨 Smart Crowd Detection (Webcam)")

# Load model once
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.markdown("Click **START** to enable webcam")

# WebRTC config (important for browser camera)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video processor
class CrowdProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize for performance
        img = cv2.resize(img, (640, 480))

        results = model(img, imgsz=320, conf=0.3, verbose=False)

        count = 0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

        # Display count
        cv2.putText(img, f"People: {count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start webcam
webrtc_streamer(
    key="crowd-detection",
    video_processor_factory=CrowdProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)
