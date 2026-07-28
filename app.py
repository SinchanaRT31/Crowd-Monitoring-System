
import streamlit as st
import cv2
import av
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="AI Smart Crowd Monitoring System", page_icon="👥", layout="wide")
st.title("👥 AI-Based Smart Crowd Monitoring System")

st.sidebar.title("⚙️ Settings")
THRESHOLD = st.sidebar.slider("Crowd Alert Threshold", 1, 20, 5)
CONFIDENCE = st.sidebar.slider("Detection Confidence", 0.10, 1.00, 0.40, 0.05)

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

CSV_FILE = "crowd_logs.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Date & Time", "People Count", "Status"])

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class CrowdProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_snapshot = ""
        self.last_log = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))

        now = datetime.now()
        current_time = now.strftime("%d-%m-%Y %H:%M:%S")

        results = model.predict(
            img,
            imgsz=640,
            conf=CONFIDENCE,
            classes=[0],
            verbose=False,
        )

        count = 0
        for r in results:
            for box in r.boxes:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"Person {score:.2f}", (x1, max(25, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if count <= THRESHOLD * 0.5:
            status = "Normal"
            color = (0,255,0)
        elif count <= THRESHOLD:
            status = "Medium"
            color = (0,255,255)
        else:
            status = "Overcrowded"
            color = (0,0,255)

        sec = now.strftime("%Y-%m-%d %H:%M:%S")
        if sec != self.last_log:
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([current_time, count, status])
            self.last_log = sec

        cv2.rectangle(img, (10,10), (410,190), (255,255,255), -1)
        cv2.putText(img, f"People: {count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.putText(img, f"Status: {status}", (20,75), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        cv2.putText(img, f"Threshold: {THRESHOLD}", (20,110), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        occ = min(count/THRESHOLD*100,100)
        cv2.putText(img, f"Occupancy: {occ:.0f}%", (20,145), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.putText(img, current_time, (20,175), cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,80,80),1)

        if count > THRESHOLD:
            cv2.putText(img, "ALERT! CROWD LIMIT EXCEEDED", (20,220),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),3)
            snap = now.strftime("%Y%m%d_%H%M%S")
            if snap != self.last_snapshot:
                cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"crowd_{snap}.jpg"), img)
                self.last_snapshot = snap

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.subheader("🎥 Live Crowd Monitoring")
webrtc_streamer(
    key="crowd-monitor",
    video_processor_factory=CrowdProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.subheader("📥 Download Logs")
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "rb") as f:
        st.download_button("Download Crowd Logs", f, "crowd_logs.csv", "text/csv")
