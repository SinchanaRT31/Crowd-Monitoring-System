# Crowd Monitoring System

## Overview

This project is a real-time crowd monitoring system developed using YOLOv8, OpenCV, and Streamlit. It detects people from a live webcam feed, counts the number of people present, and displays the crowd status on a web dashboard. The system also records the crowd count in a CSV file and captures snapshots whenever the number of people exceeds the specified threshold.

## Features

- Real-time webcam monitoring
- Person detection using YOLOv8
- Live crowd counting
- Crowd status (Normal / Overcrowded)
- Adjustable confidence threshold
- Adjustable crowd threshold
- Occupancy percentage calculation
- Automatic CSV logging
- Snapshot capture during overcrowding
- Download crowd log file

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- streamlit-webrtc

## Project Structure

```
Crowd-Monitoring-System/
│── app.py
│── requirements.txt
│── README.md
│── yolov8n.pt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/SinchanaRT31/Crowd-Monitoring-System.git
```

Move to the project directory:

```bash
cd Crowd-Monitoring-System
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Future Enhancements
- Crowd density estimation
- Occupancy dashboard
- Heatmap visualization
- Entry/Exit counting
- Alert generation
- Historical analytics


