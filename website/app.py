# app.py
import cv2
import mediapipe as mp
import gradio as gr
import joblib
import pandas as pd
import time

# Load the trained model
model = joblib.load("posture_model.pkl")  # Ensure this file is uploaded to Hugging Face

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Time-based alert interval for slouching detection
ALERT_INTERVAL = 5  # seconds
last_alert_time = time.time()

def detect_posture(frame):
    global last_alert_time
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks and pose_results.pose_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks

        # Get chin and shoulder positions
        chin = face_landmarks.landmark[152]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate average shoulder height
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2

        # Prepare data for model prediction
        input_data = pd.DataFrame([[chin.y, shoulder_avg_y]], columns=["chin_y", "shoulder_avg_y"])

        # Predict posture
        prediction = model.predict(input_data)[0]
        if prediction == 1 and (time.time() - last_alert_time > ALERT_INTERVAL):
            last_alert_time = time.time()
            posture_status = "Slouching Detected! Sit up straight!"
        else:
            posture_status = "Good Posture Detected."
            
        # Overlay feedback on frame
        cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if prediction == 1 else (0, 255, 0), 2)
    else:
        posture_status = "No Posture Detected."

    return cv2.flip(frame, 1), posture_status

# Define Gradio interface
def live_posture_monitoring(frame):
    annotated_frame, posture_status = detect_posture(frame)
    return annotated_frame, posture_status

iface = gr.Interface(
    fn=live_posture_monitoring,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs=[gr.Image(), "text"],
    live=True,
    description="Posture Monitoring System - Sit straight for good posture!"
)

iface.launch()
