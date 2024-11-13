import gradio as gr
import cv2
import mediapipe as mp
import joblib
import pandas as pd
import pygame
import time

# Load the trained model and initialize necessary components
model = joblib.load("models/posture_model.pkl")
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("assets/beep.mp3")

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

ALERT_INTERVAL = 5  # Time between alerts in seconds
last_alert_time = time.time()

def process_frame(image):
    global last_alert_time
    frame = cv2.flip(image, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    if face_results.multi_face_landmarks and pose_results.pose_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        pose_landmarks = pose_results.pose_landmarks

        chin = face_landmarks.landmark[152]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        input_data = pd.DataFrame([[chin.y, shoulder_avg_y]], columns=["chin_y", "shoulder_avg_y"])

        prediction = model.predict(input_data)[0]
        if prediction == 1 and time.time() - last_alert_time > ALERT_INTERVAL:
            alert_sound.play()
            last_alert_time = time.time()

    return frame

iface = gr.Interface(
    fn=process_frame,
    inputs=gr.inputs.Image(source="webcam", tool="editor", type="numpy"),
    outputs="image",
    live=True,
    title="Posture Monitor",
    description="Real-time posture monitoring to detect slouching and alert the user."
)

iface.launch()
