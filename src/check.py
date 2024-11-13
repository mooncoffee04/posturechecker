# src/posture_monitor_with_model.py

import cv2
import mediapipe as mp
import pygame
import joblib
import pandas as pd
import time

# Load the trained model
model = joblib.load("C:/Users/ajkan/OneDrive/Desktop/pm_final/models/posture_model.pkl")

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize Pygame for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("C:/Users/ajkan/OneDrive/Desktop/pm_final/alert.wav")

# Setup video capture
cap = cv2.VideoCapture(0)
BUFFER_DURATION = 30
ALERT_INTERVAL = 5

last_alert_time = time.time()
slouch_start_time = None

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Skipping empty frame.")
            continue

        frame = cv2.flip(frame, 1)
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

            # Prepare the input data for prediction
            input_data = pd.DataFrame([[chin.y, shoulder_avg_y]], columns=["chin_y", "shoulder_avg_y"])

            # Use the trained model for prediction
            prediction = model.predict(input_data)[0]

            if prediction == 1:  # If slouching is detected
                if slouch_start_time is None:
                    slouch_start_time = time.time()
                elif time.time() - slouch_start_time >= BUFFER_DURATION:
                    if time.time() - last_alert_time > ALERT_INTERVAL:
                        alert_sound.play()
                        last_alert_time = time.time()
                        slouch_start_time = None
            else:
                slouch_start_time = None

        cv2.imshow("Posture Monitor", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
