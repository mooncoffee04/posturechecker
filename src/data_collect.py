# src/data_collection.py

import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Setup video capture from the front camera
cap = cv2.VideoCapture(0)

# Prepare the dataset folder and file
os.makedirs("../data", exist_ok=True)
file_path = "../data/posture_data.csv"

with open(file_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["chin_y", "shoulder_avg_y", "label"])

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Skipping empty frame.")
            continue

        # Flip frame horizontally for natural webcam view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face and pose landmarks
        pose_results = pose.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks and pose_results.pose_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            pose_landmarks = pose_results.pose_landmarks

            # Extract key points: chin and shoulders
            chin = face_landmarks.landmark[152]
            left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # Calculate average shoulder height as a reference
            shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2

            # Draw chin and shoulder line
            img_height, img_width, _ = frame.shape
            chin_x, chin_y = int(chin.x * img_width), int(chin.y * img_height)
            left_shoulder_x, left_shoulder_y = int(left_shoulder.x * img_width), int(left_shoulder.y * img_height)
            right_shoulder_x, right_shoulder_y = int(right_shoulder.x * img_width), int(right_shoulder.y * img_height)

            cv2.circle(frame, (chin_x, chin_y), 5, (0, 0, 255), -1)  # Chin - red
            cv2.line(frame, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 2)

            # Display the frame
            cv2.imshow("Data Collection", frame)

            # Press 's' for straight or 'c' for slouched to label data
            key = cv2.waitKey(1)
            if key == ord('s'):
                writer.writerow([chin.y, shoulder_avg_y, "straight"])
                print("Saved straight posture sample.")
            elif key == ord('c'):
                writer.writerow([chin.y, shoulder_avg_y, "slouched"])
                print("Saved slouched posture sample.")
            elif key == 27:  # Press 'ESC' to exit
                break

cap.release()
cv2.destroyAllWindows()
