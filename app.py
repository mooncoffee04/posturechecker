import dash
from dash import html, dcc
import dash.dependencies as dd
import cv2
import mediapipe as mp
import pygame
import joblib
import pandas as pd
import time
import base64
from io import BytesIO

# Initialize Dash app
app = dash.Dash(__name__)

# Load the trained model
model = joblib.load(r"C:\Users\laava\Downloads\pm_final\pm_final\models\posture_model.pkl")

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize Pygame for sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r'C:\Users\laava\Downloads\pm_final\pm_final\src\alert.wav')

# Setup video capture
cap = cv2.VideoCapture(0)
ALERT_INTERVAL = 5  # Time between alerts (in seconds)

# Helper function to convert OpenCV image to base64 for Dash
def encode_image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

# Dash Layout with aesthetic improvements
app.layout = html.Div(
    style={
        'backgroundColor': '#f7f7f7',  # Light background
        'fontFamily': 'Arial, sans-serif',  # Clean and modern font
        'textAlign': 'center',
        'padding': '20px',
    },
    children=[
        html.H1(
            "Posture Monitor (Live)",
            style={
                'color': '#333',
                'fontSize': '36px',
                'fontWeight': 'bold',
                'marginBottom': '20px',
            }
        ),
        # We hide the video feed by not rendering the image element
        html.Div(id="alert-message", style={
            'color': '#555',
            'fontSize': '18px',
            'marginBottom': '40px',
            'fontWeight': 'lighter',
        }),
        html.Div(
            "Monitor your posture and receive alerts when slouching is detected.",
            style={
                'color': '#555',
                'fontSize': '18px',
                'marginBottom': '40px',
                'fontWeight': 'lighter',
            }
        ),
        # Store the last alert time value in dcc.Store
        dcc.Store(id="last-alert-time-store", data=time.time()),  # Initial time set here
        dcc.Interval(
            id="interval-component",
            interval=100,  # update every 100ms
            n_intervals=0
        )
    ]
)

# Define callback to update posture detection and alerts
@app.callback(
    dd.Output("alert-message", "children"),  # Update alert message
    [dd.Input("interval-component", "n_intervals")],
    [dd.State("last-alert-time-store", "data")]  # Fetch the last_alert_time from the store
)
def update_frame(n, last_alert_time):
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        return dash.no_update
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to a smaller size (e.g., 640x480)
    frame = cv2.resize(frame, (640, 480))

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
            # Check if enough time has passed to trigger the alert
            if time.time() - last_alert_time > ALERT_INTERVAL:
                alert_sound.play()  # Play sound alert
                last_alert_time = time.time()  # Update the alert time
                return "Slouching detected! Please straighten up."  # Show alert message

    # If posture is fine, show a normal message
    return "Posture is good. Keep it up!"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)