from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
from SimpleCNN import SimpleCNN, load_model  # Importing SimpleCNN class and the load_model function

app = Flask(__name__)

# Create an instance of SimpleCNN
model = SimpleCNN()
# Load the model weights
model_path = 'static/emotiondetector.h5'  # Update this with your model file path
model = load_model(model, model_path)

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to detect emotion
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = torch.Tensor(face).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            predicted_emotion = model(face)

        predicted_label = label[predicted_emotion.argmax()]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = detect_emotion(frame)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
