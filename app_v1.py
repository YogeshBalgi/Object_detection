import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Create directories for storing known faces and uploads
os.makedirs('static/known_faces', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# Dictionary to store known face encodings and names
known_persons = {}

# For webcam
camera = None
video_path = None
processing_video = False

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Set lower resolution for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

# Simplified object detection using Haar Cascades (built into OpenCV)
# This doesn't require downloading external model files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

def detect_objects(frame):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Detect bodies (less reliable, but no external model needed)
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in bodies:
        if any((x <= fx <= x+w and y <= fy <= y+h) for fx, fy, fw, fh in faces):
            continue  # Skip if overlaps with a face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Add some basic object detection (this is very simplified)
    height, width = frame.shape[:2]
    cv2.putText(frame, "Objects Detected:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"- Persons: {len(faces)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def generate_frames():
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Apply object detection
        frame = detect_objects(frame)
        
        # Add timestamp
        cv2.putText(frame, "Live Feed", (frame.shape[1]-150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_video():
    global video_path, processing_video
    
    if not video_path or not processing_video:
        return
    
    video = cv2.VideoCapture(video_path)
    
    while processing_video:
        success, frame = video.read()
        if not success:
            processing_video = False
            break
        
        # Apply object detection
        frame = detect_objects(frame)
        
        # Add timestamp
        cv2.putText(frame, "Video Processing", (frame.shape[1]-200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_processing')
def video_processing():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    release_camera()  # Release any existing camera
    cam = get_camera()  # Open a new camera
    if not cam.isOpened():
        return jsonify({'status': 'error', 'message': 'Could not open webcam'})
    return jsonify({'status': 'success'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    release_camera()
    return jsonify({'status': 'success'})

@app.route('/add_person', methods=['POST'])
def add_person():
    if 'name' not in request.form:
        return jsonify({'status': 'error', 'message': 'No name provided'})
    
    name = request.form['name']
    
    # Take a picture
    cam = get_camera()
    ret, frame = cam.read()
    
    if not ret:
        return jsonify({'status': 'error', 'message': 'Failed to capture image'})
    
    # Save the image
    image_path = f"static/known_faces/{name}.jpg"
    cv2.imwrite(image_path, frame)
    
    # Add to known persons
    known_persons[name] = image_path
    
    return jsonify({'status': 'success', 'message': f'Added {name} to known persons', 'count': len(known_persons)})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, processing_video
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video provided'})
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video selected'})
    
    # Save the video
    import time
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"video_{timestamp}.mp4"
    video_path = os.path.join('static/uploads', filename)
    video_file.save(video_path)
    
    # Start processing
    processing_video = True
    
    return jsonify({'status': 'success', 'message': 'Video uploaded successfully'})

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global processing_video
    processing_video = False
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)