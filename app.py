import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import urllib.request

app = Flask(__name__)

# Create directories for storing known faces and uploads
os.makedirs('static/known_faces', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Dictionary to store known face encodings and names
known_persons = {}
known_face_encodings = []
known_face_names = []

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

# Function to download the YOLO files if they don't exist
def download_yolo_files():
    yolo_weights = 'models/yolov3-tiny.weights'
    yolo_config = 'models/yolov3-tiny.cfg'
    coco_names = 'models/coco.names'
    
    # Check if files already exist
    if not os.path.exists(yolo_weights):
        try:
            print("Downloading YOLOv3-tiny weights...")
            urllib.request.urlretrieve(
                'https://pjreddie.com/media/files/yolov3-tiny.weights',
                yolo_weights
            )
        except Exception as e:
            print(f"Error downloading weights: {e}")
            return False
    
    if not os.path.exists(yolo_config):
        try:
            print("Downloading YOLOv3-tiny config...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
                yolo_config
            )
        except Exception as e:
            print(f"Error downloading config: {e}")
            return False
    
    if not os.path.exists(coco_names):
        try:
            print("Downloading COCO class names...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
                coco_names
            )
        except Exception as e:
            print(f"Error downloading class names: {e}")
            return False
    
    return True

# Load the YOLO model
def load_yolo():
    # Check/download YOLO files
    if not download_yolo_files():
        print("Failed to download YOLO files. Using fallback detection.")
        return None, None
    
    yolo_weights = 'models/yolov3-tiny.weights'
    yolo_config = 'models/yolov3-tiny.cfg'
    
    # Load YOLO network
    try:
        net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        try:
            # For OpenCV 4.5.4+
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            # For older OpenCV versions
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        return net, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None

# Load COCO class names
def load_coco_classes():
    coco_names = 'models/coco.names'
    try:
        with open(coco_names, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except:
        # Fallback class names if file can't be loaded
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

# Face recognition functions
def load_known_faces():
    global known_face_encodings, known_face_names, known_persons
    
    # Try to load known persons from file
    if os.path.exists('static/known_persons.txt'):
        with open('static/known_persons.txt', 'r') as f:
            for line in f:
                if line.strip():
                    name, path = line.strip().split(',')
                    if os.path.exists(path):
                        known_persons[name] = path
    
    # Load face detector and recognition model
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        # Try to create and load face recognizer
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Check if we have any known faces and a trained model
        if known_persons and os.path.exists('static/known_faces_model.yml'):
            face_recognizer.read('static/known_faces_model.yml')
            return face_detector, face_recognizer
        
        # If we have known persons but no model, train a new one
        elif known_persons:
            return face_detector, train_face_recognizer()
        
        else:
            return face_detector, face_recognizer
            
    except Exception as e:
        print(f"Error setting up face recognition: {e}")
        return face_detector, None

def train_face_recognizer():
    global known_persons
    
    if not known_persons:
        return None
    
    try:
        # Create face recognizer
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Prepare training data
        faces = []
        labels = []
        label_map = {}
        
        for i, (name, img_path) in enumerate(known_persons.items()):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            detected_faces = face_detector.detectMultiScale(gray, 1.1, 4)
            
            # If a face is detected, add it to training data
            if len(detected_faces) > 0:
                (x, y, w, h) = detected_faces[0]  # Use the first detected face
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))  # Normalize size
                faces.append(face_roi)
                labels.append(i)
                label_map[i] = name
        
        # Train the recognizer if we have faces
        if faces:
            face_recognizer.train(faces, np.array(labels))
            face_recognizer.save('static/known_faces_model.yml')
            
            # Save label map
            with open('static/label_map.txt', 'w') as f:
                for label, name in label_map.items():
                    f.write(f"{label},{name}\n")
            
            # Save known persons list
            with open('static/known_persons.txt', 'w') as f:
                for name, path in known_persons.items():
                    f.write(f"{name},{path}\n")
                    
            return face_recognizer
        
    except Exception as e:
        print(f"Error training face recognizer: {e}")
        
    return None

# Load components
yolo_net, output_layers = load_yolo()
coco_classes = load_coco_classes()
face_detector, face_recognizer = load_known_faces()

def detect_objects(frame):
    detected_objects = []
    detected_persons = []  # Track detected person names
    height, width = frame.shape[:2]
    
    # Attempt to use YOLO for general object detection
    if yolo_net is not None and output_layers is not None:
        # Prepare image for neural network
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        
        # Forward pass through network
        try:
            outputs = yolo_net.forward(output_layers)
            
            # Process detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:  # Confidence threshold
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Draw bounding boxes for detected objects
            for i in indices:
                if isinstance(i, (list, tuple)):  # Older OpenCV versions might return tuples/lists
                    i = i[0]
                
                x, y, w, h = boxes[i]
                label = str(coco_classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detected_objects.append(label)
                
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
    
    # Fallback or additional face recognition
    if face_detector is not None:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, 1.1, 4)
        
        # Try to load label map for face recognition
        label_map = {}
        if os.path.exists('static/label_map.txt'):
            with open('static/label_map.txt', 'r') as f:
                for line in f:
                    if ',' in line:
                        label, name = line.strip().split(',')
                        label_map[int(label)] = name
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face if YOLO didn't detect any people
            if 'person' not in detected_objects:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                detected_objects.append("person")
            
            # Try to recognize the face
            if face_recognizer is not None and len(label_map) > 0:
                try:
                    # Prepare face for recognition
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (100, 100))
                    
                    # Predict who this is
                    label, confidence = face_recognizer.predict(face_roi)
                    
                    # If confidence is reasonable and we have a name for this label
                    # Note: For LBPH, lower confidence value means better match
                    if confidence < 100 and label in label_map:
                        name = label_map[label]
                        # Draw the recognized name with a different color to stand out
                        cv2.putText(frame, f"{name} ({100-confidence:.1f}%)", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        detected_persons.append(name)
                        
                except Exception as e:
                    print(f"Error in face recognition: {e}")
    
    # Add object count information
    unique_objects = set(detected_objects)
    cv2.putText(frame, "Objects Detected:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    y_pos = 70
    for obj in unique_objects:
        count = detected_objects.count(obj)
        cv2.putText(frame, f"- {obj}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
    
    # Add recognized persons
    if detected_persons:
        y_pos += 10
        cv2.putText(frame, "Known Persons:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_pos += 30
        for person in set(detected_persons):
            cv2.putText(frame, f"- {person}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_pos += 30
    
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
    global known_persons, face_recognizer
    
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
    
    # Train the face recognizer with the new face
    face_recognizer = train_face_recognizer()
    
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