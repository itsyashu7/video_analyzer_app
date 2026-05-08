import cv2
import os
import torch
import threading
import winsound

from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DEVICE ----------------

def get_best_device():

    if torch.cuda.is_available():
        return "0"

    elif torch.backends.mps.is_available():
        return "mps"

    return "cpu"

device = get_best_device()

# ---------------- MODEL ----------------

model = YOLO("best.pt")

danger_detected = False

dangerous_objects = [
    "fire",
    "firearm",
    "smoke",
    "knife",
    "scissors",
    "baseball bat"
]

# ---------------- ALARM ----------------

alarm_playing = False

def play_alarm():

    global alarm_playing

    # Prevent multiple alarms at same time
    if alarm_playing:
        return

    alarm_playing = True

    try:

        # 3 beep alarm
        for _ in range(3):

            winsound.Beep(2500, 700)

    finally:

        alarm_playing = False

# ---------------- DETECTION ----------------

def run_detection(source_path):

    global danger_detected

    cap = cv2.VideoCapture(source_path)

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.4,
            device=device,
            verbose=False
        )

        danger_detected = False

        # ---------------- CUSTOM DRAWING ----------------

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])

                label = model.names[cls]

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ---------------- DANGEROUS OBJECT ----------------

                if label in dangerous_objects:

                    danger_detected = True

                    # Start alarm thread
                    threading.Thread(
                        target=play_alarm,
                        daemon=True
                    ).start()

                    color = (0, 0, 255)   # RED

                    text = f"DANGER: {label} {conf:.2f}"

                # ---------------- NORMAL OBJECT ----------------

                else:

                    color = (0, 255, 0)   # GREEN

                    text = f"{label} {conf:.2f}"

                # ---------------- DRAW BOX ----------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3
                )

                # ---------------- DRAW LABEL ----------------

                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        annotated_frame = frame

        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )

    cap.release()

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():

    return Response(
        run_detection(0),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/upload_feed/<filename>')
def upload_feed(filename):

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return "File not found", 404

    return Response(
        run_detection(filepath),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/upload_video', methods=['POST'])
def upload_video():

    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']

    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    file.save(filepath)

    return filename

@app.route('/danger_status')
def danger_status():

    return jsonify({
        "danger": danger_detected
    })

# ---------------- MAIN ----------------

if __name__ == '__main__':

    app.run(
        debug=True,
        threaded=True
    )