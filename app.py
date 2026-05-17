import cv2
import os
import torch
import threading
import winsound
import datetime
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thread management kosam
stop_event = threading.Event()
danger_detected = False
alarm_playing = False
dangerous_objects = ['fire', 'firearm', 'grenade', 'knife', 'pistol', 'rocket']

# Stats setup
hazard_stats = {
    "total_hazards": 0,
    "last_24h": 0,
    "live_camera_hazards": 0,
    "video_scans_hazards": 0,
    "avg_accuracy": 0,
    "total_conf_sum": 0,
    "recent_events": []
}

model = YOLO("best.pt")

def play_alarm():
    global alarm_playing
    if alarm_playing: return
    alarm_playing = True
    try:
        for _ in range(3): winsound.Beep(2500, 500)
    finally: alarm_playing = False

def update_stats(label, conf, source_type):
    global hazard_stats
    hazard_stats["total_hazards"] += 1
    if source_type == "LIVE":
        hazard_stats["live_camera_hazards"] += 1
    else:
        hazard_stats["video_scans_hazards"] += 1
    
    hazard_stats["total_conf_sum"] += (conf * 100)
    hazard_stats["avg_accuracy"] = round(hazard_stats["total_conf_sum"] / hazard_stats["total_hazards"], 1)
    
    new_event = {
        "type": label.capitalize(),
        "source": source_type,
        "accuracy": f"{round(conf * 100, 1)}%",
        "time": datetime.datetime.now().strftime("%I:%M:%S %p")
    }
    hazard_stats["recent_events"].insert(0, new_event)
    hazard_stats["recent_events"] = hazard_stats["recent_events"][:10]

def run_detection(source_path):
    global danger_detected
    stop_event.clear()
    cap = cv2.VideoCapture(source_path)
    source_type = "LIVE" if source_path == 0 else "VIDEO"

    while cap.isOpened() and not stop_event.is_set():
        success, frame = cap.read()
        if not success: break

        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        current_danger = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                
                if label in dangerous_objects:
                    current_danger = True
                    update_stats(label, conf, source_type)
                    if not alarm_playing:
                        threading.Thread(target=play_alarm, daemon=True).start()
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        danger_detected = current_danger
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    danger_detected = False

@app.route('/')
def index(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', stats=hazard_stats)

@app.route('/video_feed')
def video_feed():
    return Response(run_detection(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    stop_event.set()
    return jsonify({"status": "stopped"})

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

    # Nuvvu adigina lines tharvatha side panel logs support kosam JSON pampali
    return jsonify({"filename": filename})

@app.route('/upload_feed/<filename>')
def upload_feed(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(run_detection(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats(): return jsonify(hazard_stats)

@app.route('/danger_status')
def danger_status(): return jsonify({"danger": danger_detected})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)