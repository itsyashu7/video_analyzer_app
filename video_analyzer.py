import cv2
from ultralytics import YOLO
import torch
import time
from threading import Thread

# -------------------- HARDWARE ACCELERATION --------------------
def get_best_device():
    if torch.cuda.is_available():
        print("🚀 NVIDIA GPU detected! Using CUDA.")
        return "0"
    elif torch.backends.mps.is_available():
        print("🍎 Apple Silicon detected! Using MPS.")
        return "mps"
    else:
        print("💻 No dedicated GPU found. Using CPU.")
        return "cpu"

device = get_best_device()

# -------------------- MODEL --------------------
# Using Medium model for higher accuracy as requested previously
model = YOLO("best.pt") 

# -------------------- VIDEO THREAD --------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame.copy() if self.ret else None

    def release(self):
        self.running = False
        self.cap.release()

vs = VideoStream()

# -------------------- SETTINGS --------------------
dangerous_objects = ['fire', 'firearm', 'grenade', 'knife', 'pistol', 'rocket']
frame_count = 0
start_time = time.time()

print("\n--- Starting Detection Logs ---")

# -------------------- LOOP --------------------
while True:
    frame = vs.read()
    if frame is None:
        break

    display_frame = cv2.resize(frame, (640, 360))
    
    # Process every frame for high accuracy (no skipping)
    # verbose=True will show the default YOLO stats in terminal
    results = model.predict(display_frame, imgsz=640, device=device, verbose=False, stream=True)

    for r in results:
        # custom terminal print for every frame
        found_something = False
        
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            
            if label in dangerous_objects or label == "person":
                found_something = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 🔥 TERMINAL PRINTING (The detail you wanted)
                status = "⚠️ DANGER" if label in dangerous_objects else "✅ INFO"
                print(f"[{status}] Object: {label:10} | Conf: {conf:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")

                # Drawing on Video
                color = (0, 0, 255) if label in dangerous_objects else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed

    cv2.putText(display_frame, f"FPS: {int(fps)} | Hardware: {device}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Detection System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()