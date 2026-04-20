from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==========================================
# 1️⃣ Setup High-Accuracy Models
# ==========================================
print("Loading Models... (Please wait)")

device_id = 0 if torch.cuda.is_available() else -1

# A. Object Detection
yolo_model = YOLO("yolov8s.pt") 

# B. OCR (Text & Number Reader)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# C. Currency Classifier
currency_classifier = pipeline(
    "image-classification", 
    model="khalid99ml/bangladeshi-taka-classifier", 
    device=device_id
)

print("✅ All Models Loaded Successfully!")

# Shared Variables for Threading
latest_frame = None
detected_objects = []
detected_currency = None
live_text_string = "Scanning for text..."
speech_queue = "" 
is_running = True

def get_color(cls_id):
    np.random.seed(cls_id)
    return tuple([int(x) for x in np.random.randint(50, 255, 3)])

# ==========================================
# 2️⃣ Background AI Worker Thread
# ==========================================
def ai_processing_worker():
    global latest_frame, detected_objects, detected_currency, live_text_string, speech_queue, is_running
    last_spoken_text = ""
    
    while is_running:
        if latest_frame is not None:
            frame_to_process = latest_frame.copy()
            h, w = frame_to_process.shape[:2]
            
            # -------------------------------------
            # TASK 1: Object Detection
            # -------------------------------------
            results = yolo_model(frame_to_process, conf=0.45, verbose=False)[0]
            temp_objects = []
            current_labels = []
            
            for box in results.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = yolo_model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                temp_objects.append((label, conf, (x1, y1, x2, y2), cls_id))
                current_labels.append(label)

            detected_objects = temp_objects

            # -------------------------------------
            # TASK 2: OCR (No Bounding Boxes)
            # -------------------------------------
            gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray_frame)
            
            valid_spoken_texts = []
            
            for (bbox, text, prob) in ocr_results:
                if prob > 0.30 and len(text.strip()) > 1: 
                    valid_spoken_texts.append(text)
            
            if valid_spoken_texts:
                live_text_string = " | ".join(valid_spoken_texts)
            else:
                live_text_string = "Scanning for text..."

            # -------------------------------------
            # TASK 3: Currency Detection
            # -------------------------------------
            cx1, cy1 = int(w * 0.3), int(h * 0.2)
            cx2, cy2 = int(w * 0.7), int(h * 0.8)
            roi_frame = frame_to_process[cy1:cy2, cx1:cx2]
            
            try:
                rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_roi)
                currency_results = currency_classifier(pil_img)
                best_currency = currency_results[0]
                
                if best_currency['score'] > 0.65:
                    detected_currency = (best_currency['label'], best_currency['score'])
                else:
                    detected_currency = None
            except Exception:
                detected_currency = None

            # -------------------------------------
            # TASK 4: Audio Feedback Priority
            # -------------------------------------
            speech_text = ""
            if detected_currency:
                speech_text += f"{detected_currency[0]} Taka. "
            elif valid_spoken_texts:
                speech_text += f"Text says: {valid_spoken_texts[0]}. "
            elif current_labels:
                unique_items = list(set(current_labels))
                speech_text += f"{', '.join(unique_items[:2])}. "

            if speech_text and speech_text != last_spoken_text:
                speech_queue = speech_text
                last_spoken_text = speech_text
                
            time.sleep(0.3)

# ==========================================
# 3️⃣ Video Streaming Generator (UI)
# ==========================================
def generate_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            latest_frame = frame.copy()
            h, w = frame.shape[:2]

            # 1. Draw Currency Box
            cx1, cy1 = int(w * 0.3), int(h * 0.2)
            cx2, cy2 = int(w * 0.7), int(h * 0.8)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            cv2.putText(frame, "PLACE TAKA HERE", (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if detected_currency:
                curr_label, curr_conf = detected_currency
                text = f"BD TAKA: {curr_label} ({int(curr_conf*100)}%)"
                cv2.rectangle(frame, (cx1, cy2), (cx2, cy2 + 40), (0, 255, 255), -1)
                cv2.putText(frame, text, (cx1 + 10, cy2 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

            # 2. Draw Object Boxes
            for obj_name, conf, (x1, y1, x2, y2), cls_id in detected_objects:
                color = get_color(cls_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{obj_name} {int(conf * 100)}%"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. Live Text Reader Panel at the bottom (NO BOUNDING BOXES)
            cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"READ TEXT: {live_text_string}", (20, h - 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 4️⃣ Flask Routes
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_speech')
def get_speech():
    global speech_queue
    text_to_speak = speech_queue
    speech_queue = "" 
    return jsonify({"text": text_to_speak})

if __name__ == "__main__":
    ai_thread = threading.Thread(target=ai_processing_worker, daemon=True)
    ai_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)