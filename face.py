import cv2
import pyttsx3
import threading
import time
import numpy as np
from ultralytics import YOLO
import easyocr
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1️⃣ Setup Models & TTS
# ==========================================
print("Loading Fast AI Models...")

# YOLOv8 Nano: Sobcheye fast object detection model real-time er jonno
yolo_model = YOLO("yolov8n.pt") 

# EasyOCR: Text porar jonno (Lightweight)
reader = easyocr.Reader(['en'], gpu=True) # GPU thakle fast hobe, na thakle CPU te cholbe

# Text-to-Speech Setup
engine = pyttsx3.init()
engine.setProperty('rate', 160)
tts_lock = threading.Lock()

print("Models Loaded Successfully!")

# ==========================================
# 2️⃣ Shared Variables for Threading
# ==========================================
# Background thread theke data main thread e anar jonno variables
latest_frame = None
detected_objects = []
detected_texts = []
last_spoken = ""
is_running = True

# ==========================================
# 3️⃣ Background AI Worker Thread
# ==========================================
# Ei thread ti continuously AI run korbe kintu video ke lag korabe na
def ai_processing_worker():
    global latest_frame, detected_objects, detected_texts, last_spoken
    
    while is_running:
        if latest_frame is not None:
            # Process korar jonno frame er ekta copy niye nilam
            frame_to_process = latest_frame.copy()
            
            # --- Object Detection (YOLOv8) ---
            results = yolo_model(frame_to_process, verbose=False)[0]
            current_objects = []
            close_warnings = []
            frame_area = frame_to_process.shape[0] * frame_to_process.shape[1]

            temp_detected_objects = []
            
            for box in results.boxes:
                conf = box.conf[0].item()
                if conf > 0.5: # 50% confidence er upore thakle
                    cls = int(box.cls[0].item())
                    label = yolo_model.names[cls]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    temp_detected_objects.append((label, (x1, y1, x2, y2)))
                    current_objects.append(label)
                    
                    # Distance approximation: box er area diye
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area / frame_area > 0.4: # Frame er 40% cover korle mane onek kache
                        close_warnings.append(label)

            detected_objects = temp_detected_objects

            # --- Text Detection (OCR) - Proti 3 second e ekbar ba logic diye komate paren ---
            # Ekhane fast processing er jonno grayscale use kora holo
            gray_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
            ocr_results = reader.readtext(gray_frame, detail=0, paragraph=True)
            
            valid_texts = [text for text in ocr_results if len(text) > 3]
            
            # --- Audio Feedback Generation ---
            speech_text = ""
            unique_items = list(set(current_objects))
            
            if unique_items:
                speech_text += f"I see {', '.join(unique_items[:4])}. " # Maximum 4ta jinis er nam bolbe
            
            if close_warnings:
                speech_text += f"Careful, {', '.join(set(close_warnings))} is very close! "
                
            if valid_texts:
                speech_text += f"Text says: {valid_texts[0][:30]}. "

            # TTS Run kora (Eki kotha bar bar na bolar logic)
            if speech_text and speech_text != last_spoken:
                with tts_lock:
                    print(f"Speaking: {speech_text}")
                    engine.say(speech_text)
                    engine.runAndWait()
                last_spoken = speech_text
                
            # AI k processing korar jonno ektu rest dilam jate CPU overheat na hoy
            time.sleep(0.5) 

# Start background thread
ai_thread = threading.Thread(target=ai_processing_worker, daemon=True)
ai_thread.start()

# ==========================================
# 4️⃣ Main Camera Loop (Smooth Video)
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'Q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update latest frame for background thread
    latest_frame = frame.copy()

    # Draw the results calculated by the background thread
    # Eita eto fast hobe karon ei loop e kono AI processing hoche na!
    for obj_name, (x1, y1, x2, y2) in detected_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, obj_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Smooth 3rd Eye Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_running = False
        break

cap.release()
cv2.destroyAllWindows()