import torch
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import easyocr

# =====================================
# 1️⃣ Initialize Models
# =====================================

print("Loading models...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Text Recognizer
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
model.to(device)
model.eval()

# Strong Text Detector
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

print("Models loaded successfully!")

# =====================================
# 2️⃣ Start Webcam
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to exit")

# =====================================
# 3️⃣ Real-Time Loop
# =====================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    centers = []

    # =====================================
    # 4️⃣ Detect Text Regions (STRONG)
    # =====================================

    results = reader.readtext(frame)

    for (bbox, text_easy, prob) in results:

        if prob < 0.3:
            continue

        # Convert bbox points
        (tl, tr, br, bl) = bbox
        x1, y1 = int(tl[0]), int(tl[1])
        x2, y2 = int(br[0]), int(br[1])

        # Crop region
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # Convert to PIL
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # =====================================
        # 5️⃣ Recognize Text with TrOCR
        # =====================================

        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show text
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        # Center point
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # =====================================
    # 6️⃣ Distance Between Text Blocks
    # =====================================

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):

            x1, y1 = centers[i]
            x2, y2 = centers[j]

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)

            cv2.putText(frame, f"{int(distance)} px",
                        (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2)

    cv2.imshow("STRONG OCR + Distance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()