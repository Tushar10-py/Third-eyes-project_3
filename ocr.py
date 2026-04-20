import torch

import cv2

import numpy as np

from PIL import Image

from transformers import DetrImageProcessor, DetrForObjectDetection

from facenet_pytorch import MTCNN, InceptionResnetV1

import os



# ==========================================

# 1️⃣ Device Setup

# ==========================================



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)



# ==========================================

# 2️⃣ Object Detection Model

# ==========================================



processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")



detr_model.to(device)

detr_model.eval()



print("Object detection model loaded")



# ==========================================

# 3️⃣ Face Recognition Models

# ==========================================



mtcnn = MTCNN(keep_all=True, device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)



known_embeddings = []

known_names = []



face_folder = "known_faces"



if os.path.exists(face_folder):



    for file in os.listdir(face_folder):



        img = Image.open(os.path.join(face_folder, file)).convert("RGB")



        face = mtcnn(img)



        if face is not None:



            # shape fix

            if face.ndim == 3:

                face = face.unsqueeze(0)



            face = face.to(device)



            with torch.no_grad():

                embedding = resnet(face)



            known_embeddings.append(embedding)

            known_names.append(file.split('.')[0])



print("Face recognition ready")



# ==========================================

# 4️⃣ Webcam

# ==========================================



cap = cv2.VideoCapture(0)



if not cap.isOpened():

    print("Cannot open webcam")

    exit()



print("Press Q to exit")



# ==========================================

# 5️⃣ Real-Time Loop

# ==========================================



while True:



    ret, frame = cap.read()



    if not ret:

        break



    centers = []



    # ---------- Object Detection ----------



    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



    inputs = processor(images=image, return_tensors="pt").to(device)



    with torch.no_grad():

        outputs = detr_model(**inputs)
        print(outputs)

