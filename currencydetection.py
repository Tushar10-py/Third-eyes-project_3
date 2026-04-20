from transformers import pipeline
from PIL import Image

def detect_bdt_note(image_path):
    print("Model load hochhe... (Prothombar ektu somoy lagte pare)")
    
    # ১. Hugging Face theke model-ti pipeline-er maddhome load kora
    # Ekhane amra image-classification task select korechi
    classifier = pipeline("image-classification", model="khalid99ml/bangladeshi-taka-classifier")
    
    # ২. Takar chobi (Image) load kora
    try:
        image = Image.open(image_path)
        print(f"'{image_path}' chobiti successfully load hoyeche.\n")
    except Exception as e:
        print(f"Error: Chobiti khunje pawa jayni ba open kora jacche na. {e}")
        return

    # ৩. Model bebohar kore chobiti predict kora
    print("Prediction cholche...\n")
    results = classifier(image)

    # ৪. Result print kora
    print("--- Model Prediction ---")
    # Model sadharonoto top 2 ba 3 ta prediction dey confidence score soho
    for result in results:
        label = result['label']
        confidence = result['score'] * 100
        print(f"Note-er Dhoron (Label): {label} | Confidence: {confidence:.2f}%")
        
    print("\n[Sobcheye beshi confidence thaka label-tii holo model-er final uttor]")

# --- Tomar Chobi Diye Test Koro ---
# Ekhane "amar_taka.jpg" er jaygay tomar takar chobir ashol nam evong location dao
my_image_path = "download (8).jpg" 

detect_bdt_note(my_image_path)