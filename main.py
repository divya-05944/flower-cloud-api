from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import clip
import io

# -------------------------------
# APP SETUP
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# MODEL SETUP (CLIP)
# -------------------------------
FLOWER_CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

text_prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]
text_tokens = clip.tokenize(text_prompts).to(device)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict_flower(image: Image.Image):
    img_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(img_input)
        txt_feat = model.encode_text(text_tokens)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

        logits = 100.0 * img_feat @ txt_feat.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    best_idx = probs.argmax()
    return FLOWER_CLASSES[best_idx], probs

# -------------------------------
# API ENDPOINT
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    label, probs = predict_flower(image)
    confidence = float(max(probs))

    if confidence < 0.75:
        return {
            "predicted_class": "Unknown Flower",
            "confidence": confidence
        }

    return {
        "predicted_class": label,
        "confidence": confidence
    }

