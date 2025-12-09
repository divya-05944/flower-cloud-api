import torch
import clip
from PIL import Image

FLOWER_CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

text_prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]
text_tokens = clip.tokenize(text_prompts).to(device)

def predict_flower(image_path):
    image = Image.open(image_path).convert("RGB")
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
