from pathlib import Path
import io
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from model import WasteCNN

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "waste_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="EcoSort Waste Scanner")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
transform = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
model, classes = None, []

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = WasteCNN(len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Loaded trained model:", classes)
except Exception as exc:
    print("Model not loaded:", exc, "| Put waste_cnn.pth beside app.py.")

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/status")
def status():
    return {"model_loaded": model is not None}

@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded. Put waste_cnn.pth beside app.py and restart.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Unable to open this image. Try JPG or PNG.")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        confidence, index = torch.max(probs, dim=0)
    predicted = classes[index.item()]
    organic = {"food", "leaf", "leaves", "paper", "wood", "organic", "biodegradable"}
    category = "Biodegradable" if any(x in predicted.lower() for x in organic) else "Non-biodegradable"
    return {"predicted_class": predicted, "confidence": round(confidence.item() * 100, 2), "category": category}
