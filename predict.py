import sys
import torch
from PIL import Image
from torchvision import transforms
from model import WasteCNN

if len(sys.argv) < 2:
    raise SystemExit('Usage: python predict.py "path/to/image.jpg"')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("waste_cnn.pth", map_location=device)
classes = checkpoint["classes"]
model = WasteCNN(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
transform = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    probs = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
print("Predicted class:", classes[index.item()])
print(f"Confidence: {confidence.item() * 100:.2f}%")
