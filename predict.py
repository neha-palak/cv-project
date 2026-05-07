from pathlib import Path

import timm
import torch
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
LABELS_PATH = BASE_DIR / "Khana_Dataset" / "labels.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

labels = [
    label.strip()
    for label in LABELS_PATH.read_text(encoding="utf-8").splitlines()
    if label.strip()
]

model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=len(labels),
)

checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

flip_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(image):
    image = image.convert("RGB")
    inputs = torch.stack([
        base_transform(image),
        flip_transform(image),
    ]).to(device)

    logits = model(inputs).mean(dim=0)
    pred = logits.argmax().item()

    return labels[pred]
