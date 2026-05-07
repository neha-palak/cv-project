import argparse
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

MODEL_PATH = "best_model.pth"
TEST_DIR = "test_food"
IMG_PATH = "test_food/img4.jpg"
LABELS_PATH = "Khana_Dataset/labels.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transforms matching the validation transforms used during training.
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


def get_class_names(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model(model_path, num_classes):
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=num_classes,
    )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image):
    inputs = torch.stack([
        base_transform(image),
        flip_transform(image),
    ]).to(DEVICE)

    with torch.no_grad():
        logits = model(inputs).mean(dim=0)

    return int(torch.argmax(logits).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Predict food classes for local test images.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the saved model weights.")
    parser.add_argument("--test-dir", default=TEST_DIR, help="Directory containing test images.")
    parser.add_argument("--labels", default=LABELS_PATH, help="Path to labels.txt.")
    parser.add_argument("--show", action="store_true", help="Display each image with matplotlib.")
    return parser.parse_args()


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)

    class_names = get_class_names(args.labels)
    model = load_model(args.model, len(class_names))
    # print(f"Class labels and model loaded on {DEVICE}\n")

    img_path = IMG_PATH
    image = Image.open(img_path).convert("RGB")

    pred_idx = predict_image(model, image)
    pred_class = class_names[pred_idx]

    print(f"{IMG_PATH} -> {pred_class}")

    # for img_path in sorted(test_dir.iterdir()):
    #     if img_path.name.startswith(".") or not img_path.is_file():
    #         continue

    #     try:
    #         image = Image.open(img_path).convert("RGB")
    #     except Exception as e:
    #         print(f"Skipping {img_path.name}: {e}")
    #         continue

    #     pred_idx = predict_image(model, image)
    #     pred_class = class_names[pred_idx]
    #     print(f"{img_path.name:40s} -> {pred_class}")

        # if args.show:
        #     import matplotlib.pyplot as plt

        #     plt.figure(figsize=(3, 3))
        #     plt.imshow(image)
        #     plt.title(pred_class)
        #     plt.axis("off")
        #     plt.show()


if __name__ == "__main__":
    main()
