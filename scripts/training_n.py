import os
import random
import torch
import timm
from collections import Counter
from PIL import ImageFile
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

#configurations
DATA_DIR = "/Users/nehapalak/cs4912/cv-project/khana"
BATCH_SIZE = 32 # was 64 
EPOCHS = 50     # increased from 30 to give more room to converge with augmentation
LR = 5e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 6    # was 2 -- too aggressive, augmented training needs more epochs to show improvement
NUM_WORKERS = 0   # shifted to .py because of multiprocessing issues in Jupyter

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = device

# loading khana dataset and apply transformations

def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)

        if img.mode == "P":
            img = img.convert("RGBA") # fixed this warning: /Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/PIL/Image.py:1047: UserWarning: Palette images with Transparency expressed in bytes should be converted to RGBA images warnings.warn(

        return img.convert("RGB")
    
def get_dataloaders(): # added two diff transforms because val accuracy didnt reach the baseline 91%
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),          # resize slightly larger than 224 so random crop has room to work
        transforms.RandomCrop((224, 224)),       # random crop to 224 -- more varied than just resize to 224 directly
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10), # expensive on cpu
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # food images vary a lot in lighting and color, this helps generalize
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # imagenet mean and std -- required because efficientnet_b0 pretrained weights expect this exact normalization, without it the pretrained features are misaligned with our inputs
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),          # resize to 256 first, same as train
        transforms.CenterCrop((224, 224)),       # then center crop to 224 -- deterministic, no randomness for eval, keeps val/train at same resolution
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # same normalization as train -- must match or val distribution will differ from what model learned
    ])

    full_dataset = ImageFolder(DATA_DIR, loader=rgb_loader) # made full dataset without transforms to do stratified split, and then created separate datasets with transforms for train and val subsets, because if we apply transforms before splitting, then the same image with different augmentations can end up in both train and val sets, which can lead to data leakage and overestimated performance, so we want to split first and then apply transforms separately to train and val sets to ensure that they are distinct and that our evaluation is more realistic

    # doing stratified split so that we have balanced classes in both train and val sets
    random.seed(42)
    # group indices by class
    class_indices = {}

    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    train_indices = []
    val_indices = []

    # split per class
    for label, indices in class_indices.items():
        random.shuffle(indices)
        #80-20 split
        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    train_dataset = ImageFolder(DATA_DIR, transform=train_transform, loader=rgb_loader)
    val_dataset = ImageFolder(DATA_DIR, transform=val_transform, loader=rgb_loader)

    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)

    # setting parameters for dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False # setting to false because it is not supported on mac
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    return full_dataset, train_loader, val_loader, train_indices, val_indices


# adding weighted loss to handle class imbalance

def get_loss(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)

    weights = torch.zeros(len(dataset.classes))

    for i in range(len(dataset.classes)):
        weights[i] = 1.0 / class_counts[i]

    weights = weights.to(DEVICE)

    return torch.nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.0 # was 0.1 but it was hurting performance, so set to 0, but can experiment with it later (it helps prevent overconfidence in predictions and can improve generalization, but in this case it was not helping)
    )
# Loss penalizes rare class mistakes more
# model pays attention to them


# MODEL from hugging face (timm library)
# pretrained on imagenet, 80 output classes for our dataset
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html

def get_model(num_classes):
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=True,
        num_classes=num_classes
    )
    return model.to(DEVICE)


# training function 

def train():
    print("training started !!!\n")

    dataset, train_loader, val_loader, train_idx, val_idx = get_dataloaders()

    # debug class balance
    train_labels = [dataset.samples[i][1] for i in train_idx]
    val_labels = [dataset.samples[i][1] for i in val_idx]
    # print("Train distribution:", Counter(train_labels))
    # print("Val distribution:", Counter(val_labels))

    model = get_model(len(dataset.classes))
    criterion = get_loss(dataset)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # cosine annealing scheduler -- starts at LR and smoothly decays to eta_min over EPOCHS
    # this helps the model escape plateaus and fine-tune into better minima in later epochs
    # fixed LR of 5e-5 was stalling -- scheduler gives large steps early, tiny steps late
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    start_epoch = 0
    best_acc = 0

    # adding 'resume from checkpoint' functionality so that if training is interrupted, we can continue from where we left off instead of starting over, which i had to do multiple times, in case i wanted to make any changes
    if os.path.exists("checkpoint.pth"):
        print("Resuming from previous run checkpoint...")
        checkpoint = torch.load("checkpoint.pth")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # restore scheduler too so LR curve continues correctly, not reset to the beginning

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    patience_counter = 0

    # training loop 

    for epoch in range(start_epoch, EPOCHS):

        epoch_start = time.time()
        print(f"Epoch {epoch+1} started")

        # for i, (images, labels) in enumerate(train_loader):
        #     if i % 50 == 0:
        #         print(f"Batch {i}")

        # removed the debug batch-print loop that was here -- it was iterating through
        # train_loader a full extra time every epoch, doubling load time for no training benefit

        # TRAIN
        model.train()
        train_correct, train_total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # VALIDATION
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # step the scheduler once per epoch -- advances the cosine decay curve
        scheduler.step()

        # SAVE BEST MODEL
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        # SAVE CHECKPOINT
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # added scheduler state so resume works correctly
            'best_acc': best_acc
        }, "checkpoint.pth")
        
        epoch_time = time.time() - epoch_start
        print(f"train: {train_acc:.4f} -- val: {val_acc:.4f} -- time: {epoch_time:.2f}s")
        
        # adding early stopping functionality so that if the model stops improving for 2 consecutive epochs, we stop training to save time and resources
        if patience_counter >= PATIENCE:
            print("No improvement for 6 epochs, therefore stopping early")
            break

    print(f"\nTraining completed! with best validation accuracy of: {best_acc:.4f}")


if __name__ == "__main__":
    print("Using device:", device)
    train()