import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import os
import copy
from PIL import ImageFile, Image

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  

# --- Configuration ---
NUM_CLASSES = 28
CHANNELS = 3
IMAGE_RESIZE = 224
NUM_EPOCHS = 25
EARLY_STOP_PATIENCE = 5
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10

# [MODIFIED] ResNeSt101 is heavier than ResNet50. 
# Reduced Batch Size to 24 to avoid CUDA Out of Memory. 
# Increase this if you have >16GB VRAM.
BATCH_SIZE_TRAINING = 24
BATCH_SIZE_VALIDATION = 24 
BATCH_SIZE_TESTING = 16
LEARNING_RATE = 0.0001

OUTPUT_DIR = "./resnest101_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Preparation ---

# Standard ResNet/ResNeSt normalization values are identical
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
}

# Define Paths
# Ensure these folders exist: ./train/, ./valid/, ./test/
dirs = {
    'train': './Clf_data/train/',
    'valid': './Clf_data/valid/',
    'test':  './Clf_data/test/'
}

image_datasets = {x: datasets.ImageFolder(dirs[x], data_transforms[x]) 
                  for x in ['train', 'valid', 'test'] if os.path.exists(dirs[x])}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE_TRAINING, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE_VALIDATION, shuffle=True),
    'test':  DataLoader(image_datasets['test'], batch_size=BATCH_SIZE_TESTING, shuffle=False)
}

# --- 2. Model Setup (Converted to ResNeSt101) ---

print("Loading ResNeSt101 model via torch.hub...")
# [MODIFIED] Load ResNeSt101 from the official repo
# This requires an internet connection for the first run to download weights.
try:
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
except Exception as e:
    print(f"Error loading from Torch: {e}")
    raise e

# Replace the final layer to match NUM_CLASSES
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# Optimizer and Loss
# Keras used 'decay=1e-6', in PyTorch Adam this is 'weight_decay'
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()

# --- 3. Training Loop with Early Stopping Logic ---

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
patience_counter = 0
history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

print("Starting Training...")

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
            steps = STEPS_PER_EPOCH_TRAINING
        else:
            model.eval()
            steps = STEPS_PER_EPOCH_VALIDATION

        running_loss = 0.0
        running_corrects = 0
        samples_processed = 0

        # Iterate over data
        iterator = tqdm(enumerate(dataloaders[phase]), total=steps, desc=phase)
        
        for i, (inputs, labels) in iterator:
            if i >= steps: break 

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            samples_processed += inputs.size(0)

        epoch_loss = running_loss / samples_processed
        epoch_acc = running_corrects.double() / samples_processed

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'train':
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc.item())
        else:
            history['val_loss'].append(epoch_loss)
            history['val_accuracy'].append(epoch_acc.item())

            # Deep Copy Model (Checkpointing)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1

    # Early Stopping check
    if patience_counter >= EARLY_STOP_PATIENCE:
        print("Early stopping triggered.")
        break

print('Training complete.')

# Load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "resnest101_final.pth")

# --- 4. Visualization ---

plt.figure(1, figsize=(15, 8))

plt.subplot(221)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])

plt.subplot(222)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'])
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))

# --- 5. Testing & Evaluation ---

print("Starting Testing...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in tqdm(dataloaders['test'], desc="Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Get class names
class_names = image_datasets['train'].classes

# Classification Report
report = metrics.classification_report(all_labels, all_preds, target_names=class_names)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
cm = metrics.confusion_matrix(all_labels, all_preds)

# Plotting the Confusion Matrix as Percentages
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(20, 20))
sns.heatmap(cm_percentage, annot=True, cbar=False, fmt='.2%', cmap='Blues')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

# Class Labels Dictionary
class_indices = image_datasets['train'].class_to_idx
idx_to_class = {v: k for k, v in class_indices.items()}
with open(os.path.join(OUTPUT_DIR, 'class_indices.txt'), 'w') as f:
    for idx, class_name in idx_to_class.items():
        f.write(f"{idx}: {class_name}\n")
