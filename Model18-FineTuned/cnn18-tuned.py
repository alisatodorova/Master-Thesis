# Fine-tuned model (initially based on MalNet's paper)
# Normalized inputs
# lr=2.5e-5 based on Learning Rate Test
# learning rate scheduler: OneCycleLR

# Install dependencies
import subprocess
subprocess.run(['pip', 'install', 'torchvision', 'tqdm'], check=True)

# Import libraries
import os, time, sys
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchsummary import summary
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
from PIL import ImageFile, Image
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import zipfile
import io
from torch.optim.lr_scheduler import OneCycleLR

# Redirect print and debugging to log files
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
sys.stdout = open("C:\\Users\\Boss\\Downloads\\malnet-images\\output.log", "w", buffering=1)
sys.stderr = open("C:\\Users\\Boss\\Downloads\\malnet-images\\error.log", "w", buffering=1)

# Ensure reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# Configurations (based on MalNet's paper: https://github.com/safreita1/malnet-image/tree/master)
args = {
    'im_size': 256,
    'batch_size': 128,
    'num_channels': 3,
    'group': 'type',  # Options: binary, type, family
    'color_mode': 'rgb',  # Options: rgb, grayscale
    'epochs': 100,
    'model': 'resnet18',
    'weights': None,  # Options: None (random), imagenet
    'loss': 'categorical_crossentropy',  # Options: categorical_crossentropy, categorical_focal_loss
    'reweight': 'effective_num', #Effective Number Class Reweighting; Other option: None
    'reweight_beta': 0.999,
    'seed': 1,
    'train_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\train', # Train dataset
    'val_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\val', # Validation dataset
    'test_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\test', # Test dataset
    'drive_checkpoint_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\checkpoints', # Folder where the checkpoints will be saved
    'training_plot_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\plots' # Folder where the plots will be saved
}

# Set device and check if GPU is available
args['device'] = torch.device("cuda")
print("Using DataParallel on GPUs:", [i for i in range(torch.cuda.device_count())])
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 2)
    print(f"GPU {i}: {name} with {mem} MB memory")
print(f"Using device: {args['device']}")


# # Transforms
# args['transform'] = transforms.Compose([
#     transforms.Resize((args['im_size'], args['im_size'])),
#     transforms.ToTensor(),
# ])

# Transforms -- CHANGED
args['transform'] = transforms.Compose([
    transforms.Resize((args['im_size'], args['im_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # Normalize the inputs
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# This is needed due to truncated files in the datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Load the datasets and loaders
args['train_dataset'] = ImageFolder(root=args['train_dir'], transform=args['transform'])
args['val_dataset']   = ImageFolder(root=args['val_dir'], transform=args['transform'])
args['test_dataset'] = ImageFolder(root=args['test_dir'], transform=args['transform'])

args['train_loader'] = DataLoader(args['train_dataset'], batch_size=args['batch_size'], shuffle=True)
args['val_loader'] = DataLoader(args['val_dataset'], batch_size=args['batch_size'], shuffle=False)
args['test_loader'] = DataLoader(args['test_dataset'], batch_size=args['batch_size'], shuffle=False)


# Check for corrupted images
# from PIL import Image
# import os

# bad_images = []
# for path, _ in args['train_dataset'].samples:
#     try:
#         img = Image.open(path)
#         img.verify()  # verify but do not load image
#     except Exception as e:
#         bad_images.append(path)

# print(f"Found {len(bad_images)} corrupted images.")


# Class Info
args['class_labels'] = args['train_dataset'].classes
args['num_classes'] = len(args['class_labels'])
# Since we have corrupted images, we need a safe way to extract y_train without loading the images
args['y_train'] = [label for _, label in args['train_dataset'].samples]


# Effective Class Weights for Effective Number Class Reweighting (based on MalNet's paper: https://github.com/safreita1/malnet-image/tree/master)
def get_effective_class_weights(args):
    beta = args['reweight_beta']
    counts = Counter(args['y_train'])
    effective_num = [(1 - beta) / (1 - np.power(beta, counts[i])) for i in range(args['num_classes'])]
    weights = np.array(effective_num)
    weights = weights / np.sum(weights) * args['num_classes']
    return torch.tensor(weights, dtype=torch.float32).to(args['device'])

# Confusion matrix
def plot_and_save_confusion_matrix(true_labels, pred_labels, class_names, save_prefix):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{args['training_plot_dir']}/{save_prefix}.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{save_prefix.replace("_", " ").title()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{args['training_plot_dir']}/{save_prefix}.png")
    plt.close()

# Build model
def build_model(args):
    model = models.resnet18(pretrained=(args['weights'] == 'imagenet'))
    model.fc = nn.Linear(model.fc.in_features, args['num_classes'])
    return model.to(args['device'])


# Dataset Summary
print("DATASET SUMMARY")
print(f"Number of classes        : {args['num_classes']}")
print(f"Class labels             : {args['class_labels']}")
print(f"Train samples            : {len(args['train_dataset'])}")
print(f"Validation samples       : {len(args['val_dataset'])}")
print(f"Test samples             : {len(args['test_dataset'])}")
class_counts = {label: args['y_train'].count(label) for label in range(args['num_classes'])}
print(f"Train class distribution : {class_counts}")
print("-" * 50)


# Init Model, Loss, Optimizer
model = build_model(args)
print(f"Using {torch.cuda.device_count()} GPUs.")
model = nn.DataParallel(model)  # Wrap the model for multi-GPU
args['model_instance'] = model.to(args['device'])
args['class_weights'] = get_effective_class_weights(args) if args['reweight'] == 'effective_num' else None
args['criterion'] = nn.CrossEntropyLoss(weight=args['class_weights'])
args['optimizer'] = torch.optim.Adam(args['model_instance'].parameters(), lr=2.5e-5) # lr = learning rate -- CHANGED

# Scheduler based on OneCycleLR -- CHANGED
args['scheduler'] = torch.optim.lr_scheduler.OneCycleLR(
    args['optimizer'],
    max_lr=2.5e-5,
    steps_per_epoch=len(args['train_loader']),
    epochs=args['epochs'],
    anneal_strategy='linear',
    pct_start=0.3,                           # % of steps to reach max_lr
    div_factor=25,                           # initial LR = max_lr / 25
    final_div_factor=1e4                     # min LR = max_lr / 1e4
)

# Model Summary
print("MODEL SUMMARY")
sample_input_shape = (args['num_channels'], args['im_size'], args['im_size'])
summary(args['model_instance'], input_size=sample_input_shape)

# Ensure checkpoint and plot directories exist
os.makedirs(args['drive_checkpoint_dir'], exist_ok=True)
os.makedirs(args['training_plot_dir'], exist_ok=True)

## === Load from Last Checkpoint ===
checkpoint_path = os.path.join(args['drive_checkpoint_dir'], 'epoch_050.pt')  # change to latest

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=args['device'], weights_only=False)
    args['model_instance'].load_state_dict(checkpoint['model_state_dict'])
    args['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    args['scheduler'].load_state_dict(checkpoint['scheduler_state_dict'])  
    start_epoch = checkpoint['epoch']
    print(f"Loaded epoch {start_epoch}")
# === End of Load from Checkpoint ===


# Training Loop with Logging + Checkpointing
# best_val_acc = 0.0
# Lists to store metrics
args['train_acc_history'] = []
args['val_acc_history'] = []
args['train_bal_acc_history'] = []
args['val_bal_acc_history'] = []
args['loss_history'] = []
for epoch in range(start_epoch, args['epochs']):
# for epoch in range(args['epochs']):
    args['model_instance'].train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    start_time = time.time()

    for images, labels in tqdm(args['train_loader'], desc=f"[Epoch {epoch+1}/{args['epochs']}]"):
        images, labels = images.to(args['device']), labels.to(args['device'])

        args['optimizer'].zero_grad()
        outputs = args['model_instance'](images)
        loss = args['criterion'](outputs, labels)
        loss.backward()
        args['optimizer'].step()
        # Update learning rate: OneCycleLR step per batch
        args['scheduler'].step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = correct / total
    train_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    elapsed = time.time() - start_time

    # Validation
    args['model_instance'].eval()
    val_correct, val_total, val_preds, val_labels = 0, 0, [], []
    with torch.no_grad():
        for images, labels in args['val_loader']:
            images, labels = images.to(args['device']), labels.to(args['device'])
            outputs = args['model_instance'](images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    val_bal_acc = balanced_accuracy_score(val_labels, val_preds)

    # Free unused GPU memory
    torch.cuda.empty_cache()

    # Print training/validation results
    print(f"\nEpoch {epoch+1:03d} | Time: {elapsed:.1f}s | Loss: {total_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f} | Train Balanced Acc: {train_bal_acc:.4f}")
    print(f"Val   Acc: {val_acc:.4f} | Val   Balanced Acc: {val_bal_acc:.4f}")

    args['train_acc_history'].append(train_acc)
    args['val_acc_history'].append(val_acc)
    args['train_bal_acc_history'].append(train_bal_acc)
    args['val_bal_acc_history'].append(val_bal_acc)
    args['loss_history'].append(total_loss)

    # Save checkpoint every 10 epochs or at the last epoch
    if (epoch+1) % 10 == 0 or (epoch+1) == args['epochs']:
        checkpoint_path = os.path.join(args['drive_checkpoint_dir'], f"epoch_{epoch+1:03d}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': args['model_instance'].state_dict(), 
            'optimizer_state_dict': args['optimizer'].state_dict(),
            'scheduler_state_dict': args['scheduler'].state_dict(),
            },
            checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    # # Save best model
    # if val_bal_acc > best_val_acc:
    #     best_val_acc = val_bal_acc
    #     best_path = os.path.join(args['drive_checkpoint_dir'], "best_model.pt")
    #     torch.save(args['model_instance'].state_dict(), best_path)
    #     print(f"Saved best model to: {best_path}")

# Evaluation on Test Set & Confusion Matrix 
args['model_instance'].eval()
all_preds, all_labels = [], []
correct = 0
total = 0
with torch.no_grad():
    for images, labels in args['test_loader']:
        images, labels = images.to(args['device']), labels.to(args['device'])
        outputs = args['model_instance'](images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Model Accuracy: {accuracy:.2f}%")

# Free unused GPU memory
torch.cuda.empty_cache()

plot_and_save_confusion_matrix(all_labels, all_preds, args['class_labels'], 'test_confusion_matrix')

# Final Plots
plt.figure()
plt.plot(args['train_acc_history'], label='Train Acc')
plt.plot(args['val_acc_history'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f"{args['training_plot_dir']}/accuracy_vs_epochs.png")
plt.close()

plt.figure()
plt.plot(args['train_bal_acc_history'], label='Train Balanced Acc')
plt.plot(args['val_bal_acc_history'], label='Val Balanced Acc')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy vs Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f"{args['training_plot_dir']}/balanced_accuracy_vs_epochs.png")
plt.close()

plt.figure()
plt.plot(args['loss_history'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f"{args['training_plot_dir']}/loss_vs_epochs.png")
plt.close()


# Per-Class Accuracy (Helps reveal class imbalance or weak spots)
per_class_acc = []
for class_idx in range(len(args['class_labels'])):
    idxs = [i for i, label in enumerate(all_labels) if label == class_idx]
    class_acc = accuracy_score(np.array(all_labels)[idxs], np.array(all_preds)[idxs])
    per_class_acc.append(class_acc)

# Plot
plt.figure()
plt.bar(args['class_labels'], per_class_acc)
plt.xticks(rotation=90)
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.tight_layout()
plt.savefig(f"{args['training_plot_dir']}/per_class_accuracy.png")
plt.close()

# Function to save images with full names for labels and predictions
def save_images(images, labels, predicted, classes, title, save_path):
    os.makedirs(save_path, exist_ok=True)
    images = images / 2 + 0.5  # unnormalize
    npimg = images.cpu().numpy()

    fig, axs = plt.subplots(1, 6, figsize=(20, 5))
    for idx in range(6):
        axs[idx].imshow(np.transpose(npimg[idx], (1, 2, 0)))
        true_label = classes[labels[idx]]
        pred_label = classes[predicted[idx]]
        axs[idx].set_title(f'True: {true_label}\nPred: {pred_label}')
        axs[idx].axis('off')
    
    fig.suptitle(title)
    
    # Sanitize title for filename (optional)
    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    save_file = os.path.join(save_path, f"{safe_title}.png")
    
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()


# Get some random test images
dataiter = iter(args["test_loader"])
images, labels = next(dataiter)
images, labels = images.to(args["device"]), labels.to(args["device"])

# Run model on test images
args['model_instance'].eval()
with torch.no_grad():
    outputs = args['model_instance'](images)
    _, predicted = torch.max(outputs, 1)

# Save the results
save_images(images, labels, predicted, args["test_loader"].dataset.classes, 'Test Predictions', args['training_plot_dir'])
