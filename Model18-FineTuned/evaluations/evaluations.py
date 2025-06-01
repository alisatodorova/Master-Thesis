# Model based on MalNet's paper

# === Install dependencies (only necessary in some environments) ===
import subprocess
subprocess.run(['pip', 'install', 'torchvision', 'tqdm', 'scikit-learn', 'torchsummary'], check=True)

# === Import libraries ===
import os, time, random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
from tqdm import tqdm
import numpy as np
from collections import Counter
from PIL import ImageFile
from sklearn.metrics import classification_report

# === Ensure reproducibility ===
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# === Configurations ===
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
    'train_dir': '/scratch/users/atodorova/train',
    'val_dir': '/home/users/atodorova/alisa-thesis/malnet-cnn/val',
    'test_dir': '/home/users/atodorova/alisa-thesis/malnet-cnn/test',
    'drive_checkpoint_dir': '/home/users/atodorova/alisa-thesis/malnet-cnn/'
}

# === Set device ===
args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {args['device']}")

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

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Load datasets ===
args['train_dataset'] = ImageFolder(root=args['train_dir'], transform=args['transform'])
args['val_dataset'] = ImageFolder(root=args['val_dir'], transform=args['transform'])
args['test_dataset'] = ImageFolder(root=args['test_dir'], transform=args['transform'])

args['train_loader'] = DataLoader(args['train_dataset'], batch_size=args['batch_size'], shuffle=True)
args['val_loader'] = DataLoader(args['val_dataset'], batch_size=args['batch_size'], shuffle=False)
args['test_loader'] = DataLoader(args['test_dataset'], batch_size=args['batch_size'], shuffle=False)

# === Class information ===
args['class_labels'] = args['train_dataset'].classes
args['num_classes'] = len(args['class_labels'])
args['y_train'] = [label for _, label in args['train_dataset'].samples]

# === Reweighting ===
def get_effective_class_weights(args):
    beta = args['reweight_beta']
    counts = Counter(args['y_train'])
    effective_num = [(1 - beta) / (1 - np.power(beta, counts[i])) for i in range(args['num_classes'])]
    weights = np.array(effective_num)
    weights = weights / np.sum(weights) * args['num_classes']
    return torch.tensor(weights, dtype=torch.float32).to(args['device'])

# Build model
def build_model(args):
    model = models.resnet18(pretrained=(args['weights'] == 'imagenet'))
    model.fc = nn.Linear(model.fc.in_features, args['num_classes'])
    return model.to(args['device'])


# Init Model, Loss, Optimizer
model = build_model(args)
print(f"Using {torch.cuda.device_count()} GPUs.")
# model = nn.DataParallel(model)  # Wrap the model for multi-GPU
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


# === Model Summary ===
print("MODEL SUMMARY")
summary(args['model_instance'], input_size=(args['num_channels'], args['im_size'], args['im_size']))

# === Load from Checkpoint (if exists) ===
checkpoint_path = os.path.join(args['drive_checkpoint_dir'], 'finetuned-epoch_100.pt')
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=args['device'])
    try:
        args['model_instance'].load_state_dict(checkpoint['model_state_dict'])  # NOT `.module` if not using DataParallel
    except RuntimeError:
        # If model was saved using DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            new_state_dict[k.replace("module.", "")] = v
        args['model_instance'].load_state_dict(new_state_dict)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

# === EVALUATE ===
all_preds = []
all_labels = []

args['model_instance'].eval()
with torch.no_grad():
    for images, labels in args['test_loader']:
        images = images.to(args['device'])
        outputs = args['model_instance'](images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# === Classification Report ===
print("\n=== Classification Report ===")
report = classification_report(all_labels, all_preds, target_names=args['class_labels'], zero_division=0)
print(report)
