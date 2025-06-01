## EA-based targeted adversarial attacks based on https://github.com/aliotopal/EAbasedAttack

# Install dependencies
import subprocess
subprocess.run(['pip', 'install', 'torchvision', 'tqdm'], check=True)

# Import libraries
import os, time, io, sys
import random
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from collections import Counter
from PIL import ImageFile
import torch.nn.functional as F
from typing import Optional
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

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
    'batch_size': 64,
    'num_channels': 3,
    'group': 'type',  # Options: binary, type, family
    'color_mode': 'rgb',  # Options: rgb, grayscale
    'epochs': 100,
    'model': 'resnet18',
    'alpha': 1e-4, # Learning rate
    'weights': None,  # Options: None (random), imagenet
    'loss': 'categorical_crossentropy',  # Options: categorical_crossentropy, categorical_focal_loss
    'reweight': 'effective_num', #Effective Number Class Reweighting; Other option: None
    'reweight_beta': 0.999,
    'seed': 1,
    # 'device': torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else "cpu",
    # 'pseudo_benign_class_index': 0,  # 'addisplay' class is chosen as the pseudo benign class
    'epsilon': 8, # Epsilon value for the attacks, maximum allowed pixel change [-epsilon, epsilon]
    'train_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\train', # Train dataset
    'val_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\val', # Validation dataset
    'test_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\test', # Test dataset
    'attacks_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\attacks18\\' # Folder where the attacks will be saved
}

# Set device and check if GPU is available
args['device'] = torch.device("cuda")
print("Using DataParallel on GPUs:", [i for i in range(torch.cuda.device_count())])
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 2)
    print(f"GPU {i}: {name} with {mem} MB memory")
print(f"Using device: {args['device']}")


# Transforms
args['transform'] = transforms.Compose([
    transforms.Resize((args['im_size'], args['im_size'])),
    transforms.ToTensor(),
])

# This is needed due to truncated files in the datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the datasets and loaders
args['train_dataset'] = ImageFolder(root=args['train_dir'], transform=args['transform'])
args['val_dataset'] = ImageFolder(root=args['val_dir'], transform=args['transform'])
args['test_dataset'] = ImageFolder(root=args['test_dir'], transform=args['transform'])

args['train_loader'] = DataLoader(args['train_dataset'], batch_size=args['batch_size'], shuffle=True)
args['val_loader'] = DataLoader(args['val_dataset'], batch_size=args['batch_size'], shuffle=False)
args['test_loader'] = DataLoader(args['test_dataset'], batch_size=args['batch_size'], shuffle=False)

# Class Info
args['class_labels'] = args['train_dataset'].classes
args['num_classes'] = len(args['class_labels'])
# Since we have corrupted images, we need a safe way to extract y_train without loading the images
args['y_train'] = [label for _, label in args['train_dataset'].samples]

# Build model
def build_model(args):
    model = models.resnet18(pretrained=(args['weights'] == 'imagenet'))
    model.fc = nn.Linear(model.fc.in_features, args['num_classes'])
    return model.to(args['device'])


# Load model
model = build_model(args)
print(f"Using {torch.cuda.device_count()} GPUs.")
model = nn.DataParallel(model)  # Wrap the model for multi-GPU
checkpoint = torch.load("C:\\Users\\Boss\\Downloads\\malnet-images\\checkpoints\\epoch_100.pt", map_location=args['device'])
model.load_state_dict(checkpoint["model_state_dict"]) # Weights
model.to(args['device'])
model.eval()
# Function to load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = args['transform'](image)  # Shape: [3, 256, 256]
    return tensor.to(args['device'])   # Don't add unsqueeze here


## EA-based targeted adversarial attacks based on https://github.com/aliotopal/EAbasedAttack
## Modified to work with PyTorch
class EA:
    def __init__(self, klassifier, max_iter, confidence, targeted):
        self.klassifier = klassifier
        self.max_iter = max_iter
        self.confidence = confidence
        self.targeted = targeted
        self.pop_size = 50
        self.numberOfElites = 10

    def _get_class_prob(self, x_batch):
        """
        :param x_batch: an array of images to be predicted: (50, 224, 224, 3) shaped array
        :return: returns the probability of the images in an array (50, 43) where 43 is the number of classes
        """
        with torch.no_grad():
            out = self.model(x_batch) 
            return F.softmax(out, dim=1).detach().cpu().numpy()

    def _get_fitness(self, probs, class_idx):
        """
        It simply returns the CNN's probability for the images but different objective functions can be used here.
        :param probs: an array of images' probabilities of selected CNN
        :return: returns images' probabilities in an array (50,)
        """
        return probs[:, class_idx] 

    def _selection_untargeted(self, images, fitness):
        """
        Population will be divided into elite, middle_class, and didn't make it based on
        images (individuals) fitness values. The images furthest from the ancestor category will be
        closer to be in the elite.
        :param images: the population of images in an array: size (pop_size, 224, 224, 3)
        :param fitness: an array of images' propabilities of selected CNN
        :return: returns a tuple of elite, middle_class images, fitness values of elites, index number of elites
                in the population array, and random_keep images as numpy arrays.
        """
        idx_elite = fitness.argsort()[: self.numberOfElites]
        # print("IDX ELITE:", idx_elite)
        half_pop_size = images.shape[0] // 2
        idx_middle_class = fitness.argsort()[self.numberOfElites:half_pop_size]
        elite = images[idx_elite]
        middle_class = images[idx_middle_class]

        possible_idx = list(set(range(images.shape[0])) - set(idx_elite))
        idx_keep = random.sample(possible_idx, (half_pop_size - self.numberOfElites))
        random_keep = images[idx_keep]
        return elite, middle_class, random_keep

    def _selection_targeted(self, images, fitness):
        idx_elite = fitness.argsort()[-self.numberOfElites:]
        half_pop_size = images.shape[0] // 2
        idx_middle_class = fitness.argsort()[half_pop_size:-self.numberOfElites]

        elite = images[idx_elite].clone()
        middle_class = images[idx_middle_class].clone()

        possible_idx = list(set(range(0, images.shape[0])) - set(idx_elite))
        idx_keep = random.sample(possible_idx, (half_pop_size - self.numberOfElites))
        random_keep = images[idx_keep].clone()

        return elite, middle_class, random_keep

    @staticmethod
    def _get_no_of_pixels(im_size: int) -> int:
        """
        :param im_size: Original inputs' size, represented by an integer value.
        :return: returns an integer that will be used to decide how many pixels will be mutated
        in the image during the current generation.
        """
        u_factor = np.random.uniform(0.0, 1.0)
        n = 60  # normally 60, the smaller n -> more pixels to mutate
        res = (u_factor ** (1.0 / (n + 1))) * im_size
        no_of_pixels = im_size - res
        return no_of_pixels

    @staticmethod
    def _mutation(
        _x: torch.Tensor,
        no_of_pixels: int,
        mutation_group: torch.Tensor,
        percentage: float,
        boundary_min: int,
        boundary_max: int,
    ) -> torch.Tensor:
        """
        PyTorch version of mutation operator.

        :param _x: Tensor with the original input to be attacked. Shape: (C, H, W) or similar
        :param no_of_pixels: Number of pixels to mutate
        :param mutation_group: Tensor of individuals to mutate. Shape: (N, C, H, W)
        :param percentage: Fraction of individuals to mutate
        :param boundary_min: Minimum pixel value (e.g., 0)
        :param boundary_max: Maximum pixel value (e.g., 255)
        :param epsilon: Max noise range (perturbation bound)
        :return: Mutated group of individuals
        """

        # If input is a NumPy array, convert to PyTorch tensor
        if isinstance(_x, np.ndarray):
            _x = torch.tensor(_x, dtype=torch.float32)

        if isinstance(mutation_group, np.ndarray):
            mutation_group = torch.tensor(mutation_group, dtype=torch.float32)
        
        # Ensure the mutation group is on the same device as _x
        _x = _x.to(args['device'])
        mutation_group = mutation_group.to(args['device'])

        # Convert to channels-first format if currently channels-last
        if mutation_group.shape[-1] <= 4:  # heuristic: channels-last likely if last dim small
            mutation_group = mutation_group.permute(0, 3, 1, 2).contiguous()  # [N, H, W, C] -> [N, C, H, W]
        if _x.shape[-1] <= 4:
            _x = _x.permute(2, 0, 1).contiguous()  # [H, W, C] -> [C, H, W]

        mutated_group = mutation_group.clone()
        no_of_individuals = mutated_group.shape[0]

        num_to_mutate = int(no_of_individuals * percentage)
        for individual in range(num_to_mutate):
            c, h, w = _x.shape  # Assuming shape is (C, H, W)

            loc_x = torch.randint(0, c, (no_of_pixels,))
            loc_y = torch.randint(0, h, (no_of_pixels,))
            loc_z = torch.randint(0, w, (no_of_pixels,))

            deltas = torch.randint(0, 2, (no_of_pixels,)) * 2 - 1  # Random -1 or 1
            mutated_vals = mutated_group[individual, loc_x, loc_y, loc_z] - deltas.to(mutated_group.device)
            mutated_group[individual, loc_x, loc_y, loc_z] = mutated_vals

            # Clip the noise to epsilon
            noise = mutated_group[individual] - _x
            noise = torch.clamp(noise, -args['epsilon'], args['epsilon'])
            mutated_group[individual] = _x + noise

        # Mutation check: Count the number of changed pixels
        num_changed = torch.sum((mutated_group[individual] != _x).int()).item()
        print(f"\nMutation - Individual {individual}: Changed pixels = {num_changed}")

        mutated_group = torch.clamp(mutated_group, boundary_min, boundary_max)
        return mutated_group



    @staticmethod
    def _get_crossover_parents(crossover_group: np.ndarray) -> list:
        size = crossover_group.shape[0]  # size = 30
        no_of_parents = random.randrange(0, size, 2)  # gives random even number between 0 and size.
        parents_idx = random.sample(range(0, size), no_of_parents)
        return parents_idx  # returns parents indexs who will be used for corssover.


    @staticmethod
    def _crossover(_x: np.ndarray, crossover_group: np.ndarray, parents_idx: list) -> np.ndarray:
        crossedover_group = crossover_group.clone()
        initial_group = crossover_group.clone() # For debugging purposes
        for i in range(0, len(parents_idx), 2):
            parent_index_1 = parents_idx[i]
            parent_index_2 = parents_idx[i + 1]
            # ensure crossover_range is at least 1 before using it
            crossover_range = max(1, int(_x.shape[0] * 0.15)) # 15% of the image will be crossovered.
            size_x = np.random.randint(1, crossover_range + 1)
            start_x = np.random.randint(0, _x.shape[0] - size_x + 1)
            size_y = np.random.randint(1, crossover_range + 1)
            start_y = np.random.randint(0, _x.shape[1] - size_y + 1)
            z = np.random.randint(_x.shape[2])
            temp = crossedover_group[
                   parent_index_1,
                   start_x: start_x + size_x,
                   start_y: start_y + size_y,
                   z,
                   ]
            crossedover_group[
            parent_index_1, start_x: start_x + size_x, start_y: start_y + size_y, z
            ] = crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z]
            crossedover_group[parent_index_2, start_x: start_x + size_x, start_y: start_y + size_y, z] = temp

        # Crossover check: Count the number of changed pixels
        num_changes = torch.sum((crossedover_group != initial_group).int()).item()
        print(f"\nCrossover - Total pixels changed: {num_changes}")
        return crossedover_group

    def _generate(self, x: np.ndarray, y: Optional[int] = None) -> np.ndarray:
        """
        :param x: Original input image (numpy array of shape HxWxC).
        :param y: Target class index (int).
        :return: Adversarial example as a numpy array.
        """
        boundary_min = 0
        boundary_max = 255
        
        # Preprocess the input image
        def preprocess_tensor(img_tensor):
            img_tensor = img_tensor / 255.0 if img_tensor.max() > 1 else img_tensor
            return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])(img_tensor).unsqueeze(0).to(args['device'])

        # Convert the input image to a tensor
        def predict_batch(imgs):
            self.klassifier.eval()
            imgs = imgs.float() / 255.0  # Already tensor
            imgs = torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(img)
                for img in imgs
            ])
            with torch.no_grad():
                outputs = self.klassifier(imgs)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

        # Convert the input image to a batch of size pop_size
        images = torch.stack([x.clone()] * self.pop_size).to(args['device'])  # shape: (pop_size, C, H, W)

        start_time = time.time()
        count = 0

        # Get original prediction
        orig_tensor = preprocess_tensor(x)
        with torch.no_grad():
            output = self.klassifier(orig_tensor)
            ancestor_index = output.argmax().item()
            ancestor_conf = torch.softmax(output, dim=1)[0, ancestor_index].item()

        ## Dynamically choose a nearby target class
        if y is None and self.targeted:
            # Automatically pick a more reachable target class (excluding ancestor)
            top_k_classes = torch.topk(output, 5, dim=1).indices[0].tolist()
            # Remove the original class index from top_k
            top_k_classes = [cls for cls in top_k_classes if cls != ancestor_index]
            # Use the second most likely class as the new target (or pick at random)
            y = top_k_classes[0] if top_k_classes else (ancestor_index + 1) % args['num_classes']
            print(f"Auto-selected target class index: {y} ({args['class_labels'][y]})")


        # Extract original class label from path
        class_folder_name = Path(image_path).relative_to(args['test_dir']).parts[0]
        true_class_idx = args['class_labels'].index(class_folder_name)

        print(f"\n--- Original Prediction Results ---")
        print(f"Predicted class: {ancestor_index} ({args['class_labels'][ancestor_index]})\n")
        print(f"Confidence     : {ancestor_conf:.4f}\n")
        print(f"True class     : {true_class_idx} ({class_folder_name})\n")

        if self.targeted:
            print(f"Target ('benign') class: {y} ({args['class_labels'][y]})")

        best_overall_prob = 0.0 if self.targeted else 1.0 # For debugging purposes

        # Check if the original image is already classified as the target class
        while count < self.max_iter: # Stop after max_iter iterations
            preds = predict_batch(images) # (50, 43)
            # Choose the class index we want to measure fitness against
            class_idx = y if self.targeted else ancestor_index
            # Pass full predictions and the class index to get fitness scores
            fitness = self._get_fitness(preds, class_idx)
            # if self.targeted:
            #     class_probs = preds[:, y]
            # else: # untargeted
            #     class_probs = preds[:, ancestor_index]
            # fitness = self._get_fitness(class_probs) 

            # Select elite, middle, and random individuals
            if self.targeted:
                elite, middle, random_keep = self._selection_targeted(images, fitness)
            else:
                elite, middle, random_keep = self._selection_untargeted(images, fitness)

            # Get the best individual
            best_idx = np.argmax(fitness) if self.targeted else np.argmin(fitness)
            best_img = images[best_idx]
            best_prob = fitness[best_idx]
            best_class = preds[best_idx].argmax()

            # Check fitness
            if count == 0:
                best_overall_prob = best_prob
            else:
                if best_prob > best_overall_prob:
                    print(f"\nIteration {count}: New best fitness {best_prob:.4f}")
                    best_overall_prob = best_prob

            # Print progress
            elapsed = time.time() - start_time
            progress = (count + 1) / self.max_iter
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed

            print(
                f"\rIteration {count + 1}/{self.max_iter} "
                f"({progress * 100:.2f}%) | "
                f"Best class: {best_class} | "
                f"Probability: {best_prob:.4f} | "
                f"Time left: {remaining / 60:.2f} min",
                flush=True,
                end=""
            )


            if self.targeted and best_class == y and best_prob > self.confidence:
                print(f"\nTargeted adversarial image created in {count} iterations.")
                return best_img
            if not self.targeted and best_class != ancestor_index:
                print(f"\nUntargeted adversarial image created in {count} iterations.")
                return best_img

            
            # Mutation and crossover
            # Select population classes based on fitness and create 'keep' group
            # Select random pixels to mutate based on the fitness of the individuals
            elite_tensor = elite.clone().detach().to(args['device'])
            random_keep_tensor = random_keep.clone().detach().to(args['device'])
            keep = torch.cat([elite_tensor, random_keep_tensor], dim=0)
            no_of_pixels = int(self._get_no_of_pixels(x.shape[0] * x.shape[1] * x.shape[2]))
            # Reproduce individuals by mutating Elits and Middle class 
            # Mutate the middle class individuals
            mutated_middle = self._mutation(x, no_of_pixels, middle, 1.0, boundary_min, boundary_max)
            # Mutate the elite individuals
            mutated_keep1 = self._mutation(x, no_of_pixels, keep, 1.0, boundary_min, boundary_max)
            # Mutate the random individuals
            mutated_keep2 = self._mutation(x, no_of_pixels, mutated_keep1, 1.0, boundary_min, boundary_max)
            
            # Crossover
            # Select random pixels to mutate based on the fitness of the individuals
            all_mutants = torch.cat((mutated_middle, mutated_keep2), dim=0)
            # Select parents for crossover 
            parents_idx = self._get_crossover_parents(all_mutants) 
            crossover_group = self._crossover(x, all_mutants, parents_idx) 
            # Create new population by combining elite, mutated individuals, and crossover group
            elite = elite.clone().detach().to(args['device'])
            crossover_group = crossover_group.clone().detach().to(args['device'])
            # Ensure the shapes are compatible for concatenation
            if elite.shape[-1] == 3:  # Convert from NHWC to NCHW
                elite = elite.permute(0, 3, 1, 2).contiguous()

            if crossover_group.shape[-1] == 3:
                crossover_group = crossover_group.permute(0, 3, 1, 2).contiguous()

            # Combine elite, mutated individuals, and crossover group
            images = torch.cat((elite, crossover_group), dim=0)

            count += 1

        print(f"\nFailed to generate adversarial image in {self.max_iter} iterations.")
        return x  # return original if unsuccessful


## Perform attacks at different confidence levels

# Load the test image
image_path = "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\banker++trojan\\acecard\\4CDF1327209FAD450F0802CB86829F588DF41532A320E3140FF780E0998D8DCF.png"
original_image = load_image(image_path)  # Shape: [1, 3, 256, 256]

# Define parameters for the attack
max_iter = 10000
# confidence_levels = [0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Loop over confidence levels
# for conf in confidence_levels:
conf = 0.40
print(f"\n\n=== Running EA Attack at confidence: {conf} with epsilon: {args['epsilon']} ===")
attack_start = time.time()

# Initialize the EA attacker
attacker = EA(
    klassifier=model,
    max_iter=max_iter,
    confidence=conf,
    targeted=True # True for targeted attack; False for untargeted attack
)

# Generate adversarial image
# adversarial_image = attacker._generate(original_image.clone(), y=args['pseudo_benign_class_index']) #hardcoded benign class
# adversarial_image = attacker._generate(original_image.clone(), y=None) #dynamically auto-selected benign class
benign_index = args["class_labels"].index("benign")
adversarial_image = attacker._generate(original_image.clone(), y=benign_index)

# Handle tensor to NumPy + format for saving
if isinstance(adversarial_image, torch.Tensor):
    if adversarial_image.dim() == 4:
        adversarial_image = adversarial_image.squeeze(0)
    if adversarial_image.shape[0] == 3:
        adversarial_image = adversarial_image.permute(1, 2, 0)  # [C, H, W] → [H, W, C]
    adversarial_image = adversarial_image.detach().cpu().numpy()

# Save as .npy
filename_base = f"malnet_EA_conf_{int(conf * 100)}"
np.save(os.path.join(args['attacks_dir'], f"{filename_base}.npy"), adversarial_image)

# Convert to uint8 for saving as PNG
if adversarial_image.max() <= 1.0:
    adversarial_image = (adversarial_image * 255).astype(np.uint8)
else:
    adversarial_image = adversarial_image.astype(np.uint8)

# Save as image
Image.fromarray(adversarial_image).save(os.path.join(args['attacks_dir'], f"{filename_base}.png"))
print(f"Saved adversarial image for confidence={conf} at: {args['attacks_dir']}")

attack_duration = time.time() - attack_start
print(f"\nCompleted EA attack for confidence={conf} in {attack_duration / 60:.2f} minutes.")


## Test the adversarial image on the model

# Load the adversarial image just saved
adv_path = os.path.join(args['attacks_dir'], f"{filename_base}.npy")
adv_np = np.load(adv_path)

# Convert numpy array to tensor
adv_tensor = torch.from_numpy(adv_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Apply normalization used during training
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
adv_tensor = normalize(adv_tensor.squeeze()).unsqueeze(0).to(args['device'])

# Forward pass through the model
with torch.no_grad():
    output = model(adv_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    # Extract the class folder name from the image path
    class_folder_name = Path(image_path).relative_to(args['test_dir']).parts[0]
    # Get the corresponding label index
    true_class_idx = args['class_labels'].index(class_folder_name)

    print(f"\n--- Adversarial Prediction Results ---")
    print(f"Predicted class: {pred_class} ({args['class_labels'][pred_class]})\n")
    print(f"Confidence     : {confidence:.4f}\n")
    print(f"True class     : {true_class_idx} ({class_folder_name})\n")


## Visualize the original vs adversarial images ##

# Denormalize function to convert tensor back to image
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(tensor.device)
    return tensor * std + mean

# Prepare original image
orig_tensor = denormalize(original_image.clone().cpu()).clamp(0,1)
orig_np = orig_tensor.permute(1,2,0).numpy()

# Prepare adversarial image (already denormalized and saved earlier)
adv_np_disp = adversarial_image.astype(np.float32) / 255.0  # Ensure it's in range [0, 1]

# Get prediction labels
orig_label = args['class_labels'][true_class_idx]
orig_pred = args['class_labels'][torch.argmax(model(original_image.unsqueeze(0))).item()]
adv_pred = args['class_labels'][pred_class]  # Already computed
adv_label = orig_label  # True class remains same

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(orig_np)
axes[0].axis('off')
axes[0].set_title(f"Original\nPred: {orig_pred}\nTrue: {orig_label}")

axes[1].imshow(adv_np_disp)
axes[1].axis('off')
axes[1].set_title(f"Adversarial\nPred: {adv_pred}\nTrue: {adv_label}")

# Save figure
fig_path = os.path.join(args['attacks_dir'], f"{filename_base}_comparison.png")
plt.tight_layout()
plt.savefig(fig_path)
plt.close()

print(f"\nSaved side-by-side comparison image at: {fig_path}")
