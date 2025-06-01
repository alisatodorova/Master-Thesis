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
sys.stdout = open("C:\\Users\\Boss\\Downloads\\malnet-images\\attacks18\\output.log", "w", buffering=1)
sys.stderr = open("C:\\Users\\Boss\\Downloads\\malnet-images\\attacks18\\error.log", "w", buffering=1)

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
    'epsilon': 32, # Epsilon value for the attacks, maximum allowed pixel change [-epsilon, epsilon]
    'train_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\train', # Train dataset
    'val_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\val', # Validation dataset
    'test_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\test', # Test dataset
    'attacks_dir': 'C:\\Users\\Boss\\Downloads\\malnet-images\\attacks18' # Folder where the attacks will be save
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
checkpoint = torch.load("C:\\Users\\Boss\\Downloads\\malnet-images\\Model18-FineTuned\\checkpoints\\epoch_100.pt", map_location=args['device'])
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
        # print(f"\nMutation - Individual {individual}: Changed pixels = {num_changed}")

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
        # print(f"\nCrossover - Total pixels changed: {num_changes}")
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

        # ## Dynamically choose a nearby target class
        # if y is None and self.targeted:
        #     # Automatically pick a more reachable target class (excluding ancestor)
        #     top_k_classes = torch.topk(output, 5, dim=1).indices[0].tolist()
        #     # Remove the original class index from top_k
        #     top_k_classes = [cls for cls in top_k_classes if cls != ancestor_index]
        #     # Use the second most likely class as the new target (or pick at random)
        #     y = top_k_classes[0] if top_k_classes else (ancestor_index + 1) % args['num_classes']
        #     print(f"Auto-selected target class index: {y} ({args['class_labels'][y]})")


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

        benign_index = args["class_labels"].index("benign")
        benign_prob_history = []
        last_benign_prob = None
        benign_first_predicted_at = None  # For tracking when benign becomes best_class
        benign_first_confidence = None  # confidence at that iteration


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
                    # print(f"\nIteration {count}: New best fitness {best_prob:.4f}")
                    best_overall_prob = best_prob

            # Print progress
            elapsed = time.time() - start_time
            progress = (count + 1) / self.max_iter
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed

            benign_prob = preds[best_idx][benign_index]

            # Track when benign prob changes (significantly different to avoid float noise)
            if last_benign_prob is None or abs(benign_prob - last_benign_prob) > 1e-5:
                benign_prob_history.append((count, benign_prob))
                last_benign_prob = benign_prob

            # Track when benign becomes the best predicted class
            if benign_first_predicted_at is None and best_class == benign_index:
                benign_first_predicted_at = count
                benign_first_confidence = benign_prob
                print(f"\n[INFO] Benign class became best class at iteration {count} with confidence {benign_prob:.4f}!")

            print(
                f"\rIteration {count + 1}/{self.max_iter} "
                f"({(count + 1) / self.max_iter * 100:.2f}%) | "
                f"Benign prob: {benign_prob:.4f} | "
                f"Best class: {best_class} | "
                f"Time left: {remaining / 60:.2f} min",
                flush=True,
                end="",
                file = sys.stderr
            )


            if self.targeted and best_class == y and best_prob > self.confidence:
                print(f"\nTargeted adversarial image created in {count} iterations.")
                return best_img, benign_prob_history, benign_first_predicted_at, benign_first_confidence

            # if not self.targeted and best_class != ancestor_index:
            #     print(f"\nUntargeted adversarial image created in {count} iterations.")
            #     return best_img

            
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
        return x, benign_prob_history, benign_first_predicted_at, benign_first_confidence  # return original if unsuccessful


## Perform attacks at different confidence levels

# Load the test image
image_paths = ["C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\adwo\\1E0AB6E6412B20AC6BF2206663982860457E0CD728406AB6A68103FBB65DD936.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\adwo\\9F1AB44375F20858F01613ED68DCE2D4DAD8B77EAD1B14206E82600D5387AD72.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\adwo\\B841D39EA54DE3F189B2325469B5D289E7A02C126B82335F814553B334E2B709.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\genbl\\0AF07C7D62334458390E4BAA93BD5720AB0CCFE46446698EA982D0C73E9CB4DB.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\genbl\\90C94EDC3F5EE3FB2BE23ED0C3C7311EF6E892BCD7D8F21BB2D275C2EA9175EB.png", 
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\genbl\\D8E8C336C9F45BDB93F26F6ADDF7F1D70021E58E0251DA760025895CE9283398.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\apperhand\\84174EEBEE1433D774C03D6A0D2811BC3DAD171EC981E997EB609D6FA0D270A8.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\apperhand\\AB52637706F087C318D4B5A4A2E215C90AAEA86988DDFA44ECC68ADD44BFDB44.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\apperhand\\B7AE86828812EEAE5D002A60B57B9E2A353944EC6F216ED656B4912F82C31B5A.png",
               "C:\\Users\\Boss\\Downloads\\malnet-images\\test\\adwareare\\apperhand\\E7545FB7CB1D6430539E8E4AA952F7F73F09BBBBC3534AF086E0D6C4369F2A9C.png"]

# image_paths = ["",
#                "",
#                "",
#                "",
#                "", 
#                "",
#                "",
#                "",
#                "",
#                ""
#                 ]

conf = 0.55
max_iter = 10000
benign_index = args["class_labels"].index("benign")
benign_prob_changes = []  # List of lists: [[(iter, prob), ...], ...]
first_best_iters = []     # Iterations when benign first becomes best
final_benign_probs = []   # Final benign probs
final_benign_iters = []   # Final iter when benign prob was recorded

# Custom callback for tracking
def record_callback_factory():
    state = {
        "benign_prob_trace": [],
        "first_best_iter": None,
        "last_benign_prob": None,
    }


    def record_callback(current_iter, output_probs):
        benign_prob = output_probs[0, benign_index].item()
        best_class = output_probs.argmax().item()

        if state["last_benign_prob"] is None or benign_prob != state["last_benign_prob"]:
            state["benign_prob_trace"].append((current_iter, benign_prob))
            state["last_benign_prob"] = benign_prob

        if best_class == benign_index and state["first_best_iter"] is None:
            state["first_best_iter"] = current_iter

    return record_callback, state


for i, image_path in enumerate(image_paths):
    try:
        print(f"\n\n=== [{i+1}/{len(image_paths)}] Creating EA Adversarial Image for {image_path} ===")
        attack_start = time.time()

        # Initialize attacker
        attacker = EA(
            klassifier=model,
            max_iter=max_iter,
            confidence=conf,
            targeted=True
        )

        # Load and prepare image
        original_image = load_image(image_path)

        # Generate adversarial image and tracking data
        adversarial_image, benign_trace, benign_first_iter, benign_first_conf = attacker._generate(original_image.clone(), y=benign_index)
        print(f"Benign first predicted at iteration {benign_first_iter} with confidence {benign_first_conf:.4f}.")
        
        # Save results
        benign_prob_changes.append(benign_trace)

        if benign_first_iter is not None:
            first_best_iters.append(benign_first_iter)
        else:
            first_best_iters.append(max_iter)  # if never became best, use max_iter as fallback

        if benign_trace:
            final_benign_iters.append(benign_trace[-1][0])
            final_benign_probs.append(benign_trace[-1][1])
        else:
            final_benign_iters.append(0)
            final_benign_probs.append(0.0)

        # Print the benign_prob changes and their iteration
        print(f"\nBenign probability changes (iteration, prob):")
        for iter_num, prob in benign_trace:
            print(f"Iteration {iter_num}: Benign prob = {prob:.4f}")

        # Convert and save
        if isinstance(adversarial_image, torch.Tensor):
            if adversarial_image.dim() == 4:
                adversarial_image = adversarial_image.squeeze(0)
            if adversarial_image.shape[0] == 3:
                adversarial_image = adversarial_image.permute(1, 2, 0)
            adversarial_image = adversarial_image.detach().cpu().numpy()

        filename_base = f"malnet_EA_conf_{int(conf * 100)}_img{i+1}"
        np.save(os.path.join(args['attacks_dir'], f"{filename_base}.npy"), adversarial_image)

        attack_duration = time.time() - attack_start
        print(f"\n Completed Adversarial Image for image {i+1} in {attack_duration / 60:.2f} minutes.")

        # Evaluate result
        adv_path = os.path.join(args['attacks_dir'], f"{filename_base}.npy")
        adv_np = np.load(adv_path)
        adv_tensor = torch.from_numpy(adv_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        adv_tensor = normalize(adv_tensor.squeeze()).unsqueeze(0).to(args['device'])

        with torch.no_grad():
            output = model(adv_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            class_folder_name = Path(image_path).relative_to(args['test_dir']).parts[0]
            true_class_idx = args['class_labels'].index(class_folder_name)

            print(f"\n--- Adversarial Prediction Results for image {i+1}: ({image_path}) ---")
            print(f"Predicted class: {pred_class} ({args['class_labels'][pred_class]})")
            print(f"Confidence     : {confidence:.4f}")
            print(f"True class     : {true_class_idx} ({class_folder_name})")

    except Exception as e:
        print(f"\n[ERROR] Error to generate adversarial image for {i+1} ({image_path}): {e}")
        continue # skip to next image


## Plots ##
# Interpolate all benign_prob curves to the same length
max_points = max(len(trace) for trace in benign_prob_changes)
aligned_iters = np.linspace(0, max_iter, max_points)

all_interpolated_probs = []

for trace in benign_prob_changes:
    iters, probs = zip(*trace)
    interp_probs = np.interp(aligned_iters, iters, probs)
    all_interpolated_probs.append(interp_probs)

# Compute average curve
avg_probs = np.mean(all_interpolated_probs, axis=0)

# Compute average first-best iteration and final benign point
avg_first_best_iter = np.mean(first_best_iters)
avg_final_iter = np.mean(final_benign_iters)
avg_final_prob = np.mean(final_benign_probs)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(aligned_iters, avg_probs, label="Average benign prob", linewidth=2)

# Dot when benign first becomes best class
plt.scatter([avg_first_best_iter], 
            [np.interp(avg_first_best_iter, aligned_iters, avg_probs)], 
            color="green", marker="o", label="Average First benign best", zorder=5)

# Star at final benign probability
plt.scatter([avg_final_iter], 
            [avg_final_prob], 
            color="red", marker="*", s=150, label="Average Final benign prob", zorder=5)

plt.xlabel("Iteration")
plt.ylabel("Benign Class Probability")
plt.title("Average Confidence Curve for Benign Class (EA Attack)")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(args['attacks_dir'], "avg_benign_confidence_plot.png")
plt.savefig(save_path)
plt.close()


