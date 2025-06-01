## EA-based targeted adversarial attacks based on https://github.com/aliotopal/EAbasedAttack

# Install dependencies
import subprocess
subprocess.run(['pip', 'install', 'torchvision', 'tqdm'], check=True)

# Import libraries
import os, time, sys
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

# # Redirect print and debugging to log files
# sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
# sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
# sys.stdout = open("C:\\Users\\Boss\\Downloads\\malnet-images\\attacks-adware-grayware\\output-adware-grayware.log", "w", buffering=1)
# sys.stderr = open("C:\\Users\\Boss\\Downloads\\malnet-images\\attacks-adware-grayware\\error-adware-grayware.log", "w", buffering=1)

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
    # 'device': torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else "cpu",
    # 'pseudo_benign_class_index': 0,  # 'addisplay' class is chosen as the pseudo benign class
    'epsilon': 32, # Epsilon value for the attacks, maximum allowed pixel change [-epsilon, epsilon]
    'train_dir': '/scratch/users/atodorova/train', # Train dataset
    'val_dir': '/home/users/atodorova/alisa-thesis/malnet-cnn/val', # Validation dataset
    'test_dir': '/home/users/atodorova/alisa-thesis/malnet-cnn/test', # Test dataset
    'attacks_dir': '/home/users/atodorova/alisa-thesis/malnet-cnn/18-tuned-riskware' # Folder where the attacks will be saved
}

#Set device and check if GPU is available
args['device'] = torch.device("cuda")
# print("Available CUDA devices:", torch.cuda.device_count())
print("Using device:", args['device'])
print("Device name:", torch.cuda.get_device_name(args['device']))
mem = round(torch.cuda.get_device_properties(args['device']).total_memory / 1024**2, 2)
print(f"with {mem} MB memory")

# # Set device and check if GPU is available
# args['device'] = torch.device("cuda")
# # args['device'] = torch.device("cpu")
# print("Using DataParallel on GPUs:", [i for i in range(torch.cuda.device_count())])
# for i in range(torch.cuda.device_count()):
#     name = torch.cuda.get_device_name(i)
#     mem = round(torch.cuda.get_device_properties(i).total_memory / 1024**2, 2)
#     print(f"GPU {i}: {name} with {mem} MB memory")
# print(f"Using device: {args['device']}")


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
checkpoint = torch.load("/home/users/atodorova/alisa-thesis/malnet-cnn/finetuned-epoch_100.pt", map_location=args['device'])
# Strip 'module.' from keys
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint["model_state_dict"].items():
    new_k = k.replace("module.", "")
    new_state_dict[new_k] = v
model = build_model(args)
model.load_state_dict(new_state_dict)# Weights
print(f"Using {torch.cuda.device_count()} GPUs.")
# model = nn.DataParallel(model)  # Wrap the model for multi-GPU
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
# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/0C6E96B3E63C6D40BF9FC4F10261BA6161996FDCD17C751088AC05CDB42DFD81.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/3C120FB0F5F4D667AD5606E825378EC330708FE72EF1BE4B1830C4A773C981B4.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/9C7C35DB2086EE350F1135ADD56A56312B2EA4C03C71A3C3BB9D81550946779D.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/54F8A5354F0ACBC6ECFB3B52EB23A64F2F6A1EC72D02657A64CA80490BC943D5.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/A4AE99AA9DC326F25B61E0FDC368A47455AC34E772CDC20FF65D0F8CBD4783E2.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/D75BE19FD391D5C256DB88A797800F9AF251094BEE3BAD5F704DB9F89025CF29.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/D878DC5CE17C5536F9C17D12C03659E97A0701C3583A713694DAA1705A60C5AB.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/EB2BD8D032FC1DA7C9F0BBCB948D4C1174CA99284C0B355F9C63654CF92C5B56.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/FB529B1AF323BB34B996128793F3CB2D85A7646A8160A8DAF5D28096BC941B41.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/addisplay++adware/genbl/FE4CAE8DE236766462C4901D1879AB46B0878800BE2ED5B414B8C70B3C20D315.png", 
#                 ]
# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/artemis/0D84D7F915B1FC1C72A60BF2D67E001326B78A0EC1960D92244B33848F01FE0B.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/dowgin/67D97A3811C4928ED647A26AF03E8A29DF17B78AD2C34BB92FD28CE01CD9B29E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/trjdown/4AE15C43B54ACC77EBAF4FA6789DBAD600109C89A75CE7E1BFA3A19D18D36BEA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/trjdown/C0648839B24B5C3758A53328EBBA9EE170714D9F8C0C79F167E4BF28CA133CF0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/wkload/5AF8014C0A370D95205CCB637606454A8899F0D9D61D238EB53A7BAF6D26DD69.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/wkload/BA0F6ED1BDCFAD8C87660518FD963139C0657B9E72473C46CBE4E000EAE1121E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/artemis/4019781C24F2F232BAB48375872C443454EE8CDF6B5F36B583668894D38CB424.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/artemis/AFB4F939F7F6386EFA23CA3AE986CAA12D5DFEBE78677D335B934250057F655E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/artemis/BE4CFE2CDCB6C98C3F21DDE1B5E154FDF2D9C93A021479525D9E25335F09DD82.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adload/artemis/FEEB7AA1AA76D288DC5F9CEE07BB82D38643E1E0C4912A9E089DB7E179936568.png"]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/spybot/449FAE366E10B33E0CCD8F8BBE8DE02D969AC128D281DE6655FE9205B26BD712.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/ramnit/37D408853915C1D36304101FFCADDAC0FA1AAA5E9189BEDE9D924D37680AE23A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/picsys/2CF16087FBA070A57D10F8DD26B8F54254FEFDD3940656C8F8556CDA6DE7D83C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/mabezat/B76D59887280DE7AA5F24A857E9A4397B2EC14B94537F6E48A03A1838C50BCCB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/klez/59FA0DE7BF1CD0A94B45EBDD76FC9482ADEA571F6E48A381B523CDF2185761CD.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/downadup/9FD90F3E0DD660D7D34EBEBA00EDD765D293586E51B3AFAFE40081DD2D21C11E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/downadup/C0B8A06A5F6A14983AB8772D83ACBFB0E5D289C2A14F6D61D8804B764239AAAC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/allaple/EBF13E616116FEE02148DC554703555F3CA4A7C4494859F2F074E39F874ED9B7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/allaple/BD6EA6B1C5CBC6C8537B653C4476D47B993F03C4E557EFE50833E9976C69EEED.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/worm/allaple/209F9F353B36C9602C61D2B3335F321DA6C5D8CF4169029972769B5DA0AC4E67.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/scrinject/4E26BD77C5FF3A672BD0F8D689A049A41FA2A9EAD1B6FC11B3868AEA5781CD5F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/scrinject/0AF7DB9F0ACD953D2235D092F56FF1092AE4782D1B1293E1B59EF35AD289FCF0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/lotoor/119FABBCB7029DBDA69D47580C073FC1A3911BD2469D71B59947D1C01ADD563F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/lotoor/878349488AABAF518FCC0BAF9D98DE01BCFFF85CE5D5082F7DB18FDDE4ED5686.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/evilclass/F9A3BC7E6D3BE3C8ED4AD53C15DF0D4E35480F22804C5395147E6BB46775B58C.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/evilclass/C63CE27BEDBDCA1793C09F416B27DB5C629972CE99A475426724FE9688177676.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/evilclass/78EDF505898AD874E832A80F97C4636DBAA68C6349ECFBC7F9359A846EC8307C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/evilclass/4A098454EDEE36C1843FA79A71BF9F955F73E3754875047AC60E3CC9CFE83F39.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/scrinject/3088FF484F1162447C6655D00FF98B13FE26BA69FEE68B142E891DCBCAC1635A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/virus/scrinject/C162E070652C0857A114CD7523A61A07BFC37AF9A8EE3CD55E50ECE2AEA22C3C.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/vbna/4F5F344623D51EA6AA0A1ABFE77A86AE32B29884807CA0ADE86506B9542AB039.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/sysn/FE29D36990FC06DF6F0380DBF7EDCBC56D05F822037B82293534C7A55839443E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/sysn/CD316BE70EB0035EA401B2C5A63A0BE7398BB4D2968DF806295A2B9FB0075C2D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/dowgin/E321C16DAB744E68C13CEA266B478042F6986DE395A9A9E80B47C7FDFDD12266.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/dowgin/62B91954324A244524D8CD5E6DEA5DE80C87569211C654A495D48656361C88C2.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/kuguo/8AD3BA958A3A1B84F99961D6B9D995FBE6F22485B1A355FAEC1CACFE3F16B646.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/kuguo/23BD9308C0A18D9CDBAC0B0F9B5C084C0B1BA5407EB43A373536301D2691FDFC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/kuguo/E87998BE67C7F688CD476F1D29713506A40DE54A6265E1065585E33B98350871.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/sysn/1036D75E938F9047E9D2E1850EF4B7715FFFA08C0F0DDE9856530A93410381C2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandropper/sysn/41714F65D05B2DE2C7F4C1E47E878B4E3489BB2D6DBA708BD2B1270D48CD1DD7.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/bbie/D9B1635E8F204AC5DFD77117B89A7F5C0356002400E5E2F9E746FBB7C0A12717.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/axde/9E90B374429BE437B7EB0EF9DB0CC344F87354CA17C8C3BA3249DADFFD1DFEAE.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/bawb/F405743C14CED404AC83A5FC78D56C8125FC77094581ED358864964077D6B58C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/bawb/2CF54FAE178168BA292F19DE020FD26DF30B4AFA95796FBA58519791F8CA14E6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/bbax/3C7B18BD5AF72FAC6BDD3083CC615837675D7B31F7DE524C037FA8A8A2B0D957.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/bbax/92D32975E9C3471D6090FF753C1BE97F36503BDF1A4B0DFD0CA6164B0FC331F7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/auzz/299E7A766896EBD325DF058AEE284004B423DCFEC3FA818370711488FD369A3D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/auzz/F9A952DEAE7289174F5504E8EA7FF8FEC30D747FF95C35E54C0AE45B1A934C4D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/awnu/2EBF9B2E92F75373CE9058BF33FEE9C032029444720396B416E51B85FCFEF414.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojandownloader/auek/5425FFD877CF0BFDAEB60C8405E005F8EDA148D8B8FA183AF64ACF47B62544F8.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/appoffer/5AF230604DAD826B36D33AD39017CF380459FA96F06AC85BE1342707E38F518F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/badnews/5E564B280C9D24FD14832A4281BF41942E8EC36866D14A6A4C67C173C47A2BE1.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/cimsci/001B1411EF318525B58EFD9809866E587F3F7CD8973C550474403ABCFCD0B744.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/darrrd/61DD8EFC56AA826624575BB0B6548F761F70B47BBD1AE15D5865F72543B0A256.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/elnx/FAA5A4B292D5A44D6BD8CB54A491C2E981ABFC1830B80EDA75E6C57878760670.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/kyview/1B4BBF912AD6BCF9042B7A2CFDB1DC86186F04C11AD03D699CBBD4BCF1C188A5.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/lootor/B1297FABD603BE93C995F1EF4A8C5FFEB78A55EE3F5BBFFB5723F261E650D5D4.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/rusms/3FFD209DAACF4803135FCB8A13C52DCCF8E82D4A8278FADE287CB16AA7DA6104.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/secapkpacker/7DF56638408EE194EFB19207498CD0CFECE6F8FD1C216214431F5BBA8D9505A9.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trojan/smcc/8597F7EAEB7D46A9F9ABF4698D583552380887A13B8A269139D9C146D594A86D.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/stopsms/0AA9CA4BD4BE2766ED14A1F4965AD98CA89ECB29A3D73199198A927BD22436EF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/monca/007D0FEB96FF7A01F3B04D70BC5D0D73CB61CE399605FEB3872C5656F9633D8F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/malform/54E6DCA0448AC85400076EB091AA25C371A2BC03973CC8C950AC12CA7F37533E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/luomao/4223740ECE2A17B05A2F5C2B07A45BB778964AFE560E84C52E0FC3A05CED4E11.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/jsmshider/634AD188272F67A57DE54FB79F78C92DB2CE069E75D2BAA4AFFE69B92AC402BF.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/goldentouch/9ED75A81D387C8E66291950B4192B36654312D94DF44FD7D83FE5ADEEC95B169.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/ginmaster/24FF5AB6274FC70DB597424A97CF2480312BCD761AED72DDAAD2B4A097342EA8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/fcbhzik/3EEE480D72F75189DCB044264E8B992CCF19ECF706D6B2DC26F50865337D0904.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/fakeflash/2B729490F296EC651F67D365944956F5AF28628583EA8674ECA615B086CD0625.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/troj/eldorado/331BFDE012A910AEC00A1A54281F8B67074FDF52CEECE306EABABDEE3D443E55.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/wkload/1F3396D39BA0DB327448555C53F750565AD509A5AB2AA143E0D7420EB5392A81.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/stopsms/1FF204816093AF601428AC0F7E7D8FE81CEBF904E20F72DAADC3F9D17F90319E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/stopsms/6B4FBDEB427F7820EAEE21FFDCE1A63151FA4AEA4D16A9B161A12607D3903CED.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/slocker/48F591F3784B9DED80546F819FB99DFCD3CE98DA75DE8ABFC2D665CA5AF3EE98.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/revo/193D168C144ECB1081526CABD1E554F2EE94A418FDEF04B25EC80D1EFAC02F6C.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/revo/76646288E19A53869976B75B9759AD1316A386C75F05448A7EE3B61CD340F822.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/fakengry/E5784A0555FA0E8DECC8CBAD2407E6E501059D45B33F515877A8E29ED761959E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/cardserv/88FC4E78D625F549F05538246C2D7C5993D88DD6D5782062B11B260E27055C1B.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/artemis/583C33C3BB7D8C1B718D5E592B6B8E7BF67D72AF6A3AE8C28DB46E5F9054A352.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/trj/andup/8A6476E2C7C19391A1BC8F693CB6D42B0253C502D42DDA78C74B0CD763BFF716.png"
#                 ]

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/plankton/86ECD8958D6A13192A7B927511F38F812BA0FFBD22DC2CD79D15EFAB6E1820CC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/plankton/A824739CF0BE0720148A37454EB5F0B9FED6C6E4506D801D0D6E95EB68CAB78C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/ginmaster/A58A24F8D731DC47AFF89C9938553E328E0B5E29B150E322C423A864A37CAAFA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/ginmaster/C5AD9DD993B506D2E07B74E1A3685408D14FF4AD1F9C6DAA1F4F64A56769AC9F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/dowgin/0C11BF1D0C117B10BEC9CBAB196F6B147A1696C21FED5F10D84D3395503B5B0E.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/dowgin/0D61A34C75D8391803C1846D89B0D1B8F204CA8A241465499A5CB96B9CF98125.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/axent/5DAE85FDB32C06358CD0A27BC92C716FEFAA9D7C92A49D0397A997FD3546DC4C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/axent/D5789CE52E287A99F0A0BDA8CE51C12F49402A0995C032F0137A595810A72BD4.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/adwo/4DE33B1C6FBD33DE049E19D0AFBC6AA8AA5BDD0A8370A11518FD4DC43CF8DE56.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spyware/admogo/5C3F8C3A6B37F841244A34DB63DEFC5221DF9BD18F7DBA428749FE69C30CB632.png"
#                 ]   

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/artemis/48ABC2A80C639C3B412C585A5BCBC7B25CB5E0CD33507743B412ECB2EC996A04.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/artemis/234FEA9076CF773E57BB94CC64FFDB679CFB051648B614E84DCF2C2CD54778EC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/deng/B295612A8A8EDC15720CAD6BBE603510057496FD03020DCA91BBD08ED8FC767D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/deng/BD0B25A95F9D4553441E7401B3652919CB72E97C29C56DF7ECB56BA0B1434251.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/smforw/4EB9BB991A8900A552623114413C33BECE265548361153AC5A3C19F76A3367A5.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/smforw/ED76F9B4BB8A6226F402B73FCF7F0F4966E1605DBEBF7E8B4C39E8FA43029365.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/smsspy/D8A19BC7B427BF70C313C0A1833F2926087AAB478A02D49A15808DDC051E2953.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/smsthief/40E2EAB4A0D4A378807E76C412D79536DD76595A8339DDF272C557FD44D2F699.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/smsthief/51406C386DD85FC184D268F115413AB49E3F8213FE3A6FA062004250400636C1.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy++trojan/smsthief/1E045384A6E1D0C058CCFD628224FF70D7D29AC32D9D0E08195FCD91B5F7E089.png"
#                 ]                            

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/spyagent/0DE708B51454900F6F516FEE1704F34EE47E9411C627C1CFE7C504E6AC9B9797.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/genbl/1C1710264D222656D874E6B32B482275B8F6D112FBDACA8AEC6AF6C8047C3157.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/drkswf/1C01C420AB31F248CC012B5511F33736550C6E4F2BA6C66610B834A4B1C9BC3A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/drkswf/C1087527FF38B6D6F13E9F59A7B36F04106B42F56888CEDC3851ED7507E2707D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/drkswf/DA2183DC98377EC3702267C451705ADFE0C3C5E8EA6F3149346E6C534F3F97BC.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/deng/5CA982538BAC74A1CCB2B9152354C3B1A894D440DD8ED71C0151F31002164F13.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/deng/15DB5455F05618B753969CA35FF50EC4A1F429361FF84E304D3FAC41F1E03FE2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/cxsxzo/66A35EE77572DEE10A9C907F9780D3920F6E60298097D4CDF89A2651EB80B884.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/cxsxzo/B5ADACEA44FDAA4D4089F36F8999BCBAB76F19007250A434ACCE46379777A7FC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spy/artemis/0BAE9E902C9694832CFA595CCD596457D1B4F1CFFA9F5094D096C167AE3975DA.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/cxsxzo/2F8B98D894BF54E07BFA6E719CE9794731B8539D00122050883D449D1F23F9C7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/cimsci/7D559AB6DEA3F26A948048A944A2A7C7917CF26D6891905D1C55FB45032913F0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/ginmaster/9C91211F6E2278C44DF5222C29F78DED35CE1CBB7330578D11CC0E1772A0852A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/mmarket/4BE63279CE76635D90146B6744721133DC5340BF8495629F13931D5507CD87C4.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/monca/580FCE8D4D9E9099A474F3E581921CEC244A53B38E36E36188C55D359D4B436C.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/tachi/0C04D8090B7D154CC07C15DB422371AFC2AF0D2DC721256729A013E6BD2FED1C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/xinhua/7C480404A7B8A974936A666A615862D0C391DEBE0FD278A2556D1709979120CE.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/xinhua/924EC7FC9EA5238EC34B1C3CADAE39A6C540875BE17E72CFD8B8908B38A67473.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/ysapk/5A19BE02D5DAE89FF349BC6A9AA0F1B0515F2ABDB1D7A9A6C4E55CA827DDAD86.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/spr/zypush/1F40EFA869247F1A35D7A0552D26B031410AC8FDD8F1262A0183D108CA097565.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/artemis/00E4B998B21FD405174BBEB6E40C02D4AD8B6643883849DCDD8DF276414484D6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/artemis/1EFD259A4A5574EA615A31DDE222BC85E7FEB4BD60EDE754B9132B6AC3A6D45A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/darrny/50EE669DFEB1730DDC0EA7D9A52F35CA2ACBA85E5E3B39E9F62F47F7FF44EE2A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/darrny/0558492C9C60FEEBB61711C750932B24562456227254752DBF0AEB0AD35D499D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/dvqvuz/0ABE07F4EF7C41DFF684D71EE7655D0925D10D5155AB30858EF470106BEC6DCB.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/eexeas/3ED7AD961119E469F9448B64C5146353FA2F0E873D0264C4E155641DF59EB0DD.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/eexeas/A0D53D424A100FE3A5621A8C83345E264BB9AB785498701DD30A5631F8D373C6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/smspay/9E9BF7F9E826BD53D4D1C7753B6076845615BA8DC1BA1C2EA8BE0375A293C242.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/smspay/3510131A71CFCBB0416BFD529F2BBFF2D6EEDBD8B2987530AC8C98761272BFC2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend++trojan/smspay/F54234A8203FD84543E6C17D57764A46635A47FEB0E3B41ABDF582FC73BE7EA7.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/artemis/0F7E30F4CD49DC2E2D12F65AC7B32CDB2FCB277B125D8C6BAB1C746B347311DE.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/artemis/1E78B7A044FAC9C2D36774F8EBC0EAA871F70EB2C840FBD5B02A0B7C7972FFDC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/igexin/0A2304EA57D027D9CFF3758C2654291548D8E65AFB256E9E82EF10E89042F573.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/igexin/0BBF56F4CF22838DB5DA9571C651E835335FA1972A35FE5CA95B72BC97259C47.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/igexin/00ED6D1821B7DF8782DCE627233815FC42C0808795AFAFBF3972BE0513230F3C.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/koomer/2D9CBF3475740B0892F9B74BA9B5E36D15FD63C75C03115477894BCDD002BFBD.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/koomer/C9DD7B4F7F8F8A3208064AECC6DA1E03040A8BF323724E1A1BAACDE056326294.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/minimob/59B9BEA8A43CD84578BD007963D81DE1C6A98BE5965F3C08BFFD76C84317F344.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/minimob/84E145B07E4F76CA3D82FB1E7A8D5F1CD2C9FC5E05B399756A8F2C8765D0E1AB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/smssend/minimob/475C5DEF53F731AD2B1E9E15F8F6A6A8BE7C48777F7DBA6485B7AC738BA373A1.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/dowgin/88C33F6C3FFF4499DCE63788125CBB0C6230038C7A67E63DDDFED714C65DA5ED.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/dowgin/A46E1E9CF1CDA41D0D7C15E060426B8D00E37326935AB3ADE292503B97735DBC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/dowgin/C1EEF5CF58BD4F80B67486DA6737C648E5F91C0A50E15D68D98852FB0F15433C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/dzbmfp/32F1D174E4437A8615753E7748831857A77C06DD2D499FA50AA8C6C9AE81FF36.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/dzbmfp/C0DC3F23BB5E73DD0E308E1B7F4D8E6F26964F5232A8EB5E92AC99D6FD28CFBC.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/igexin/061DD1CD2EA6815F30712BE29E4C133C97F64AE31F66F0364B9F24C99528CC0F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/igexin/F3E41EA1BD4F1D941ED637A9D92BC23C44598333F36B44187F264A5FDD3D72BD.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/kuguo/CB44618F32305099D0CD352023D8319D198F0FF5EE829D5FCD9084D52847E1DA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/kuguo/7F8C9F177199DAFE3C3D47AE526E659E583CE5CB90E71A5D61A86CFFCD0E50CE.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rootnik++trojan/tencentprotect/0A9382FBE3F2931CC4163B0E6188E0574179C66C928A0F1E572F01B7717227C6.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/modad/129522A9D27E7B09A1868D0451BA5BACBA47E6BAB6D062F63343AEF35C6B8473.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/modad/566795E2AB2C691C1F6CD618422508B25BD41D289EB88D15640176B93B6B4E98.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/genpua/87B1F7351360A06D20C44D552EE2A9979488210305DDAEC4FD68C8C3C095988D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/genpua/BF89FEC21FD7005FA072ADCE0BF6C8F28D3E831AC67D1103D6383150E4142AC3.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/gappusinac/1AB923521BD84DA899520C942F76D6CE09F40505F5578787A4B67BA818F4E4CE.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/gappusinac/8EA362F9D59A47872F661F085A491657E7354B835F9B9622E28BF87B9394E83B.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/evilcert/3CF44A420137DE071C569189443DC6E616E05089F9648591EA0C329DCBCD520A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/evilcert/90A87DAAC6CF85F8BE38E86F66758AC4218B519E4FAADEABC0B8150CE7C446B8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/droidad/16D9A08FFA6F7F316D98D16C86270370412112F02590A88D2430496135CFC46B.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/rog/droidad/BB4B330554EA7E4EAC23CD33B318EE15877BF8CD636B62E755A56D736C37AD97.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/artemis/27DF40B33433D35227861CF78C3B80432C142C1325F11C42B28A1F5107B09FF5.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/artemis/88BC39C62D2EF5F02B63B74EF7FE6F093B7DC79EAF0308993E72ED393669D7E3.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/deng/80D4228C7184A631A0616973C06685C7B7BEE1332B9CD52852938F92B9F160A2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/deng/930C35F29AC49447C4D529C9BF579ED2D1FE4F23341B97C1E0D43DCC1C662797.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/dowgin/28C1875F912F62897E9B9D488123FAC0C4CE6E6CDE27B97F1BE55CB61CCCEF17.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/dowgin/40B9CC0238ACD6B4F6F4FA6F816B7F43EDAD75417227F09CCBE33636B40D0023.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/genbl/4A7A59DD075251C26F3E7F5034F55D339A91CF172F65DCB37BD1C40FC59E0AC4.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/genbl/BF1D89F831698EC6A4C38B36E68FAE4CBB222B974B9DC9F2ED278589BA3CB750.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/gyzz/58309875F5375A33035EBCDC0572697AF4DC367A10197BFD306838DC45C89F93.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware++smssend/gyzz/EC61ED08101A87C3353DAA7E640009DC85AB95AF01CE885D818D5A81C2CAADE7.png"
#                 ] 

image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/adwo/3AA7C7795E2492EDDAFF2D102C61354269D2C5BFEE039F772F390DFBABD28401.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/adwo/088C5B402F9E217574DBCB64885824606A553ABBF2D2129B7EE0FB7CEEEBC7BF.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/anroidosfictus/0B0880956E7FE187D44D4D7F5519ED86378CFC81FDE87562737C726732A5B02D.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/anroidosfictus/6C4B78C92F1BA59C877A1D024CB4B7D409A0DFF9E1A0533080820BF1B4472D3A.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/avagent/5DCCE85A2A5BEC8A9BCAC33BF7D417A173444BB20C48D1A9E83F12B6879748D7.png", 
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/avagent/BA0F333E79BEAC2E80C61CDD1060D0FCD6D6233F35EB1FAE3FD8F55BB398B95A.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/bauts/1BAE34D86CEA58D9D40230B6BCBECD56066CB4F208340A4A4122026FA686E95A.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/bauts/D2FE2773679F4B479FCCDC8F36B72131C56E5D030BC4B7B92C983BC9807CC0CA.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/deng/0A551C7C07459E2E6D41B16F002D9388E2D7DFBF750339CBFE0D7777EDFB9BC5.png",
               "/home/users/atodorova/alisa-thesis/malnet-cnn/test/riskware/deng/1D30FCA8C2EBD78D384D37F2E55276E8104CAED82DC961080A659C4E18BF0DAD.png"
                ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/badpac/1E054F0849B3DDE0FB30E10BE265BC8A606D1A9D33ECC8B96B02E49A44486446.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/badpac/601D4116EDD7D6906B19EF9E368B4502ACD5EF6C63B0F8ECADAD8C292645A5EC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/badpac/32050F0FFAC3C5A46146068EAB37585236E2BF2F946D09B0C2EE490E04FFFFD8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/badpac/101182A89D8B89FE85763AFA5BAABC4325C7724A8AB11128C3930B27F64F6582.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/badpac/A69BF8AE68FC7B5A739E358D931B19989DF5EF1655EB4825036DEAF96D49B053.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/jiagu/04D6F5D44B64ABDAB4AF5FABD41AF5CE5D87321E1DC9CE91C84146D637868D14.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/jiagu/DD9869FC7B5ED2710D5B94E964BE9BF32C91BBC7D61E3EF8C9ADF45FD20AB3F7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/jiagu/A1C5A032A5923ACBC3B807E6A95DBE3050A7E9D5EB0EFC8EFCCF7C88509D6256.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/skymobi/CF362489F596AC5E9C974777E59A8106723EAB1FAD13517F045D0E894241544A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/risktool++riskware++virus/skymobi/47C1AA9096D8063E15156B82C45FC2273A2A0C20FB5B3D062B6F19AFDA546354.png"
#                 ] 


# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dcjebu/5E8CE8B6C48019E406D6842AA45DBE931568237917388B1FA1B91C517533CA82.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dcjebu/23C1F329CFB55A53A572D13B443E55BB37E70E37F0D625FC911771EA93868A08.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dcjebu/028D7D355F349848F4BBB5CE2427AF211CC1660744FC12E22FDDF0A0A7D4A41E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dcpvpa/1D726F88D38BE40582F9B2A48FD48399C5A11CF461004082A4EECCB0396F74F7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dcpvpa/7BCB0563B02D999CD1344CF8862ACB6ACAC3C8E73DEB7923B4694C55CA47CD58.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dononq/9CF42141290F43BD093B6EBEEA53CBA1EB228EFE43E5CA02D92C05706A3C56AD.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dononq/5341394FA6D08B06A2D03FF717F0A58709503AEFF8EDC45B656B95879F70B358.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/doqwkd/2B04F9286BD97E1A477F33FB1CA4F8FC42510CF3515C2A6846FC66F84544B518.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/doqwkd/7B01C008A4E62FA13E77AF4E9227DC71CBB70C1FA6E292F513E974018FC1E4FB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/ransom++trojan/dyrlac/1C1E0AC6C5B9E42DCC3D9FBE122CB8EF745C1A4DCDA7D1DF49823DDA997CFC2F.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/artemis/0BF449D0AAA83E726E4CA2813A2C3497BE6D0BE611F3BF971A68A3A90FD472B0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/artemis/1B273B651B3900090052E073221488F3958B89182B40FF65A0D2D9C5E5B9CED6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/artemis/3B94E5419870FF7D749825F45408238793FAF315961140C6DE4EF99C5BB61055.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/deng/0FD7FA14ED621F5619CE002003C8FDEE09519CC0A8EC3B72022109805DF0E5B7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/deng/88126D4DF0597F9EEBEC47456914E3D86A6CB25BA6DB8A9A6FADCCD5F9EF0C38.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/itracker/28F34F5B5CE7A57C0A5FDA22665AF7FF6F02DABE4F5A254EFA4870837D2670F8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/itracker/FAD869B0C2140C16F9E18465F0AFBE0F3F1718D4DF24A9409B92AC1FD93DC9DB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/smsspy/1A4C56B420CB8922A3C1D3BEDF49E39965813DC27D7BC04BE55F8C27D2191AA0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/smsspy/49B0D3C176E6F6505C5076EF58723E6670C45790A6DCC2A88E4764B214615971.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/monitor/spyagent/2A3AE8F8FF3B8DA780CA2788C06BC98677727800E619DC434B1499D43A3DEAF9.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/0AFFB92524A5D2C4121D67BC68E1E90EBEF3C6094FBD9AC6DD972F1784D02BD2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/0DD0BB31B02C561C5C757CFC73BA219B1F75339E74DB940DE1D3C148D614195F.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/3DECD5EECE31402564EC6DA88C895BF6584842ED7BE6FCE23333C973CC677521.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/4B52E82C280E4B622CA8DF169B77B3549AC6AA87F92860BBF8B3DA02FAD7417E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/5BC0CC57DDE8DFA47EE271E68A6697E3987C69620A88F9B75DA764A639BE354C.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/6DE5AFD1F03821AF8742343700E759B68A5C178E230075A00074A157C51153A4.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/72C3AD3DF616077D89E1EAF995E55D70338F30FF9D76B6BEA916E7AC16F1B29B.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/1250E416A83D8602514A60A7EA83EFCBA277ECADAB96360AE6BC1EDBBBD8F846.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/67629E0997594B151E4C3B21E2DD095F028ED309A8C05A2F2063135FDBBB0B51.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware++trj/awnu/B5803484CD9A576194DBECFBB48D16905ED4C0CA2FE7C6E1D4179E37218C7A4A.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/anydown/5F34914BCF5CC4C377F36778B42898487410C68F994EA5BBF469FD879015B0EB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/anydown/7F88B0C574CA29BE98E550F4CDFE58E5E5D7DDFF289E2508354186BB723688CF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/artemis/12F5C8B198CCB7EF6266CB3E81F2D576AEC5AC4762F63039105D329FEE2B1FC5.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/artemis/CD6FBDEEADCCB301676A3061D6CA12B09DC07F9131D9EA380D6DD24E5D3CD7A3.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/azshouyou/9CB2CF0DEE882839F7BCDE4140F30AB8FA4615C697570380AED9D92F7C829A1E.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/azshouyou/E431C61C1148D3EDA5E3A91CCD43C6087ECFF3A7951C9C031CA88E8EB0A736F9.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/cve/20CBB1FAC2F3ED1073F9E87A0B8814382E9A6AA675D6350839AD7FE9CB827320.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/cve/7204692C5A8771650DF25C7859CD773E886D85DB9910B5270199F924673ED877.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/sdi/1C45DCC352FBEFDF76268388F86C9F0F0EBEA062D57EF8FFB0DD8570DEC669BE.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/malware/sdi/3BBE742D307C8A3753B38247A69761829D0C39737A7B981A92634FE42D6A6EE9.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/floatgame/1F340D156C2A7A58415C2D0479A414DD76B280D4FA7CAE02383D7C8CBE369343.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/floatgame/6CC7741E65D285C293FC840EBB9B13AC322D2FB174C0E825F1F03F7D049C6F8A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/jiagu/0C0318FB07F10481711A3C22ED38A05B252F8718EC94C8D260C6C4132A43F728.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/jiagu/7DC7BFD0306E526FC3A64ADEBDF354B00ED48ED46194DA98A3848B50CB07F6AF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/sexpay/76A4D7BB2C10EF972C915FC4AD6D33A9C6B721EC826EBCBF7C0E58142A8ABA77.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/sexpay/C7FA186FD541E17A24374921E5170BC0BAF1B23578FF3DD47C1754A32CDE4EC6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/tatic/6AA12FDF130A5AB0D209B4F5F30DA7AE2C994107BBBBDAD77198AABAC9471BD2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/tatic/9C1FF4D89E98A1367F6A1C32567A808746D41C0EC641B114ECC4A0D9452D1716.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/teldown/135EFA60F9148F48CBC924E8221144BDD002021B8E913B62C9B172A070723FAE.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/gray/teldown/41186E972ECFB6039A6F2404F0281CA6423457010348E5C4F82EBE73B339CAE9.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/inmobi/16C8A40210A05F50C1542EE56D9C640E96EF75666A3A5B3E49C69F9F5FE6A002.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/inmobi/C35A02EF1859A3E70064206AB0FBE7C4E125CBCF65B3540852143D77375483EF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/penetho/2D9E429BCDEDA803E9A9469B2C1C153D17296DAFF424E50847CE9D0E4571E083.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/penetho/95BCDD99D7A7F7D72C145D200F149C41E953DE52642489E51DB916F1790571E5.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/genpua/0E389BB6ECDCE3E7B1412D1B56240C2F181FB325E0CF79D379612CF6F047AF25.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/genpua/858BC3A08B02D8C94C1F1B855F5FC625566A8B69830C0752BFD519A20195D549.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/artemis/30AB52531C2095DDE5B21D907A35D7C52433A8683DB8DA8E438955F902ABC6D5.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/artemis/FF0784D7E3053CA1034DCFEE2B007C49AF9C1E6471870F8C1D5E049EA101C637.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/appinventor/1D5FD104790543CA8D114313BFB6AB3CE993DAC42B44F81279A68ACC1BC30B97.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/hacktool/appinventor/0382890C42683F267DB47D7D5F64A0B7B90CFCF8E90BA73B27AD41C52EF69E21.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/deng/018D41D3F84A6887BDE1C486D28E03E82FB486D95B20C0E9A69B7545B0950335.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/deng/25D7FCC10189348C709B3F2E2DA43B62DA502E48F58B05DC0882AD08F53CD2BA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/dcshui/3C1DDDFC4B0B0FD9B54ACFF410487378756C577EEABCDBB5DC7C8DD2744E9D95.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/dcshui/1DC30B5DCF309B38204DA6BF8014058FB5CE16BF08E9068376DDFF891C3DCB7C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/cwzgsb/8AAAC80E6834CFE53097F6B614A85679CE0EB3E23300844F89B829CCD23F2703.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/cwzgsb/3AA4D5F1F6BF8278E44E8999E261CC3CE9D0A3DE1BB19D209FA79A59C267CB3C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/boxer/4DAE15113EB12AFB727961A3C853D1B28970CD5EA9C29E010E881472ADA3A2A3.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/boxer/8D81C47F04B6F122513D4D0260A5707FFAE1D022838D8CB4D60A5C4A985CE999.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/artemis/1B20CB30519D370C5236218DA0452715DEC87E6BE74CD4D33FCDABDB96863A46.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeinst++trojan/artemis/6BF08446F93B07E7FA8601DE40332C18BBE2DD25A12DE1163198B8AAF98703AD.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/1C11A8DA1FE475F1803074835FDEDA48032069087A4C179B53587EDBD1B44B31.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/5D86F5C77137CEC5295E826F6C300F483BB91E95080857ACDD0A9D047AED28EA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/6E16EC97DACA63471CA68F3EDD27718F54EF5E3AF388E4B294461309E5F13D85.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/08AC820F5C1528B50C7E68ED1CAC4C6C12EA91BFB3397046857AA56CEB6F4F3B.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/95DA001BF4D4C3D162C0D564C0F98E91A03DDFC6575EA66D2D50250B846B6FC6.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/112F5BB6799DF52ECA274A9C6CD2DBA290867AB1695A5B4D5EF31A36835D4117.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/A3E886953E5AEE4B8E11B5A8B9C66D45EFAACC1B6F12F7B346CFC7E0134D68C0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/B3938D52BAB376AA04477F9BD497967CA3174930B783C861A7B37E466D4D1EE7.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/CBD9CEF89DA3C356C921D6FF94BED125A6AB8CDF5AC9219FE72853D2E97DFDAB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp++trojan/eaeoaa/F84DAF54F2802C7B012F565360D40B49396CEE2756E57D0D79A01FBAD432D9AE.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/genbl/6BAE2F4DBB8788FF19CD74E06CF6F540258C9778FE5EE125E4DB255B64D2F5D2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/genbl/88579BA4B2B9BE10E4338CC4AF027A6CE78D12ABFB42004226B29B1D34A1C441.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/fakeflash/0A1E52B6C758C2AB1433FD635585B188F8C4279EBF4083C5EBC74BCA5698D8BB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/fakeflash/3A79ED26A3F38E4F6CBB302DCAE424D5B69691227A2626D060B20ED252729CA6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/dowgin/0BA61BD7D028683C8C2A11AE5BAAF9E97CC7BD53BC998D9D7EF98DF56C6CBC05.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/dowgin/271C7234C85692092E4356EEA8EF4A7EB4C63DDB883222B40863AE867CBD7648.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/deng/8C98B267ED826E0837C23D2F2CD77966FF75B50EEBAFCDF89487F3B088D5566E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/deng/EA91E67DC5A8AA481244550643D9E3EE9FDC63359B59A2664A068E72FCF5CBF2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/artemis/6FECA038F3FD553012A272CED1B3779172CF95D9DD8F02357E1C12E55C3B9849.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeapp/artemis/1746EFFBC0F1827AFB580CFA92F773EDB8DB5BAECF72360517A1C45994A822C6.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/artemis/05CEEA4B1965CE17D0E2642B7B63602A215B7EE7D52B9255A0F101948CB49DFA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/artemis/C052538D6BE72DA1BB203429DC8A95CBFBBD4A7C872CCF6F3EAF01C888A09682.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/artemis/D46A169CCD4BE06F221473F899F3E83C354F8579E7AF8A953E9A9C8A3FEC05E2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/0C4602835176D1C9C3E778742490C7423F76EE2133097667EE8946FBD414F136.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/6EBFA44F091D261D75582699B19D81251ADB16D686314FE4A9087EB68095F8CF.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/9ACEF0CBF24A0B141FAD0A8D045A3FED349143842FD5C78CC7314C886F6E7005.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/126954AEDB74C7F2464FBC10F5D24F29A6427BF64C3327A6A0B4EDD7DF7EFE10.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/B725E9BB931FAFCDECB6A6FE3144148931C18485F0756CDB3A3CBBBC6B099B81.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/F023C8A3D5A71822D5AB72BBB6CB95D0B78C41AB2B88E2955240CBFECD346142.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/fakeangry/umeng/F75C6A76343C522F18524E75103C8E2CC3F91683D7B1B1BB416A14E63492D83E.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/deng/1A34B5790495A099C0E63C7FC55E1AB4BD2D59D6FC894284505F7D0542947590.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/deng/594B5C01C8EBE18F6458C55AC30FE87A937B5A7379F31FDF23A38C30AA7549CC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/dupentry/0EE310CD6130FB6A2E0A25D869BD70409490ADC74EF1D0B3C732E52A55E2A9FF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/dupentry/952B9A9CE23A7C8E42819CE3599A90F71C784CAC3D47C64CA03AC5322E64B278.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/dupentry/D4C35C8C6C041DCC64CB296A7C690E936D4993B9CF3CDE7E51908A48ED21F236.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/gingerbreak/1E873D659FA04E8E2165F05A94A0ED8B92671F82BDF686DE0AD41F0BD9066A05.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/gingerbreak/5B920D0A1CAA8221D3EF4327383A0564791568FC877DD7CD53D5C04D6AE48EDC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/loopb/0B00025027C83EC7049CBB83118E4ABF8B526D4115F55A9E4B9E92164DA5B797.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/loopb/1DB3061B615CA99020291D61E7DC26ABAF8924A84DE5D8F3DF8D34D11F857B96.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/exploit/loopb/3AD4869A0D9044BBF6CD937E7C63574A1A51DEE0E913A174F052AECEAE2FBCC3.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/adshell/1EC746D3A9603F2B7C2EBA358EC8AC95A2E47A25D48EA410148BC7BF8958B809.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/artemis/0D8AAC2884AF9242977F043D737D797E6288C24143D842BD91595C860D96D655.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/artemis/2C04E47D9493338C7C41F1997F2D9B867559987BE2CF5E0346373D99BD8151D6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/deng/4BBD7225E351BCF67526C607F555DDF13911E385293878576D381C920BDBB2D2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/dowgin/0FB1F7CF5DA073A7DD05DAA6AC57C3503E0ACB5DDC2E2DE3C96A469A959A67C0.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/dowgin/7106F4E3D476DE6C342B6D441486EBED32813966551E4BE8A25B524BC62F2EF2.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/jfpush/2F86F046FE5DA66D53F4913243DAAA0EADD38D2938E6EDCEBE54E14522F7DB7C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/jfpush/54D517E661FA58FBBA95219F25048E2D1012E48ED604657931A0B4E487C41453.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/jfpush/AECF5678AB0AB3D7125A57D41513D73ABD954EA5B5F95D7FB951B3FD2C443351.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/dropper++trojan/jiagu/0CCE38CC6B4CF7C7F38A67FC0577CCA08D357AE26B5EE172450D1056C344C097.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/genpua/0B9DF6D33939EE950E7EC32862592AE2E7BEE3A9308A3AAE5FE4C5A4D5C05BCA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/genpua/0D1F3BCA5A1ADF1636E9EBC0FA9A43C35B22BA4E8ED8976223258B39AC41316A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/hiddenapps/1A76E9A0290ADD5E1165668C367FE25952DF5F3EF16BBC7E8CC41E86A35891C8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/hiddenapps/5B8052A1E55717D2AC0080AA3E69A95D035266D52D45F45F4B76FE53FFA9BB11.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/jiagu/0BE71AAB19AEFFE84B9E8B38F25BD4D3E462EAA7E2ADED10D9AA93FD6DDBD562.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/jiagu/1A09B97EA30FCD7902A379DD23B0071C6C0C1C34D11D67A5986E6B4772ECFFCB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/openconnection/0A9032DD56B2599B3027AE78E63D0D23555C76F78348402007041F278F3A1716.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/openconnection/0C1C3FE15C3588239738B2F2991ED62F70C3BCF93104539162B44DD502E810FB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/tencentprotect/0EC11455F84D9F7A9761B78FFC4080D26F643E0BC8D2A3151ED9EBD9A969B4CB.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/downloader/tencentprotect/4B7560420EDE159023FB404B9E990E92AF85B93D7074A781E96C63C354532C2B.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/artemis/1EDF12D2122BF33183CF56D8EC8B9C1E349AACBDA43B02ADCC02583EF8FBACD0.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/artemis/6C1221B37C395C92F7A40417B9A687A34E7316B668314864A04932BA7DA0102E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/artemis/18C64BD30673F5E26FBD46A2EF03554565C0BB8B31640F448C1728C43C0CA584.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/cve/6B330020108B7EDB6DE21B4BFB0307C1B55E42564C6448562FA6A4AF78B28EA1.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/cve/C820FF5EE23B87663F9EB5F47C30E51C8F384CE1E96C560206E3C411029288A7.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/cxqisx/0A2C349E1DE707447CB62001C54E4EED391D0163A3CDDDF1F0F085FEE50D40B3.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/cxqisx/4FA3FCBC04413808E0239596E654AF74F55EB723C70D4C65604AF3922F59181D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/monca/22BCFE4409B40551CD6EE83BAC70B9C42AE5CFE3FD89D78E13A3C457E8D5787A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/monca/BA7653E2C129B3301829A3CFC389D9DC298769A284D98F4B55AE6D0720BEEB6A.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clickfraud++riskware/genbl/8D1859E92AADA8B0EE8C685FE727F40E8636C96E53A34D0A1C5C611A68FEFD2A.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/dowgin/0C0F4D01706E91A57C511FC2853ED2FA5A4A6983A9746190BF3081BCD54EAFE8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/dowgin/0C0FE38282D6B3B42E97A8DA0C4A275A755AE1AE4419BC74DA2C1607BEC8AD06.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/ganlet/14B52E704A44990CF34C0630F763A4642B718CB0153B12F6409A1143F937A244.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/ganlet/083EF46F4DB067985C3CB6D4D1AE6CCB278F42E662497D12C1BA605DADB2B1D5.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/ganlet/9674BBEEE923530EB37549AB2148D1EFE5032E5E7918D6D826F4C12713EA804C.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/ganlet/CA574B9AA2E93B7E7684C2C06FDF7D0AAC828731F9CF7635964EFAA6DD4393CC.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/kuguo/1932DB8107D365109CC9CAAF8AB7AAC0789B94FE8F494FC0E4479465328922FF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/kuguo/BA46F1F14A8CC3A156010C172545B9BDB7A7EE110E3F5B2B3F86E2A266269348.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/kuguo/FAADBB745CFAB4861A889B7C56B0B2EC29CBD7AD2A7085780D44331974A35236.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/clicker++trojan/kuguo/8626D78ACC002762E56B36DAA473150BF3D3B143864A7256F14B8DAC6AC57039.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/admobads/00FC79832B84F9BCD391A29FB2EFC46C139FA03C827E91FB51BBDFA3003868B3.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/admobads/3B018898C017360D8562D8D4CDB2DA360BBC1898A7239501DB2D64C0434A01C8.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/ads/50BB1F0E7BE545E139F63D8E6E401095BD485993676030E476A1DAB7382A5BBA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/ads/D104FAFD9F33D17D2BCD6F415D575676B9E156E3E0BDA9CB1CCEC4173190D454.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/adwhirlads/1F767FEA4474CCF0AA75D7018BEAEA7BEFABB20FE59D3BA6F4AD50105B8A2FE8.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/adwhirlads/9BB0E6DE7AF300F7552E87CF258DD60CBFAD5F54E5A2D303C691ABDD7BC2725C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/caulyads/2EBC6865F19186AAF04F7B9C87825ADD33B0D99F87C9D46C74F003C167FE1228.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/caulyads/D5DD8BC1E008B65566A20ADC93F216800681D66885E8C00D4D97D1A5E47FC4CA.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/genbl/B315F6719D944C76971A686BC0A9BDF513512F71FB8C3B1DE8019A3842A01863.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/adsware/genbl/B112A86B554938FD1A06153AEF7B84A8955DBEC56C85A5AE0584A88C7FFB6588.png"
#                 ] 

# image_paths = ["/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/aqin/118B0000BEF537F62FD9A57D3670B211D5A20C3001EE1E4CC30F1EC5C287F35E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/aqin/E01B2CFE97E87B4236346730A2430628C21DD87987B42BD657685B124B1446AF.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/aqin/F13969A1A0B759AD3375A9F4D7D6320AAC6F90968F7FB0D70CD0CDE0E450E83C.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/droidkungfu/37F04C728BB0A3DE3EDD3C5D7D8420E78610307EBBF8AF38713B5FB56E22DAD6.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/droidkungfu/BD588C5A6C704C2E73DB012E81FA355768D90D1785B398AB54082BA9D4BDDF89.png", 
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/droidkungfu/E2E20AFB8D2725795E266CC74F9CB50F025FCBC1078C50E5B94FB618E554C56D.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/golddream/1DD721D9C8694E48486495ADDE74F93D9B1786920322B91296603C2E8F2C4F63.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/golddream/772ED13B87668A0E3311DB5C6C114F3D22367B01389057E8E7614720795BD735.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/golddream/FACB3905D198BF14CE5DD6D1639503C4D01871B92761C0E85C6B66CDEA71D08E.png",
#                "/home/users/atodorova/alisa-thesis/malnet-cnn/test/backdoor/hupigon/3BFE07135EB705162C6BA16126410F47D22AC713E098DEB786EE192AF519B4B2.png"
#                 ] 

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
