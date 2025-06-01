# Master Thesis
**Title:** A Comparative Study for Malware Image Detection Models Against Evolutionary Algorithm Based Adversarial Attacks

Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of Master in Information and Computer Sciences of the University of Luxembourg

## Abstract
The relentless and continuously evolving malware threats pose significant challenges for traditional detection techniques. Although Convolutional Neural Networks (CNNs) show great potential for image-based malware classification, their practical implementation is challenged by the demands of training on large-scale datasets, as well as by the need to ensure robustness against sophisticated adversarial attacks, particularly in the presence of class imbalance. This paper addresses these issues by training ResNet18 (from scratch) and ResNet50 (via transfer learning) on a novel 1.2 million image, 47-class malware dataset, called MalNet-Image. We detail the practical considerations and observed performance across High Performance Computing (HPC), custom-built computer, and personal laptop environments. Furthermore, we conduct a comprehensive analysis of model robustness using targeted Evolutionary Algorithm (EA)-based adversarial attacks, attempting 460 attacks per CNN to misclassify malware as benign. Our results demonstrate that distributed training is practical for large-scale malware datasets and reveal that model performance varies significantly due to class imbalance. Our adversarial attacks consistently achieved a 100% success rate on the fine-tuned ResNet18, contrasting with varied success rates on ResNet50. This disparity suggests that ResNet18, even after fine-tuning, possesses reduced adversarial resilience compared to ResNet50, which benefits from a deeper architecture and a more robust feature representation. Moreover, we find cases where high-confidence original predictions contribute to greater robustness against attacks. This research also provides a comprehensive literature review covering malware detection methods, CNN-based image classification techniques for malware, and evolutionary adversarial attack strategies in deep learning.

## Repository Organization
This repository includes the following folders:

- **HPC:** Contains the YAML configuration file for the virtual environment used in the University of Luxembourg's High Performance Computing (ULHPC) Facility, along with an example SLURM job script for job submission. To use HPC, you need an authorized account and allocated space. For more information, refer to [HPC's documentation] (https://hpc-docs.uni.lu/).

- **Model18-Original:** Contains resources for the initial ResNet18 model, including its Python training script, training output logs, report plots, and evaluation scripts with their respective output logs.

- **Model18-FineTuned:** Contains resources for the fine-tuned ResNet18 model, including its Python training script, training output logs, report plots, and evaluation scripts with their respective output logs.

- **Model50:** Contains all resources for the initial ResNet18 model, including its Python training script, training output logs, the final model (which is model50.pt), report plots, and evaluation scripts with their respective output logs.

- **Adversarial-Attacks:** Contains all attack-related data for initial ResNet18 (Model18-Original), fine-tuned ResNet18 (Model18-FineTuned), and ResNet50 (Model50). For each class, this includes the Python attack scripts, all generated adversarial images, plots visualizing the average confidence of the benign target class for successful EA attacks over iterations, and an XLSX table summarizing all results as presented in Section Results of our Report.   

**Note:** Due to GitHub's file size limitations, the trained models (model18Original.pt and model18-finetuned.pt) for ResNet18 (Model18-Original) and fine-tuned ResNet18 (Model18-FineTuned), respectively, could not be directly uploaded. They are instead accessible via the following public Google Drive link: [Google Drive Link] (https://drive.google.com/drive/folders/1HflLjjLm--6s8vNc2cKdVDaBWhaKydKd?usp=sharing).
