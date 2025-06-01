# Master Thesis
**Title:** A Comparative Study for Malware Image Detection Models Against Evolutionary Algorithm Based Adversarial Attacks

Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of Master in Information and Computer Sciences of the University of Luxembourg

## Abstract


## Repository Organization
This repository includes the following folders:

- **HPC:** Contains the YAML configuration file for the virtual environment used in the University of Luxembourg's High Performance Computing (ULHPC) Facility, along with an example SLURM job script for job submission. To use HPC, you need an authorized account and allocated space. For more information, refer to [HPC's documentation] (https://hpc-docs.uni.lu/).

- **Model18-Original:** Contains resources for the initial ResNet18 model, including its Python training script, training output logs, report plots, and evaluation scripts with their respective output logs.

- **Model18-FineTuned:** Contains resources for the fine-tuned ResNet18 model, including its Python training script, training output logs, report plots, and evaluation scripts with their respective output logs.

- **Model50:** Contains all resources for the initial ResNet18 model, including its Python training script, training output logs, the final model (which is model50.pt), report plots, and evaluation scripts with their respective output logs.

- **Adversarial-Attacks:** Contains all attack-related data for initial ResNet18 (Model18-Original), fine-tuned ResNet18 (Model18-FineTuned), and ResNet50 (Model50). For each class, this includes the Python attack scripts, all generated adversarial images, plots visualizing the average confidence of the benign target class for successful EA attacks over iterations, and an XLSX table summarizing all results as presented in Section Results of our Report.   

**Note:** Due to GitHub's file size limitations, the trained models (model18Original.pt and model18-finetuned.pt) for ResNet18 (Model18-Original) and fine-tuned ResNet18 (Model18-FineTuned), respectively, could not be directly uploaded. They are instead accessible via the following public Google Drive link: [Google Drive Link] (https://drive.google.com/drive/folders/1HflLjjLm--6s8vNc2cKdVDaBWhaKydKd?usp=sharing).
