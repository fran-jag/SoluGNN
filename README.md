
# GCN Solubility Prediction

This project implements and deploys a **Graph Convolutional Network (GCN)** model for predicting solubility values of molecular compounds based on their **SMILES (Simplified Molecular Input Line Entry System)** representations. This GCN architecture builds on graph-based representations of molecules to provide accurate predictions of solubility.

## Project Overview

### Objective
To create a robust GCN model capable of predicting solubility (e.g., solvation free energy) of chemical compounds by leveraging molecular graph representations. The project includes:
- Training and optimization using Bayesian techniques.
- Deployment for real-time predictions.

### Key Features

1. **Model Architecture**:
   - Implements GCN layers with graph convolutions and pooling operations.
   - Flexible hyperparameter tuning for layers, dropout, and learning rates.
   - Designed for scalable training on molecular datasets.

2. **Bayesian Optimization**:
   - Utilizes `Ax-platform` for Bayesian optimization.
   - Finds the optimal hyperparameters (e.g., number of layers, learning rates) to minimize prediction error (RMSE).

3. **Prediction Service**:
   - Supports predictions using single or ensembles of models.
   - Generates outputs with associated uncertainty (mean and standard deviation for ensemble models).

4. **Molecular Data Handling**:
   - Converts SMILES strings to graph representations using **RDKit**.
   - Creates feature and adjacency matrices for GCN input.

---

## Folder Structure

- **`model.py`**:
  Contains the implementation of the GCN model, including convolution, pooling layers, and utilities for training and testing.

- **`optimization.py`**:
  Automates hyperparameter tuning and model optimization using Bayesian optimization.

- **`runner.py`**:
  Initializes the prediction service and makes predictions based on input SMILES strings.

- **`collection.py`**:
  Manages the dataset, including molecule-to-graph conversion and data loading utilities.

- **`model_service.py`**:
  Provides a unified service for loading models, running predictions, and managing single or multi-model setups.

- **`config.py`**:
  Defines project settings and configurations using environment variables.

---

## Setup Instructions

### Dependencies  

1. Clone the repository
2. Install Poetry if not already installed:
```bash
pip install poetry
```
3. Install dependencies:
```bash
poetry install
```

### Configuration
Modify `.env` or `config.py` to set paths and default parameters:
- `data_file_name`: Path to the dataset CSV.
- `model_path`: Directory to save/load trained models.
- `node_vec_len`, `max_atoms`: Feature lengths for molecular graphs.
- Other hyperparameters such as `learning_rate` and `n_epochs`.

---

## Workflow

1. **Training and Optimization**:
   - Use `runner.py` for automatic model optimization:
     ```bash
     poetry run python runner.py
     ```

---

## Example Prediction

**Input SMILES**: `CC(=O)N1CCCC1`  
**Expected Output**:
```
SMILES: CC(=O)N1CCCC1
Solubility free energy: -4.92 ± 0.669
```

---
## Notes
- This project is a **work in progress**. Further refinements to the model architecture and deployment are ongoing.
- Ensure the `data/train.csv` dataset file is available for training.
