
# GCN Solubility Prediction

This project aims to implement and deploy a Graph Convolutional Network (GCN) model for predicting the solubility of molecular compounds, based on their SMILES (Simplified Molecular Input Line Entry System) representations. This model architecture was chosen as the best performer in a previous exploratory study on solubility prediction. See ![Solvation Free Energy](https://github.com/fran-jag/solvation_free_energy). 

Currently, the project is a **work in progress**.

## Project Overview

### Objective
The main objective of this project is to build a robust GCN model that can predict solubility values of chemical compounds from their graph structures. It includes training, optimizing, and deploying the GCN model for real-time predictions.

### Key Components

1. **Model Definition**: Implements a GCN with graph convolutional and pooling layers to extract features from molecular graphs. The model architecture includes options for defining convolutional and hidden layers and applies dropout for regularization.
2. **Optimization**: Uses Bayesian optimization to determine optimal hyperparameters, aiming to minimize the Root Mean Squared Error (RMSE).
3. **Prediction Service**: A service class handles model loading, prediction on single or multiple models, and the generation of prediction statistics (mean, standard deviation) based on SMILES strings.
4. **Data Handling**: Utilizes RDKit for SMILES to graph conversion, creating an adjacency matrix and feature matrices for each molecule.
   
## Folder Structure
- **model_service.py**: Contains `ModelService`, which manages model loading, building, and prediction.
- **optimization.py**: Performs Bayesian optimization on the GCN, using `AxClient` to find the best hyperparameters for minimizing RMSE.
- **runner.py**: Script for initializing the `ModelService`, loading models, and generating predictions.
- **collection.py**: Manages the dataset and graph representation of molecules, including dataset loaders and utility functions.
- **model.py**: Defines the GCN model architecture with convolutional and pooling layers, training, testing, and data standardization functionalities.

## Usage

To use the model:
1. **Train and Optimize**:
   Run `optimization.py` to train the model and perform Bayesian optimization.
   
2. **Deploy and Predict**:
   Use `runner.py` to initialize the `ModelService` with the optimized model. It accepts SMILES strings as input and provides predicted solubility values with uncertainty (mean ± standard deviation for multiple models).

## Example

Example usage to predict the solubility free energy for a molecule (SMILES: `CC(=O)N1CCCC1`):

```python
python runner.py
```

Expected output:
```
SMILES: CC(=O)N1CCCC1
Solvation free energy: -4.92 ± 0.669
```

## Dependencies
- **PyTorch**: For defining and training the GCN.
- **Ax-platform (Meta)**: For Bayesian optimization.
- **RDKit**: For chemical data processing (SMILES to graph).

## Notes
- This project is a **work in progress**. Further refinements to the model architecture and deployment are ongoing.
- Ensure the `data/train.csv` dataset file is available for training.
