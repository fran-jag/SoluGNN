"""
Graph Convolutional Network (GCN) Model for Molecular Data

This file implements a Graph Convolutional Network (GCN) to predict
properties of molecules based on their graph structure. It includes
convolutional and pooling layers for graph representation learning,
utilities for data standardization, and methods to train, test,
and save the model.

Classes:
    - ConvolutionLayer: A layer that performs convolution on graph nodes.
    - PoolingLayer: Aggregates node features after convolution.
    - ChemGCN: Defines the full GCN model for molecular data.
    - Standardizer: Handles data standardization and restoration.

Functions:
    - initialize_model: Sets up the ChemGCN model with customizable
        architecture parameters.
    - run_epoch: Performs training for a single epoch.
    - get_outputs: Extracts outputs from a dataset for standardization.
    - initialize_standardizer: Initializes a standardizer for model outputs.
    - initialize_optimizer: Sets up an optimizer for model training.
    - initialize_loss: Returns the mean squared error loss function.
    - train_all_epochs: Trains the model over a specified number of epochs.
    - train_model: Configures and trains the model, returns training metrics.
    - fix_random_seeds: Sets random seeds for reproducibility.
    - save_model: Saves the trained model to a specified directory.
    - test_model: Evaluates the model on a test dataset
        and returns performance metrics.

Dependencies:
    - PyTorch: For model definition and training.
    - Scikit-learn: For calculating performance metrics like MAE and RMSE.
    - NumPy: For fixing random seeds.

Usage:
    Run the script to train and evaluate a GCN model on molecular data:
    ```
    python this_file.py
    ```

    Ensure `collection` module is available for `get_split_dataset_loaders`.
"""


from config import settings
from loguru import logger
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch
import torch.nn as nn


class ConvolutionLayer(nn.Module):
    """
    A convolutional layer for a Graph Convolutional Network.

    The ConvolutionLayer essentially does three things:
    - Calculation of inverse diagonal degree matrix from the adjacency matrix
    - Multiplication of the four matrices (D⁻¹ANW)
    - Application of a non-linear activation function to the layer output.
    """

    def __init__(self, node_in_len: int, node_out_len: int) -> None:
        """
        Initialize the ConvolutionLayer.

        Args:
            node_in_len (int): The size of the input node features.
            node_out_len (int): The size of the output node features.
        """
        super().__init__()
        # Create linear layer for node matrix
        self.conv_linear = nn.Linear(node_in_len, node_out_len)
        # Create activation function
        self.conv_activation = nn.LeakyReLU()

    def forward(self, node_mat, adj_mat):
        """
        Forward pass of the ConvolutionLayer.

        Args:
            node_mat (torch.Tensor): Node feature matrix
                shape: (batch_size, num_nodes, node_in_len).
            adj_mat (torch.Tensor): Adjacency matrix
                shape: (batch_size, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Updated node features after convolution.
        """
        # Calculate number of neighbors
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True)
        # Create identity tensor
        idx_mat = torch.eye(
            adj_mat.shape[-2],
            adj_mat.shape[-1],
            device=n_neighbors.device
        )
        # Add new (batch) dimension and expand
        idx_mat = idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        # Get inverse degree matrix
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)
        # Perform matrix multiplication (D⁻¹AN)
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)
        # Perform linear transformation to node features (node_fea * W)
        node_fea = self.conv_linear(node_fea)
        # Apply activation
        node_fea = self.conv_activation(node_fea)
        return node_fea


class PoolingLayer(nn.Module):
    """
    A pooling layer for aggregating node features in a graph.
    """

    def __init__(self):
        """
        Initialize the PoolingLayer.
        """
        super().__init__()

    def forward(self, node_fea):
        """
        Forward pass of the PoolingLayer.

        Args:
            node_fea (torch.Tensor): Node features
                shape: (batch_size, num_nodes, feature_len).

        Returns:
            torch.Tensor: Pooled node features
                shape: (batch_size, feature_len).
        """
        # Pool the node matrix
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea


class ChemGCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) model for chemical data.
    """

    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.1,
    ):
        """
        Initialize the ChemGCN model.

        Args:
            node_vec_len (int): Length of the input node vector.
            node_fea_len (int): Length of the node feature vector.
            hidden_fea_len (int): Length of the hidden feature vector.
            n_conv (int): Number of convolutional layers.
            n_hidden (int): Number of hidden layers after pooling.
            n_outputs (int): Number of output features.
            p_dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)
        # Convolution layers
        self.conv_layers = nn.ModuleList(
            [
                ConvolutionLayer(
                    node_in_len=node_fea_len,
                    node_out_len=node_fea_len,
                )
                for _ in range(n_conv)
            ]
        )
        # Pool convolution outputs
        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len
        # Pooling activation
        self.pooling_activation = nn.LeakyReLU()
        # From pooling layers to hidden layers
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden - 1):
            self.hidden_layers.append(nn.Linear(hidden_fea_len, hidden_fea_len))  # noqa: E501
        # Activation function
        self.hidden_activation = nn.LeakyReLU()
        # Dropout
        self.dropout = nn.Dropout(p=p_dropout)
        # Final layer going to output
        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)

    def forward(self, node_mat, adj_mat):
        """
        Forward pass of the ChemGCN model.

        Args:
            node_mat (torch.Tensor): Node feature matrix
                shape: (batch_size, num_nodes, node_vec_len).
            adj_mat (torch.Tensor): Adjacency matrix
                shape: (batch_size, num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, n_outputs).
        """
        # Perform initial transform on node_mat
        node_fea = self.init_transform(node_mat)
        # Perform convolutions
        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)
        # Perform pooling
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)
        # First hidden layer
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)
        # Subsequent hidden layers
        for hidden_layer in self.hidden_layers:
            hidden_node_fea = hidden_layer(hidden_node_fea)
            hidden_node_fea = self.hidden_activation(hidden_node_fea)
            hidden_node_fea = self.dropout(hidden_node_fea)
        # Output
        out = self.hidden_to_output(hidden_node_fea)
        return out


class Standardizer:
    """
    A class for standardizing data (mean normalization and scaling).
    """

    def __init__(self, X):
        """
        Initialize the Standardizer with data.

        Args:
            X (torch.Tensor): Data to compute mean and standard deviation from.
        """
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        """
        Standardize the data using the computed mean and std.

        Args:
            X (torch.Tensor): Data to standardize.

        Returns:
            torch.Tensor: Standardized data.
        """
        Z = (X - self.mean) / self.std
        return Z

    def restore(self, Z):
        """
        Restore the data from standardized form.

        Args:
            Z (torch.Tensor): Standardized data.

        Returns:
            torch.Tensor: Original data.
        """
        X = self.mean + Z * self.std
        return X

    def state(self):
        """
        Get the state (mean and std) of the Standardizer.

        Returns:
            dict: Dictionary containing 'mean' and 'std'.
        """
        return {'mean': self.mean, 'std': self.std}

    def load(self, state):
        """
        Load the state (mean and std) into the Standardizer.

        Args:
            state (dict): Dictionary containing 'mean' and 'std'.
        """
        self.mean = state['mean']
        self.std = state['std']


def initialize_model(
    node_vec_len: int = settings.node_vec_len,
    node_fea_len: int = settings.node_vec_len,
    hidden_fea_len: int = settings.node_vec_len,
    n_conv: int = settings.n_conv,
    n_hidden: int = settings.n_hidden,
    n_outputs: int = settings.n_outputs,
    p_dropout: float = settings.p_dropout,
    use_GPU: bool = settings.use_GPU,
) -> ChemGCN:
    """
    Initialize the ChemGCN model.

    Args:
        node_vec_len (int, optional): Length of input node vector.
            Defaults to 60.
        node_fea_len (int, optional): Length of node feature vector.
            Defaults to 60.
        hidden_fea_len (int, optional): Length of hidden feature vector.
            Defaults to 60.
        n_conv (int, optional): Number of convolutional layers.
            Defaults to 4.
        n_hidden (int, optional): Number of hidden layers.
            Defaults to 2.
        n_outputs (int, optional): Number of output features.
            Defaults to 1.
        p_dropout (float, optional): Dropout probability.
            Defaults to 0.1.
        use_GPU (bool, optional): Whether to use GPU.
            Defaults to True.

    Returns:
        ChemGCN: Initialized ChemGCN model.
    """
    # Model
    model = ChemGCN(
        node_vec_len,
        node_fea_len,
        hidden_fea_len,
        n_conv,
        n_hidden,
        n_outputs,
        p_dropout,
    )
    # Transfer to GPU
    if use_GPU:
        model.cuda()
    return model


def run_epoch(
    epoch,
    model,
    training_dataloader,
    optimizer,
    loss_fn,
    standardizer,
    use_GPU: bool = True,
    max_atoms: int = 100,
    node_vec_len: int = 60,
    verbose=False,
):
    """
    Run one epoch of training.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        training_dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating the model parameters.
        loss_fn (callable): Loss function.
        standardizer (Standardizer): Standardizer for the outputs.
        use_GPU (bool, optional): Whether to use GPU.
            Defaults to True.
        max_atoms (int, optional): Maximum number of atoms.
            Defaults to 100.
        node_vec_len (int, optional): Length of node vector.
            Defaults to 60.
        verbose (bool, optional): Whether to print progress.
            Defaults to False.

    Returns:
        tuple: Average loss and average mean absolute error (MAE).
    """
    # Variables to store losses and error
    avg_loss = 0
    avg_mae = 0
    count = 0
    # Switch model to train mode
    model.train()
    # Go over each batch
    for i, dataset in enumerate(training_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]
        # Reshape inputs
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)
        # Standardize output
        output_std = standardizer.standardize(output)
        # Package inputs, outputs; check GPU
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
            nn_output = output_std.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output_std
        # Compute output from network
        nn_prediction = model(*nn_input)
        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        avg_loss += loss
        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        avg_mae += mae
        # Set zero gradients for all tensors
        optimizer.zero_grad()
        # Do backward propagation
        loss.backward()
        # Update optimizer
        optimizer.step()
        # Increase count
        count += 1
    # Calculate avg loss and MAE
    avg_loss = avg_loss / count
    avg_mae = avg_mae / count
    if verbose is True:
        if epoch % 10 == 0:
            print(
                'Epoch: [{0}]\tTraining Loss: [{1:.2f}]\t'
                'Training MAE: [{2:.2f}]'.format(epoch, avg_loss, avg_mae)
            )
    # Return loss and MAE
    return avg_loss.item(), avg_mae


def get_outputs(dataset):
    """
    Extract outputs from the dataset.

    Args:
        dataset: Dataset containing inputs and outputs.

    Returns:
        list: List of outputs from the dataset.
    """
    return [data[1] for data in dataset]


def initialize_standardizer(outputs,
                            log: bool = True):
    """
    Initialize the Standardizer with outputs.

    Args:
        outputs (list or torch.Tensor): Outputs to compute mean and std from.

    Returns:
        Standardizer: Initialized Standardizer.
    """
    if log:
        logger.info('Created standardizer from outputs')
    return Standardizer(torch.Tensor(outputs))


def initialize_optimizer(model,
                         lr: float = settings.learning_rate):
    """
    Initialize the optimizer.

    Args:
        model (nn.Module): Model whose parameters to optimize.
        lr (float, optional): Learning rate. Defaults to 0.01.

    Returns:
        Optimizer: Initialized optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr)


def initialize_loss():
    """
    Initialize the loss function.

    Returns:
        Loss function: Mean squared error loss.
    """
    return torch.nn.MSELoss()


def train_all_epochs(
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    standardizer,
    n_epochs,
    verbose: bool = False,
):
    """
    Train the model for all epochs.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating the model parameters.
        loss_fn (callable): Loss function.
        standardizer (Standardizer): Standardizer for the outputs.
        n_epochs (int): Number of epochs to train.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        tuple: Lists of losses, MAEs, and epochs.
    """
    loss = []
    mae = []
    epoch = []
    for i in range(n_epochs):
        epoch_loss, epoch_mae = run_epoch(
            epoch=i,
            model=model,
            training_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            standardizer=standardizer,
            verbose=verbose,
        )
        loss.append(epoch_loss)
        mae.append(epoch_mae)
        epoch.append(i)
    return loss, mae, epoch


def train_model(
    model,
    dataset,
    train_dataloader,
    verbose: bool = True,
    n_epochs: int = settings.n_epochs,
    lr: float = settings.learning_rate,
    log: bool = True
):
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        dataset: The dataset containing training data.
        train_dataloader (DataLoader): DataLoader for the training data.
        verbose (bool, optional): Whether to print progress.
            Defaults to True.
        n_epochs (int, optional): Number of epochs to train.
            Defaults to 100.
        lr (float, optional): Learning rate.
            Defaults to 0.01.

    Returns:
        tuple: Lists of losses, MAEs, epochs, standardizer, and loss function.
    """
    # Standardizer
    outputs = get_outputs(dataset)
    standardizer = initialize_standardizer(outputs, log=log)
    # Optimizer
    optimizer = initialize_optimizer(model, lr)
    # Loss function
    loss_fn = initialize_loss()
    # Train model
    loss, mae, epoch = train_all_epochs(
        model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        standardizer=standardizer,
        verbose=verbose,
        n_epochs=n_epochs,
    )
    return loss, mae, epoch, standardizer, loss_fn


def fix_random_seeds(seed: int = 42):
    """
    Fix the random seeds for reproducibility.

    Args:
        seed (int, optional): Seed value.
            Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model,
               path=settings.model_path,
               name: str = 'model.pth'):
    """
    Save the model.

    Args:
        model (nn.Module): The model to save.
        path (str, optional): Directory path to save the model.
            Defaults to '../models/'.
        name (str, optional): Name of the model file.
            Defaults to 'model.pth'.
    """
    full_path = f"{path.as_posix()}/{name}"
    torch.save(model, full_path)


def test_model(
    model,
    test_dataloader,
    standardizer,
    loss_fn,
    use_GPU: bool = settings.use_GPU,
    max_atoms: int = settings.max_atoms,
    node_vec_len: int = settings.node_vec_len,
):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_dataloader (DataLoader): DataLoader for the test data.
        standardizer (Standardizer): Standardizer used during training.
        loss_fn (callable): Loss function.
        use_GPU (bool, optional): Whether to use GPU.
            Defaults to True.
        max_atoms (int, optional): Maximum number of atoms.
            Defaults to 100.
        node_vec_len (int, optional): Length of node vector.
            Defaults to 60.

    Returns:
        tuple: Average loss, MAE, and RMSE on the test set.
    """
    # Store loss and error
    test_loss = 0
    test_mae = 0
    count = 0
    # Store all outputs and predictions for RMSE
    all_outputs = []
    all_predictions = []
    # Switch to evaluation mode
    model.eval()
    # Go over batches of test set
    for i, dataset in enumerate(test_dataloader):
        # Unpack data
        node_mat = dataset[0][0]
        adj_mat = dataset[0][1]
        output = dataset[1]
        # Reshape
        first_dim = int((torch.numel(node_mat)) / (max_atoms * node_vec_len))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)
        # Standardize output
        output_std = standardizer.standardize(output)
        # Package inputs and outputs to GPU
        if use_GPU:
            nn_input = (node_mat.cuda(), adj_mat.cuda())
            nn_output = output_std.cuda()
        else:
            nn_input = (node_mat, adj_mat)
            nn_output = output_std
        # Compute output
        nn_prediction = model(*nn_input)
        # Calculate loss
        loss = loss_fn(nn_output, nn_prediction)
        test_loss += loss
        # Calculate MAE
        prediction = standardizer.restore(nn_prediction.detach().cpu())
        mae = mean_absolute_error(output, prediction)
        test_mae += mae
        # Store predictions and actual values for RMSE
        all_predictions.extend(prediction.numpy().flatten())
        all_outputs.extend(output.numpy().flatten())
        # Increase count
        count += 1
    # Calculate avg loss, MAE, RMSE
    test_loss = test_loss / count
    test_mae = test_mae / count
    test_rmse = root_mean_squared_error(all_outputs, all_predictions)
    return test_loss, test_mae, test_rmse


if __name__ == '__main__':
    # Workflow
    from collection import get_split_dataset_loaders

    for seed in [6969]:
        # Fix Seeds
        fix_random_seeds(seed)

        # Create dataloaders
        dataset, train_loader, test_loader = get_split_dataset_loaders()

        # Initialize and train model
        model = initialize_model()
        loss, mae, epoch, standardizer, loss_fn = train_model(model,
                                                              dataset,
                                                              train_loader,
                                                              verbose=True)
        # Save trained model
        # save_model(model, name=f'test_gcn_v0_seed_{seed}.pt')

        # Test trained model
        test_loss, test_mae, test_rmse = test_model(model,
                                                    test_loader,
                                                    standardizer,
                                                    loss_fn,
                                                    )
        # Print final results
        print(f"Results for seed {seed}")
        print(f"Training Loss: {loss[-1]:.2f}")
        print(f"Training MAE: {mae[-1]:.2f}")
        print(f"Test Loss: {test_loss:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print('-'*12)
