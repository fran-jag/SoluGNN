# Construction of convolution layer


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class Convolutionlayer(nn.Module):
    """
    Note:
    The ConvolutionLayer essentially does three things
    - Calculation of inverse diagonal degree matrix from the adjacency matrix
    - Multiplication of the four matrices (D⁻¹ANW)
    - Application of a non-linear activation function to the layer output.
    """

    def __init__(self,
                 node_in_len: int,
                 node_out_len: int,
                 ) -> None:
        super().__init__()

        # Create linear layer for node matrix
        self.conv_linear = nn.Linear(node_in_len, node_out_len)

        # Create activation function
        self.conv_activation = nn.LeakyReLU()

    def forward(self,
                node_mat,
                adj_mat,
                ):
        # Calculate number of neighbors
        n_neighbors = adj_mat.sum(dim=-1, keepdims=True)

        # Create identity tensor
        self.idx_mat = torch.eye(
            adj_mat.shape[-2],
            adj_mat.shape[-1],
            device=n_neighbors.device
        )

        # Add new (batch) dimension and expand
        idx_mat = self.idx_mat.unsqueeze(0).expand(*adj_mat.shape)
        # Get inverse degree matrix
        inv_degree_mat = torch.mul(idx_mat, 1 / n_neighbors)

        # Perform matrix multiplication (D⁻¹AN)
        node_fea = torch.bmm(inv_degree_mat, adj_mat)
        node_fea = torch.bmm(node_fea, node_mat)

        # Perfom linear transformation to node features (node_fea * W)
        node_fea = self.conv_linear(node_fea)

        # Apply activation
        node_fea = self.conv_activation(node_fea)

        return node_fea


class PoolingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                node_fea):
        # Pool the node matrix
        pooled_node_fea = node_fea.mean(dim=1)
        return pooled_node_fea


class ChemGCN(nn.Module):
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
        super().__init__()

        # Define layers
        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        # Convolution layers
        self.conv_layers = nn.ModuleList(
            [Convolutionlayer(node_in_len=node_fea_len,
                              node_out_len=node_fea_len,
                              )
                for i in range(n_conv)]
        )

        # Pool convolution outputs
        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len

        # Pooling activation
        self.pooling_activation = nn.LeakyReLU()

        # From pooling layers to hidden layers
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)

        # Hidden layer activation function
        self.hidden_activation = nn.LeakyReLU()

        # Hidden layer dropout
        self.dropout = nn.Dropout(p=p_dropout)

        # If hidden layer > 1, add more hidden layers
        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList(
                [self.hidden_layer for _ in range(n_hidden - 1)]
            )
            self.hidden_activation_layers = nn.ModuleList(
                [self.hidden_activation for _ in range(n_hidden - 1)]
                )
            self.hidden_dropout_layers = nn.ModuleList(
                [self.dropout for _ in range(n_hidden - 1)]
            )

        # Final layer going to output
        self. hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)

    def forward(self, node_mat, adj_mat):
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

        # Subsequent hidden layer
        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)  # noqa:E501
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)  # noqa:E501
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)  # noqa:E501
        # Output
        out = self.hidden_to_output(hidden_node_fea)

        return out


class Standardizer:
    def __init__(self, X):
        self.mean = torch.mean(X)
        self.std = torch.std(X)

    def standardize(self, X):
        Z = (X - self.mean) / self.std
        return Z

    def restore(self, Z):
        X = self.mean + Z * self.std
        return X

    def state(self):
        return {'mean': self.mean, 'std': self.std}

    def load(self, state):
        self.mean = state['mean']
        self.std = state['std']


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
                'Training MAE: [{2:.2f}]'
                .format(
                    epoch, avg_loss, avg_mae
                )
            )

    # Return loss and MAE
    return avg_loss.item(), avg_mae


def initialize_model(node_vec_len: int = 60,
                     node_fea_len: int = 60,
                     hidden_fea_len: int = 60,
                     n_conv: int = 4,
                     n_hidden: int = 2,
                     n_outputs: int = 1,
                     p_dropout: float = 0.1,
                     use_GPU: bool = True,
                     ) -> ChemGCN:

    # Model
    model = ChemGCN(node_vec_len,
                    node_fea_len,
                    hidden_fea_len,
                    n_conv,
                    n_hidden,
                    n_outputs,
                    p_dropout)
    # Transfer to GPU
    if use_GPU:
        model.cuda()

    return model


def get_outputs(dataset):
    return [dataset[i][1] for i in range(len(dataset))]


def initialize_standardizer(outputs):
    return Standardizer(torch.Tensor(outputs))


def initialize_optimizer(model,
                         lr: float = 0.01):
    return torch.optim.Adam(model.parameters(),
                            lr)


def initialize_loss():
    return torch.nn.MSELoss()


def train_all_epochs(model,
                     train_dataloader,
                     optimizer,
                     loss_fn,
                     standardizer,
                     n_epochs,
                     verbose: bool = False
                     ):
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


def train_model(model,
                dataset,
                train_dataloader,
                verbose: bool = True,
                n_epochs: int = 100,
                lr: float = 0.01
                ):

    # Standardizer
    outputs = get_outputs(dataset)
    standardizer = initialize_standardizer(outputs)

    # Optimizer
    optimizer = initialize_optimizer(model, lr)

    # Loss function
    loss_fn = initialize_loss()

    # Train model
    loss, mae, epoch = train_all_epochs(model,
                                        train_dataloader=train_dataloader,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        standardizer=standardizer,
                                        verbose=verbose,
                                        n_epochs=n_epochs
                                        )

    return loss, mae, epoch, standardizer, loss_fn


def fix_random_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model,
               path: str = '../models/',
               name: str = None):
    full_path = path + name
    torch.save(model.state_dict(), full_path)


def test_model(
        model,
        test_dataloader,
        standardizer,
        loss_fn,
        use_GPU: bool = True,
        max_atoms: int = 100,
        node_vec_len: int = 60,
        ):

    # Store loss and error
    test_loss = 0
    test_mae = 0
    count = 0

    # Store all outputs and predictions for R2
    all_outputs = []
    all_predictions = []

    # Switch to inference mode
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

        # Store predictions and actual values for R²
        all_predictions.extend(prediction.numpy().flatten())
        all_outputs.extend(output.numpy().flatten())

        # Increase count
        count += 1

    # Calculate avg loss, MAE, R2
    test_loss = test_loss / count
    test_mae = test_mae / count
    test_rmse = root_mean_squared_error(all_outputs, all_predictions)

    return test_loss, test_mae, test_rmse


if __name__ == '__main__':
    # Workflow
    from collection import get_split_dataset_loaders

    for seed in [0, 42, 64, 13]:
        # Fix Seeds
        fix_random_seeds(seed)

        # Create dataloaders
        dataset, train_loader, test_loader = get_split_dataset_loaders()

        # Initialize and train model
        model = initialize_model()
        loss, mae, epoch, standardizer, loss_fn = train_model(model,
                                                              dataset,
                                                              train_loader,
                                                              verbose=False)
        # Save trained model
        save_model(model, name=f'test_gcn_v0_seed_{seed}.pt')

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
