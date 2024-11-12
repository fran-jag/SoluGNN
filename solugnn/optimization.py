"""
Module to perform Bayesian optimization on a Graph Convolutional Network (GCN)
using the Ax platform. Finds the optimal hyperparameters for the GCN to
minimize the root mean squared error (RMSE) on a given dataset.

The optimization process includes:
- Defining the search space for hyperparameters:
    number of hidden nodes, number of convolutional layers,
    the number of hidden layers, and learning rate.
- Using the AxClient to manage the Bayesian optimization experiment.
- Training and testing the GCN model with different sets of hyperparameters.
- Selecting the best hyperparameters based on the RMSE metric.
- Optionally, optimizing, training multiple models and saving them.

Functions:
- train_test: Train and test the GCN model with given parameters.
- initialize_client: Initialize the AxClient for Bayesian optimization.
- create_experiment: Set up experiment with specified parameters & objectives.
- run_base_trial: Run a trial with default parameters.
- run_trials: Run multiple optimization trials.
- get_trials_dataframe: Retrieve the trials data as a pandas DataFrame.
- get_best_parameters: Extract the best hyperparameters from the experiment.
- auto_optimize: Single function to automate procedure.

Usage:
- Run this script directly to perform optimization and obtain the best model.
- Modify the 'parameters' and 'objectives' in 'auto_optimize'
    to customize the optimization.
"""

from collection import get_split_dataset_loaders
from model import (
    fix_random_seeds,
    initialize_model,
    train_model,
    test_model,
    save_model,
)

import numpy.random as random
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties


def train_test(parameters, output_model: bool = False):
    """
    Train and test the GCN model with the given parameters.

    Args:
        parameters (dict): Dictionary containing the model parameters.
        output_model (bool, optional): If True, returns the trained model along
            with the test RMSE. Defaults to False.

    Returns:
        float or tuple:
            - If output_model is False, returns the test RMSE (float).
            - If output_model is True, returns a tuple (model, test_rmse).
    """
    # Unpack parameters
    hidden_nodes = parameters['hidden_nodes']        # Default = 60
    n_conv_layers = parameters['n_conv_layers']      # Default = 4
    n_hidden_layers = parameters['n_hidden_layers']  # Default = 2
    learning_rate = parameters['learning_rate']      # Default = 0.01

    # Create dataloaders
    dataset, train_loader, test_loader = get_split_dataset_loaders()

    # Initialize and train model
    model = initialize_model(
        node_fea_len=hidden_nodes,
        n_conv=n_conv_layers,
        n_hidden=n_hidden_layers,
    )
    loss, mae, epoch, standardizer, loss_fn = train_model(
        model,
        dataset,
        train_loader,
        lr=learning_rate,
        verbose=False,
    )

    # Test trained model
    test_loss, test_mae, test_rmse = test_model(
        model,
        test_loader,
        standardizer,
        loss_fn,
    )

    if output_model:
        return model, test_rmse
    else:
        return test_rmse


def initialize_client():
    """
    Initialize the AxClient for Bayesian optimization.

    Returns:
        AxClient: An instance of AxClient.
    """
    return AxClient()


def create_experiment(
    client,
    name: str,
    parameters: list,
    objectives: dict,
    overwrite: bool = False,
):
    """
    Create an experiment in the AxClient.

    Args:
        client (AxClient): The AxClient instance.
        name (str): Name of the experiment.
        parameters (list): List of parameter configurations.
        objectives (dict): Dictionary of objectives to optimize.
        overwrite (bool, optional): Overwrite an existing experiment.
            Defaults to False.
    """
    client.create_experiment(
        name=name,
        parameters=parameters,
        objectives=objectives,
        overwrite_existing_experiment=overwrite,
    )


def run_base_trial(client, base_params):
    """
    Run a base trial with default parameters.

    Args:
        client (AxClient): The AxClient instance.
        base_params (dict): Dictionary of base parameters.
    """
    client.attach_trial(parameters=base_params)
    client.complete_trial(
        trial_index=0,
        raw_data=train_test(base_params),
    )


def run_trials(client, n_trials: int = 20):
    """
    Run multiple optimization trials.

    Args:
        client (AxClient): The AxClient instance.
        n_trials (int, optional): Number of trials to run. Defaults to 20.
    """
    for _ in range(n_trials):
        parameters, trial_index = client.get_next_trial()
        client.complete_trial(trial_index=trial_index,
                              raw_data=train_test(parameters),
                              )


def get_trials_dataframe(client) -> pd.DataFrame:
    """
    Retrieve the trials data as a pandas DataFrame.

    Args:
        client (AxClient): The AxClient instance.

    Returns:
        pd.DataFrame: DataFrame containing the trials data.
    """
    return client.get_trials_data_frame()


def get_best_parameters(client):
    """
    Get the best parameters from the experiment.

    Args:
        client (AxClient): The AxClient instance.

    Returns:
        dict: Dictionary of the best parameters found.
    """
    parameters, _ = client.get_best_parameters()
    return parameters


def auto_optimize(n_models: int = 1,
                  save: bool = False,
                  model_name: str = 'gcn_v0.pt'):
    """
    Automatically optimize the GCN model using Bayesian optimization.

    Train and save optimized models. If n_models = N, train N models with
    N random seeds and save it to path as 'optimized_gcn_seed_{seed}.pt'.

    Args:
        n_models (int, optional): Number of models to train with the best
            parameters. Defaults to 1.
        save (bool, optional): If True, saves the trained model(s).
            Defaults to False.

    Returns:
        tuple:
            - If n_models > 1, returns a tuple (models, rmses).
            - If n_models == 1, returns a tuple (model, rmse).
    """
    fix_random_seeds()
    optimizer = initialize_client()

    parameters = [
        {
            "name": "hidden_nodes",
            "type": "range",
            "bounds": [10, 100],
            "value_type": "int",
            "log_scale": False,
        },
        {
            "name": "n_conv_layers",
            "type": "range",
            "bounds": [1, 10],
            "value_type": "int",
        },
        {
            "name": "n_hidden_layers",
            "type": "range",
            "bounds": [1, 10],
            "value_type": "int",
        },
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [1e-6, 0.1],
            "value_type": "float",
            "log_scale": True,
        },
    ]
    objectives = {"rmse": ObjectiveProperties(minimize=True)}

    create_experiment(
        optimizer,
        name="GCN optimization",
        parameters=parameters,
        objectives=objectives,
    )

    base_parameters = {
        'hidden_nodes': 60,
        'n_conv_layers': 4,
        'n_hidden_layers': 2,
        'learning_rate': 0.01,
    }

    run_base_trial(optimizer, base_parameters)
    run_trials(optimizer)
    best_parameters = get_best_parameters(optimizer)

    if n_models > 1:
        models = []
        rmses = []
        seeds = random.randint(0, 256, n_models)

        for i in seeds:
            fix_random_seeds(i)
            model, rmse = train_test(best_parameters, output_model=True)

            if save:
                save_model(
                    model,
                    name=f'optimized_gcn_seed_{i}.pt',
                )
                print(f"Model optimized_gcn_seed_{i}.pt saved")
            models.append(model)
            rmses.append(rmse)

        return models, rmses

    model, rmse = train_test(best_parameters, output_model=True)

    if save:
        save_model(
            model,
            name=model_name,
        )
        print(f"Model saved as {model_name}")

    return model, rmse


if __name__ == "__main__":
    model, rmse = auto_optimize()
    print(rmse)
