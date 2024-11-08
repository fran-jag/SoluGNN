"""
Script to optimize the GCN via Ax platform's bayessian optimization
"""

from collection import get_split_dataset_loaders
from model import fix_random_seeds, initialize_model, \
            train_model, test_model, save_model

import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties


def train_test(parameters,
               output_model: bool = False):

    # Unpack parameters
    hidden_nodes = parameters['hidden_nodes']  # Default = 60
    n_conv_layers = parameters['n_conv_layers']  # Default = 4
    n_hidden_layers = parameters['n_hidden_layers']  # Default = 2
    learning_rate = parameters['learning_rate']  # Default = 0.01

    # Create dataloaders
    dataset, train_loader, test_loader = get_split_dataset_loaders()

    # Initialize and train model
    model = initialize_model(node_fea_len=hidden_nodes,
                             n_conv=n_conv_layers,
                             n_hidden=n_hidden_layers
                             )
    loss, mae, epoch, standardizer, loss_fn = train_model(model,
                                                          dataset,
                                                          train_loader,
                                                          lr=learning_rate,
                                                          verbose=False,
                                                          )
    # Test trained model
    test_loss, test_mae, test_rmse = test_model(model,
                                                test_loader,
                                                standardizer,
                                                loss_fn,
                                                )
    if output_model:
        return model, test_rmse
    else:
        return test_rmse


def initialize_client():
    return AxClient()


def create_experiment(client,
                      name: str,
                      parameters: list,
                      objectives: dict,
                      overwrite: bool = False,
                      ):

    client.create_experiment(
        name=name,
        parameters=parameters,
        objectives=objectives,
        overwrite_existing_experiment=overwrite,
    )


def run_base_trial(client,
                   base_params):

    client.attach_trial(parameters=base_params)
    client.complete_trial(trial_index=0,
                          raw_data=train_test(base_params))


def run_trials(client,
               n_trials: int = 20):
    for i in range(n_trials):
        parameters, trial_index = client.get_next_trial()
        client.complete_trial(trial_index=trial_index,
                              raw_data=train_test(parameters))


def get_trials_dataframe(client) -> pd.DataFrame:
    return client.get_trials_data_trame()


def get_best_parameters(client):
    return client.get_best_parameters()


if __name__ == '__main__':

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
            "value_type": "int"
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

    create_experiment(optimizer,
                      name="GCN optimization",
                      parameters=parameters,
                      objectives=objectives)

    base_parameters = {'hidden_nodes': 60,
                       'n_conv_layers': 4,
                       'n_hidden_layers': 2,
                       'learning_rate': 0.01}

    run_base_trial(optimizer, base_parameters)
    run_trials(optimizer)

    best_parameters = get_best_parameters(optimizer)

    for i in [12, 24, 48, 64, 69]:
        fix_random_seeds(i)
        model, rmse = train_test(best_parameters, output_model=True)
        save_model(model,
                   name=f'optimized_gcn_seed_{i}.pt')
