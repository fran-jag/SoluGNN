"""
ModelService for loading, optimizing, and making predictions with GCN.

This module provides the ModelService class, which includes methods for
loading single or multiple models, building models if they do not exist,
and making predictions based on input SMILES strings.
The models are Graph Convolutional Networks (GCNs) trained on molecular data.

Classes:
    ModelService: Handles loading, building, and predicting with models.

Functions:
    __init__(multi_model: bool):
        Initializes ModelService with single or multi-model support.
    load_model(model_name: str):
        Loads a single model or prompts to build one if not found.
    load_models(model_path: str):
        Loads multiple models or prompts to build models if none are found.
    get_pred(smile: str, model: torch.nn.Module):
        Predicts using a single model on a molecular SMILES string.
    get_preds(smile: str):
        Predicts using multiple models on a SMILES string,
        provides mean and standard deviation of predictions.

Usage:
    # Initialize with single or multiple model support
    service = ModelService(multi_model=True)

    # Load or build models
    service.load_models()  # For multiple models
    service.load_model('model_v1.pt')  # For a single model

    # Make predictions
    mean, std = service.get_preds(smile='CCO')  # For multiple models
    pred = service.get_pred(smile='CCO')  # For a single model
"""
from optimization import auto_optimize
from loguru import logger
from model import (initialize_standardizer,
                   get_outputs,
                   ChemGCN,
                   ConvolutionLayer,
                   PoolingLayer)
from collection import retrieve_dataset, Graph
from config import settings
import torch


class ModelService:
    """
    A service class for loading, building, and making predictions.

    This class provides methods to load a single or multiple models,
    optimize and build models if they don't exist, and make predictions
    for molecular structures represented as SMILES strings.

    Attributes:
        model (torch.nn.Module): A single loaded model instance.
        models (list): A list of loaded models when multi_model is True.
        multi_model (bool): A flag indicating single or multiple model mode.
        outputs: Output data.
        standardizer: A standardizer instance from trained model.
    """

    def __init__(self, multi_model: bool = False) -> None:
        """
        Initializes the ModelService with optional multi-model support.

        Args:
            multi_model (bool): If True, supports multiple models.
        """
        self.model = None
        self.models = []
        self.multi_model = multi_model
        self.outputs = get_outputs(retrieve_dataset())
        self.standardizer = initialize_standardizer(self.outputs)

    def _set_globals(self):
        """
        Sets global permissions to safely load PyTorch models.
        """
        global_list = [set, ChemGCN, ConvolutionLayer, PoolingLayer,
                       torch.nn.Linear, torch.nn.modules.container.ModuleList,
                       torch.nn.modules.activation.LeakyReLU,
                       torch.nn.modules.dropout.Dropout,
                       ]

        logger.debug(f'Following globals set as safe: {global_list}')

        torch.serialization.add_safe_globals(global_list)

    def load_model(self):
        """
        Loads or builds a single model from a specified file path.

        Args:
            model_name (str): The name of the model file to load.

        Returns:
            str: A message indicating whether the model was loaded or built.

        Raises:
            AssertionError: If `multi_model` is set to True.
        """
        assert self.multi_model is False, "multi_models is set to True. Use load_models()"  # noqa: E501

        model_path = f"{settings.model_path}/{settings.model_name}"

        if not model_path.exists():
            print("Model doesn't exist.")
            answer = input("Do you want to build a model? (y/n)")
            if answer.lower() == 'y':
                model, rmse = auto_optimize(save=True,
                                            model_name=settings.model_name)
                logger.info(f'GCN model built\nTrain RMSE: {rmse}')
                logger.info(f'Model saved at {model_path}')
                self.model = model
                return None
            elif answer.lower() == 'n':
                return "Model won't be built."
            else:
                return "Unknown input. Try again."

        # Load model and set to evaluation mode
        self.model = torch.load(model_path, weights_only=True)
        self.model.eval()
        logger.info(f'Model {settings.model_name} loaded.')

    def load_models(self,
                    model_path=settings.model_path):
        """
        Loads or builds multiple models from a specified directory.

        Args:
            model_path (str): The directory path containing the model files.

        Returns:
            str: A message indicating the number of models loaded or built.

        Raises:
            AssertionError: If `multi_model` is set to False.
        """
        assert self.multi_model is True, "multi_models is set to False. Use load_model()"  # noqa: E501

        model_names = [file for file in model_path.glob('*.pt')
                       if file.is_file()]

        if len(model_names) == 0:
            print("No models in directory.")
            answer = input("Do you want to build models? (y/n)")
            if answer.lower() == 'y':
                n_models = int(input("How many models? (1-5)"))

                assert n_models != 1, "To load or build 1 model use load_model() instead"  # noqa: E501

                models, rmses = auto_optimize(save=True, n_models=n_models)
                logger.info(f'{n_models} models built.')
                logger.info(f'Saved at {model_path}')
                mean_rmse = sum(rmses) / len(rmses)
                logger.info(f'Mean RMSE: {mean_rmse:.2f}')
                self.models = [model.eval() for model in models]
                return None

            elif answer.lower() == 'n':
                return "Model won't be built."
            else:
                return "Unknown input. Try again."

        self.models = [torch.load(model_name, weights_only=True)
                       for model_name in model_names]

        self.models = [model.eval() for model in self.models]

        logger.info(f'{len(self.models)} models found and loaded.')

    def get_pred(self, smile, model=None):
        """
        Makes a prediction using a single model for a given SMILES string.

        Args:
            smile (str): SMILES string representing the molecular structure.
            model (torch.nn.Module): Optional; the model to use for prediction.
                                     Defaults to the loaded single model.

        Returns:
            float: The predicted output after standardization.
        """
        if self.multi_model is False:
            logger.info(f'Predicting for SMILE: {smile}')

        if model is None:
            model = self.model

        mol_graph = Graph(smile)
        node_mat = mol_graph.node_mat
        adj_mat = mol_graph.adj_mat
        max_atoms = mol_graph.max_atoms
        node_vec_len = mol_graph.node_vec_len

        # Reshape inputs
        first_dim = int((torch.numel(torch.Tensor(node_mat)) /
                         (max_atoms * node_vec_len)))
        node_mat = node_mat.reshape(first_dim, max_atoms, node_vec_len)
        adj_mat = adj_mat.reshape(first_dim, max_atoms, max_atoms)

        # Package inputs; send to GPU
        nn_input = (torch.Tensor(node_mat).cuda(),
                    torch.Tensor(adj_mat).cuda())

        # Predict
        output = model(*nn_input)
        output = self.standardizer.restore(output.detach().cpu())
        return output.item()

    def get_preds(self, smile):
        """
        Makes predictions using multiple models for a given SMILES string, and
        calculates the mean and standard deviation of the predictions.

        Args:
            smile (str): SMILES string representing the molecular structure.

        Returns:
            tuple: The mean and standard deviation of the predictions.

        Raises:
            AssertionError: If `multi_model` is set to False.
        """

        assert self.multi_model is True, "multi_models is set to False. Use get_pred()"  # noqa: E501
        logger.info(f'Predicting for SMILE: {smile}')

        outputs = []

        for model in self.models:
            outputs.append(self.get_pred(smile, model=model))

        outputs_mean = sum(outputs) / len(outputs)
        outputs_SS = sum([(x - outputs_mean)**2 for x in outputs])
        outputs_var = outputs_SS / len(outputs)
        outputs_std = outputs_var ** 0.5

        logger.info(('solvation free energy:'
                     f'{outputs_mean:.2f} ± {outputs_std:.3f}'))

        return outputs_mean, outputs_std
