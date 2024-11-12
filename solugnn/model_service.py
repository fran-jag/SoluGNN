from optimization import auto_optimize
from model import initialize_standardizer, get_outputs
from collection import retrieve_dataset, Graph

from pathlib import Path
import torch


class ModelService:

    def __init__(self, multi_model: bool = False) -> None:
        self.model = None
        self.models = []
        self.multi_model = multi_model
        self.outputs = get_outputs(retrieve_dataset())
        self.standardizer = initialize_standardizer(self.outputs)

    def load_model(self, model_name='gcn_v1.pt'):

        assert self.multi_model is False, "multi_models is set to True. Use load_models()"  # noqa: E501

        model_path = Path(f'../models/{model_name}')

        if not model_path.exists():
            print("Model doesn't exist.")
            answer = input("Do you want to build a model? (y/n)")
            if answer.lower() == 'y':
                model, rmse = auto_optimize(save=True,
                                            model_name=model_name)
                print(f'GCN model built\nTrain RMSE: {rmse}')
                print(f'Model saved at {model_path}')
                self.model = model
                return f'Model {model_name} built and loaded.'
            elif answer.lower() == 'n':
                return "Model won't be built."
            else:
                return "Unknown input. Try again."

        # Load model and set to evaluation mode
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()
        print(f'Model {model_name} loaded.')

    def load_models(self, model_path='../models/'):

        assert self.multi_model is True, "multi_models is set to False. Use load_model()"  # noqa: E501

        model_path = Path(model_path)
        model_names = [file for file in model_path.glob('*.pt')
                       if file.is_file()]

        if len(model_names) == 0:
            print("No models in directory.")
            answer = input("Do you want to build models? (y/n)")
            if answer.lower() == 'y':
                n_models = int(input("How many models? (1-5)"))

                assert n_models != 1, "To load or build 1 model use load_model() instead"  # noqa: E501

                models, rmses = auto_optimize(save=True,
                                              n_models=n_models)
                print(f'{n_models} models built.')
                print(f'Saved at {model_path.parent}')

                mean_rmse = sum(rmses)/len(rmses)
                print(f'Mean RMSE: {mean_rmse:.2f}')

                self.models = models
                self.models = [model.eval() for model in self.models]
                return f'{len(self.models)} models built and loaded.'

            elif answer.lower() == 'n':
                return "Model won't be built."
            else:
                return "Unknown input. Try again."

        self.models = [torch.load(model_name) for model_name in model_names]
        self.models = [model.eval() for model in self.models]
        print(f'{len(self.models)} models found and loaded.')

    def get_pred(self,
                 smile,
                 model=None,
                 ):

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

    def get_preds(self,
                  smile,
                  ):

        assert self.multi_model is True, "multi_models is set to False. Use get_pred()"  # noqa: E501

        outputs = []

        for model in self.models:
            outputs.append(self.get_pred(smile,
                           model=model))

        outputs_mean = sum(outputs) / len(outputs)
        outputs_SS = sum([(x - outputs_mean)**2 for x in outputs])
        outputs_var = outputs_SS / len(outputs)
        outputs_std = outputs_var ** 0.5

        return outputs_mean, outputs_std
