"""
Module that contains classes for a chemical graph and dataset.

This module provides functionalities to convert SMILES
strings into graph representations using RDKit and to handle datasets
of such graphs for machine learning purposes.
"""

from db.db_model import SolvationMolecules
import numpy as np
import pandas as pd
from sqlalchemy import select
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from config.config import settings, engine

from rdkit import Chem
from rdkit.Chem import rdmolops, rdDistGeom
from torch.utils.data import Dataset, DataLoader

from loguru import logger


class Graph:
    """
    Represents a chemical graph derived from a SMILES string.

    This class transforms a SMILES string into a graph representation
    where nodes represent atoms and edges represent bonds,
    incorporating both adjacency and distance information.

    Attributes:
        smiles (str): SMILES representation of the molecule.
        node_vec_len (int): Length of the node feature vector.
        max_atoms (int or None): Maximum number of atoms in the graph. If None,
            the number of atoms in the molecule is used.
        mol (rdkit.Chem.Mol or None): RDKit molecule object with hydrogens.
        node_mat (np.ndarray): Node matrix of shape (max_atoms, node_vec_len).
        adj_mat (np.ndarray): Adjacency matrix with inverse bond lengths of
            shape (max_atoms, max_atoms).
        std_adj_mat (np.ndarray): Standard adjacency matrix
            of shape (max_atoms, max_atoms).
    """

    def __init__(
        self,
        molecule_smiles: str,
        node_vec_len: int = settings.node_vec_len,
        max_atoms: int = settings.max_atoms,
    ) -> None:
        """
        Initialize Graph object by converting a SMILES string to a graph.

        Args:
            molecule_smiles (str): SMILES representation of the molecule.
            node_vec_len (int): Length of the node feature vector.
            max_atoms (int, optional): Maximum number of atoms in the graph.
                If None, the number of atoms in the molecule is used.
                Defaults to None.
        """
        # Store properties
        self.smiles = molecule_smiles
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        # Convert SMILES to RDKit mol
        self.smiles_to_mol()

        # Check if valid mol was created and generate graph
        if self.mol:
            self.smiles_to_graph()

    def smiles_to_mol(self) -> None:
        """
        Converts the SMILES string to RDKit molecule object with hydrogens.

        Sets the `mol` attribute to the RDKit molecule if successful;
        otherwise, sets it to None.
        """
        mol = Chem.MolFromSmiles(self.smiles)

        if mol is None:
            self.mol = None
            return

        self.mol = Chem.AddHs(mol)

    def smiles_to_graph(self) -> None:
        """
        Generates the node feature matrix and adjacency matrix
        from the RDKit molecule.

        The node feature matrix encodes atomic numbers, and
        the adjacency matrix incorporates inverse bond lengths.
        Both matrices are padded to match `max_atoms` if necessary.
        """
        # Get list of atoms in molecule
        atoms = self.mol.GetAtoms()

        # If max_atoms is not provided, max_atoms = len(atoms)
        if self.max_atoms is None:
            n_atoms = len(list(atoms))
        else:
            n_atoms = self.max_atoms

        # Create empty node matrix
        node_mat = np.zeros((n_atoms, self.node_vec_len))

        # Iterate over atoms and add to matrix
        for atom in atoms:
            # Get atom index and atomic number
            atom_index = atom.GetIdx()
            atom_no = atom.GetAtomicNum()

            # Assign to node matrix
            if atom_no < self.node_vec_len:
                node_mat[atom_index, atom_no] = 1
            else:
                # Handle cases where atomic number exceeds node_vec_len
                # This can be adjusted based on specific requirements
                pass

        # Get adjacency matrix using RDKit
        adj_mat = rdmolops.GetAdjacencyMatrix(self.mol)
        self.std_adj_mat = np.copy(adj_mat)

        # Get distance matrix using RDKit
        dist_mat = rdDistGeom.GetMoleculeBoundsMatrix(self.mol)
        dist_mat[dist_mat == 0.0] = 1.0  # Avoid division by zero

        # Get modified adjacency matrix with inverse bond lengths
        adj_mat = adj_mat * (1.0 / dist_mat)

        # Pad the adjacency matrix with zeros if necessary
        dim_add = n_atoms - adj_mat.shape[0]
        if dim_add > 0:
            adj_mat = np.pad(
                adj_mat,
                pad_width=((0, dim_add), (0, dim_add)),
                mode='constant',
            )

        # Add identity matrix to adjacency matrix
        # making each atom its own neighbor
        adj_mat = adj_mat + np.eye(n_atoms)

        # Save adjacency and node matrices
        self.node_mat = node_mat
        self.adj_mat = adj_mat


class GraphDataset(Dataset):
    """
    Dataset for chemical graphs.

    This dataset class handles loading molecular data from a CSV file,
    where each entry contains a SMILES string and an experimental output value.
    It transforms each SMILES string into a graph representation
    suitable for machine learning models.

    Attributes:
        node_vec_len (int): Length of the node feature vector.
        max_atoms (int): Maximum number of atoms in each graph.
        indices (List[int]): List of dataset indices.
        smiles (List[str]): List of SMILES strings.
        outputs (List[float]): List of experimental output values.
    """

    def __init__(
        self,
        node_vec_len: int = settings.node_vec_len,
        max_atoms: int = settings.max_atoms,
    ) -> None:
        """
        Initializes the GraphDataset by loading data from a CSV file.

        Args:
            dataset_path (str): Path to the dataset CSV file.
                The CSV should contain at least 'smiles' and 'expt' columns.
            node_vec_len (int): Length of the node feature vector.
            max_atoms (int): Maximum number of atoms in each graph.
        """
        self.node_vec_len = node_vec_len
        self.max_atoms = max_atoms

        # Load dataset file
        df = load_from_db()

        # Extract columns
        self.indices = df.index.tolist()
        self.smiles = df['smiles'].tolist()
        self.outputs = df['expt'].tolist()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.indices)

    def __getitem__(self, i: int):
        """
        Retrieves the i-th sample from the dataset.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            tuple:
            - tuple of torch.Tensor: Node feature matrix and adjacency matrix.
            - torch.Tensor: Experimental output value.
            - str: SMILES string of the molecule.
        """
        # Get SMILES string
        smile = self.smiles[i]

        # Create graph using the Graph class
        mol = Graph(
            molecule_smiles=smile,
            node_vec_len=self.node_vec_len,
            max_atoms=self.max_atoms
        )

        # Get the matrices
        node_mat = torch.Tensor(mol.node_mat)
        adj_mat = torch.Tensor(mol.adj_mat)

        # Get output
        output = torch.Tensor([self.outputs[i]])

        return (node_mat, adj_mat), output, smile


def load_from_db():
    logger.info("Loading data from database")
    query = select(SolvationMolecules)
    return pd.read_sql(query, engine)


def collate_graph_dataset(dataset: Dataset):
    """
    Custom collate function for DataLoader to batch graph datasets.

    This function aggregates individual samples into batched tensors
    suitable for training models with DataLoader.

    Args:
        dataset (Dataset): The dataset to collate.

    Returns:
        tuple:
        - tuple of torch.Tensor: Batched node, feature and adjacency matrices.
        - torch.Tensor: Batched experimental output values.
        - list of str: List of SMILES strings in the batch.
    """
    # Initialize lists to hold batch data
    node_mats = []
    adj_mats = []
    outputs = []
    smiles = []

    # Iterate over the dataset and collect components
    for i in range(len(dataset)):
        (node_mat, adj_mat), output, smile = dataset[i]
        node_mats.append(node_mat)
        adj_mats.append(adj_mat)
        outputs.append(output)
        smiles.append(smile)

    # Concatenate node and adjacency matrices along the batch dimension
    node_mats_tensor = torch.stack(node_mats, dim=0)
    adj_mats_tensor = torch.stack(adj_mats, dim=0)
    outputs_tensor = torch.stack(outputs, dim=0)

    # Return the batched tensors and SMILES strings
    return (node_mats_tensor, adj_mats_tensor), outputs_tensor, smiles


def retrieve_dataset(
    max_atoms: int = settings.max_atoms,
    node_vec_len: int = settings.node_vec_len,
    log: bool = True
) -> GraphDataset:
    """
    Retrieves the GraphDataset from the given CSV file path.

    Initializes a GraphDataset object with the specified parameters.

    Args:
        dataset_path (str): Path to the dataset CSV file.
            The CSV should contain at least 'smiles' and 'expt' columns.
        max_atoms (int): Maximum number of atoms in each graph.
        node_vec_len (int): Length of the node feature vector.

    Returns:
        GraphDataset: The loaded graph dataset.
    """

    dataset = GraphDataset(
        max_atoms=max_atoms,
        node_vec_len=node_vec_len,
    )
    return dataset


def get_indices(dataset):
    """
    Generates an array of sequential indices for the given dataset.

    Args:
        dataset (Dataset): The dataset for which to generate indices.

    Returns:
        numpy.ndarray: An array of indices for the dataset.
    """
    return np.arange(0, len(dataset), 1)


def get_sizes(dataset,
              train_frac: float = 0.8):
    """
    Calculates samples for training and testing based on the given fraction.

    Args:
        dataset (Dataset): The dataset for which to determine train/test sizes.
        train_frac (float, optional): Fraction of samples for training.
            Defaults to 0.8.

    Returns:
        tuple: Tuple containing:
            - int: Number of samples for training.
            - int: Number of samples for testing.
    """
    train_size = int(np.round(train_frac * len(dataset)))
    test_size = len(dataset) - train_size
    return train_size, test_size


def sample_train_test(dataset_indices, train_size, log: bool = True):
    """
    Randomly samples indices for training and testing.

    Args:
        dataset_indices (numpy.ndarray): Array of dataset indices.
        train_size (int): Number of samples to allocate for training.

    Returns:
        tuple: Tuple containing:
            - numpy.ndarray: Indices for training.
            - numpy.ndarray: Indices for testing.
    """
    train_indices = np.random.choice(dataset_indices,
                                     size=train_size,
                                     replace=False)
    test_indices = np.array(list(set(dataset_indices) - set(train_indices)))
    if log:
        logger.info(f'Split data in {train_indices.size} points for training,'
                    f' and {test_indices.size} for testing.')
    return train_indices, test_indices


def get_dataloader(dataset,
                   batch_size: int = settings.batch_size):
    """
    Creates DataLoader objects for single dataset.

    Args:
        dataset (Dataset): The dataset to be loaded.

        batch_size (int, optional): Number of samples per batch.
            Defaults to 32.

    Returns:
        DataLoader: DataLoader for data.
    """

    # Create DataLoader
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=collate_graph_dataset)

    return loader


def get_dataloaders(dataset,
                    train_indices,
                    test_indices,
                    batch_size: int = settings.batch_size):
    """
    Creates DataLoader objects for training and testing.

    Args:
        dataset (Dataset): The dataset to be loaded.
        train_indices (numpy.ndarray): Indices for training samples.
        test_indices (numpy.ndarray): Indices for testing samples.
        batch_size (int, optional): Number of samples per batch.
            Defaults to 32.

    Returns:
        tuple: Tuple containing:
            - DataLoader: DataLoader for training data.
            - DataLoader: DataLoader for testing data.
    """
    # Randomly sample from indices
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create DataLoaders
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=collate_graph_dataset)
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=test_sampler,
                             collate_fn=collate_graph_dataset)

    return train_loader, test_loader


def get_split_dataset_loaders(log=True):
    """
    Retrieves dataset and splits it into train and test DataLoader objects.

    Returns:
        tuple: Tuple containing:
            - GraphDataset: The loaded dataset.
            - DataLoader: DataLoader for training data.
            - DataLoader: DataLoader for testing data.
    """
    # Get sizes
    dataset = retrieve_dataset(log=log)
    dataset_indices = get_indices(dataset)
    train_size, test_size = get_sizes(dataset)

    # Randomly sample train and test indices
    train_indices, test_indices = sample_train_test(dataset_indices,
                                                    train_size,
                                                    log=log)

    # Create dataloaders
    train_loader, test_loader = get_dataloaders(dataset,
                                                train_indices,
                                                test_indices,
                                                )

    return dataset, train_loader, test_loader


if __name__ == '__main__':

    dataset = retrieve_dataset()
    dataset_indices = get_indices(dataset)
    train_size, test_size = get_sizes(dataset)
    print(f'Train size: {train_size}')
    print(f'Test size: {test_size}')
