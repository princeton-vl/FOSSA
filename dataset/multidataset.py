import warnings
import numpy as np
from .base import BaseDataset
from importlib import import_module
import math

def get_data(data_name):
    module_name = 'dataset.' + data_name.lower()
    module = import_module(module_name)

    return getattr(module, data_name)

warnings.filterwarnings("ignore", category=UserWarning)

class MultiDataset(BaseDataset):
    def __init__(self, args, mode="train"):
        # only support mixing training for now
        assert mode == "train"

        self.dataset_names = args.train_data_name.split('+')
        self.datasets = []
        for dataset_name in self.dataset_names:
            data = get_data(dataset_name)
            self.datasets.append(data(args, mode=mode))

        self.num_datasets = len(self.datasets)

        # Force total length (number of samples per epoch) to be specified
        # We set this to Hypersim's length for now for consistency with earlier experiments, and resample indices each epoch to get full coverage
        self.total_length = args.mixed_dataset_total_length

        self.deterministic = args.deterministic

        # Each dataset contributes equally to the total length
        # Round up make sure we get no index out of bounds
        self.subset_length = math.ceil(self.total_length / self.num_datasets)

        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        if self.deterministic:
            np.random.seed(0)
        for dataset in self.datasets:
            print(f"Generating indices for dataset: {type(dataset).__name__} with {len(dataset)} samples.")
            if len(dataset) <= self.subset_length:

                # Guarantee that all samples are seen at least once
                dataset_indices = range(len(dataset))
                remaining_indices_to_sample = self.subset_length - len(dataset)

                if remaining_indices_to_sample > 0:
                    # Sample remaining indices with replacement
                    dataset_indices = list(dataset_indices) + list(
                        np.random.choice(range(len(dataset)), size=remaining_indices_to_sample, replace=True)
                    )

            else:
                dataset_indices = np.random.choice(range(len(dataset)), size=self.subset_length, replace=False)
            indices.append(dataset_indices)
            
            
            print(f"Dataset {type(dataset).__name__} contributing this many unique samples: {len(set(dataset_indices))} out of {len(dataset)}")
        return indices
    
    def get_dataset_name(self, idx):
        dataset_idx = idx % self.num_datasets
        return type(self.datasets[dataset_idx]).__name__

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = idx % self.num_datasets
        sample_idx = idx // self.num_datasets
        sample =  self.datasets[dataset_idx][self.indices[dataset_idx][sample_idx]]
        return sample

    def refresh_indices(self):
        self.indices = self._generate_indices()