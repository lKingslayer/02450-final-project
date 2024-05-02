import numpy as np

from .dataset import Dataset, DictDataset, TeachableWrapperDataset

class KinaseDataset(Dataset):
    def __init__(self, X, y, groups):
        # Initialize parent class
        super().__init__()
        # Initialize KinaseDataset specific attributes
        self._X = X
        self._y = y
        self._groups = groups

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def shape(self):
        # Implementing the required abstract property 'shape'
        if hasattr(self, '_X'):
            return self._X.shape  # Assuming X is a numpy array or similar
        else:
            return (0,)

    def __len__(self):
        # Return the number of samples, assuming y has a shape attribute
        return len(self._y)

    def __getitem__(self, index):
        # Return a sample at a particular index
        return {'X': self._X[index], 'y': self._y[index]}

    def find(self, value, first=True):
        # Method to find a value in y
        indices = np.where(self._y == value)[0]
        return indices[0] if first and indices.size > 0 else indices
    
    def split(self, labeled_fraction):
        """
        Splits the dataset into labeled and unlabeled datasets based on the specified fraction.

        Parameters:
        labeled_fraction (float): The fraction of the dataset to remain labeled (e.g., 0.4 for 40%).

        Returns:
        labeled_dataset (KinaseDataset): The labeled portion of the dataset.
        unlabeled_dataset (KinaseDataset): The unlabeled portion where the labels are set to None.
        """
        total_samples = len(self)
        labeled_count = int(np.floor(labeled_fraction * total_samples))

        # Shuffling indices to randomly select for labeled data
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        labeled_indices = indices[:labeled_count]
        unlabeled_indices = indices[labeled_count:]

        labeled_dataset = KinaseDataset(self._X[labeled_indices], self._y[labeled_indices], self._groups[labeled_indices])
        # For unlabeled data, we could consider setting 'y' to None or another appropriate value indicating unlabeled status
        unlabeled_dataset = KinaseDataset(self._X[unlabeled_indices], self._y[unlabeled_indices], self._groups[unlabeled_indices])

        return labeled_dataset, unlabeled_dataset
