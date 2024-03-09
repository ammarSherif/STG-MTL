"""A modular abstract  base class for DatasetModule,  which would be used within
our method"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com>
# License: MIT
# ==============================================================================

# ==============================================================================
# The file includes a template classes that you need to implement  for your data
# to interact with our code
# ------------------------------------------------------------------------------
# The vision is that  each dataset will have two classes that  need to be imple-
# mented:
# 1- DatasetModule class:  it is responsible for loading the data and initializ-
#    ing the dataloaders for training, validation, and test datasets.
# 2- Dataset class: it is used by the DatasetModule class  and contains the code
#    that specifies how to generate the task labels from such dataset.
# ==============================================================================
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import numpy as np
# ==============================================================================
# Create a  custom Base  Dataset class that  wraps the  individual tasks of your
# task.
# ==============================================================================
class BaseDataset(Dataset, ABC):
    """
    The class specifies how each label will be generated from your  Dataset, and
    generates the data such that it could be used to generate Datamaps.
    ----------------------------------------------------------------------------
    NOTE You need to inherit this class implementing these methods  according to
         your particular dataset and tasks
    ----------------------------------------------------------------------------
    """
    def __init__(self, data, task_ids):
        self._task_ids = task_ids
        self._data = data

    # --------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    # --------------------------------------------------------------------------
    @abstractmethod
    def __getitem__(self,idx):
        """
        The method returns:
            - idx: the  index of the data point to help identify it while gener-
                   ating data maps
            - data_point: the data point from the data_set
            - label: a torch tensor of the labels of that point according to the
                   list of tasks
        """
        pass


# ==============================================================================
# Create a  DatasetModule class that generates  loaders for the individual tasks
# of your Base Dataset class.
# ==============================================================================
class BaseDataModule(ABC):
    """
    The class contains the code needed  to load your dataset. It is also respon-
    sible for splitting the  dataset into train, validation, and test sets along
    with initializing their dataloaders.
    ----------------------------------------------------------------------------
    Within the constructor, you must modify _pos_weights and _task_names accord-
    ing to your particular datasets before calling the parent constructor.
    ----------------------------------------------------------------------------
    """
    # --------------------------------------------------------------------------
    # If some tasks are imbalanced, define the weight of the  positive points to
    # balance your learning. These weights will be applied to the loss functions
    # to have an efficient learning.  Its length should be equivalent to the max
    # number of tasks.
    # --------------------------------------------------------------------------
    _pos_weights = np.array([1])
    # --------------------------------------------------------------------------
    # List of all the task names
    # --------------------------------------------------------------------------
    _task_names = np.array(['T1'])
    def __init__(self, data_root:str, task_ids, train_val_test_split):
        """
        The method initializes your data module with:
            - data_root: the location in which your data resides
            - task_ids: the subset of tasks this module will generate
            - train_val_test_split:  percentage of  train vs  validation vs test
                    datasets.
        """
        self._data_root = data_root
        self._task_ids = task_ids
        if train_val_test_split is not None:
            self._train_val_test_split = np.array(train_val_test_split)
        else:
            self._train_val_test_split = None

        self._data_train = None
        self._data_val = None
        self._data_test = None

        self._imbalance_pos_weights= BaseDataModule._pos_weights[self._task_ids]
        self._initialize()

    # --------------------------------------------------------------------------
    @abstractmethod
    def _initialize(self) -> None:
        """
        The  private method initializes  the data_train, data_val, and data_test
        """
        pass

    # --------------------------------------------------------------------------
    def get_task_names(self):
        """
        This method returns the name of tasks of this data module.
        """
        return BaseDataModule._task_names[self._task_ids]
    
    # --------------------------------------------------------------------------
    def get_task_ids(self):
        """
        This method returns the ids of tasks of this data module.
        """
        return self._task_ids

    # --------------------------------------------------------------------------
    def __get_dataloader(self, dataset:BaseDataset, batch_size:int, 
                        shuffle:bool = False, num_workers: int = 0,
                        pin_memory:bool = True) -> DataLoader:
        """
        This method returns a dataloader of the dataset

        ------------------------------------------------------------------------
        Inputs:
            - dataset: a dataset object from <BaseDataset> type
            - batch_size: the batch size 
            - shuffle: whether to shuffle or not
                - default: False
            - num_workers: the number of workers
                - default: 0
            - pin_memory: whether to pin the in the memory or not
                - default: True
        """
        return DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            pin_memory = pin_memory
        )

    # --------------------------------------------------------------------------
    def get_train_dataloader(self, batch_size:int, shuffle:bool = True,
                             num_workers: int = 0,
                             pin_memory:bool = True) -> DataLoader:
        """
        This method returns a data loader of the training dataset
        """
        return self.__get_dataloader(self._data_train, batch_size, shuffle,
                                     num_workers, pin_memory)

    # --------------------------------------------------------------------------
    def get_val_dataloader(self, batch_size:int, shuffle:bool = False,
                            num_workers: int = 0,
                            pin_memory:bool = True) -> DataLoader:
        """
        This method returns a data loader of the validation dataset
        """
        return self.__get_dataloader(self._data_val, batch_size, shuffle,
                                     num_workers, pin_memory)

    # --------------------------------------------------------------------------
    def get_test_dataloader(self, batch_size:int, shuffle:bool = False,
                            num_workers: int = 0,
                            pin_memory:bool = True) -> DataLoader:
        """
        This method returns a data loader of the test dataset
        """
        return self.__get_dataloader(self._data_test, batch_size, shuffle,
                                     num_workers, pin_memory)

    # --------------------------------------------------------------------------
    def get_imbalance_weights(self):
        """
        This method returns the weights to balance the training of each tasks
        """
        return self._imbalance_pos_weights

    # --------------------------------------------------------------------------
    def get_train_size(self) -> int:
        """return the size of the training set"""
        return len(self._data_train)
    # --------------------------------------------------------------------------
    def get_val_size(self) -> int:
        """return the size of the validation set"""
        return len(self._data_val)
    # --------------------------------------------------------------------------
    def get_test_size(self) -> int:
        """return the size of the test set"""
        return len(self._data_test)
