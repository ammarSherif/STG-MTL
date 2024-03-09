"""An example on CIFAR10 DataModule implementing the abstract methods"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com>
# License: MIT
# ==============================================================================

# ==============================================================================
# The file includes Classes that wraps our datasets used in our experiments.
# ------------------------------------------------------------------------------
# The vision is that  each dataset will have two classes that  need to be imple-
# mented:
# 1- DatasetModule class:  it is responsible for loading the data and initializ-
#    ing the dataloaders for training, validation, and test datasets.
# 2- Dataset class: it is used by the DatasetModule class  and contains the code
#    that specifies how to generate the task labels from such dataset.
# ==============================================================================
from .base import *
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
import torch
# ==============================================================================
# Create a custom Dataset class that wraps the individual tasks of CIFAR10.
# ==============================================================================
class CIFAR10Dataset(BaseDataset):
    """
    The class specifies how each label  will be generated from  CIFAR10 Dataset,
    and generates the data such that it could be used to generate Datamaps.
    ----------------------------------------------------------------------------
    The tasks generated are: airplane,  automobile, bird, cat, deer, dog,  frog,
        horse, ship, truck,  Living being,  Odd-numbered class label,  Donwside,
        Not Living being, and Random.
    ----------------------------------------------------------------------------
    """
    def __init__(self, data, task_ids, transforms=None):
        super().__init__(data, task_ids)
        # ----------------------------------------------------------------------
        # Store the transforms
        # ----------------------------------------------------------------------
        self.__transforms = transforms
        self.__random_labels = None
        # ----------------------------------------------------------------------
        # Generate flags for the random flip task
        # ----------------------------------------------------------------------
        self.__flipper = torch.randint(low=0, high=2, size=(len(self._data),1),
                                      dtype=torch.float32,
                                      generator= torch.Generator()\
                                          .manual_seed(67))\
                                              .squeeze(dim=1)
        if (14 in task_ids):
            self.__random_labels = torch.randint(low=0, high=2,
                                                 size=(len(self._data),1),
                                                 dtype=torch.float32,
                                                 generator= torch.Generator()\
                                                    .manual_seed(61))\
                                                        .squeeze(dim=1)

    # --------------------------------------------------------------------------
    def __get_label(self, idx):
        # ----------------------------------------------------------------------
        # Get the class number and the random label if we have a random task
        # ----------------------------------------------------------------------
        cifar_label = self._data[idx][1]
        random_label = self.__random_labels[idx] if 14 in self._task_ids else 0
        flip_flag = self.__flipper[idx] if 12 in self._task_ids else 0
        labels = []
        # ----------------------------------------------------------------------
        # Now loop over each task and add its label in the label list
        # ----------------------------------------------------------------------
        for task in self._task_ids:
            label = (task < 10 and cifar_label == task) or \
                    (task == 10 and cifar_label>=2 and cifar_label<=7) \
                    or (task == 11 and (cifar_label%2 == 1)) or \
                    (task == 12 and flip_flag) or \
                    (task == 13 and (cifar_label<2 or cifar_label>7)) or \
                    (task == 14  and random_label)

            labels.append(int(label))
        # ----------------------------------------------------------------------
        # Return the labels as a torch tensor
        # ----------------------------------------------------------------------
        return torch.Tensor(labels)

    # --------------------------------------------------------------------------
    def __getitem__(self,idx):
        """
        The method returns:
            - idx: the  index of the data point to help identify it while gener-
                   ating data maps
            - data_point: the image from CIFAR10
            - label: a torch tensor of the labels of that point according to the
                   list of tasks
        """
        # ----------------------------------------------------------------------
        # We flip each odd-numbered image and return it along with the labels
        # ----------------------------------------------------------------------
        img_flip_t = transforms.RandomVerticalFlip(p=1)
        image = self._data[idx][0]
        # ----------------------------------------------------------------------
        # Apply the transform if any
        # ----------------------------------------------------------------------
        if self.__transforms is not None:
            image = self.__transforms(image)
        
        if self.__flipper[idx]:
            image = img_flip_t(image)
        return idx, image, self.__get_label(idx)


# ==============================================================================
# Create a  DatasetModule class that generates  loaders for the individual tasks
# of CIFAR10.
# ==============================================================================
class C10DataModule(BaseDataModule):
    """
    The class  contains the code needed  to load the CIFAR10 dataset. It is also
    the responsible for splitting the  dataset into train, validation,  and test
    sets along with initializing their dataloaders.
    ----------------------------------------------------------------------------
    """
    # --------------------------------------------------------------------------
    # Initialize the positive weights applied to the loss to handle imbalance of
    # the data.
    # --------------------------------------------------------------------------
    def __init__(self, data_root, task_ids, train_val_test_split,
                 transforms:tuple|list|None = None):
        self.__train_transforms = None
        self.__validation_transforms = None
        self.__test_transforms = None

        # ----------------------------------------------------------------------
        # Unpack the transforms
        # ----------------------------------------------------------------------

        if type(transforms) == tuple or type(transforms) == list:
            if len(transforms) != 3:
                msg = f"Number of transforms should be 3; you passed "+\
                      f"{len(transforms)}.\n"+"One for each of datasets\n"
                raise IndexError(msg)
            else:
                self.__train_transforms, \
                    self.__validation_transforms, \
                    self.__test_transforms = transforms
        else:
            self.__train_transforms = transforms
        # ----------------------------------------------------------------------
        # Update the parent  class attributes according to our  dataset as below
        # ----------------------------------------------------------------------
        BaseDataModule._pos_weights = np.array([10, 10, 10, 10, 10, 10, 10, 10,
                                                 10, 10, 1, 1, 1, 1, 1])
        BaseDataModule._task_names = np.array(['airplane', 'automobile','bird',
                            'cat', 'deer','dog','frog','horse', 'ship', 'truck',
                            'Living being', 'Odd-numbered', 'Downside',
                            'Not Living being', 'Random'])
        # ----------------------------------------------------------------------
        # Now initiate the parent constructor. Implicitly, it calls the
        # _initialize method.
        # ----------------------------------------------------------------------
        super().__init__(data_root, task_ids, train_val_test_split)

        # ----------------------------------------------------------------------
        # Wrap the datasets in our defined class
        # ----------------------------------------------------------------------
        self._data_train = CIFAR10Dataset(self._data_train, self._task_ids,
                                          transforms= self.__train_transforms)
        self._data_val = CIFAR10Dataset(self._data_val, self._task_ids,
                                        transforms=self.__validation_transforms)
        self._data_test = CIFAR10Dataset(self._data_test, self._task_ids,
                                         transforms= self.__test_transforms)

    # --------------------------------------------------------------------------
    def __load_dataset(self):
        torchvision.datasets.CIFAR10(root= self._data_root,
                                             train= True,
                                             download= True)
        torchvision.datasets.CIFAR10(root= self._data_root,
                                             train= False,
                                             download= True)

    # --------------------------------------------------------------------------
    def _initialize(self):
        """
        The private function initializes the data_train, data_val, and data_test
        """
        # ----------------------------------------------------------------------
        # First, load the datasets
        # ----------------------------------------------------------------------
        self.__load_dataset()
        # ----------------------------------------------------------------------
        # Apply the transforms to the dataset and loads it
        # ----------------------------------------------------------------------
        data_train = torchvision.datasets.CIFAR10( root= self._data_root,
                                                  train= True)
        self._data_test = torchvision.datasets.CIFAR10( root= self._data_root,
                                                  train= False)
        size = len(data_train)
        total_ratio = self._train_val_test_split[[0,1]].sum()
        # ----------------------------------------------------------------------
        # Splits the dataset into training and validation datasets
        # ----------------------------------------------------------------------
        ratios = np.array(self._train_val_test_split[[0,1]]/total_ratio*size,
                          dtype = int)
        self._data_train, self._data_val = random_split(
                dataset= data_train, lengths = ratios,
                generator=torch.Generator().manual_seed(13))

    # --------------------------------------------------------------------------
