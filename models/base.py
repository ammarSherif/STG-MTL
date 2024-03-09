"""A modular abstract base class for NN models, to be used within our method"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com>
# License: MIT
# ==============================================================================

# ==============================================================================
# The file includes a template classes that you need to implement for your model
# to interact with our code. These models will be later used to generate datamap
# evaluate our grouping algorithm.
# ------------------------------------------------------------------------------
# The vision is that each task model will have two classes  that need to be  im-
# plemented:
# 1- BaseModel class:  it contains your model along with its architecture
# 2- BaseMetrics class: it is used by the BaseModel and it  includes the code to
#       generate the metrics we want for our tasks.
# ==============================================================================
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import enum

# ==============================================================================
# Define an enum to help clarifying the losses type, in case our model can oper-
# ate on multiple losses.
# ==============================================================================
class LossType(enum.Enum):
    BCEWithLogits = 0
    BCE = 1
    CrossEntropy = 2
    MSE = 3
    MAE = 4


# ==============================================================================
# Our base model with the methods we should implement
# ==============================================================================
class BaseModel(nn.Module, ABC):
    """
    The Base class  specifies the necessary  methods and attributes that need to
    be implemented to interface with our code.
    """
    def __init__(self, task_ids, device = 'cpu'):
        super(BaseModel,self).__init__()
        # ----------------------------------------------------------------------
        # Initially, the image is 3x32x32
        # ----------------------------------------------------------------------
        # Now, structure the CNN to have:
        # - 1 convolution BN + MaxPool
        # - 2 convolution BN + MaxPool
        # - 2 Fully connected layers
        # ----------------------------------------------------------------------
        self._model = None
        # ----------------------------------------------------------------------
        # Also, initialize the metric class, loss functions, num of tasks
        # ----------------------------------------------------------------------
        self._metrics = None
        self._losses = []
        self._task_ids = task_ids
        self._device = device
        self._intialize_metrics_generator(device)
    # --------------------------------------------------------------------------
    def forward(self, x):
        """
        The method computes the forward pass of our model passing the input x to
        our model and generating the output.
        """
        # ----------------------------------------------------------------------
        # NOTE Re-define the method in case you would like to customize the  be-
        #      havior even more.
        # ----------------------------------------------------------------------
        return self._model(x)

    # --------------------------------------------------------------------------
    @abstractmethod
    def _intialize_metrics_generator(self,device = 'cpu'):
        """
        The method initializes our metric generator class,  which is responsible
        for calculating the necessary metrics after each epoch
        ------------------------------------------------------------------------
        """
        pass

    # --------------------------------------------------------------------------
    def get_metrics_generator(self):
        """
        returns list of metric generator objects to be dealt with while training
        """
        return self._metrics

    # --------------------------------------------------------------------------
    def get_losses(self) -> list:
        """
        The method returns the list  of loss functions to be  used for each task
        """
        return self._losses

    # --------------------------------------------------------------------------
    @abstractmethod
    def get_task_output(self, batch_outputs, task_id):
        """
        The function  returns the output of  a particular task with a  <task_id>
        from an output of an input batch of data points.
        ------------------------------------------------------------------------
        Inputs:
            - batch_outputs: an output of  multiple tasks  of a particular batch
                    of inputs, be it produced by the model or not.
            - task_id: an id of the task which we want its model output.
        ------------------------------------------------------------------------
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def get_probabilities_true_target(self, outputs, targets):
        """
        The method computes the probability of the true targets for each task
        
        ------------------------------------------------------------------------
        Inputs:
            - outputs: the output from the model after the forward pass
            - targets: the true targets of the batch items.
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def initialize_losses(self):
        """
        The method initializes a list of loss functions such that each loss will
        handle a separate task.
        """
        pass

# ==============================================================================
# Create a Base class whose responsibility is to define the  required attributes
# and operations to calculate the output metrics for our model
# ==============================================================================
class BaseMetrics(ABC):
    """
    The class specifies the  output metrics we would like to calculate per each\
    epoch depending on our tasks. It calculates the metrics for a  single epoch.

    ----------------------------------------------------------------------------
    Inputs:
        - num_of_tasks: specifies the number of tasks of our model
        - device: specifies the device on which we store our results
    """
    def __init__(self, num_of_tasks: int, device:str|torch.device= 'cpu')->None:
        self._num_of_tasks = num_of_tasks
        self._device = device
        self._metric_names = None
        self.initialize()
    
    def get_metric_names(self) -> list:
        return self._metric_names
    
    # --------------------------------------------------------------------------
    @abstractmethod
    def get_state(self) -> dict:
        """
        The method returns a dictionary holding the generator state for future \
        loading at the end of a particular epoch
        """
        pass
    
    # --------------------------------------------------------------------------
    @abstractmethod
    def load_state(self, state_dict: dict) -> None:
        """
        The method loads the Generator state for future loading from previously\
        stored state via <get_state>
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def get_unsqueeze_concatenation_dims(self) -> tuple:
        """
        The method returns two dictionaries holding the dimensions to unsqueeze\
        upon if needed and the dimensions to concatenate upon

        ------------------------------------------------------------------------
        Outputs:
            - unsqueeze_dims: dictionary of dims to unsqueeze where the key ind\
                \bicate the index within the metrics, and the value is the dim
            - concat_dims: dictionary of dims to concatenate upon where the key\
                \bicate the index within the metrics, and the value is the dim
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def initialize(self) -> None:
        """
        The method initializes the variables at the beginning of each epoch
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def update_metrics_iteration(self, results: torch.Tensor,
                                targets: torch.Tensor,#):
                                ensemble_weights: torch.Tensor|None = None):
        """
        The method records the necessary information from each singel batch of \
        a single epoch. It is called after each iteration

        ------------------------------------------------------------------------
        Inputs:
            - results: the predicted output from the model\n\
                shape(batch_size, n_tasks, n_models)
            - targets: the true output from the dataset\n\
                shape(batch_size, n_tasks)
            - ensemble_weights: the weights of the ensemble to merge the outputs
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def compute_metrics_epoch(self) -> None:
        """
        The method computes  our metrics based on the previously recorded infor\
        \bmation from the update_metrics_batch method. It is called at the end \
        of each epoch.

        ------------------------------------------------------------------------
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def get_metrics(self) -> tuple|list:
        """
        The method returns the computed metrics in an iteratable  object like a\
        list of a tuple.

        ------------------------------------------------------------------------
        """
        pass

    # --------------------------------------------------------------------------
    @abstractmethod
    def get_num_of_metrics(self) -> int:
        """
        The method returns the number of metrics being computed.

        ------------------------------------------------------------------------
        """
        pass

    # --------------------------------------------------------------------------
