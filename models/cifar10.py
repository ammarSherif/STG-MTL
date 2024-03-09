"""An example on CIFAR10 Model implementing the abstract methods"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com>
# License: MIT
# ==============================================================================

# ==============================================================================
# The file includes the model used to generate  datamaps and evaluate our group-
# ing algorithm.
# ==============================================================================

import torch
import torchvision.models as models
from .base import *

# ==============================================================================
# Define an enum to help clarifying the model type
# ==============================================================================
class ModelType(enum.Enum):
    CUSTOM_1 = 0
    RESNET18 = 1
    RESNET34 = 2
    RESNET50 = 3
# ==============================================================================
# Create a typical Conv based architecture
# ==============================================================================
class C10Model(BaseModel):
    """
    The class specifies the model architecture we are  using on CIFAR10 dataset,
    which we use in both generating the datamaps and evaluating our pipeline.
    ----------------------------------------------------------------------------
    Inputs:
        - task_ids: specifies the ids of the tasks that our model generates.
        - pos_weights: the weights to balance the training of the tasks.
        - t: the threshold value to compute our metrics
        - device: the device on which we train and store our model
    ----------------------------------------------------------------------------
    """
    def __init__(self, task_ids, pos_weights, t = 0.5, device = 'cpu', 
                pretrain= False, model_type:ModelType = ModelType.CUSTOM_1):
        # ----------------------------------------------------------------------
        # Initialize the number of outputs (tasks) and the metric class
        # ----------------------------------------------------------------------
        self.__t = t
        # ----------------------------------------------------------------------
        # Initialize the balance weights according to our task_ids
        # ----------------------------------------------------------------------
        self.__pos_weights = pos_weights
        super().__init__(task_ids, device)
        # ----------------------------------------------------------------------
        # Build the model
        # ----------------------------------------------------------------------
        self._model = self.__build_model(model_type, pretrain).to(device)
        # ----------------------------------------------------------------------
        # Finally, initialize the losses for our model
        # ----------------------------------------------------------------------
        self.initialize_losses()

    # --------------------------------------------------------------------------
    def __build_model(self, model_type:ModelType = ModelType.CUSTOM_1,
                        pretrain: bool = False):
        """
        The method builds a model and return it according to the passed type
        """
        if model_type == ModelType.CUSTOM_1:
            # ------------------------------------------------------------------
            # Initially, the image is 3x32x32
            # ------------------------------------------------------------------
            # Now, structure the CNN to have:
            # - 1 convolution BN + MaxPool
            # - 2 convolution BN + MaxPool
            # - 2 Fully connected layers
            # ------------------------------------------------------------------
            return nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
                # --------------------------------------------------------------
                # Add a BatchNorm layer
                # --------------------------------------------------------------
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # --------------------------------------------------------------
                # After this the output should be 28x28
                # --------------------------------------------------------------
                nn.MaxPool2d(2,2),
                # --------------------------------------------------------------
                # After the pooling layer the output should be 14x14
                # --------------------------------------------------------------
                nn.Conv2d(in_channels = 32, out_channels = 64, padding = 1,
                        kernel_size = 3),
                # --------------------------------------------------------------
                # We  use padding in the second,  so our dimension remains
                # --------------------------------------------------------------
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels = 64, padding = 1,
                        kernel_size = 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Flatten(),
                # --------------------------------------------------------------
                # Use dropout layers to make sure the network will not overfit
                # --------------------------------------------------------------
                nn.Dropout(0.2),
                # --------------------------------------------------------------
                # After applying the pooling the resulting would be: 64 x 7 x 7
                # --------------------------------------------------------------
                # Therefore,  we define a Fully  Connected layer  with number of
                # features of 64*7*7 = 3136
                # --------------------------------------------------------------
                nn.Linear(3136, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, len(self._task_ids)),
                # --------------------------------------------------------------
                # No need for  the  sigmoid, as we  use the  loss function  with 
                # logits.
                # --------------------------------------------------------------
                # nn.Sigmoid()
            )
        elif model_type == ModelType.RESNET34:
            w = None if not pretrain else models.ResNet34_Weights.DEFAULT
            model = models.resnet34(weights= w, num_classes=len(self._task_ids))
            model.fc = torch.nn.Linear(512, len(self._task_ids), bias=False)
            return model
        elif model_type == ModelType.RESNET18:
            w = None if not pretrain else models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights= w, num_classes=len(self._task_ids))
            model.fc = torch.nn.Linear(512, len(self._task_ids), bias=False)
            return model
        elif model_type == ModelType.RESNET50:
            w = None if not pretrain else models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights= w, num_classes=len(self._task_ids))
            model.fc = torch.nn.Linear(2048, len(self._task_ids), bias=False)
            return model
    # --------------------------------------------------------------------------
    def _intialize_metrics_generator(self, device = 'cpu'):
        """
        The method initializes our metric generator class,  which is responsible
        for calculating the necessary metrics after each epoch
        ------------------------------------------------------------------------
        """
        # Define three metric generators for training, validation, and test
        self._metrics = []
        self._metrics.append(C10Metrics(len(self._task_ids), device, self.__t))
        self._metrics.append(C10Metrics(len(self._task_ids), device, self.__t))
        self._metrics.append(C10Metrics(len(self._task_ids), device, self.__t))

    # --------------------------------------------------------------------------
    def initialize_losses(self):
        """
        The method initializes a list of loss functions such that each loss will
        handle a separate task.
        """
        self._losses = []
        for pos_weight in self.__pos_weights:
            p_weight = torch.tensor([pos_weight]).to(self._device)
            self._losses.append(nn.BCEWithLogitsLoss(pos_weight= p_weight))

    # --------------------------------------------------------------------------
    def get_task_output(self, batch_outputs:torch.Tensor, task_id):
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
        # ----------------------------------------------------------------------
        # if task_id == 0 and len(self._task_ids) ==1:
        #     return batch_outputs
        # elif task_id < len(self._task_ids):
        #     return batch_outputs[:, task_id]
        if task_id < len(self._task_ids):
            return batch_outputs[:, task_id]
        else:
            raise IndexError("Error task id: inside get_task_output")

    # --------------------------------------------------------------------------
    def get_probabilities_true_target(self, outputs, targets):
        """
        The method computes the probability of the true targets for each task
        
        ------------------------------------------------------------------------
        Inputs:
            - outputs: the output from the model after the forward pass
            - targets: the true targets of the batch items.
        """
        # ----------------------------------------------------------------------
        # In out case,  after applying the sigmoid,  we generate the probability
        # of being 1. Therefore,  the probability of  the true target  is simply
        # the below equation.
        # ----------------------------------------------------------------------
        results = nn.Sigmoid()(outputs)

        return (1 - (targets - results).abs())

# ==============================================================================
# Create a class to calculate the output metrics for our model
# ==============================================================================
class C10Metrics(BaseMetrics):
    """
    The class specifies the  output metrics we would like  to calculate per each
    epoch depending on our tasks. This is an implementation for the C10Model
    ----------------------------------------------------------------------------
    This class will calculate accuracy along with the balanced accuracy per task
    for our C10Model for 1 single epoch.
    ----------------------------------------------------------------------------
    Inputs:
        - num_of_tasks: specifies the number of tasks of our model
        - device: specifies the device on which we store our results
        - t: the threshold value for binary output
    ----------------------------------------------------------------------------
    """
    def __init__(self, num_of_tasks, device = 'cpu', t = 0.5):
        self.__t = torch.Tensor([t])
        self.__t = self.__t.to(device)

        self.__accuracy_per_task = None
        self.__balanced_accuracy = None
        self.__task_correct = None
        self.__tp = None
        self.__fp = None
        self.__tn = None
        self.__fn = None

        self.__num_of_items = 0

        super().__init__(num_of_tasks, device)
        self._metric_names = ["Accuracy","Balanced Accuracy","F1"]

    # --------------------------------------------------------------------------
    def get_state(self):
        """
        The method saves  the Generator state  for future loading  at the end of
        a particular epoch
        """
        gen_state = {
            "t": self.__t,
            "num_of_tasks": self._num_of_tasks,
            "metric_names": self._metric_names,
        }

        return gen_state

    def load_state(self, state_dict):
        """
        The method loads the Generator state for future loading from  previously
        stored state via <get_state>
        """
        self.__t = state_dict["t"]
        self._num_of_tasks = state_dict["num_of_tasks"]
        self._metric_names = state_dict["metric_names"]

    # --------------------------------------------------------------------------
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
        # ----------------------------------------------------------------------
        # For all the metrics, we generate a tensor of <num_of_tasks>.  To merge
        # across different epochs, we want to have <num_of_tasks,epochs>,  so we
        # unsqueeze and concatenate on dim = 1
        # ----------------------------------------------------------------------
        unsqueeze_dims = {}
        concat_dims = {}
        for i in range(len(self._metric_names)):
            unsqueeze_dims[i] = 1
            concat_dims[i] = 1
        return unsqueeze_dims, concat_dims

    # --------------------------------------------------------------------------
    def initialize(self):
        """
        The method initializes the variables at the beginning of each epoch
        """
        self.__accuracy_per_task = torch.zeros(self._num_of_tasks
                                               ).to(self._device)
        self.__balanced_accuracy = torch.zeros(self._num_of_tasks
                                               ).to(self._device)
        self.__f1 = torch.zeros(self._num_of_tasks).to(self._device)
        self.__task_correct = torch.zeros(self._num_of_tasks).to(self._device)

        self.__tp = torch.zeros(self._num_of_tasks).to(self._device)
        self.__fp = torch.zeros(self._num_of_tasks).to(self._device)
        self.__tn = torch.zeros(self._num_of_tasks).to(self._device)
        self.__fn = torch.zeros(self._num_of_tasks).to(self._device)
        self.__num_of_items = 0

    # --------------------------------------------------------------------------
    def update_metrics_iteration(self, results, targets, ensemble_weights=None):
        # ----------------------------------------------------------------------
        # Move the results and targets to the device
        # ----------------------------------------------------------------------
        results = results.to(self._device)
        targets = targets.to(self._device)

        # ----------------------------------------------------------------------
        # Note we discarded the last sigmoid layer  due to the loss  function we
        # use, so we insert such layer
        # ----------------------------------------------------------------------
        results = nn.Sigmoid()(results)

        # ----------------------------------------------------------------------
        # Now compute the ensemble
        # ----------------------------------------------------------------------
        if len(results.shape)==3 and ensemble_weights is None:
            ensemble_weights = torch.ones(results.shape[1], results.shape[2])
            ensemble_weights = ensemble_weights/results.shape[2]
        if len(results.shape)==3:
            ensemble_weights = ensemble_weights.to(self._device)
            results = results*ensemble_weights
            results = results.sum(dim=2)

        # ----------------------------------------------------------------------
        # Calculate the number of crrect results
        # ----------------------------------------------------------------------
        results = (results > self.__t).float()
        self.__task_correct += (results == targets.float()).sum(axis=0)
        self.__num_of_items += len(targets)

        # ----------------------------------------------------------------------
        # Compute the TP, TN, FP, and FN to compute balanced_acc
        # ----------------------------------------------------------------------
        self.__tp += ((targets==1) * (results==1)).sum(axis=0)
        self.__tn += ((targets==0) * (results==0)).sum(axis=0)
        self.__fn += ((targets==1) * (results==0)).sum(axis=0)
        self.__fp += ((targets==0) * (results==1)).sum(axis=0)

    # --------------------------------------------------------------------------
    def compute_metrics_epoch(self):
        self.__accuracy_per_task = self.__task_correct / self.__num_of_items
        recall = self.__tp / (self.__tp + self.__fn)
        precision = self.__tp / (self.__tp + self.__fp)
        specificity = self.__tn / (self.__tn + self.__fp)

        self.__balanced_accuracy = (recall + specificity) / 2
        self.__f1 = 2 * precision * recall / (precision + recall)

    # --------------------------------------------------------------------------
    def get_metrics(self) -> tuple|list:
        return self.__accuracy_per_task.detach().clone().cpu(),\
                self.__balanced_accuracy.detach().clone().cpu(),\
                self.__f1.detach().clone().cpu()

    # --------------------------------------------------------------------------
    def get_num_of_metrics(self) -> int:
        """
        The method returns the number of metrics being computed. We return 3 as\
        we are computing the accuracy, the balanced accuracy,  and F1.
        """
        return 3

    # --------------------------------------------------------------------------
