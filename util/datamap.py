# ==============================================================================
# The file  includes the code used to generate datamaps using [Welford's] online
# algorithm
# ==============================================================================
# Reference links:
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
# https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation
# ==============================================================================
import torch

class DataMapRecorder():
    """
    The class is responsible for generating  datamap recorder objects that keep\
    track of values needed to generate datamaps using Welford's online algorithm

    ----------------------------------------------------------------------------
    Inputs
        - num_tasks: the number of tasks of our data,  which is also the number\
            of datamaps generated (1 datamap for each task)
        - num_points: the number of datapoins within the datamap
        - device: the datamap of the tensors
    """
    # --------------------------------------------------------------------------
    def __init__(self, num_tasks: int, num_points: int,
                 device: torch.device) -> None:
        # ----------------------------------------------------------------------
        # Initlize the values
        # ----------------------------------------------------------------------
        self.__num_tasks:int = num_tasks
        self.__num_points:int = num_points
        self.__device:torch.device = device
        # ----------------------------------------------------------------------
        # Initialize the variables used within generation
        # ----------------------------------------------------------------------
        self.__count = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)
        self.__mean = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)
        self.__std = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)
        self.__m2 = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)

    # ==========================================================================

    def update(self, confidence:torch.Tensor, indxs:torch.Tensor) -> None:
        """
        The method updates the history of the values with a new batch of confid\
        ence, representing the model confidence of the true class of  the data-\
        points, given in <confidence> and  <indxs> representing the  indices of\
        datapoints

        ------------------------------------------------------------------------
        Inputs:
            - confidence: model confidence, probability of the true class, of a\
                batch of datapoints
                - shape: batch size x num tasks => (batch size, num tasks)
            - indxs: the indices of the datapoints that constitute the batch
                - shape: batch size
        """
        # ----------------------------------------------------------------------
        # reshape to be num_tasks x batch_size instead of batch size x num tasks
        # ----------------------------------------------------------------------
        conf = torch.transpose(confidence, 0, 1)
        # ======================================================================
        # Update the history values for the selected range of indices only.
        # ----------------------------------------------------------------------
        # Below we are implementing the update part of Welford's algorithm
        # ======================================================================
        self.__count[:,indxs] = self.__count[:,indxs]+1
        delta = conf - self.__mean[:,indxs]
        self.__mean[:,indxs] += delta / self.__count[:,indxs]
        delta2 = conf - self.__mean[:,indxs]
        self.__m2[:,indxs] += delta * delta2

    # ==========================================================================

    def reset(self) -> None:
        """
        The method resets the history, so as to start recording a new experiment
        """
        # ----------------------------------------------------------------------
        # Initialize the variables used within generation
        # ----------------------------------------------------------------------
        self.__count = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)
        self.__mean = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)
        self.__std = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)
        self.__m2 = torch.zeros(self.__num_tasks, self.__num_points,
                                    device=self.__device, requires_grad=False)

    # ==========================================================================

    def __compute(self):
        """
        The function computes the required values to compute the  datamap given\
        the recorded history
        """
        self.__std = (self.__m2/(self.__count-1)).sqrt()
    
    # ==========================================================================

    def get_mean(self) -> torch.Tensor:
        """
        returns the mean confidence according to the encountered values

        ------------------------------------------------------------------------
        Outputs:
            - the mean of the encountered confidence for each data point
                - shape: num tasks x num datapoints (num tasks, num datapoints)
        """
        return self.__mean
    
    # ==========================================================================

    def get_std(self):
        """
        returns the standard deviation of the confidence

        ------------------------------------------------------------------------
        Outputs:
            - the std of the encountered confidence for each data point
                - shape: num tasks x num datapoints (num tasks, num datapoints)
        """
        return self.__std
    
    # ==========================================================================

    def get_state(self) -> dict[str,int|torch.Tensor]:
        """
        The method saves  the recorder state  for future loading

        ------------------------------------------------------------------------
        Outputs:
            - the recorder state
        """
        rec_state:dict[str,int|torch.Tensor] = {
            "num_tasks": self.__num_tasks,
            "num_points": self.__num_points,
            "count": self.__count,
            "mean": self.__mean,
            "m2": self.__m2
        }

        return rec_state

    # ==========================================================================

    def load_state(self, state_dict:dict[str,int|torch.Tensor]):
        """
        The method loads the Recorder state from previously stored state via \
        <get_state>

        ------------------------------------------------------------------------
        Inputs:
            - state_dict: the recorder state dictionary
        """
        self.reset()
        self.__num_tasks = state_dict["num_tasks"]
        self.__num_points = state_dict["num_points"]
        self.__count = state_dict["count"]
        self.__mean = state_dict["mean"]
        self.__m2 = state_dict["m2"]
    
    # ==========================================================================

    def get_datamap(self):
        """
        The method returns a tensor of the computed datamap according to encoun\
        tered values of the confidence

        ------------------------------------------------------------------------
        Outputs:
            - the datamap of the encountered confidence values
                - shape: num tasks x num datapoints x 2 that is \
                    (num tasks, num datapoints, 2)
        """
        with torch.no_grad():
            self.__compute()
            return torch.cat((torch.unsqueeze(self.__mean,dim=-1),
                            torch.unsqueeze(self.__std,dim=-1)), dim=-1)\
                                .to(torch.device("cpu"))

