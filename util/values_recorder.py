# ==============================================================================
# The file  includes the code used to  record the history of  values of interest
# ==============================================================================

import torch
import os
from .util_enum import ValueStoringMode

class ValuesRecorder():
    """
    The class is responsible for generating recorder objects that keep track of\
    any values of interest with the option to compact these values into tensors\
    for efficiency

    ----------------------------------------------------------------------------
    Inputs:
        - metric_names: the metric names to be traced
        - id: to possibly identify this recorder like 0 or "training"
        - unsqueeze_dims: dictionary of items whose keys identify the index of \
            values within <num_metrics> and an associated value to identify the\
            -dimension to unsqueeze for all the values being updated before sto\
            ring them
        - concat_dims: similar to <unsqueeze_dims>, but to identify the dim to \
            concatenate upon
        - metric_names: list  of strings to identify the names of the recorded \
            metrics
    """
    def __init__(self, metric_names: list[str], id:int|str,
                 unsqueeze_dims:dict[int,int] = {},
                 concat_dims:dict[int,int] = {}) -> None:
        self.__num_metrics = len(metric_names)
        self.__metric_names= metric_names
        self.__unsqueeze_dims = unsqueeze_dims
        self.__concat_dims = concat_dims
        self.__id = id
        self.__value_history:list[list[torch.Tensor]] = \
            [[] for i in range(self.__num_metrics)]
        self.__data_paths:list[str] = []

    # --------------------------------------------------------------------------
    def get_num_metrics(self) -> int:
        """
        Returns the number of metrics being recorded
        """
        return self.__num_metrics

    # --------------------------------------------------------------------------
    def get_metric_names(self) -> list[str]:
        """
        Returns the metric ames th[Storing just attributes]at is being recorded
        """
        return self.__metric_names

    # --------------------------------------------------------------------------
    def get_id(self) -> int|str:
        """
        Returns the id of this recorder
        """
        return self.__id
    
    # --------------------------------------------------------------------------
    def set_id(self, id:int|str) -> None:
        """
        The method sets the id of this recorder

        ------------------------------------------------------------------------
        Inputs:
            - id: to possibly identify this recorder like 0 or "training"
        """
        self.__id = id
    
    # --------------------------------------------------------------------------
    def __process_unsqueeze(self, tensor_value: torch.Tensor, metric_index:int)\
            -> torch.Tensor:
        """
        The method unsqueezes a record before storing it if needed

        ------------------------------------------------------------------------
        Inputs:
            - tensor_value: the record to be updated
            - metric_index: the index that identifies the metric we  will be up\
                dating

        ------------------------------------------------------------------------
        Output:
            the updated record, if needed
        """
        if metric_index in self.__unsqueeze_dims:
            tensor_value = torch.unsqueeze(tensor_value,
                                        dim=self.__unsqueeze_dims[metric_index])
        return tensor_value

    # --------------------------------------------------------------------------
    def reset(self, metric_index: int|None = None) -> None:
        """
        The method resets all the history,  or optionally of a particular metric

        ------------------------------------------------------------------------
        Inputs:
            - metric_index: the index that identifies the metric we  will be de\
                leting [Optional]
        """
        if metric_index is not None:
            self.__value_history[metric_index].clear()
            self.__value_history[metric_index] = []
        else:
            self.__value_history.clear()
            self.__value_history = [[] for i in range(self.__num_metrics)]

    # --------------------------------------------------------------------------
    def update(self, tensor_value: torch.Tensor, metric_index:int) -> None:
        """
        The method updates the history of a particular metric by adding a record
        <tensor_value> into the metric values whose index is <metric_index>

        ------------------------------------------------------------------------
        Inputs:
            - tensor_value: the value to be recorded
            - metric_index: the index that identifies the metric we  will be up\
                dating
        """
        # ----------------------------------------------------------------------
        # First, process the unsqueeze if needed
        # ----------------------------------------------------------------------
        tensor_value = self.__process_unsqueeze(tensor_value, metric_index)

        # ----------------------------------------------------------------------
        # Add the value to the list of values after moving it to the cpu
        # ----------------------------------------------------------------------
        self.__value_history[metric_index].append(tensor_value.detach().clone()\
            .to(torch.device("cpu")))

    # --------------------------------------------------------------------------
    def concatenate(self, metric_index:int) -> torch.Tensor:
        """
        The method generates a concatenated tensor holding all the history

        ------------------------------------------------------------------------
        Inputs:
            - metric_index: the index that identifies the metric we  will be up\
                dating

        Outputs:
            - A concatenated tensor of all the metric history
        """
        if metric_index not in self.__concat_dims:
            raise KeyError("this dimension is not defined in the concat_dims")
        elif metric_index not in range(len(self.__value_history)):
            raise IndexError("wrong metric index")
        elif len(self.__value_history[metric_index]) == 0:
            print(f"No values has been recorded in \""+
                    f"{self.__metric_names[metric_index]}\""+
                    f"of {self.__id} history")
            return torch.Tensor([]) 
        else:
            return torch.cat(self.__value_history[metric_index],
                            dim= self.__concat_dims[metric_index])
    
    # --------------------------------------------------------------------------
    def __check_empty(self, metric_index:int) -> bool:
        """
        The returns true if the history of metric_index is empty

        ------------------------------------------------------------------------
        Inputs:
            - metric_index: the index that identifies the metric we  will be up\
                dating

        Outputs:
            - True if it is empty; otherwise, return false
        """
        if metric_index not in range(len(self.__value_history)):
            raise IndexError("Wrong metric index")
        elif len(self.__value_history[metric_index]) == 0:
            return True
        else:
            return False

    # --------------------------------------------------------------------------
    def get_last_value(self, metric_index:int) -> object|None:
        """
        The method returns the last added value in a particular metric

        ------------------------------------------------------------------------
        Inputs:
            - metric_index: the index that identifies the metric we  will be re\
                ading
        """
        if len(self.__value_history[metric_index]) > 0:
            return self.__value_history[metric_index][-1].detach().clone()
        else:
            return None
    # --------------------------------------------------------------------------
    def get_state(self) -> dict:
        """
        The method saves the recorder state for future loading, without the data

        ------------------------------------------------------------------------
        Outputs:
            - the recorder state
        """
        rec_state = {
            "num_metrics" : self.__num_metrics,
            "metric_names": self.__metric_names,
            "unsqueeze_dims" : self.__unsqueeze_dims,
            "concat_dims" : self.__concat_dims,
            "id" : self.__id,
            'data_paths': self.__data_paths
        }

        return rec_state
    # --------------------------------------------------------------------------
    def load_state(self, state_dict:dict):
        """
        The method loads the Recorder state from previously stored state via \
        <get_state>

        ------------------------------------------------------------------------
        Inputs:
            - state_dict: the recorder state dictionary
        """
        self.__num_metrics = state_dict["num_metrics"]
        self.__metric_names= state_dict["metric_names"]
        self.__unsqueeze_dims = state_dict["unsqueeze_dims"]
        self.__concat_dims = state_dict["concat_dims"]
        self.__id = state_dict["id"]
        self.__data_paths = state_dict['data_paths']
        self.reset()
    # --------------------------------------------------------------------------
    def save_state(self, base_path:str, fname:str, current_epoch:int = 0,
                   prev_epoch:int = -1, desc_dict:dict={}, 
                   mode:ValueStoringMode = ValueStoringMode.OVERWRITE) -> str:
        """
        The method saves the internal state of the ValueRecorder into a file

        ------------------------------------------------------------------------
        Inputs:
            - base_path: the base directory of the states
            - fname: the file name of the recorded state
            - current_epoch: the number of the current epoch
                - This is needed if working in the DIFFERENCE mode
            - prev_epoch: the epoch number we recorded our data last time
                - default: -1 for the first time of recording 
            - desc_dict: dictionary of external information to be added to the \
                state; this is useful for independent reading regardless of the\
                trainer itself
            - mode: the storing mode; checkout ValuesStoringMode for details
                - default: `ValuesStoringMode.OVERWRITE`
        
        ------------------------------------------------------------------------
        Outputs:
            - state dictionary
        """
        # ----------------------------------------------------------------------
        # Check the epochs
        # ----------------------------------------------------------------------
        if type(prev_epoch) != int or type(current_epoch) != int:
            raise TypeError("The value of the epochs should be integers")
        elif prev_epoch>current_epoch:
            raise ValueError("The value of the current epoch should be greater"\
                + " than the previous epoch")
        # ----------------------------------------------------------------------
        # Create a folder with the id of this recorder
        # ----------------------------------------------------------------------
        data_path = os.path.join(base_path, str(self.__id))
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        
        # ----------------------------------------------------------------------
        # Add the desc_dict to the state and insert the prev_epoch&current_epoch
        # ----------------------------------------------------------------------
        state = {}
        state.update(desc_dict)
        state.update(self.get_state())
        state["prev_epoch"] = prev_epoch
        state["current_epoch"] = prev_epoch
        # ======================================================================
        # Overwrite mode, we store in one single file
        # ======================================================================
        if mode == ValueStoringMode.OVERWRITE:
            fname = f"data.pt"
            new_hist = [[] for i in range(self.__num_metrics)]
            for metric_index in range(len(self.__value_history)):
                data = self.concatenate(metric_index).detach().clone()
                new_hist[metric_index].append(data)
            # ------------------------------------------------------------------
            # Now, save the data
            # ------------------------------------------------------------------
            data_path = os.path.join(data_path, fname)
            torch.save(new_hist, data_path)
        # ======================================================================
        # Normal mode, we store all the values into the given file
        # or DIFFERENCE with no previously recorded files
        # ======================================================================
        elif mode == ValueStoringMode.NORMAL or \
            (mode == ValueStoringMode.DIFFERENCE and prev_epoch < 0):
            # ------------------------------------------------------------------
            # do nothing, just save the original __value_history
            # ------------------------------------------------------------------
            data_path = os.path.join(data_path, fname)
            torch.save(self.__value_history, data_path)
        # ======================================================================
        # DIFFERENCE mode, we store the difference only
        # ======================================================================
        elif mode == ValueStoringMode.DIFFERENCE:
            diff_hist = [[] for i in range(self.__num_metrics)]
            for metric_index in range(len(self.__value_history)):
                # --------------------------------------------------------------
                # if the metric is empty, continue, as there is no need for pro-
                # cessing
                # --------------------------------------------------------------
                if self.__check_empty(metric_index):
                    continue
                # --------------------------------------------------------------
                # Otherwise, process the diff_hist
                # --------------------------------------------------------------
                indices = torch.arange(prev_epoch+1, current_epoch+1,
                    dtype=torch.int)
                metric_data = self.concatenate(metric_index).detach().clone()
                res = torch.index_select(metric_data,
                         dim= self.__concat_dims[metric_index],  index= indices)
                diff_hist[metric_index].append(res.detach().clone())
            # ------------------------------------------------------------------
            # Now save the diff_hist
            # ------------------------------------------------------------------
            data_path = os.path.join(data_path, fname)
            if data_path in self.__data_paths:
                raise ValueError("The file already exists in difference mode"+\
                    ". We cannot overwrite it.\n"+f"data path: {data_path}")
            torch.save(diff_hist, data_path)
        # ======================================================================
        # Finally, save the data path
        # ======================================================================
        if data_path not in self.__data_paths:
            self.__data_paths.append(data_path)
        state['data_paths'] = self.__data_paths
        state['last_data_path'] = data_path
        
        state['mode'] = mode

        print(f"Recorder {self.__id} has been saved successfully in {data_path}")
        return state
    
    # --------------------------------------------------------------------------
    def load_state_file(self, state:dict) -> None:
        """
        The method loads the  internal state of the  ValueRecorder from a state\
        dictionary 

        ------------------------------------------------------------------------
        Inputs:
            - state: the state
        """

        internal_keys = ["num_metrics", "unsqueeze_dims", "concat_dims", "id",
                         "current_epoch", "metric_names", "prev_epoch", "mode",
                         "data_paths", "value_history", "last_data_path"]
        # ----------------------------------------------------------------------
        # Print the external information if any
        # ----------------------------------------------------------------------
        print(f"External Information of {self.__id}")
        for k in state:
            if k not in internal_keys:
                print(f"{k}: {state[k]}")
        # ----------------------------------------------------------------------
        # Load the state
        # ----------------------------------------------------------------------
        self.load_state(state)

        # ----------------------------------------------------------------------
        # Load the data if there is mode
        # ----------------------------------------------------------------------
        mode = state['mode'] if "mode" in state else ValueStoringMode.NORMAL
        # ----------------------------------------------------------------------
        # We read one single file if it is the overwrite mode,  the normal mode,
        # or if it is even the first epoch of a DIFFERENCE mode.
        # ----------------------------------------------------------------------
        if mode == ValueStoringMode.OVERWRITE or \
           mode == ValueStoringMode.NORMAL or \
            (mode == ValueStoringMode.DIFFERENCE and state['prev_epoch'] < 0):
            data_path = state['last_data_path']
            self.__value_history = torch.load(data_path)
        # ----------------------------------------------------------------------
        # Alternatively if it were the difference mode, and  not the first  one,
        # we iterate building them again
        # ----------------------------------------------------------------------
        elif mode == ValueStoringMode.DIFFERENCE:
            for data_path in self.__data_paths:
                batch_values:list[torch.Tensor] = torch.load(data_path)
                for indx in range(len(self.__value_history)):
                    self.__value_history[indx] += batch_values[indx]