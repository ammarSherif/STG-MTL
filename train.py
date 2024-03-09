# ==============================================================================
# The file  includes the code used to train  a model on  a dataset and possibily
# generating a datamap.
# ==============================================================================

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import enum
import copy
import numpy as np
import random
from .models.base import BaseModel, BaseMetrics
from .data_modules.base import BaseDataModule
from .util.datamap import DataMapRecorder
from .util.values_recorder import ValuesRecorder
from .util.util_enum import ValueStoringMode, DatasetType

import math
import os

# ==============================================================================
# Define an enum to help clarifying the optimizer type
# ==============================================================================
class OptimizerType(enum.Enum):
    ADAM = 0
    NADAM = 1
    # SGD_Momentum = 2
# ==============================================================================
# Define an enum to help clarifying the optimizer type
# ==============================================================================
class LRDecayType(enum.Enum):
    NONE = -1
    COS = 0


# ==============================================================================
# Create a trainer class
# ==============================================================================
class Trainer():
    """
    The class is used to train a model on a dataset module and to generate data-
    maps if needed along with the training results
    ----------------------------------------------------------------------------
    Inputs:
        - model: the model that will be trained, and or evaluated.
        - data_module: a DatasetModule class that contains the dataset we will \
                be using.
        - args: a dictionary holding the necessary arguments to run; it includes
            - Learning rate-related arguments:
                - lr: the initial learning rate
                - lr_sched: scheduling type; one of the options in LRDecayType
                - warmup_epochs: the warmup epochs before changing the lr
                - min_lr: the minimum learning rate
            - num_of_epochs: the number of epochs
            - batch_size: the batch size used while training
            - DataLoader related args:
                - num_workers: the number of workers in the dataloaders
                - pin_memory: the pin_memory option for dataloaders
            - device: the device to train upon
            - print_per_epoch: how many print statements per one epoch 
            - base_dir: the experiment name, dir, where the logs and chekpoints\
                will reside
            - checkpoint_enable: True to enable checkpoint recording
            - overwrite_datamaps: True to enable overwrite datamaps after each\
                epoch when checkpoint_enable and datamap_concat_save are True
            - datamap_concat_save: True to save the history of the datamaps so\
                far. Otherwise, it saves the datamap of the current epoch only
        - debug_mode: True to print helpful messages for debugging
        - epochs_data_map: a list of epochs in which we record the datamaps
        - optimizer: the optimizer to be used. None to generate a one
        - opt_type: the type of optimizer to be created in case it was not pass\
                ed
        - training_set: indicates the which dataset we will train on. It is the\
                training dataset by default, by you can indicate None
        - eval_set: indicates the datasets to be evaluated upon. It is the vali-
                dation set by default
        - task_weights: the weights applied to the loss of each task. None to 1.
            - shape: #tasks
    ----------------------------------------------------------------------------
    """

    def __init__(self, model: BaseModel, data_module: BaseDataModule,
                 args: dict, debug_mode = False,
                 save_mode:ValueStoringMode = ValueStoringMode.OVERWRITE,
                 epochs_data_map:list[int]|None = None,
                 optimizer:torch.optim.Optimizer|None = None,
                 opt_type:OptimizerType = OptimizerType.ADAM,
                 training_set:DatasetType =DatasetType.TRAIN,
                 eval_set:DatasetType = DatasetType.VALIDATION,
                 task_weights:torch.Tensor|None = None):
        # ----------------------------------------------------------------------
        # Initialize the learning hyperparameters.
        # ----------------------------------------------------------------------
        self.__epoch:int = -1
        self.__args:dict = args
        self.__save_mode:ValueStoringMode = save_mode
        self.__model:BaseModel = model
        self.__metrics_gen:list[BaseMetrics] = self.__model.get_metrics_generator()
        self.__data_module:BaseDataModule = data_module
        self.__debug:bool = debug_mode
        self.__epochs_data_map:list[int] = epochs_data_map
        if self.__epochs_data_map is None:
            self.__epochs_data_map = []
        self.__optimizer:torch.optim.Optimizer|None = optimizer
        self.__opt_type:OptimizerType = opt_type
        # ----------------------------------------------------------------------
        # Initialize the number of tasks
        # ----------------------------------------------------------------------
        self.__num_of_tasks:int = len(self.__data_module.get_task_ids())

        self.__train_loader:DataLoader|None = None
        self.__eval_loader:DataLoader|None = None

        self.__train_set:DatasetType = training_set
        if self.__train_set is None:
            self.__train_set = DatasetType.NONE

        self.__eval_set:DatasetType = eval_set
        if self.__eval_set is None:
            self.__eval_set = DatasetType.NONE

        if self.__args['device'] is None:
            self.__args['device'] = self.__get_device()

        self.__task_weights:torch.Tensor|None = task_weights

        self.__datamap_recorder:DataMapRecorder|None = None
        self.__history_recorders = []
        # ----------------------------------------------------------------------
        # Initialize any missing args
        # ----------------------------------------------------------------------
        self.__init_args()

        self.initialize()

    # --------------------------------------------------------------------------
    def __init_args(self) -> None:
        """
        The method initializes the missing args if there were any missing ones.
        """
        critical_keys = ['lr', 'num_of_epochs', 'batch_size', 'base_dir',
                         'checkpoint_enable']
        for k in critical_keys:
            if k not in self.__args:
                raise KeyError(f"Key [{k}] is missing in your args")
        default_args = {
            'lr_sched': LRDecayType.NONE,
            'warmup_epochs': self.__args['num_of_epochs']+1,
            'min_lr': 0,
            'num_workers': 0,
            'pin_memory': True,
            'device': self.__get_device(),
            'print_per_epoch': 0
        }

        for def_k in default_args:
            if def_k not in self.__args:
                self.__args[def_k] = default_args[def_k]

    # --------------------------------------------------------------------------
    def __get_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    def update_args(self,args):
        """
        The method updates the args
        """
        self.__args.update(args)

    # --------------------------------------------------------------------------
    def update_datamap_epochs(self, epochs:list[int]) -> None:
        """
        The method updates the datamap epochs adding new number of epohcs

        ------------------------------------------------------------------------
        Inputs:
            - epochs: a list of integers representing epoch numbers ranging from
                1 and including  any positive integer where 1 represents the 2nd
                epoch. During training, if the epoch matches any  integer in the
                list, it records the datamap in that epoch.
        """
        new_val = []
        for v in epochs:
            if type(v) != int or v<1:
                raise TypeError("All the values must be at least 1")
            else:
                new_val.append(v)
        self.__epochs_data_map += new_val

    # --------------------------------------------------------------------------
    def save_checkpoint(self, init_epoch:int=0) -> str|None:
        """
        The function  stores  a checkpoint to  enable resume training from this\
        point later on

        ------------------------------------------------------------------------
        Inputs:
            - init_epoch: the initial epoch of the training
        """
        # ----------------------------------------------------------------------
        # If there is no base directory for this experiment throw an error
        # ----------------------------------------------------------------------
        if 'base_dir' not in self.__args.keys():
            raise KeyError("\'base_dir\' is not defined in the args")
        elif self.__args['base_dir'] is None:
            raise ValueError("\'base_dir\' has wrong dir")
        # ----------------------------------------------------------------------
        # By default, the checkpoint_dir name is "checkpoints"
        # ----------------------------------------------------------------------
        checkpoint_dir = "checkpoints"
        if 'checkpoint_dir' in self.__args.keys() and \
            type(self.__args['checkpoint_dir']) == str and \
            self.__args['checkpoint_dir'] is not None:
            checkpoint_dir = self.__args['checkpoint_dir']

        # ----------------------------------------------------------------------
        # Create the path if it is not there
        # ----------------------------------------------------------------------
        checkpoint_path = os.path.join(self.__args['base_dir'], checkpoint_dir)
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        # ----------------------------------------------------------------------
        # Define the checkpoint dictionary
        # ----------------------------------------------------------------------
        checkpoint = {
            'epoch': self.__epoch,
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'args': self.__args,
            'save_mode': self.__save_mode
        }
        # ----------------------------------------------------------------------
        # Store the metric generator states [Storing just attributes]
        # ----------------------------------------------------------------------
        gen_states = []
        for gen in self.__metrics_gen:
            gen_states.append(gen.get_state())
        checkpoint['gen_states'] = gen_states
        # ----------------------------------------------------------------------
        # Store the history recorder states [Heavy, so we store it separately]
        # ----------------------------------------------------------------------
        init_state = {
            "init_epoch": init_epoch,
            "last_epoch": self.__epoch
        }
        rec_states = []
        for rec in self.__history_recorders:
            rec_states.append(rec.save_state(base_path=checkpoint_path,
                        fname=f"{self.__epoch}.pt", current_epoch= self.__epoch,
                        prev_epoch= self.__epoch-1, desc_dict=init_state, 
                        mode= self.__save_mode))
        checkpoint['recorder_states'] = rec_states
        # ----------------------------------------------------------------------
        # Store the datamap state
        # ----------------------------------------------------------------------
        checkpoint['datamap_state'] = self.__datamap_recorder.get_state() \
                                if self.__datamap_recorder is not None else None

        # ----------------------------------------------------------------------
        # Store the datamap 
        # ----------------------------------------------------------------------
        datamap_path = self.save_datamap(base_dir= checkpoint_path, 
                                        fname=f"{self.__epoch}.pt",
                                        epochs= self.__epochs_data_map)
        checkpoint['datamap_path'] = datamap_path
        # checkpoint['datamap'] = self.__data_map
        
        # ----------------------------------------------------------------------
        # Store the random states
        # ----------------------------------------------------------------------
        rnd_state_path = self.__save_random_state(path= checkpoint_path)
        checkpoint['rnd_state_path'] = rnd_state_path
        # ----------------------------------------------------------------------
        # By default, the checkpoint_dir name is "checkpoints"
        # ----------------------------------------------------------------------
        checkpoint_dir = "checkpoints"
        if 'checkpoint_dir' in self.__args.keys() and \
            self.__args['checkpoint_dir'] is not None:
            checkpoint_dir = self.__args['checkpoint_dir']

        # ----------------------------------------------------------------------
        # Store the checkpoint
        # ----------------------------------------------------------------------
        checkpoint_name = f'{self.__epoch}.pt'
        checkpoint_path = os.path.join(checkpoint_path, checkpoint_name)
        torch.save(checkpoint,checkpoint_path)

        print(f"checkpoint saved successfully in {checkpoint_path}")
        print("__"*40)
        return checkpoint_path

    # --------------------------------------------------------------------------
    def __save_random_state(self, path:str) -> str:
        """
        The method saves the random state of all  and every  possible evironment
        for reproducibility. The states are saved in `random_states` directory. 

        ------------------------------------------------------------------------
        Inputs:
            - path: the checkpoint path; 
        """
        # ======================================================================
        # NOTE We store the random states in a separate file so as not to be ma-
        #      pped along with the model state, which produces an error if wrong
        #      ly mapped.
        # ======================================================================
        # By default, the checkpoint_dir name is "random_states"
        # ----------------------------------------------------------------------
        random_states_dir = "random_states"
        # ----------------------------------------------------------------------
        # Create the path if it is not there
        # ----------------------------------------------------------------------
        random_states_path = os.path.join(path, random_states_dir)
        if not os.path.isdir(random_states_path):
            os.makedirs(random_states_path)
        
        # ----------------------------------------------------------------------
        # Save the random states
        # ----------------------------------------------------------------------
        rnd_states = {}
        rnd_states['torch_rng_state'] = torch.get_rng_state()
        rnd_states['cuda_rng_state'] = None
        if torch.cuda.is_available():
            rnd_states['cuda_rng_state'] = torch.cuda.get_rng_state()
        rnd_states['cuda_all_rng_states'] = None
        if torch.cuda.is_available():
            rnd_states['cuda_all_rng_states'] = torch.cuda.get_rng_state_all()
        rnd_states['np_rng_state'] = np.random.get_state()
        rnd_states['random_state'] = random.getstate()
        rnd_states['cuda_benchmark'] = torch.backends.cudnn.benchmark
        rnd_states['cuda_deterministic']=torch.backends.cudnn.deterministic
        # ----------------------------------------------------------------------
        # save the file and return the path
        # ----------------------------------------------------------------------
        file_name = f'{self.__epoch}.pt'
        random_states_path = os.path.join(random_states_path, file_name)
        torch.save(rnd_states, random_states_path)

        return random_states_path

    # --------------------------------------------------------------------------
    def __write_summary(self) -> None:
        """
        This method is used to write down the summary of the results after the \
        training is done
        """
        if 'base_dir' not in self.__args.keys():
            raise KeyError("\'base_dir\' is not defined in the args")
        elif self.__args['base_dir'] is None:
            raise ValueError("\'base_dir\' has wrong dir")
        
        summary_s = ""
        summary = {}
        for recorder in self.__history_recorders:
            summary_s += str(recorder.get_id()) + " Results\n" + "=="*40 + "\n"
            rec_values = {}
            for m_id, m_name in enumerate(recorder.get_metric_names()):
                summary_s += f"\tMetric \"{m_name}\":\n"
                summary_s += "\t" + "--"*38 + "\n"
                rec_values[m_name] = recorder.concatenate(m_id)
                summary_s += "\t\t" + str(rec_values[m_name]) + "\n"
                summary_s += "\t" + "__"*38 + "\n"
            summary[recorder.get_id()] = rec_values
        # ----------------------------------------------------------------------
        # By default, the checkpoint_dir name is "log"
        # ----------------------------------------------------------------------
        log_dir = "log"
        log_path = os.path.join(self.__args['base_dir'], log_dir)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        # ----------------------------------------------------------------------
        # Store the log
        # ----------------------------------------------------------------------
        log_name = f'{self.__epoch}.pt'
        dict_log_path = os.path.join(log_path, log_name)
        torch.save(summary,dict_log_path)
        summary_s += "=="*40 + "\n"+ f"Summary dictionary path: {dict_log_path}"
        summary_s += "\n"+"=="*40 + "\n"
        # ----------------------------------------------------------------------
        # store the summary text into a text file
        # ----------------------------------------------------------------------
        log_name = f'{self.__epoch}.txt'
        txt_log_path = os.path.join(log_path, log_name)
        if os.path.isdir(txt_log_path):
            print("The file is already there")
            print(f"File path: {txt_log_path}")
            print("Suggest a new name: ",end="")
            log_name = str(input())
            txt_log_path = os.path.join(log_path, log_name)
        with open(txt_log_path,'w') as f:
            f.write(summary_s)
            f.close()
        print(f"log saved successfully in {txt_log_path}")

    # --------------------------------------------------------------------------
    def load_checkpoint(self, path: str) -> None:
        """
        The function loads a checkpoint.

        ------------------------------------------------------------------------
        Inputs:
            - path: checkpoint directory path
        """
        device = self.__args['device']
        checkpoint = torch.load(path, map_location=device)

        self.__epoch = checkpoint['epoch']
        self.__model.load_state_dict(checkpoint['model_state_dict'])
        self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.__save_mode:ValueStoringMode = checkpoint['save_mode']
        if 'args' in checkpoint:
            # ------------------------------------------------------------------
            # Printing ARGS for reference
            # ------------------------------------------------------------------
            print("\n","=="*40)
            print("Printing the old args [only for reference]")
            for k in checkpoint['args']:
                print(f"\t{k}: {checkpoint['args'][k]}")
            print("--"*40)
            print("To update any of the trainer configurations, please, "+\
                "use `obj.update_args()`")
            print("along with `obj.update_datamap_epochs`")
            print("--"*40)
            print(f"Current save mode is [{self.__save_mode.name}]")
            print("=="*40)
            # ------------------------------------------------------------------
        #     self.__args = checkpoint['args']
        # self.__args['device'] = device
        for i, gen_state in enumerate(checkpoint['gen_states']):
            self.__metrics_gen[i].load_state(gen_state)

        for i, rec_state in enumerate(checkpoint['recorder_states']):
            self.__history_recorders[i].load_state_file(rec_state)
        # ----------------------------------------------------------------------
        # Store the datamap state
        # ----------------------------------------------------------------------
        if checkpoint['datamap_state'] is not None \
            and self.__datamap_recorder is not None:
            self.__datamap_recorder.load_state(checkpoint['datamap_state'])
        res = Trainer.load_datamap_from_checkpoint(path,
                                    self.__epochs_data_map,
                                    mode= self.__save_mode)
        if res is not None:
            self.__data_map = res
        
        # ----------------------------------------------------------------------
        # Finally, load the random states
        # ----------------------------------------------------------------------
        self.__load_random_state(checkpoint['rnd_state_path'])
   
    # --------------------------------------------------------------------------
    def __load_random_state(self, path:str) -> None:
        """
        The method loads the random state of all  and every  possible evironment
        for reproducibility. 

        ------------------------------------------------------------------------
        Inputs:
            - path: the random states file path; 
        """
        # ----------------------------------------------------------------------
        # Load the random states
        # ----------------------------------------------------------------------
        rnd_states = torch.load(path)
        
        # ----------------------------------------------------------------------
        # Load the random states
        # ----------------------------------------------------------------------
        torch.set_rng_state(rnd_states['torch_rng_state'])
        if rnd_states['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(rnd_states['cuda_rng_state']) 
        if rnd_states['cuda_all_rng_states'] is not None:
            torch.cuda.set_rng_state_all(rnd_states['cuda_all_rng_states']) 
        np.random.set_state(rnd_states['np_rng_state']) 
        random.setstate(rnd_states['random_state']) 
        torch.backends.cudnn.benchmark = rnd_states['cuda_benchmark']
        torch.backends.cudnn.deterministic = rnd_states['cuda_deterministic']
 
    # --------------------------------------------------------------------------
    def save_datamap(self,base_dir:str, fname:str, epochs:list[int]|str) -> str:
        """
        The method stores the datamap into a file in  under base directory; we \
        create a new directory "datamaps" under which we store the datamap after
        moving it to cpu

        ------------------------------------------------------------------------
        Inputs:
            - base_dir: the base directory under which we are creating our data\
                map directory
            - fname: the file name
            - epochs: the list that includes the epoch numbers when we recorded\
                these datamaps
        
        ------------------------------------------------------------------------
        Outputs:
            - the file path or 'NONE' if it did not store
        """
        # ----------------------------------------------------------------------
        # if we did not have this epoch, do not store anything
        # ----------------------------------------------------------------------
        if self.__epoch not in epochs:
            return 'NONE'
        # ----------------------------------------------------------------------
        # Create a folder with the name of "datamaps"
        # ----------------------------------------------------------------------
        datamap_path = os.path.join(base_dir, "Datamaps")
        if not os.path.isdir(datamap_path):
            os.makedirs(datamap_path)
        # ----------------------------------------------------------------------
        # Add the file name and store it
        # ----------------------------------------------------------------------
        # Save the datamap as it is in case we are using the OVERWRITE
        # ----------------------------------------------------------------------
        if self.__save_mode == ValueStoringMode.OVERWRITE:
            final_path = os.path.join(datamap_path, "datamap.pt")
            dm = self.get_data_map().detach().clone().cpu()
        # ----------------------------------------------------------------------
        # Save the datamap as it is in case we are using the NORMAL
        # ----------------------------------------------------------------------
        elif self.__save_mode == ValueStoringMode.NORMAL:
            final_path = os.path.join(datamap_path, fname)
            dm = self.get_data_map().detach().clone().cpu()
        # ----------------------------------------------------------------------
        # Alternatively, we are in the  <DIFFERENCE>  mode, so we store the last
        # epoch only
        # ----------------------------------------------------------------------
        elif self.__save_mode == ValueStoringMode.DIFFERENCE:
            final_path = os.path.join(datamap_path, fname)
            dm = self.get_data_map()[:,-1:,:,:].detach().clone().cpu()
        torch.save(dm, final_path)

        # ----------------------------------------------------------------------
        # Create a descriptor file if it is not already there
        # ----------------------------------------------------------------------
        desc_fname = "descriptor.txt"
        desc_path = os.path.join(datamap_path, desc_fname)
        content = f"Epochs: {epochs}\nPath: {final_path}\n" + "=="*40 + "\n"
        if os.path.exists(desc_path):
            with open(desc_path, "r+") as f:
                old = f.read()
                f.seek(0) 
                f.write(content + old)
        else:
            with open(desc_path, "w") as f: 
                f.write(content)
        if self.__debug:
            print(f"Datamap successfully stored in \"{final_path}\""+ 
                f" & updated details in \"{desc_fname}\"")
        
        return final_path
 
    # --------------------------------------------------------------------------
    def __update_lr(self, optimizer:torch.optim.Optimizer, epoch: float) ->None:
        """
        The function is used to decay the learning rate with half-cycle cosine
        after warmup
        """
        if self.__args['lr_sched'] == LRDecayType.NONE:
            pass
        elif self.__args['lr_sched'] == LRDecayType.COS:
            if epoch < self.__args['warmup_epochs']:
                lr = self.__args['lr'] * epoch / self.__args['warmup_epochs']
            else:
                lr = self.__args['min_lr'] + (self.__args['lr'] - \
                    self.__args['min_lr']) * 0.5 * (1. + math.cos(math.pi * \
                        (epoch - self.__args['warmup_epochs']) / \
                        (self.__args['epochs'] - self.__args['warmup_epochs'])))
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
 
    # --------------------------------------------------------------------------
    def __initialize_optimizer(self) -> None:
        """
        The function initializes  the optimizer according to  the specified type
        to be used while training.
        """
        if self.__optimizer is not None:
            return
        elif self.__opt_type == OptimizerType.ADAM:
            self.__optimizer = optim.Adam(self.__model.parameters(),
                                          lr = self.__args['lr'])
        elif self.__opt_type == OptimizerType.NADAM:
            self.__optimizer = optim.NAdam(self.__model.parameters(),
                                          lr = self.__args['lr'])
        else:
            # ERROR
            self.__optimizer = optim.Adam(self.__model.parameters(),
                                          lr = self.__args['lr'])

    # --------------------------------------------------------------------------
    def __get_loader(self, loader_type: DatasetType) -> DataLoader:
        """
        The function returns the dataloader according to the passed type
        """
        loader_args= {
            'batch_size': 1,
            'num_workers': 0,
            'pin_memory': True
        }

        for k in list(loader_args.keys()):
            if k in self.__args:
                loader_args[k] = self.__args[k]
            else:
                del loader_args[k]
        if loader_type == DatasetType.TRAIN:
            return self.__data_module.get_train_dataloader(**loader_args)
        elif loader_type == DatasetType.VALIDATION:
            return self.__data_module.get_val_dataloader(**loader_args)
        elif loader_type == DatasetType.TEST:
            return self.__data_module.get_test_dataloader(**loader_args)
        elif loader_type == DatasetType.TRAIN_VAL:
            return self.__get_loader(DatasetType.TRAIN), \
                self.__get_loader(DatasetType.VALIDATION)
        elif loader_type == DatasetType.ALL:
            return self.__get_loader(DatasetType.TRAIN), \
                self.__get_loader(DatasetType.VALIDATION), \
                self.__get_loader(DatasetType.TEST)
        else:
            return None
 
    # --------------------------------------------------------------------------
    def initialize(self):
        """
        The function enables initializing the losses, the optimizer, train load-
        er, evaluation data loader, task weights, and some  other attributes be-
        fore training the model.
        """
        # ----------------------------------------------------------------------
        # Do some initializations for the current object
        # ----------------------------------------------------------------------
        self.__data_map = None
        if self.__epochs_data_map is not None\
            and type(self.__epochs_data_map) == list \
            and len(self.__epochs_data_map) != 0:
            self.__datamap_recorder = DataMapRecorder(self.__num_of_tasks,
                                            self.__data_module.get_train_size(),
                                            self.__args['device'])
        metric_names = self.__metrics_gen[0].get_metric_names()
        metric_names.append("Loss")
        recorder_ids = ["Training", "Validation", "Testing"]      
        # ----------------------------------------------------------------------
        # We concate the losses on the same dimension, and therefore,  we do not
        # unsqueeze
        # ----------------------------------------------------------------------
        dims = self.__metrics_gen[0].get_unsqueeze_concatenation_dims()  
        unsqueeze_dims, concat_dims = dims
        concat_dims[len(metric_names)-1] = 0
        
        self.__history_recorders =[ValuesRecorder(metric_names=metric_names,
                id=recorder_ids[i], unsqueeze_dims=unsqueeze_dims,
                concat_dims=concat_dims) for i in range(len(recorder_ids))]
        self.__list_of_losses = []

        # ----------------------------------------------------------------------
        # Initialize the loss functions and the optimizer
        # ----------------------------------------------------------------------
        self.__list_of_losses = self.__model.get_losses()
        self.__initialize_optimizer()
        if self.__task_weights is None:
            self.__task_weights = torch.ones(len(self.__list_of_losses))
        else:
            self.__task_weights = self.__task_weights

        self.__train_loader = self.__get_loader(self.__train_set)
        self.__eval_loader = self.__get_loader(self.__eval_set)
 
    # --------------------------------------------------------------------------
    def __train_epoch(self, epoch:int):
        """
        The function is primarily used to train a single epoch
        """
        itr = 0
        max_itr = len(self.__train_loader) 
        max_itr_len = len(str(max_itr))
        train_loss = 0
        self.__model.train()
        for indxs, inputs, targets in self.__train_loader:
            # ------------------------------------------------------------------
            # Move to the Device
            # ------------------------------------------------------------------
            inputs = inputs.to(self.__args['device'])
            targets = targets.to(self.__args['device'])

            # ------------------------------------------------------------------
            # clear the parameter gradients
            # ------------------------------------------------------------------
            self.__optimizer.zero_grad()
            self.__update_lr(self.__optimizer, epoch+(itr/max_itr))
            # ------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------
            outputs = self.__model(inputs)
            # ------------------------------------------------------------------
            # Calculate the batch loss
            # ------------------------------------------------------------------
            loss = 0
            for task_id, t_weight, loss_func in zip(range(
                        self.__num_of_tasks), self.__task_weights,
                        self.__list_of_losses):
                task_out = self.__model.get_task_output(outputs, task_id)
                task_target = self.__model.get_task_output(targets, task_id)
                loss += t_weight * loss_func(task_out, task_target)
            # ------------------------------------------------------------------
            # Backward pass and optimize the parameters
            # ------------------------------------------------------------------
            loss.backward()
            self.__optimizer.step()

            # ------------------------------------------------------------------
            # If we will generate a data map; store the results
            # ------------------------------------------------------------------
            if self.__datamap_recorder is not None:
                with torch.no_grad():
                    self.__fill_data_results(indxs, outputs, targets)

            # ------------------------------------------------------------------
            # Update the dynamics of the metrics  generator for the  current
            # batch.
            # ------------------------------------------------------------------
            with torch.no_grad():
                self.__metrics_gen[0].update_metrics_iteration(outputs, targets)
            train_loss += loss.item()
            itr += 1
            if self.__debug and 'print_per_epoch' in self.__args and \
                itr%(max_itr//self.__args['print_per_epoch']) == 0:
                data = {
                    'itr': itr, 'max_itr_len': max_itr_len, 'max_itr':max_itr,
                    'loss': loss.item(), 'train_loss': train_loss
                }
                self.__print_debug_messages("iteration_update", data)
        return train_loss

    # --------------------------------------------------------------------------
    def __print_debug_messages(self, id:str, data:dict[str, object]|None =None):
        """
        This method prints some useful messages based on the context of the <id>

        ------------------------------------------------------------------------
        Inputs:
            - id: a string indicating the context of the debug
                - "initial": for printing initial info
                - "epoch_update": to do an epoch update
                - "iteration_update": to do an iteration update
        """
        # ----------------------------------------------------------------------
        # Printing the initial values
        # ----------------------------------------------------------------------
        if id == "initial":
            print('=='*40)
            print("   Batch size =", self.__args['batch_size'])
            print("       Epochs =", self.__args['num_of_epochs'])
            print("Learning rate =", self.__args['lr'])
            print('--'*40)
        # ----------------------------------------------------------------------
        # Printing the epoch update values
        # ----------------------------------------------------------------------
        elif id == "epoch_update":
            metric_names = self.__metrics_gen[0].get_metric_names()
            max_len = max([len(n) for n in metric_names])
            num_of_metrics = self.__history_recorders[0].get_num_metrics()
            loss_indx = num_of_metrics-1
            print('--'*40)
            epoch_str = f"Epoch ({data['epoch']})"
            print(f"{epoch_str:>{max_len}}: [Train Loss = "+
                  f"{data['train_loss']:.2f}", end='')
            v_loss = self.__history_recorders[1].get_last_value(loss_indx)
            v_loss = 0 if v_loss is None else v_loss.item()
            print(f', Validation Loss = {v_loss:.2f}]')
            print("Training Metrics")
            for i, metric in enumerate(self.__metrics_gen[0].get_metrics()):
                print(f"{metric_names[i]:>{max_len}}:", metric)
            print("Validation Metrics")
            for i, metric in enumerate(self.__metrics_gen[1].get_metrics()):
                print(f"{metric_names[i]:>{max_len}}:", metric)
            print('--'*40)
        # ----------------------------------------------------------------------
        # Printing an iteration update
        # ----------------------------------------------------------------------
        elif id == "iteration_update":
            print(f"Iteration [{data['itr']:0{data['max_itr_len']}}/"+
                        f"{data['max_itr']}]: batch loss"+
                        f" = {data['loss']:5.2f}"+
                        f"\t total loss = {data['train_loss']:.2f}")

    # --------------------------------------------------------------------------
    def train(self):
        """
        The procedure to train our model
        """
        # ----------------------------------------------------------------------
        if self.__debug:
            self.__print_debug_messages("initial")
        # ----------------------------------------------------------------------
        # Move the model the device
        # ----------------------------------------------------------------------
        self.__model.to(self.__args['device'])
        self.__task_weights = self.__task_weights.to(self.__args['device'])
        # ----------------------------------------------------------------------
        # Get the number of metrics
        # ----------------------------------------------------------------------
        num_of_metrics = self.__history_recorders[0].get_num_metrics()
        loss_indx = num_of_metrics-1
        # ----------------------------------------------------------------------
        # Start training
        # ----------------------------------------------------------------------

        init_epoch = self.__epoch
        for epoch in range(init_epoch+1, init_epoch + 1 +\
                                            self.__args['num_of_epochs']):
            for metric_gen in self.__metrics_gen:
                metric_gen.initialize()

            train_loss = self.__train_epoch(epoch)
            # ==================================================================
            # HISTORY UPDATE
            # ==================================================================
            # loss_indx = -1, last one
            self.__history_recorders[0].update(torch.Tensor([train_loss]), 
                                                loss_indx)
            # ------------------------------------------------------------------
            # Compute the training metrics for this epoch
            # ------------------------------------------------------------------
            self.__metrics_gen[0].compute_metrics_epoch()
            # ------------------------------------------------------------------
            # Store the metrics of this epoch
            # ------------------------------------------------------------------
            for metric_id,metric in enumerate(self.__metrics_gen[0]\
                                                                .get_metrics()):
                # ==============================================================
                # HISTORY UPDATE
                # ==============================================================
                self.__history_recorders[0].update(metric, metric_id)

            # ==================================================================
            # Evaluate on the validation dataset
            # ==================================================================
            if self.__eval_loader is not None:
                self.evaluate()

            if self.__debug:
                data = {
                    "epoch": epoch+1,
                    "train_loss": train_loss,

                }
                self.__print_debug_messages("epoch_update", data)

            # ==================================================================
            # Generate a data map according to the
            # ==================================================================
            if epoch in self.__epochs_data_map:
                # --------------------------------------------------------------
                # We get the datamap and concatenate it
                # --------------------------------------------------------------
                dm= torch.unsqueeze(self.__datamap_recorder.get_datamap(),dim=1)
                if self.__data_map is None:
                    self.__data_map = dm
                elif isinstance(self.__data_map, torch.Tensor) and \
                    len(self.__data_map.shape) == 4:
                    self.__data_map = torch.cat((self.__data_map,dm), dim=1)
                else:
                    raise Exception("Wrong dimension in datamap")
            # ==================================================================
            # Update epoch
            # ==================================================================
            self.__epoch = epoch
            # ==================================================================
            # Save checkpoint
            # ==================================================================
            if self.__args['checkpoint_enable']:
                self.save_checkpoint(init_epoch= init_epoch)
        # ======================================================================
        # Finally, write the training summary
        # ======================================================================
        self.__write_summary()

    # --------------------------------------------------------------------------
    def evaluate(self, model:BaseModel = None):
        """
        The procedure to evaluate our model  in case an evaluation set has been\
        specified.

        ------------------------------------------------------------------------
        """

        if self.__eval_loader is None:
            raise ValueError("Undefined evaluation set")
        # ----------------------------------------------------------------------
        # In case we do not specify a model, then we are evaluating our own.
        # ----------------------------------------------------------------------
        if model is None:
            model = self.__model

        # ----------------------------------------------------------------------
        # Initialize the values before recording the results
        # ----------------------------------------------------------------------
        self.__metrics_gen[1].initialize()
        val_loss = 0
        num_of_metrics = self.__history_recorders[0].get_num_metrics()
        loss_indx = num_of_metrics-1
        model.eval()
        for indxs, inputs, targets in self.__eval_loader:
            # ------------------------------------------------------------------
            # Move to the Device
            # ------------------------------------------------------------------
            inputs = inputs.to(self.__args['device'])
            targets = targets.to(self.__args['device'])
            # ------------------------------------------------------------------
            # Forward pass
            # ------------------------------------------------------------------
            outputs = self.__model(inputs)
            # ------------------------------------------------------------------
            # Calculate the batch loss
            # ------------------------------------------------------------------
            loss = 0
            for task_id, t_weight, loss_func in zip(range(
                        self.__num_of_tasks), self.__task_weights,
                        self.__list_of_losses):
                task_out = self.__model.get_task_output(outputs, task_id)
                task_target = self.__model.get_task_output(targets, task_id)
                loss += t_weight * loss_func(task_out, task_target)

            # ------------------------------------------------------------------
            # Update the dynamics of the metrics generator for the current batch
            # ------------------------------------------------------------------
            self.__metrics_gen[1].update_metrics_iteration(outputs, targets)
            val_loss += loss.item()
        # ----------------------------------------------------------------------
        # Compute the metrics for this epoch
        # ----------------------------------------------------------------------
        self.__metrics_gen[1].compute_metrics_epoch()
        # ======================================================================
        # HISTORY UPDATE
        # ======================================================================
        self.__history_recorders[1].update(torch.Tensor([val_loss]), loss_indx)
        # ----------------------------------------------------------------------
        # Store the metrics of this epoch
        # ----------------------------------------------------------------------
        for metric_id, metric in enumerate(self.__metrics_gen[1].get_metrics()):
            # ==================================================================
            # HISTORY UPDATE
            # ==================================================================
            self.__history_recorders[1].update(metric, metric_id)

    # --------------------------------------------------------------------------
    def evaluate_ensemble(self, models_base_dir:list[str],
                            epochs:int|list[int], 
                            dataloaders_type: DatasetType|list[DatasetType] \
                                = DatasetType.ALL,
                            model_weights:torch.Tensor|None = None):
        """
        This procedure evaluates an  ensemble of models whose  directory is in \
        <models_base_dir>. (Notice that it is expected for the every base dir. \
        to include a 'checkpoints' directory that includes the checkpoints).
        The ensemble is evaluated on the data in <dataloader> for <epochs> with\
        <model_weights> as weights for each model in order 

        ------------------------------------------------------------------------
        Inputs:
            - models_base_dir: list of strings indicating the model  base_dirs 
                - example: ['./models/m1/', './models/m2/'] for two models m1 &\
                    m2
            - epochs: the number of epochs in which we create the ensemble, or \
                the list of file names without the extension.
                - example: 5, hence, there should be one file for each version,\
                    equivalent to `epochs = [0, 1, 2, 3, 4]`
                - example: ['0','5','10'] to load ['0.pt','5.pt','10.pt']
            - dataloader_type: the dataloader types on which we evaluate the en\
                semble
                - example: DatasetType.TRAIN to test on the training loader
                - DatasetType.ALL to test on all the data loaders
                - default value is DatasetType.ALL
            - model_weights: the weights of the ensemble as torch.Tensor;
                - shape: #tasks x #clusters
                - None for uniform, the default value
        """
        dataloaders = []
        if type(dataloaders_type) == list:
            for dl_type in dataloaders_type:
                dataloaders.append(self.__get_loader(dl_type))
            dataloaders = tuple(dataloaders)
        else:
            dataloaders = self.__get_loader(dataloaders_type)
        with torch.no_grad():
            self.__evaluate_ensemble(models_base_dir = models_base_dir,
                            epochs= epochs, dataloaders = dataloaders,
                            model_weights = model_weights)

    # --------------------------------------------------------------------------
    def __evaluate_ensemble(self, models_base_dir:list[str],
                            epochs:int|list[int], 
                            dataloaders: DataLoader|tuple[DataLoader],
                            model_weights:torch.Tensor|None) -> None:
        """
        This procedure evaluates an  ensemble of models whose  directory is in \
        <models_base_dir>. (Notice that it is expected for the every base dir. \
        to include a 'checkpoints' directory that includes the checkpoints).
        The ensemble is evaluated on the data in <dataloader> for <epochs> with\
        <model_weights> as weights for each model in order 

        ------------------------------------------------------------------------
        Inputs:
            - models_base_dir: list of strings indicating the model  base_dirs 
                - example: ['./models/m1/', './models/m2/'] for two models m1 &\
                    m2
            - epochs: the number of epochs in which we create the ensemble, or \
                the list of file names without the extension.
                - example: 5, hence, there should be one file for each version,\
                    equivalent to `epochs = [0, 1, 2, 3, 4]`
                - example: ['0','5','10'] to load ['0.pt','5.pt','10.pt']
            - dataloader: a dataloader of the data on which we evaluate the\
                the ensemble
            - model_weights: the weights of the ensemble as torch.Tensor;
                - shape: #tasks x #clusters
                - None for uniform
            - logger_index: the index of the logger to be used
                - Default: 0
        """
        # ----------------------------------------------------------------------
        # initialize the model weights to be uniform if not specified.
        # ----------------------------------------------------------------------
        if model_weights is None:
            num_clusters = len(models_base_dir)
            model_weights = torch.ones(self.__num_of_tasks, num_clusters)
            model_weights = model_weights/num_clusters

        model_weights = model_weights.to(self.__args['device'])
        # ----------------------------------------------------------------------
        # initialize the base model and the number of epochs
        # ----------------------------------------------------------------------
        base_model = copy.deepcopy(self.__model)
        if type(epochs) == int:
            epochs = list(range(epochs))
        
        if type(dataloaders) == DataLoader:
            dataloaders = (dataloaders)
        # ======================================================================
        # Loop over the files and evaluate them
        # ======================================================================
        init_epoch = None
        for epoch, name in enumerate(epochs):
            if init_epoch is None:
                init_epoch = name
            f_name = f"{name}.pt"

            # ------------------------------------------------------------------
            # load the models
            # ------------------------------------------------------------------
            models = []
            for path in models_base_dir:
                model_path = os.path.join(path, "checkpoints")
                model_path = os.path.join(model_path, f_name)
                m = Trainer.load_model_from_checkpoint(model_path, base_model)
                m.to(self.__args['device'])
                m.eval()
                models.append(m)
            
            # ------------------------------------------------------------------
            # Loop over the batches in each data loader
            # ------------------------------------------------------------------
            for loader_indx, loader in enumerate(dataloaders):
                if loader_indx >= len(self.__metrics_gen):
                    raise IndexError("Large number of data loaders for the en"+
                    f"semble. It should be maximum of \
                        [{len(self.__metrics_gen)}]")
                # --------------------------------------------------------------
                # Initialize the values before recording the results
                # --------------------------------------------------------------
                self.__metrics_gen[loader_indx].initialize()
                # model.eval()
                for indxs, inputs, targets in loader:
                    # ----------------------------------------------------------
                    # Move to the Device
                    # ----------------------------------------------------------
                    inputs = inputs.to(self.__args['device'])
                    targets = targets.to(self.__args['device'])
                    # ----------------------------------------------------------
                    # Forward pass
                    # ----------------------------------------------------------
                    outputs = torch.zeros_like(targets)
                    # ----------------------------------------------------------
                    # Ensemble across the models
                    # ----------------------------------------------------------
                    for m_indx, model in enumerate(models):
                        outputs += model(inputs)*model_weights[:,m_indx]
                    # ----------------------------------------------------------
                    # Update the dynamics of the metrics generator for the curr-
                    # ent batch
                    # ----------------------------------------------------------
                    self.__metrics_gen[loader_indx].update_metrics_iteration( \
                                                               outputs, targets)
                # --------------------------------------------------------------
                # Compute the metrics for this epoch
                # --------------------------------------------------------------
                self.__metrics_gen[loader_indx].compute_metrics_epoch()
                # ==============================================================
                # HISTORY UPDATE
                # ==============================================================
                for metric_id, metric in enumerate(\
                    self.__metrics_gen[loader_indx].get_metrics()):
                    self.__history_recorders[loader_indx].update(metric,
                                                                      metric_id)
            if self.__debug:
                data = {
                    "epoch": int(name)+1,
                    "train_loss": 0,

                }
                self.__print_debug_messages("epoch_update", data)
            # ==================================================================
            # Update epoch
            # ==================================================================
            self.__epoch = epoch
            # ==================================================================
            # Save checkpoint
            # ==================================================================
            if self.__args['checkpoint_enable']:
                self.save_checkpoint(init_epoch= init_epoch)
        # ======================================================================
        # Finally, write the training summary
        # ======================================================================
        self.__write_summary()

    # --------------------------------------------------------------------------
    def __fill_data_results(self, indxs, outputs, targets) -> None:
        """
        The method fills in the probabilities of the true label in each task for
        each data point. We use that data to generate the data maps.
        ------------------------------------------------------------------------
        Inputs:
            - data_results: a 2D list contains (num_of_tasks, num_of_points, pr)
                where pr is the number of occurrence of each  point in each task
            - indxs: holds the indices of each data point of the batch
            - outputs: holds the model output of the batch
            - targets: holds the true targets of the batch
        ------------------------------------------------------------------------
        Therefore, the method will add the probabilities of the true  targets of
        the batch according to the <outputs> into <data_results>
        """
        # ----------------------------------------------------------------------
        # Get the probabilities of true target of the batch and update the reco-
        # rder
        # ----------------------------------------------------------------------
        confidence = self.__model.get_probabilities_true_target(outputs,targets)
        self.__datamap_recorder.update(confidence, indxs)

    # --------------------------------------------------------------------------
    def get_data_map(self) -> torch.Tensor:
        """
        The method returns the data map tensor whose dimension is
                (#tasks, #epochs, #data_points, 2)
        The last dimension includes two values that is the confidence and std
        """
        return self.__data_map

    # --------------------------------------------------------------------------
    @staticmethod
    def load_history_from_checkpoint(path: str) -> list[ValuesRecorder]:
        """
        The method loads a history of values from a checkpoint and return  it. \
        it returns a list of <ValuesRecorder> objects. 

        ------------------------------------------------------------------------
        Common usage hints of ValuesRecorder:
            - obj.get_metric_names(): returns ordered list of  metric names of \
                the recorded values
            - obj.get_id(): returns the id of the recorder (e.g. "Training")
            - obj.concatenate(i): returns a tensor of all the history of  the i\
                th metric.\n
        You are encouraged to read the documentation of ValuesRecorder for other
        methods
        ------------------------------------------------------------------------
        Inputs:
            - path: checkpoint path
        ------------------------------------------------------------------------
        Outputs:
            - list of value recorders
        """
        # ----------------------------------------------------------------------
        # read the checkpoint
        # ----------------------------------------------------------------------
        checkpoint = torch.load(path, map_location= torch.device("cpu"))
        # ----------------------------------------------------------------------
        # create a list of dummy recorders until being filled
        # ----------------------------------------------------------------------
        history_recorders = [ValuesRecorder(metric_names=[], id="",
            unsqueeze_dims={}, concat_dims={})\
                for i in range(len(checkpoint['recorder_states']))]
        # ----------------------------------------------------------------------
        # read the recorders
        # ----------------------------------------------------------------------
        for i, rec_state_path in enumerate(checkpoint['recorder_states']):
            history_recorders[i].load_state_file(rec_state_path)
        
        return history_recorders

    # --------------------------------------------------------------------------
    @staticmethod
    def load_datamap_from_checkpoint(path:str, epochs_data_map:list[int] = [],
                     mode:ValueStoringMode = ValueStoringMode.OVERWRITE,
                     device:torch.device = torch.device("cpu"))\
                        -> torch.Tensor|None:
        """
        The function loads a checkpoint and returns the datamap

        ------------------------------------------------------------------------
        Inputs:
            - path: checkpoint path
            - epochs_data_map: the epochs in which we were recording datamap to\
                safely re-construct the datamap
            - mode: the storing mode; check `ValueStoringMode` for more info.
                - default: ValueStoringMode.OVERWRITE
            - device: the device to which we move our datamap after reading it
                - default: cpu
        ------------------------------------------------------------------------
        Outputs:
            - datamap: a tensor holding the datamap
                - shape: #tasks, #epochs, #data_points, 2
        """
        # ----------------------------------------------------------------------
        # Load the checkpoint
        # ----------------------------------------------------------------------
        checkpoint = torch.load(path, map_location= device)
        dm_path = checkpoint['datamap_path']
        # ----------------------------------------------------------------------
        # If the path is NONE, return None
        # ----------------------------------------------------------------------
        if dm_path =="NONE":
            return None
        
        # ----------------------------------------------------------------------
        # get the epoch of the file
        # ----------------------------------------------------------------------
        is_digit = dm_path.split("/")[-1].split(".")[0].isdigit()
        f_name = dm_path.split("/")[-1].split(".")[0] if not is_digit else \
                int(dm_path.split("/")[-1].split(".")[0])
        
        datamap_path = '/'.join(dm_path.split("/")[:-1])
        # ----------------------------------------------------------------------
        # If we are reading the datamap in OVERWRITE or NORMAL modes, or even if
        # the name is not the the list epochs, return the file 
        # ----------------------------------------------------------------------
        if mode == ValueStoringMode.OVERWRITE or mode ==ValueStoringMode.NORMAL\
            or ((f_name not in epochs_data_map) and \
                (str(f_name) not in epochs_data_map)):
            return torch.load(dm_path, map_location= device)
        else:
            # ------------------------------------------------------------------
            # Otherwise, we are in the DIFFERENCE mode,  
            # ------------------------------------------------------------------
            # Otherwise,  we need to know the  index of the  file in the list of
            # epochs
            # ------------------------------------------------------------------
            if f_name in epochs_data_map:
                f_indx = epochs_data_map.index(f_name) 
            elif str(f_name) in epochs_data_map:
                f_indx = epochs_data_map.index(str(f_name))
            # ------------------------------------------------------------------
            # if the index is 0, this is  the first epoch,  so just  return  the
            # file as it is
            # ------------------------------------------------------------------
            if f_indx == 0:
                return torch.load(dm_path, map_location= device)
            # ------------------------------------------------------------------
            # Otherwise,  loop over all  the previous files to  re-construct the
            # datamap safely 
            # ------------------------------------------------------------------
            else:
                p = os.path.join(datamap_path, f"{epochs_data_map[0]}.pt")
                dm = torch.load(p, map_location= device)
                for i in range(1,f_indx+1):
                    p = os.path.join(datamap_path, f"{epochs_data_map[i]}.pt")
                    dm = torch.cat((dm, torch.load(p, map_location= device)),
                                    dim=1)
                return dm
    
    # --------------------------------------------------------------------------
    @staticmethod
    def load_model_from_checkpoint(path:str, model:BaseModel, 
                                    device:torch.device = torch.device("cpu"))\
                                         -> BaseModel:
        """
        The function loads a model from a checkpoint returning a <model> clone \
        with the loaded state.

        ------------------------------------------------------------------------
        Inputs:
            - path: checkpoint path
            - model: the model that includes the architecture where the data is\
                loaded
        ------------------------------------------------------------------------
        Outpus:
            - a cloned of <model> with the same state in the checkpoint
        """
        m = copy.deepcopy(model)
        checkpoint = torch.load(path, map_location= device)
        m.load_state_dict(checkpoint['model_state_dict'])
        return m
    
    # --------------------------------------------------------------------------