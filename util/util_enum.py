import enum

# ==============================================================================
# Define an enum to help clarifying the storing type
# ==============================================================================
class ValueStoringMode(enum.Enum):
    """
    This signifies the modes of storing values; it has the below options:
        - `ValuesStoringMode.OVERWRITE`: writes the values in a single file,  so
            the content of the epoch `N` overwrites the records in `N-1`
        - `ValuesStoringMode.DIFFERENCE`: writes the values into separate chunks
            where all the records are distributed among these chunks
        - `ValuesStoringMode.NORMAL`: the  normal mode  where you write all the\
            values everytime, which might not be very efficient
    """
    OVERWRITE = 0
    DIFFERENCE = 1
    NORMAL = 2

# ==============================================================================
# Define an enum to help clarifying the dataset type
# ==============================================================================
class DatasetType(enum.Enum):
    """
    This enum represents the different possibilities for choosing a dataset type
        - `DatasetType.NONE`: for no dataset
        - `DatasetType.TRAIN`: to load the loader of the training dataset
        - `DatasetType.VALIDATION`: to load the loader of the validation set
        - `DatasetType.TEST`: to load the load of the test set
        - `DatasetType.TRAIN_VAL`: to load both the training an validation data
        - `DatasetType.ALL`: to load all the three types
    """
    NONE = -1
    TRAIN = 0
    VALIDATION = 1
    TEST = 2
    TRAIN_VAL = 3
    ALL = 4