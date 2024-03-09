"""An example on CIFAR10 Model implementing the abstract methods"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com>
# License: MIT
# ==============================================================================

# ==============================================================================
# The file includes the model used to generate  datamaps and evaluate our group-
# ing algorithm.
# ==============================================================================

from .cifar10 import *

# ==============================================================================
# Create an interface to the same C10Model
# ==============================================================================
class C100Model(C10Model):
    pass

# ==============================================================================
# Create an interface to the same C10Metrics
# ==============================================================================
class C100Metrics(C10Metrics):
    pass
