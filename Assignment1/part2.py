import random
from typing import List, Tuple

# CONSTANTS
N = 8 # Batch Size
C = 4 # Number of input maps/filters channels
H = W = M = 32 # input map height and width, number of output maps/filters
R = S = 5 # Filter height and width
U = 2 # Stride Length
E = F = 14 # output map height and width