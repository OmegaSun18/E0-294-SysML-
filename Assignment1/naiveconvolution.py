import random
from typing import List, Tuple

# CONSTANTS
N = 8 # Batch Size
C = 4 # Number of input maps/filters channels
H = W = M = 32 # input map height and width, number of output maps/filters
R = S = 5 # Filter height and width
U = 2 # Stride Length
E = F = 14 # output map height and width

def create_arrays() -> Tuple[List[List[List[List[float]]]]]:
    """Creates the input maps and the filter maps"""
    inputMaps = [[[[0 for itr4 in range(H)] for itr3 in range(W)] for itr2 in range(C)] for itr1 in range(N)]
    helper = [1, -1]
    filterWeights = [[[[0 for itr4 in range(R)] for itr3 in range(S)] for itr2 in range(C)] for itr1 in range(M)]
    for itr1 in range(N):
        for itr2 in range(C):
            for itr3 in range(W):
                for itr4 in range(H):
                    rand = random.random()
                    multiplier = random.choice(helper)
                    inputMaps[itr1][itr2][itr3][itr4] = (rand*multiplier)
    for itr1 in range(M):
        for itr2 in range(C):
            for itr3 in range(S):
                for itr4 in range(R):
                    rand = random.random()
                    multiplier = random.choice(helper)
                    filterWeights[itr1][itr2][itr3][itr4] = (rand*multiplier)
    return inputMaps, filterWeights

def naiveConvolution(inputMaps: List[List[List[List[float]]]], filterWeights: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
    """Performs a naive convolution given the input maps and the filter weights"""
    outputMapsNaive = [[[[0 for itr4 in range(E)] for itr3 in range(F)] for itr2 in range(M)] for itr1 in range(N)]
    for n in range(N):
        for m in range(M):
            for x in range(F):
                for y in range(E):
                    for i in range(R):
                        for j in range(S):
                            for k in range(C):
                                outputMapsNaive[n][m][x][y] += inputMaps[n][k][U*x + i][U*y + j] * filterWeights[m][k][i][j]
    return outputMapsNaive

inputMaps, filterWeights = create_arrays()
outputMapsNaive = naiveConvolution(inputMaps, filterWeights)