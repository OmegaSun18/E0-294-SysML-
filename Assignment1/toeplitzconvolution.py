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

def filterWeightstoToeplitz(filterWeights: List[List[List[List[float]]]]) -> List[List[float]]:
    """Converts the filter weights matrix from a 4D matrix to its toeplitz form"""
    filterWeightsforToeplitz = [[] for itr1 in range(M)]
    for m in range(M):
        temp = [0 for i in range(C*R*S)]
        innerCtr = 0
        for i in range(R):
            for j in range(S):
                for k in range(C):
                    temp[R*S*k + innerCtr] = filterWeights[m][k][i][j]
                innerCtr += 1
        filterWeightsforToeplitz[m] = temp
    return filterWeightsforToeplitz

def inputMapstoToeplitz(inputMaps: List[List[List[List[float]]]]) -> List[List[float]]:
    """Converts the input maps from a 4D matrix to its toeplitz form"""
    inputMapsforToeplitz = [[] for itr2 in range(int(N*E*F))]
    outerCtr = 0
    for n in range(N):
        for y in range(E):
            for x in range(F):
                temp = [0 for itr in range(C*R*S)]
                innerCtr = 0
                for i in range(R):
                    for j in range(S):
                        for k in range(C):
                            temp[R*S*k + innerCtr] = inputMaps[n][k][U*x + i][U*y + j]
                        innerCtr += 1
                inputMapsforToeplitz[outerCtr] = temp
                outerCtr += 1
    return inputMapsforToeplitz

def toeplitzConvolution(inputMapsforToeplitz: List[List[float]], filterWeightsforToeplitz: List[List[float]]) -> List[List[float]]:
    """Performs convolution(Matrix Multiplication) on the input maps and filter weights shaped in the toeplitz format"""
    outputMapsforToeplitz = [[] for itr1 in range(M)]
    for m in range(M):
        temp = [0 for itr in range(N*F*E)]
        for i in range(N*F*E):
            for j in range(C*S*R):
                temp[i] += inputMapsforToeplitz[i][j] * filterWeightsforToeplitz[m][j]
        outputMapsforToeplitz[m] = temp
    return outputMapsforToeplitz

inputMaps, filterWeights = create_arrays()
inputMapsforToeplitz = inputMapstoToeplitz(inputMaps)
filterWeightsforToeplitz = filterWeightstoToeplitz(filterWeights)
outputMapsforToeplitz = toeplitzConvolution(inputMapsforToeplitz, filterWeightsforToeplitz)