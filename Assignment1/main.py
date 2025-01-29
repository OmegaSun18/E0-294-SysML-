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
    inputMaps = [[[[[] for itr4 in range(H)] for itr3 in range(W)] for itr2 in range(C)] for itr1 in range(N)]
    helper = [1, -1]
    filterWeights = [[[[[] for itr4 in range(R)] for itr3 in range(S)] for itr2 in range(C)] for itr1 in range(M)]
    for itr1 in range(N):
        for itr2 in range(C):
            for itr3 in range(W):
                for itr4 in range(H):
                    rand = random.random()
                    multiplier = random.choice(helper)
                    inputMaps[itr1][itr2][itr3][itr4].append(rand*multiplier)
    for itr1 in range(M):
        for itr2 in range(C):
            for itr3 in range(S):
                for itr4 in range(R):
                    rand = random.random()
                    multiplier = random.choice(helper)
                    filterWeights[itr1][itr2][itr3][itr4].append(rand*multiplier)
    return inputMaps, filterWeights

def naiveConvolution(inputMaps: List[List[List[List[float]]]], filterWeights: List[List[List[List[float]]]]) -> List[List[List[List[float]]]]:
    """Performs a naive convolution given the input maps and the filter weights"""
    outputMapsNaive = [[[[[0] for itr4 in range(E)] for itr3 in range(F)] for itr2 in range(M)] for itr1 in range(N)]
    for n in range(N):
        for m in range(M):
            for x in range(F):
                for y in range(E):
                    for i in range(R):
                        for j in range(S):
                            for k in range(C):
                                outputMapsNaive[n][m][x][y][0] += inputMaps[n][k][U*x + i][U*y + j][0] * filterWeights[m][k][i][j][0]
    return outputMapsNaive

def filterWeightstoToeplitz(filterWeights: List[List[List[List[float]]]]) -> List[List[float]]:
    """Converts the filter weights matrix from a 4D matrix to its toeplitz form"""
    filterWeightsforToeplitz = [[] for itr1 in range(M)]
    for m in range(M):
        temp = [0 for i in range(C*R*S)]
        innerCtr = 0
        for i in range(R):
            for j in range(S):
                for k in range(C):
                    temp[R*S*k + innerCtr] = filterWeights[m][k][i][j][0]
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
                            temp[R*S*k + innerCtr] = inputMaps[n][k][U*x + i][U*y + j][0]
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

def convertOutputMaps(outputMapsNaive: List[List[List[List[float]]]]) -> List[List[float]]:
    """Converts the naive convolution output maps(4D matrix) to a 2D matrix for comparison with the toeplitz convolution method"""
    convertedOutputMaps = [[] for itr1 in range(M)]
    for m in range(M):
        temp = [0 for itr in range(N*F*E)]
        ctr = 0
        for n in range(N):
            for y in range(E):
                for x in range(F):
                    temp[ctr] = outputMapsNaive[n][m][x][y][0]
                    ctr += 1
        convertedOutputMaps[m] = temp
    return convertedOutputMaps

def comparison(outputMapsforToeplitz: List[List[float]], convertedOutputMaps: List[List[float]]) -> str:
    """Iterates through the two output matrices for both the methods and checks if both elements are the same.
    It checks with a toleraance of 1e-13 as python rounding may give two outputs which differ at the 14th decimal or so."""
    ctr = 0
    tol = 1e-13
    for itr1 in range(M):
        for itr2 in range(len(outputMapsforToeplitz[0])):
            if abs(outputMapsforToeplitz[itr1][itr2] - convertedOutputMaps[itr1][itr2]) > tol:
                ctr += 1
    if ctr == 0:
        return "The two methods produce the same outputs."
    return "The two methods don't produce the same outputs."


inputMaps, filterWeights = create_arrays()
outputMapsNaive = naiveConvolution(inputMaps, filterWeights)
inputMapsforToeplitz = inputMapstoToeplitz(inputMaps)
filterWeightsforToeplitz = filterWeightstoToeplitz(filterWeights)
outputMapsforToeplitz = toeplitzConvolution(inputMapsforToeplitz, filterWeightsforToeplitz)
convertedOutputMaps = convertOutputMaps(outputMapsNaive)
print(comparison(outputMapsforToeplitz, convertedOutputMaps))