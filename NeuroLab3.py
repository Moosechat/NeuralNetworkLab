# Лабораторная Работа №2

import numpy as np
import torch


def sigma(x):
    return 1 / (1 + np.exp(-x))


def ClearInputInOutput(x, y, z):
    return np.matmul(x, y) + z


def SigmoidFunctionDerivative(x):
    return 1 - x


def NewWeightsCalculation(x, y, z):
    return x - np.matmul(y, z)


Input_Vector = torch.Tensor([1, 0, 1, 0])
OutputDifference = torch.Tensor([-0.13, -0.13, -0.13])
Inputs = torch.Tensor([0.79, 0.79, 0.73])
SigmoidFunction = torch.Tensor([0.87, 0.87, 0.87])
InitialWeights = torch.Tensor([0.71, 0.16, 0.57])
LR = ([0.5, 0.5, 0.5])
SigmoidDerivative = SigmoidFunctionDerivative(SigmoidFunction)
print('Sigmoid Derivative = ', SigmoidDerivative)
OutputDerivative = torch.mul(OutputDifference * SigmoidFunction, SigmoidDerivative * Inputs)
print('Output Derivative = ', OutputDerivative)
NewWeights = NewWeightsCalculation(InitialWeights, LR, OutputDerivative)
print('New Weights = ', NewWeights)
