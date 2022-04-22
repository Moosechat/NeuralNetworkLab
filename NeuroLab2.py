# Лабораторная Работа №2

import numpy as np
import torch


def sigm(x):
    return 1 / (1 + np.exp(-x))


def ClearInputInOutput(x, y, z):
    return np.matmul(x, y) + z
Input_Vector = torch.Tensor([1, 0, 1, 0])
Target_Output = torch.Tensor([1])
WeightsInHiddenLayer = torch.Tensor([[0.49, 0.72, 0.38], [0.35, 0.9, 0.43], [0.44, 0.58, 0.39], [0.8, 0.92, 0.21]])
BiasAtHiddenLayer = torch.Tensor([0.4, 0, 0.22])
InitialWeights = torch.Tensor([0.71, 0.16, 0.57])
OutputBias = 0.83
print(Input_Vector)
print(Target_Output)
print(WeightsInHiddenLayer)
print(BiasAtHiddenLayer)
InputTimesWeight = np.matmul(Input_Vector, WeightsInHiddenLayer)
print('input times weight = ', InputTimesWeight)
NetInput = np.add(InputTimesWeight, BiasAtHiddenLayer)
print('Net Input = ', NetInput)
fInput = sigm(NetInput)
print('(f) Input) = ', fInput)
NetInputAtOutput = ClearInputInOutput(fInput, InitialWeights, OutputBias)
print('Net Input At Output = ', NetInputAtOutput)
fNetInput = sigm(NetInputAtOutput)
print('(f) Net Input) = ', fNetInput)
