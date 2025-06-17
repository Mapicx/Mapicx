from random import randint
from typing import List, Optional
import numpy as np

class Neuron:
    def __init__(
        self,
        inputs: List[float],
        weights: Optional[List[float]] = None,
        bias: float = 0.0
    ):
        self.inputs = inputs
        self.weights = weights or [randint(-3, 3) for _ in inputs]
        self.bias = bias
        self.output = 0.0

    def find_output(self):
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output
    
nrn = Neuron(inputs=[1.0, 2.0, 3.0, 2.5], weights=[0.2, 0.8, -0.5, 1], bias=2)
print(nrn.find_output())