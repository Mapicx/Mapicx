import numpy as np

class Relu:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = np.maximum(0, input)

    def backward(self, dvalues):
        self.dinput = dvalues.copy()
        self.dinput[self.input <= 0] = 0

class Softmax:
    def __init__(self):
        pass

    def forward(self, input):
        # Improved numerical stability
        max_vals = np.max(input, axis=1, keepdims=True)
        exp_values = np.exp(input - max_vals)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # Clip to prevent log(0) in loss calculations
        self.output = np.clip(probabilities, 1e-7, 1 - 1e-7)

    def backward(self, dvalues):
        # Allocate gradient array
        self.dinput = np.empty_like(dvalues)
        
        # For each sample
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Convert softmax output to column vector
            single_output = single_output.reshape(-1, 1)
            # Build Jacobian matrix
            jacobian = np.diagflat(single_output) - single_output.dot(single_output.T)
            # Calculate gradient w.r.t. inputs
            self.dinput[index] = jacobian.dot(single_dvalues)

