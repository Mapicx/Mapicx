from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:

        losses = self.forward(y_pred, y_true)
        return float(np.mean(losses))

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:


        samples = y_pred.shape[0]

        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if y_true.ndim == 1:
            
            correct_confidences = y_pred[np.arange(samples), y_true]
        elif y_true.ndim == 2:
           
            correct_confidences = np.sum(y_pred * y_true, axis=1)
        else:
            raise ValueError(
                f"y_true must be 1D or 2D, got ndim={y_true.ndim}"
            )

        return -np.log(correct_confidences)
    
    def backward(self, dvalue, y_true):
        samples = len(dvalue)
        labels = len(dvalue[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinput = -y_true/dvalue
        self.dinput = self.dinput/ samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def forward(self, inputs, y_true):
        # Softmax activation
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # Save ground truth
        self.y_true = y_true

        # Calculate loss
        samples = len(inputs)

        # Clip values to prevent log(0)
        clipped_output = np.clip(self.output, 1e-7, 1 - 1e-7)

        # If labels are one-hot encoded, pick only the correct class probs
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(clipped_output * y_true, axis=1)
        else:
            correct_confidences = clipped_output[range(samples), y_true]

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        # If labels are one-hot, convert to discrete labels
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Compute gradient
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
