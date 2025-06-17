import numpy as np
from layer import Dense, Layers

class SGD:
    def __init__(self, _learning_rate = 1.0, _decay = 0.0, momentum = 0.0) -> None:
        self.learning_rate = _learning_rate
        self.current_learning_rate = _learning_rate
        self.decay = _decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_update = self.momentum * layer.weight_momentums \
                            - self.current_learning_rate * layer.dweight
            layer.weight_momentums = weight_update

            bias_update = self.momentum * layer.bias_momentums - \
                            self.current_learning_rate * layer.dbaises
            layer.bias_momentums = bias_update

        else:
            weight_update = -self.current_learning_rate * layer.dweight
            bias_update = -self.current_learning_rate * layer.dbaises

        layer.weights += weight_update
        layer.biases += bias_update

    def post_update_params(self):
        self.iterations += 1
