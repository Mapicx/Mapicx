# Mapicx - High-Performance Neural Network Framework

[![PyPI version](https://img.shields.io/pypi/v/Mapicx.svg)](https://pypi.org/project/Mapicx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mapicx is a lightweight yet powerful neural network framework designed for both educational purposes and practical deep learning applications. Built from scratch in Python, Mapicx provides intuitive interfaces for building, training, and deploying various types of neural networks.

## Key Features

- 🚀 **Multi-Architecture Support**
  - Artificial Neural Networks (ANN)
  - Convolutional Neural Networks (CNN) 
  - Recurrent Neural Networks (RNN/LSTM) 
  - Hybrid architectures
- 💻 **Intuitive API**
  - Keras-like syntax for easy adoption
  - Modular layer-based design
- ⚡ **High Performance**
  - Optimized matrix operations with NumPy
  - GPU acceleration support (coming soon)
- 📦 **Extensible Design**
  - Easily customizable layers and activation functions
  - Simple plugin system for custom components

## Installation

```bash
pip install Mapicx
```

## Quick Start
```py
from Mapicx import Mapicx
from Mapicx.datasets import spiral_data
from Mapicx.optimizers import SGD

# Load dataset
X, y = spiral_data(samples=1000, classes=3)

# Create model
model = Mapicx()
model.add(2, 128, layer='Dense', activation='Relu')
model.add(128, 64, layer='Dense', activation='Relu')
model.add(64, 3, layer='Dense', activation='Softmax')

# Compile with optimizer
optimizer = SGD(_learning_rate=0.1, _decay=1e-4, momentum=0.9)
model.compile(optimizer=optimizer)

# Train the model
model.fit(X, y, epochs=1000, print_every=100)

# Make predictions
predictions = model.predict(X)
```

## Custom Neural Architecture
```py
model = Mapicx()
model.add(784, 256, activation='Relu')
model.add(0, 0, layer='Dropout', rate=0.4)
model.add(256, 128, activation='LSTM')  # RNN layer coming soon!
model.add(128, 10, activation='Softmax')
```

## Key Components
  Layers
  Dense (Fully Connected)
  Dropout (Regularization)
  Activation Functions
  ReLU, Leaky ReLU
  Sigmoid, Tanh
  Softmax

## Optimizers
  SGD (with Momentum and Decay)
  Adam (Upcoming)
  RMSprop (Upcoming)
  Loss Functions
  Categorical Crossentropy
  Binary Crossentropy (Upcoming)
  MSE (Upcoming)

## Documentation
Explore the full documentation at mapicx.readthedocs.io

## Contributing
We welcome contributions! Please see our Contribution Guidelines for details.

