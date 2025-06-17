from nnfs.datasets import spiral_data
import numpy as np
import nnfs

nnfs.init()

import matplotlib.pyplot as plt

x, y = spiral_data(samples=100, classes=3)

