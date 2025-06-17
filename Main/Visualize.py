import matplotlib.pyplot as plt
from Data import x, y


plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
plt.title("Spiral Dataset Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Class')
plt.show()