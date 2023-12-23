import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a grid of points
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)

# Combine the x and y values into complex numbers and compute sin(z)
Z = X + 1j*Y
W = np.sin(Z)

# Separate W into its real and imaginary parts
W_real = np.real(W)
W_imag = np.imag(W)

# Create a 3D plot for the real part of sin(z)
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, W_real, color='r')
ax.set_title('Real part of sin(z)')

# Create a 3D plot for the imaginary part of sin(z)
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, W_imag, color='b')
ax.set_title('Imaginary part of sin(z)')

plt.show()
