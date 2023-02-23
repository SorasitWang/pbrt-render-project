from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *

coordX = []
coordY = []
coordZ = []
values = []

def addCoord(x, y, z, val):
    coordX.append(x)
    coordY.append(y)
    coordZ.append(z)
    values.append(val)

def draw():
    print(values)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.scatter3D(coordX, coordY, coordZ, c=values, cmap="Greens")
    plt.show()
