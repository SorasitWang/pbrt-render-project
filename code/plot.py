from tokenize import Number
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
import random

# 2D


def plot2d(
    string, alpha, delta=1, init_direction=np.array([1.0, 0]), colors={}, gen=True
):
    plt.gca().set_aspect("equal", adjustable="box")
    allPos = []
    pos = np.zeros(2, dtype=np.float64)
    direction = init_direction
    color = "k"
    saved_states = []
    currentNo = -1
    for x in string:
        if x == "F":
            new_pos = pos + direction
            plt.plot([pos[0], new_pos[0]], [pos[1], new_pos[1]], c=color)
            # print("leaf" if color == "g" else "bark",currentNo)
            allPos.append(
                ShapeL("leaf" if color == "g" else "bark", pos, new_pos, currentNo)
            )
            pos = new_pos
            # allPos.append(pos)
        elif x == "+":
            direction = rotate(direction, alpha)
        elif x == "-":
            direction = rotate(direction, -alpha)
        elif x == "*":
            direction *= delta
        elif x == "/":
            direction /= delta
        elif x == "|":
            direction = rotate(direction, 180)
        elif x == "[":
            saved_states.append((pos, direction))
        elif x == "]":
            pos, direction = saved_states.pop()
        elif x in colors:
            color = colors[x]
        elif x.isnumeric():
            currentNo = int(x)
    if not gen:
        plt.show()
    return allPos


# 3D


def axis_equal_3d(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def plot3d(string, alpha, delta=1, init_direction=np.array([1.0, 0, 0]), colors={}):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pos = np.zeros(3, dtype=np.float64)
    direction = init_direction
    color = "k"
    allPos = []
    f = 0
    fStat = []
    saved_states = []
    for x in string:
        # alpha = random.randint(5, 15) / 10 * alpha_
        # delta = random.randint(5, 15) / 10 * delta_
        if x == "F":
            f += 1
            new_pos = pos + direction
            ax.axes.set_xlim3d(left=-6, right=1)
            ax.axes.set_ylim3d(bottom=-2, top=5)
            ax.axes.set_zlim3d(bottom=0, top=15)
            ax.plot(
                [pos[0], new_pos[0]],
                [pos[1], new_pos[1]],
                [pos[2], new_pos[2]],
                c=color,
            )
            allPos.append(ShapeL("sect", pos, new_pos, f))
            pos = new_pos
        elif x == "+":
            direction = rotate_x(direction, alpha)
        elif x == "-":
            direction = rotate_x(direction, -alpha)
        elif x == "&":
            direction = rotate_y(direction, alpha)
        elif x == "^":
            direction = rotate_y(direction, -alpha)
        elif x == "<":
            direction = rotate_z(direction, alpha)
        elif x == ">":
            direction = rotate_z(direction, -alpha)
        elif x == "*":
            direction = direction * delta
        elif x == "/":
            direction = direction / delta
        elif x == "|":
            direction = rotate_x(direction, 180)
            direction = rotate_y(direction, 180)
            direction = rotate_z(direction, 180)
        elif x == "[":
            saved_states.append((pos, direction))
            fStat += [f]
        elif x == "]":
            pos, direction = saved_states.pop()
            f = fStat.pop()

        elif x in colors:
            color = colors["g"]
    axis_equal_3d(ax)
    plt.show()
    return allPos


def plotA3d(
    string,
    step,
    alpha_,
    delta=1,
    init_direction=np.array([1.0, 0, 0]),
    colors={},
    gen=True,
):
    allPos = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pos = np.zeros(3, dtype=np.float64)
    direction = init_direction
    color = "k"
    f = 0
    ff = 0
    maxF = -1
    fStat = []
    topStack = []
    prev = None
    leafStack = []
    saved_states = []
    ax.axes.set_xlim3d(left=-6, right=1)
    ax.axes.set_ylim3d(bottom=-2, top=5)
    ax.axes.set_zlim3d(bottom=0, top=15)
    for x in string:
        alpha = random.randint(7, 13) / 10 * alpha_
        if x == "F":

            f += 1
            new_pos = pos + direction
            new_pos_ = new_pos

            if color == "g":
                direction_ = rotate_y(direction, alpha)
                direction_ = rotate_z(direction_, alpha)
                new_pos_ = pos + direction_
            ax.plot(
                [pos[0], new_pos_[0]],
                [pos[1], new_pos_[1]],
                [pos[2], new_pos_[2]],
                c=color,
                lw=2,
            )

            allPos.append(ShapeL("leaf" if color == "g" else "bark", pos, new_pos, f))

            if color != "g":
                if f < ff and prev != None:
                    prev.core = False
                    # print("T")
                prev = allPos[-1]
                # print(pos, new_pos, ff, f)
                ff = f
            else:
                leafStack.append(allPos[-1])

            maxF = max(maxF, ff)
            if allPos[-1].type == "bark":
                if topStack == []:
                    topStack.append(allPos[-1])
                else:
                    topStack[-1] = allPos[-1]
                # print(pos, new_pos, topStack)
            pos = new_pos
        elif x == "+":
            direction = rotate_x(direction, alpha)
        elif x == "-":
            direction = rotate_x(direction, -alpha)
        elif x == "&":
            direction = rotate_y(direction, alpha)
        elif x == "^":
            direction = rotate_y(direction, -alpha)
        elif x == "<":
            direction = rotate_z(direction, alpha)
        elif x == ">":
            direction = rotate_z(direction, -alpha)
        elif x == "*":
            direction = direction * delta
        elif x == "/":
            direction = direction / delta
        elif x == "|":
            direction = rotate_x(direction, 180)
            direction = rotate_y(direction, 180)
            direction = rotate_z(direction, 180)
        elif x == "[":
            saved_states.append((pos, direction))
            fStat += [f]
            topStack += [None]
        elif x == "]":
            pos, direction = saved_states.pop()
            f = fStat.pop()
            # change top is topBark
            if topStack != [] and topStack[-1] is not None:
                topStack[-1].type = "topBark"
                if len(leafStack) >= 2:
                    leafStack[-1].type = "topLeaf"
                    leafStack[-2].type = "topLeaf"
                    leafStack = []
            if len(topStack) != 0:
                top = topStack.pop()
        elif x in colors:
            color = colors[x]
        elif x.isnumeric():
            currentNo = int(x)
    prev.core = False
    axis_equal_3d(ax)
    if not gen:
        plt.show()
    return allPos
