import numpy as np
import math


def to_radian(angle):
    return angle * np.pi / 180


def rotate(vector, angle):
    theta = to_radian(angle)
    matrix = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return np.dot(matrix, vector)


def rotate_x(vector, angle):
    theta = to_radian(angle)
    matrix = [
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)],
    ]
    return np.dot(matrix, vector)


def rotate_y(vector, angle):
    theta = to_radian(angle)
    matrix = [
        [np.cos(theta), 0, -np.sin(theta)],
        [0, 1, 0],
        [np.sin(theta), 0, np.cos(theta)],
    ]
    return np.dot(matrix, vector)


def rotate_z(vector, angle):
    theta = to_radian(angle)
    matrix = [
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ]
    return np.dot(matrix, vector)


class ShapeL:
    def __init__(self, type, prevPos, pos, child):
        self.type = type
        self.pos = pos
        self.prevPos = prevPos
        self.child = child
        self.core = True


def get_intersections(x0, y0, r0, x1, y1, r1):

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # none intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(r0**2 - a**2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return (x3, y3, x4, y4)


class TreeComponent:
    def __init__(self, position, rotation, scale):
        self.position = position
        self.rotation = rotation
        self.scale = scale


def modifyFile(name, dim, value):
    path = "D:/univ/4/PBR/l/pbrt"
    startIdx = -1
    endIdx = -1
    with open("{}/{}.pbrt".format(path, name), "r+") as f:
        lines = f.read().split("\n")
        for idx, line in enumerate(lines):
            if "start{}d".format(dim) in line:
                # print(line)
                startIdx = idx
            if startIdx != -1:
                # find end after found start
                if "end{}d".format(dim) in line:
                    endIdx = idx
        if startIdx == -1 or endIdx == -1:
            print("Invalid file : no start/end label")
            return
        newTxt = "\n".join(lines[: startIdx + 1])
        newTxt += "\n" + value
        newTxt += "\n".join(lines[endIdx:])
        f.seek(0)
        f.write(newTxt)
        f.close()


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
            [0, 0, 0, 1],
        ]
    )


def translate_matrix(t):
    return np.array([[1, 0, 0, t[0]], [0, 1, 0, t[1]], [0, 0, 1, t[2]], [0, 0, 0, 1]])


def scale_matrix(sx, sy, sz):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """

    return np.array([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]])
