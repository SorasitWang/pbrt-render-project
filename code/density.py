import numpy as np
import math
from util import *
import animate as an
from multiprocessing import Pool

pbrtFaces = []
pbrtVertices = []
vertices = []
faces = []
OFFSET = 0.5
bb = []
initSize = {
    "bark": scale_matrix(0.8, 0.8, 1),
    "topBark": scale_matrix(0.8, 0.8, 0.5),
    "leaf": scale_matrix(1.5, 1.5, 1.5),
    "topLeaf": scale_matrix(1.5, 1.5, 1.5),
    "sect": scale_matrix(0.1, 0.1, 1),
}


def createVexel():
    # return vexel, num, bb
    global bb
    vexel = []

    x = [-6, 6.2, 0.1]
    y = [0, 6.25, 0.1]
    z = [-2, 22, 0.1]

    bb = [x[0], x[1], y[0], y[1], z[0], z[1]]

    numX = len(np.arange(x[0], x[1], x[2]))
    numY = len(np.arange(y[0], y[1], y[2]))
    numZ = len(np.arange(z[0], z[1], z[2]))

    for k in np.arange(z[0], z[1], z[2]):
        for j in np.arange(y[0], y[1], y[2]):
            for i in np.arange(x[0], x[1], x[2]):
                vexel.append([i, j, k])

    return vexel, [numX, numY, numZ], [[x[0], y[0], z[0]], [x[1], y[1], z[1]]]


def writeVolume(num, bb, density, file, name):
    density: np
    pMin, pMax = bb
    nx, ny, nz = num

    init = """MakeNamedMedium "{}"   "string type" [ "heterogeneous" ] """.format(name)
    resol = """"integer nx" {x} "integer ny" {y} "integer nz" {z}\n""".format(
        x=int(nx), y=int(ny), z=int(nz)
    )
    box = """"point p0" {pMin} "point p1" {pMax}\n""".format(
        pMin=str(pMin), pMax=str(pMax)
    )
    densityTxt = """"float density" [{density}]\n""".format(
        density=" ".join(str(item) for item in density)
    )
    text = init + resol + box + densityTxt
    with open("D:/univ/4/PBR/l/res/{}.pbrt".format(file), "w+") as f:
        f.seek(0)
        f.write(text)
        f.close()


class Shape:
    def __init__(self, type, center, axis, angle, scale, initScale):
        self.type = type
        self.center = center
        self.axis = axis
        self.angle = angle

        self.scale = scale
        t = translate_matrix(center)
        if angle == 0:
            r = np.identity(4)
        else:
            r = rotation_matrix(axis, math.radians(angle))
        s = scale_matrix(scale[0], scale[1], scale[2])

        self.m = np.matmul(t, np.matmul(r, np.matmul(s, initScale)))
        self.mInv = np.identity(4)
        self.mInv = np.linalg.inv(self.m)
        self.localBb = [
            self.center[0] - max(2, scale[0]),
            self.center[0] + max(2, scale[0]),
            self.center[1] - max(2, scale[1]),
            self.center[1] + max(2, scale[1]),
            self.center[2] - max(2, scale[2]),
            self.center[2] + max(2, scale[2]),
        ]


def insideBb(pos, bb):
    return (
        pos[0] > bb[0]
        and pos[0] < bb[1]
        and pos[1] > bb[2]
        and pos[1] < bb[3]
        and pos[2] > bb[4]
        and pos[2] < bb[5]
    )


class Object:
    def __init__(self):
        # minX, maxX, minY, maxY, minZ, maxZ
        self.boundingBox = np.array([10000, -10000, 10000, -10000, 10000, -10000])
        self.shapes = []
        self.offset = 5
        self.isAddOffset = False

        self.mInv = np.identity(4)
        self.m = np.identity(4)

    def addTransform(self, center, rotates, scale, have=True):

        """
        type
        param
            translate = [x,y,z]
            rotate = [x,y,z,angle]
            scale = [x,y,z]
        """
        if have:
            t = translate_matrix(center)
            r2 = np.identity(4)
            r1 = rotation_matrix(rotates[0][0], math.radians(rotates[0][1]))
            if len(rotates) == 2:
                r2 = rotation_matrix(rotates[1][0], math.radians(rotates[1][1]))
            s = scale_matrix(scale[0], scale[1], scale[2])

            m = np.matmul(t, np.matmul(r2, np.matmul(r1, s)))
        else:
            m = np.identity(4)

        for shape in self.shapes:
            shape.m = np.matmul(m, shape.m)
            newCenter = shape.m.dot(shape.mInv.dot(np.array(shape.center + [1])))[:-1]
            shape.mInv = np.linalg.inv(shape.m)

            box = self.boundingBox
            self.boundingBox = [
                min(box[0], newCenter[0]),
                max(box[1], newCenter[0]),
                min(box[2], newCenter[1]),
                max(box[3], newCenter[1]),
                min(box[4], newCenter[2]),
                max(box[5], newCenter[2]),
            ]
            shape.center = newCenter.tolist()
            shape.localBb = [
                shape.center[0] - max(2, 2 * shape.scale[0]),
                shape.center[0] + max(2, 2 * shape.scale[0]),
                shape.center[1] - max(2, 2 * shape.scale[1]),
                shape.center[1] + max(2, 2 * shape.scale[1]),
                shape.center[2] - max(2, 2 * shape.scale[2]),
                shape.center[2] + max(2, 2 * shape.scale[2]),
            ]

    def inside(self, pos):
        if not self.isAddOffset:
            self.isAddOffset = True
            self.boundingBox += np.array(
                [
                    -self.offset,
                    self.offset,
                    -self.offset,
                    self.offset,
                    -self.offset,
                    self.offset,
                ]
            )
        bb = self.boundingBox
        return insideBb(pos, bb)

    def addShape(self, shape):
        shape: Shape
        self.shapes.append(shape)


faces = []
vertices = []
objects = []


def dot2(v):
    return np.dot(v, v)


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def halfSigmoid(x, a):
    if x >= 0.3:
        return 1
    return 2 / (1 + math.exp(-a * x)) - 1


def udSphere(p):
    r = 1
    return np.linalg.norm(p) - r


def udCylinder(p):

    h = 1
    r = 1
    t = np.array([np.linalg.norm(np.array([p[0], p[2]])), p[1]])
    t[t < 0] = 0
    d = t - np.array([h, r])
    tmp = min(max(d[0], d[1]), 0.0)
    d[d < 0] = 0
    return tmp + np.linalg.norm(d)


def udLeaf(p):
    """
    p : point position in object space
    """

    # radius and height
    h = [1, 0.15]
    k = np.array([-0.8660254, 0.5, 0.57735])
    p = np.abs(p)
    p[0:2] -= 2 * min(np.dot(k[0:2], p[0:2]), 0.0) * k[0:2]

    d = np.array(
        [
            np.linalg.norm(
                p[0:2] - np.array([np.clip(p[0], -k[2] * h[0], k[2] * h[0]), h[0]])
            )
            * np.sign(p[1] - h[0]),
            p[2] - h[1],
        ]
    )
    aComponent = min(max(d[0], d[1]), 0.0)
    d[d < 0] = 0
    return aComponent + np.linalg.norm(d)


def udBark(p):
    h = 2
    r = 1
    d = np.abs(np.array([np.linalg.norm(np.array([p[0], p[2]])), p[1]])) - np.array(
        [h, r]
    )
    aComponent = min(max(d[0], d[1]), 0.0)
    d[d < 0] = 0
    return aComponent + np.linalg.norm(d)


def udTriangle(p, a, b, c):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = np.cross(ba, ac)

    return math.sqrt(
        min(
            min(
                dot2(ba * clamp(np.dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                dot2(cb * clamp(np.dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb),
            ),
            dot2(ac * clamp(np.dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc),
        )
        if np.sign(np.dot(np.cross(ba, nor), pa))
        + np.sign(np.dot(np.cross(cb, nor), pb))
        + np.sign(np.dot(np.cross(ac, nor), pc))
        < 2.0
        else np.dot(nor, pa) * np.dot(nor, pa) / dot2(nor)
    )


def udQuad(p, a, b, c, d):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    dc = d - c
    pc = p - c
    ad = a - d
    pd = p - d
    nor = np.cross(ba, ad)
    return np.sqrt(
        min(
            min(
                min(
                    dot2(ba * clamp(np.dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                    dot2(cb * clamp(np.dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb),
                ),
                dot2(dc * clamp(np.dot(dc, pc) / dot2(dc), 0.0, 1.0) - pc),
            ),
            dot2(ad * clamp(np.dot(ad, pd) / dot2(ad), 0.0, 1.0) - pd),
        )
        if (
            np.sign(np.dot(np.cross(ba, nor), pa))
            + np.sign(np.dot(np.cross(cb, nor), pb))
            + np.sign(np.dot(np.cross(dc, nor), pc))
            + np.sign(np.dot(np.cross(ad, nor), pd))
            < 3.0
        )
        else np.dot(nor, pa) * np.dot(nor, pa) / dot2(nor)
    )


def udBox(p, b):

    q = np.abs(p) - b
    tmp = min(max(q.x, max(q.y, q.z)), 0.0)
    q[q < 0] = 0
    return np.linalg.norm(q) + tmp


def checkObjects(pos):
    minObj = 100000
    for obj in objects:
        obj: Object
        if False:
            pass
        if not obj.inside(pos):
            # print(pos)
            continue
        else:
            for shape in obj.shapes:
                shape: Shape

                if not insideBb(pos[:-1], shape.localBb):
                    continue
                objPos = shape.mInv.dot(pos)[:-1]

                if "eaf" in shape.type:
                    tmpMin = udLeaf(objPos)
                elif "ark" in shape.type:
                    tmpMin = udBark(objPos)
                elif "sphere" in shape.type:
                    tmpMin = udSphere(objPos)

                elif "Box" in shape.type:
                    tmpMin = udBox(objPos)
                elif "cylinder" in shape.type:
                    tmpMin = udCylinder(objPos)

                # print(tmpMin)
                if tmpMin < minObj:
                    minObj = tmpMin
                    thatShape = shape
                    # it is inside some shape, can suddenly return
                    if minObj < 0:
                        # print(minObj)
                        return 0
                    # accepted small distance , no neccessary to find whether there are smaller distance
                    elif minObj < 0.1:
                        return 1
    if minObj < 0:
        return 0
    if minObj > 0.3:
        minObj = 3
    return minObj


def loopFindDist(posArr, num, bb):

    global maxY, faces, vertices

    denseMatrix = np.zeros(num[0] * num[1] * num[2])
    # print(num, len(posArr))
    idx = 0
    offset = 1
    maxX_ = maxX + offset
    minX_ = minX - offset
    maxY_ = maxY + offset
    minY_ = minY - offset
    maxZ_ = maxZ + offset
    minZ_ = minZ - offset

    for idx, pos in enumerate(posArr):

        minGround = 100000
        pos = np.array(pos + [1])
        # ground

        # since terrain don't have transformation
        pos = pos[:-1]

        if (
            pos[0] > maxX_
            or pos[0] < minX_
            or pos[1] > maxY_
            or pos[1] < minY_
            or pos[2] > maxZ_
            or pos[2] < minZ_
        ):
            minGround = 1
        else:
            minGround = checkGround(pos)

        if minGround > 1:
            minGround = 1
        calDist = 1 - minGround
        # other shape

        minObj = checkObjects(pos)
        calDist = min(calDist, minObj)
        # if calDist == 1:
        #     print(calDist)
        # an.addCoord(pos[0], pos[1], pos[2], calDist)
        denseMatrix[idx] = halfSigmoid(calDist, 5)
        idx += 1

    writeVolume(num, bb, denseMatrix, "fog_src_test_allFog", "fog")
    return denseMatrix


groundM = rotation_matrix([0, 0, 1], 180)
groundMInv = np.linalg.inv(groundM)


def checkGround(pos):

    global pbrtVertices, faces, pbrtFaces, groundM
    pos = groundMInv.dot(pos)[:-1]
    minGround = 1000
    w = 1.3
    # print(len(pbrtVertices))
    distance = np.zeros(len(pbrtFaces))
    for faces in pbrtFaces:
        minGround_ = w * minGround
        v = [0, 0, 0]
        for i in range(3):
            if distance[faces[i]] == 0:
                distance[faces[i]] = np.linalg.norm(pbrtVertices[faces[i]] - pos)
            v[i] = distance[faces[i]]
        if v[0] > minGround_ and v[1] > minGround_ and v[2] > minGround_:
            continue
        # print(pos, vertexL)
        minGround = min(
            minGround,
            # (v[0] + v[1] + v[2]) / 3
            udTriangle(
                pos,
                pbrtVertices[faces[0]],
                pbrtVertices[faces[1]],
                pbrtVertices[faces[2]],
            ),
        )
        if minGround < 0:
            return 0

    return min(minGround, 1)  # , minGround


def addShapes(val: Shape):
    global shapes
    shapes.append(val)


def addObject(val: Object):
    global objects
    objects.append(val)


def printShapes():
    global shapes
    print(shapes)


def readObj():
    minX = 10000
    maxX = -10000
    minY = 10000
    maxY = -10000
    minZ = 10000
    maxZ = -10000

    global vertices, faces
    import re

    flt = r"-?[\d\.]+"
    vertex_pattern = r"v\s+(?P<x>{})\s+(?P<y>{})\s+(?P<z>{})".format(flt, flt, flt)
    face_pattern = r"f\s+((\d+/\d+/\d+\s*){3,})"

    with open("D:/univ/4/PBR/l/res/ground1.obj", "r") as file:
        for line in map(str.strip, file):
            match = re.match(vertex_pattern, line)
            if match is not None:
                vertices.append(tuple(map(float, match.group("x", "y", "z"))))
                if vertices[-1][0] > maxX:
                    maxX = vertices[-1][0]
                elif vertices[-1][0] < minX:
                    minX = vertices[-1][0]

                if vertices[-1][1] > maxY:
                    maxY = vertices[-1][1]
                elif vertices[-1][1] < minY:
                    minY = vertices[-1][1]

                if vertices[-1][2] > maxZ:
                    maxZ = vertices[-1][2]
                elif vertices[-1][2] < minZ:
                    minZ = vertices[-1][2]

            match = re.match(face_pattern, line)
            if match is not None:
                faces.append(
                    tuple(
                        tuple(map(int, vertex.split("/")))
                        for vertex in match.group(1).split()
                    )
                )
    print(maxY)
    print(f"The first face has {len(faces[0])} vertices:")
    for vertex in faces[0]:
        print(vertex)

    print(
        "\nWe do not care about the texture coordinate indices or normal indices, so we get the three vertex indices:"
    )
    for vertex in faces[0]:
        print(vertex[0])

    print(
        "\nAfter substituting the indices for their values, the face is made up of these coordinates:"
    )
    return minX, maxX, minY, maxY, minZ, maxZ


def readPbrt():
    minX = 10000
    maxX = -10000
    minY = 10000
    maxY = -10000
    minZ = 10000
    maxZ = -10000

    global vertices, faces, pbrtVertices, pbrtFaces
    import re

    inPos = False
    inFace = False
    with open("D:/univ/4/PBR/l/res/stone/terrain_only.pbrt", "r") as file:
        for line in map(str.strip, file):

            if "point3" in line:
                inPos = True
            elif inPos:
                if "]" in line:
                    inPos = False
                    continue
                pbrtVertices.append(np.array(line.split(" ")).astype(np.float))
                pbrtVertices[-1][1], pbrtVertices[-1][2] = (
                    pbrtVertices[-1][2],
                    pbrtVertices[-1][1],
                )
                if pbrtVertices[-1][0] > maxX:
                    maxX = pbrtVertices[-1][0]
                elif pbrtVertices[-1][0] < minX:
                    minX = pbrtVertices[-1][0]

                if pbrtVertices[-1][1] > maxY:
                    maxY = pbrtVertices[-1][1]
                elif pbrtVertices[-1][1] < minY:
                    minY = pbrtVertices[-1][1]

                if pbrtVertices[-1][2] > maxZ:
                    maxZ = pbrtVertices[-1][2]
                elif pbrtVertices[-1][2] < minZ:
                    minZ = pbrtVertices[-1][2]
            elif "indices" in line:
                inFace = True
            elif inFace:
                if "]" in line:
                    inFace = False
                    break
                pbrtFaces.append(np.array(line.split(" ")).astype(np.int32))

    return minX, maxX, minY, maxY, minZ, maxZ


src = ["D:/univ/4/PBR/l/res/bark_src.pbrt"] * 2
src += ["D:/univ/4/PBR/l/res/bamboo_src.pbrt"] * 7

treeTransform = [
    [
        [3, 2, 10],
        [
            [
                [
                    -1,
                    0,
                    0,
                ],
                90,
            ]
        ],
        [0.1, 0.1, 0.1],
    ],
    [
        [-9, 3.4, 3],
        [
            [
                [
                    -1,
                    0,
                    0,
                ],
                90,
            ],
            [
                [
                    0,
                    0,
                    1,
                ],
                -70,
            ],
        ],
        [0.1, 0.1, 0.1],
    ],
]

bambooTransform = []
objTransform = treeTransform + bambooTransform


def wrapAddTransform(obj, t):
    """
    t : translate,axis,angle,scale
    """
    obj: Object
    obj.addTransform(t[0], t[1], t[2])


def srcPbrt():
    global objects
    for i in range(len(objTransform)):
        obj = Object()
        center = [0, 0, 0]
        axis = [1, 0, 0]
        angle = 0
        scale = [1, 1, 1]
        type = ""
        print(src[i])
        with open(src[i], "r") as file:
            for line in map(str.strip, file):
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    continue
                p = line.split()

                if len(p) > 1:
                    if p[0] == "ObjectInstance":
                        type = p[1][1:-1]
                    else:
                        p[1 : len(p) + 1] = list(map(float, p[1 : len(p) + 1]))
                        if p[0] == "Translate":
                            center = p[1:4]
                        elif p[0] == "Scale":
                            scale = p[1:4]
                        elif p[0] == "Rotate":
                            axis = p[2:5]
                            angle = p[1]

                elif "End" in line:

                    obj.addShape(
                        Shape(type, center, axis, angle, scale, initSize[type])
                    )
        wrapAddTransform(obj, objTransform[i])
        objects.append(obj)


# Sphere
def additionShape():
    global objects
    mon = Object()
    mon.addShape(
        Shape(
            "sphere", [-3.05, 3.8, 16], [1, 0, 0], 0, [0.75, 0.75, 0.75], np.identity(4)
        )
    )
    objects.append(mon)
    mon.addTransform(None, None, None, False)

    pond = Object()
    pond.addShape(
        Shape("sphere", [3, 1.4, 8], [0, 0, 1], 0, [3.75, 0.15, 3.75], np.identity(4))
    )
    objects.append(pond)
    pond.addTransform(None, None, None, False)

    b1 = Object()
    b1.addShape(
        Shape("sphere", [0, 2.5, 12], [0, 1, 0], -60, [3, 1, 1], np.identity(4))
    )
    objects.append(b1)
    b1.addTransform(None, None, None, False)

    b2 = Object()
    b2.addShape(Shape("sphere", [2, 2.5, 6], [0, 1, 0], 0, [1, 1, 1], np.identity(4)))
    objects.append(b2)
    b2.addTransform(None, None, None, False)

    b3 = Object()
    b3.addShape(
        Shape("sphere", [2, 1.5, 4], [0, 1, 0], 20, [4, 1, 1.5], np.identity(4))
    )
    objects.append(b3)
    b3.addTransform(None, None, None, False)

    b4 = Object()
    b4.addShape(
        Shape("sphere", [-2, 2.5, 7], [0, 1, 0], 20, [0.5, 0.5, 1.5], np.identity(4))
    )
    objects.append(b4)
    b4.addTransform(None, None, None, False)


if __name__ == "__main__":
    additionShape()
    srcPbrt()
    vexel, num, bb = createVexel()
    loopFindDist(vexel, num, bb)
