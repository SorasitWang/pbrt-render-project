from enum import Enum
from turtle import distance
import numpy as np
from util import *
import math
from density import Shape, Object, addObject, loopFindDist, createVexel

atrbBegin = "AttributeBegin\n"
atrbEnd = "AttributeEnd\n"
debugShape = """Shape "sphere" "float radius" [0.05]"""

initSize = {
    "bark": scale_matrix(0.6, 0.6, 1),
    "topBark": scale_matrix(0.6, 0.6, 0.5),
    "leaf": scale_matrix(1.5, 1.5, 1.5),
    "topLeaf": scale_matrix(1.5, 1.5, 1.5),
    "sect": scale_matrix(0.1, 0.1, 1),
}


def init():
    bark = """ObjectBegin "bark"\n\tMaterial "matte" "rgb Kd" [1 0 0]\n\tScale 0.3 0.3 1\n\tInclude "./res/bark.pbrt\n\nObjectEnd\n"""
    leaf = """ObjectBegin "leaf"\n\tMaterial "matte" "rgb Kd" [0 1 0]\n\tShape "cylinder" "float radius" [0.05]\nObjectEnd\n"""
    return bark + leaf


def wrapAtrb(value):
    return atrbBegin + value + atrbEnd


def findRotationParam(posA, posB):
    """ 
        https://math.stackexchange.com/questions/3268979/rotate-one-3d-vector-on-another
    """
    destVec = posA - posB
    destVec[destVec == 0] = 0.0000000000001
    # normalize
    destVec = destVec / np.sqrt(np.sum(destVec**2))
    originVec = np.asarray([0, 1, 0])
    crossVec = np.cross(originVec, destVec)
    axis = crossVec / (np.linalg.norm(crossVec))
    angle = math.degrees(np.arcsin(np.clip(np.linalg.norm(crossVec), -1, 1)))
    if destVec[1] < 0:
        angle += 180
        axis *= -1
    return axis, angle


def genPbrt(pos, step, threeD=False):

    """
    pos : [[x,y,z]] => [[1.5, 3 , 0.5]]
    """
    shapes = []
    instancing = ""
    debug = """Material "matte" "rgb Kd" [1 0 0]\n"""
    obj = Object()
    addObject(obj)
    # print(pos)
    for p in pos[0:]:
        p: ShapeL
        p.pos = np.asarray(p.pos)

        # order of operation scale=>rotate=>translate , order of command will be inversed

        # 1. find middle point and distance between 2 pos
        if p.type == "leaf":
            center = (2 * p.pos + 0 * p.prevPos) / 2
        elif p.type == "topLeaf":
            center = (0.5 * p.pos + 1.5 * p.prevPos) / 2
        else:
            center = (p.pos + p.prevPos) / 2
        distance = np.linalg.norm(p.pos - p.prevPos)
        # 2. find degree of rotation and axis
        axis, angle = findRotationParam(p.pos, p.prevPos)
        if distance == 0:
            continue
        child = pow(0.98, p.child)
        # child = 1
        size = [child, np.around(distance / 2.0, 5), child]
        if p.type == "leaf":
            # for leaf, more decreasing size
            child = 1.5 * pow(child, 1.05)
            size = [child, child, child]
        scale = "Scale {} {} {}\n".format(size[0], size[1], size[2])
        axis = np.around(axis, 5)
        center = np.around(center, 5)
        rotate = "Rotate {} {} {} {}\n".format(
            round(angle, 3), axis[0], axis[1], axis[2]
        )

        one = ""

        if threeD:
            translate = "Translate {} {} {}\n".format(center[0], center[1], center[2])
        else:
            translate = "Translate {} {} 0\n".format(center[0], center[1])
        one += translate
        one += rotate
        one += scale
        if p.type == "leaf":
            one += """\tObjectInstance "leaf"\n"""
        elif p.type == "topBark" and not p.core:
            one += """\tObjectInstance "topBark"\n"""
        elif p.type == "topLeaf":
            one += """\tObjectInstance "topLeaf"\n"""
        else:
            one += """\tObjectInstance "bark"\n"""
        obj.addShape(
            Shape(p.type, center, axis, round(angle, 3), size, initSize[p.type])
        )
        instancing += wrapAtrb(one)

        ball = """Translate {} {} {} \n""".format(
            (p.prevPos[0] + p.pos[0]) / 2,
            (p.prevPos[1] + p.pos[1]) / 2,
            0 if not threeD else (p.prevPos[2] + p.pos[2]) / 2,
        )

        ball = """Translate {} {} {} \n""".format(
            p.pos[0],
            p.pos[1],
            0 if not threeD else p.pos[2],
        )

        ball += """Shape "sphere" "float radius" [0.08]\n"""
        debug += wrapAtrb(ball)

    vexel, num, bb = createVexel()
    # loopFindDist(vexel, num, bb)
    modifyFile("density", 3, instancing)
    return init() + instancing


def genPbrtBamboo(pos, step, threeD=False):

    """
    pos : [[x,y,z]] => [[1.5, 3 , 0.5]]
    """
    shapes = []
    instancing = ""
    debug = """Material "matte" "rgb Kd" [1 0 0]\n"""
    obj = Object()
    addObject(obj)
    # print(pos)
    for p in pos[0:]:
        p: ShapeL
        p.pos = np.asarray(p.pos)

        # order of operation scale=>rotate=>translate , order of command will be inversed

        # 1. find middle point and distance between 2 pos
        # print(p.pos, lastP.pos)

        center = (p.pos + p.prevPos) / 2
        distance = np.linalg.norm(p.pos - p.prevPos)
        # print(distance, p.pos - p.prevPos, center)
        # 2. find degree of rotation and axis
        axis, angle = findRotationParam(p.pos, p.prevPos)
        if distance == 0:
            continue
        child = pow(0.98, p.child)
        # child = 1
        size = [child, np.around(distance / 2.0, 5), child]
        scale = "Scale {} {} {}\n".format(size[0], size[1], size[2])
        axis = np.around(axis, 5)
        center = np.around(center, 5)
        rotate = "Rotate {} {} {} {}\n".format(
            round(angle, 3), axis[0], axis[1], axis[2]
        )

        one = ""

        if threeD:
            translate = "Translate {} {} {}\n".format(center[0], center[1], center[2])   
        else:
            translate = "Translate {} {} 0\n".format(center[0], center[1])
        one += translate
        one += rotate
        one += scale
        obj.addShape(
            Shape(p.type, center, axis, round(angle, 3), size, initSize[p.type])
        )
        one += """\tObjectInstance "sect"\n"""
        instancing += wrapAtrb(one)

        ball = """Translate {} {} {} \n""".format(
            (p.prevPos[0] + p.pos[0]) / 2,
            (p.prevPos[1] + p.pos[1]) / 2,
            0 if not threeD else (p.prevPos[2] + p.pos[2]) / 2,
        )

        ball = """Translate {} {} {} \n""".format(
            p.pos[0],
            p.pos[1],
            0 if not threeD else p.pos[2],
        )

        ball += """Shape "sphere" "float radius" [0.08]\n"""
        debug += wrapAtrb(ball)
    # loopFindDist(createVexel())
    modifyFile("bamboo_src", 3, instancing)
    return init() + instancing

