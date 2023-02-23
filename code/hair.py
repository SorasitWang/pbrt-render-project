from tokenize import Number
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
import math
from matplotlib import animation


# https://davis.wpi.edu/~matt/courses/hair/hrstlmodeling.html

# gravityDir must be normalize

gravityForce = 1

D = 1
d = D
const = 10
length = 3  # 8
THETASTART = math.pi / 4
THETASTEP = math.pi / 2
THETAEND = 5 * math.pi / 4
ITER = 3
dampling = 0.99
radius = 1


class Particle:
    def __init__(self, dim, idx, pos):
        self.force: np = np.zeros((dim))
        self.pos: np = pos
        self.idx: int = idx
        self.velocity: np = np.zeros((dim))
        self.correction: np = np.zeros((dim))


class Strand:
    def __init__(self, dim, origin, d):
        self.particles = []
        self.origin = origin
        self.d = d

    def addParticle(self, p: Particle):
        self.particles.append(p)

    def addForce(self, force: list):
        force = np.array(force)
        for i in range(2, len(self.particles)):
            self.particles[i].force += force


def exportStrand(strands):
    pbrTxt = ""
    end = "0.005"
    start = "0.08"
    for strand in strands:
        strand: Strand
        mergePos = ""
        for particle in strand.particles:
            particle: Particle
            mergePos += "{} {} {} ".format(
                particle.pos[0], particle.pos[2], particle.pos[1]
            )
        # print(str(joints))

        pbrTxt += (
            'Shape "curve"\n'
            + '"float width1" [{}]\n'.format(end)
            + '"float width0" [{}]\n'.format(start)
            + '"point3 P" [{}]\n'.format(mergePos)
        )
    path = "D:/univ/4/PBR/l/pbrt"
    with open("{}/{}.pbrt".format(path, "hair_src2"), "w+") as f:
        f.write(pbrTxt)
        f.close()
    print(pbrTxt)


# Shape "curve"
#     "float width1" [ 0.000007 ]
#     "float width0" [ 0.00008 ]
#     "point3 P" [ 0.005968 0.095212 -0.003707 0.006467 0.099219 -0.005693 0.007117
#                  0.102842 -0.008281 0.007995 0.105534 -0.011779 ]

windForce = np.array([1, 0, 0])
windVelo = np.array([1, 0, 0])
timeStep = 0.2


def movement(x):
    return x + timeStep * windVelo + pow(timeStep, 2) * windForce


def moment(tangent, i, dim=3):
    global d, k
    gravityDir = np.array([0, 0, -1])
    if dim == 2:
        gravityDir = np.array([0, -1])
    # tangent should be normalized

    # up = gravityForce * 100 / (1 + math.exp(-1.1 * i + 12))
    up = 0  # gravityForce * 500 / (1 + math.exp(-4 * i + 15))
    gForce = (
        -gravityForce
        * d
        * np.cos(
            np.dot(tangent, gravityDir) / (np.linalg.norm(tangent))
        )  # cos (angle between force and hair)
        / 2
        * pow(length - i + 1, 2)
    )
    # print(i, gForce, up)
    # if i >= 7:
    #     return gravityForce * i
    return gForce + up


def displacement(tangent, i, dim):
    # print(moment(tangent, i))
    return -0.5 * moment(tangent, i, dim) / const * pow(d, 2)


def calculateForce(idx, p: Particle, dim):
    if idx <= 1:
        # root can not move
        return np.zeros(dim)
    # windForce + hardcode gravity
    # Actually, windForce is acc because not mutliply with mass (=1)
    # but gravity need to calculate mass from the rest particles on strand
    return windForce + 0.1 * np.array([0, 0, 1])
  
 

def dynamic(hair: Strand, dim):
    """ """
    for idx, p in enumerate(hair.particles):
        p: Particle
        p.force = calculateForce(idx, p, dim)
        backup = np.copy(p.pos)

        p.pos = p.pos + timeStep * p.velocity + pow(timeStep, 2) * p.force
        static(hair, idx, dim)
        p.velocity = (p.pos - backup) / timeStep
        if idx < len(hair.particles) - 1:
            p.velocity += -dampling * hair.particles[idx + 1].correction / timeStep
        p.correction = p.pos - backup
        p.force = 0
    return


def static(hair: Strand, i, dim):
    """
        1. move pi+1 to resolve distant const with pi => move to nearest point on suface of sphere
        2. if that pi+1 collision, resolve => find intersect between head and surface of sphere that nearest
    """

    def circleInter(theta, anchor, dist=True):
        p_i = c_i + r_i * (t * np.cos(theta) + b * np.sin(theta))
        if not dist:
            return p_i
        return np.linalg.norm(p_i - anchor)

    particles = hair.particles
    p1: np = particles[i].pos
    p0: np = particles[i - 1].pos
    if i == 0:
        return p1
    center = np.array([0] * dim)

    # dist constraint
    dist = np.linalg.norm(p1 - p0)
    if dist > hair.d + 0.0001:
        p1 = p0 + ((p1 - p0) / dist) * d
    particles[i].pos = p1
    # col constraint
    dist = np.linalg.norm(p1 - center)
    if dist < radius:
        # resolve col

        if dim == 2:
            col = get_intersections(p0[0], p0[1], d, center[0], center[1], radius)
            if col == None:
                # should not happen
                print(col)
                print("Not col Catch")
            else:
                if np.linalg.norm(np.array(col[0:2]) - p1) < np.linalg.norm(
                    np.array(col[2:4]) - p1
                ):
                    p1 = np.array(col[0:2])
                else:
                    p1 = np.array(col[2:4])
        else:
            # https://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection
            # https://stackoverflow.com/questions/60717881/sphere-sphere-intersection
            # distant graph
            # https://www.desmos.com/calculator/wkxbegmmya?lang=th
            h = 0.5 + (pow(radius, 2) - pow(d, 2)) / (2 * pow(dist, 2))
            c_i = center + h * (p0 - center)
            # print(i, p1, p0, radius, h, dist, d)
            if radius * radius - pow(h, 2) * pow(dist, 2) >= 0:
                print(i, np.linalg.norm(p1 - p0), d)
                r_i = math.sqrt(radius * radius - pow(h, 2) * pow(dist, 2))
                n_i = (p0 - center) / d
                axis = np.array([1, 0, 0])

                if (n_i / np.linalg.norm(n_i) == axis).all():
                    axis = np.array([0, 0, 1])
                t = np.cross(axis, n_i)
                b = np.cross(t, n_i)

                # mid_point
                thetaStart = THETASTART
                thetaEnd = THETAEND
                for _ in range(ITER):
                    reStart = circleInter(thetaStart, p1)
                    reEnd = circleInter(thetaEnd, p1)
                    if reStart > reEnd:
                        thetaEnd = (thetaEnd + thetaStart) / 2
                    else:
                        thetaStart = (thetaEnd + thetaStart) / 2
                p1 = (
                    circleInter(thetaStart, None, False)
                    if reStart < reEnd
                    else circleInter(thetaEnd, None, False)
                )
    particles[i].pos = p1


def colHandler(p, dim):
    """
        return False,numpy[3] if p is not inside head model
        return True,numpy[3] for nearest point on the head's surface that nearest p

        https://math.stackexchange.com/questions/324268/how-do-i-find-the-nearest-point-on-a-sphere
    """
    # For now assume head is sphere (should be ellipsoid)
    center = np.array([0] * dim)
    radius = 1
    dist = np.linalg.norm(p - center)
    if dist < radius:
        # should return intersect point of circle/sphere and circle/sphere/ellipsoid, but maybe costly
        # just find neares point on head model
        return True, (p - center) / dist * radius
    return False, np.zeros(3)

def render(hair: Strand, dim):
    particles = hair.particles
    for i in range(0, len(particles) - 1):
        if dim == 3:
            ax.plot(
                [particles[i].pos[0], particles[i + 1].pos[0]],
                [particles[i].pos[1], particles[i + 1].pos[1]],
                [particles[i].pos[2], particles[i + 1].pos[2]],
                c="#FF0000",
            )
        else:
            plt.plot(
                [particles[i].pos[0], particles[i + 1].pos[0]],
                [particles[i].pos[1], particles[i + 1].pos[1]],
                c="#FF0000",
            )


def outStyle(hair: Strand):
    offset = 1
    for p in hair.particles:
        p: Particle
        if p.pos[2] > center[2] + radius + offset:
            return True
    return False


def gloom(hair: Strand):
    if dim < 2 or dim > 3:
        print("Error dimension")
        return
    # gravityDir = np.array([0, 0, -1])
    gravityDir = np.array([0] * (dim - 1) + [-1])
    origin = hair.origin
    # origin is also normal since sphere's radius is 1
    start = origin + hair.d * origin
    tangent = origin
    hair.addParticle(Particle(dim, 0, origin))
    hair.addParticle(Particle(dim, 1, start))
    for i in range(2, length):

        y = displacement(tangent, i, dim)

        yi = hair.particles[-1].pos + y * gravityDir
        ei = yi - hair.particles[-2].pos

        tangent = ei / np.linalg.norm(ei)
        pos_ = hair.particles[-1].pos + hair.d * tangent

        hair.addParticle(Particle(dim, i, pos_))
        static(hair, i, dim)
        tangent = (hair.particles[-1].pos - hair.particles[-2].pos) / np.linalg.norm(
            hair.particles[-1].pos - hair.particles[-2].pos
        )   
        tangent[dim - 1] = -abs(tangent[dim - 1])

    if outStyle(hair):
        return False
    return True

if __name__ == "__main__":
    dim = 3
    offset = np.array([0.5] * dim)
    center = np.array([0] * dim)
    allPos = []
    strands = []
    s = 10
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(s, s, s, c="white")
        ax.scatter(s, s, -s, c="white")
        ax.scatter(s, -s, -s, c="white")
        ax.scatter(s, -s, s, c="white")
        ax.scatter(-s, s, s, c="white")
        ax.scatter(-s, s, -s, c="white")
        ax.scatter(-s, -s, s, c="white")
        ax.scatter(-s, -s, -s, c="white")
        (f1,) = ax.plot([], [], "r-", lw=1)  # plot1
    plt.xlim([-s, s])
    plt.xlim([-s, s])

    for idx in range(50):
        # +- 50%
        d = D * (0.5 + 1.5 * np.random.rand())
        pos = 2 * (np.random.rand(dim) - offset)
        # sphere's radius = 1
        pos /= np.linalg.norm(pos)
        # print(pos)
        pos += center
        strands.append(Strand(dim, pos, d))

        if not gloom(strands[-1]):
            strands.pop()
    for hair in strands:
        render(hair, dim)
    plt.show()
