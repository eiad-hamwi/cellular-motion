import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def PlotCells(x, i, size):  # ellipse plotting module for cells (not final)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)

    for j in range(np.size(x[i], axis=1)):
        width = 2 * x[i][0, j] + x[i][1, j]
        length = 2 * x[i][0, j]
        c = np.cos(x[i][4, j])
        s = np.sin(x[i][4, j])

        # rotate corner of rectangle
        rx, ry = np.dot([[c, -s],
                         [s, c]],
                        [-width / 2, -length / 2])

        # shift corner to cell location
        rx, ry = rx + x[i][2, j], ry + x[i][3, j]

        ax.add_artist(
            Rectangle((rx, ry), width, length, 180 / np.pi * x[i][4, j],
                      fc="none", ec="blue")
        )

    plt.show()

    return np.size(x[i], axis=1)


def PlotFewCells(x, t, J):  # ellipse plotting module for cells (not final)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    X = x[t][2, J]
    Y = x[t][3, J]
    A = max(x[t][0, 0], x[t][0, 1])

    ax.set_xlim(min(X) - 2 * A, max(X) + 2 * A)
    ax.set_ylim(min(Y) - 2 * A, max(Y) + 2 * A)

    for j in J:
        ax.add_artist(
            Ellipse((x[t][2, j], x[t][3, j]), 2 * x[t][0, j] + x[t][1, j],
                    2 * x[t][0, j], 180 / np.pi * x[t][4, j], fc="none",
                    ec="blue"))

    return plt.show()


#   this initializes the Figure, Canvas, & Axes for the animation
def anim_init(size):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_aspect(1)

    return fig, canvas, ax


#   this adds the cells to the axes and returns the filled axes
def ells(x, i, ax):
    for j in range(np.size(x[i], axis=1)):
        ax.add_artist(
            Ellipse((x[i][2, j], x[i][3, j]), 2 * x[i][0, j] + x[i][1, j],
                    2 * x[i][0, j], 180 / np.pi * x[i][4, j], fc="none",
                    ec="blue"))
    return ax


# rectangles
def rect(x, i, ax):
    for j in range(np.size(x[i], axis=1)):
        width = 2 * x[i][0, j] + x[i][1, j]
        length = 2 * x[i][0, j]
        c = np.cos(x[i][4, j])
        s = np.sin(x[i][4, j])

        # rotate corner of rectangle
        rx, ry = np.dot([[c, -s],
                         [s, c]],
                        [-width / 2, -length / 2])

        # shift corner to cell location
        rx, ry = rx + x[i][2, j], ry + x[i][3, j]

        ax.add_artist(
            Rectangle((rx, ry), width, length, 180 / np.pi * x[i][4, j],
                      fc="none", ec="blue")
        )
    return ax


#   this generates all the image files, writes the GIF animation, then deletes the image file
def animate(x, size, filename, frames):
    image_list = []
    for i in range(0, len(x), len(x) // frames):
        fig, canvas, ax = anim_init(size)  # initialize Figure
        ax = rect(x, i, ax)  # add ellipses
        fig.savefig('plots/test.png', dpi=300)  # save plot to temporary image file
        image_list.append(imageio.imread('plots/test.png'))  # transform image file into NumPy array
    os.remove('plots/test.png')  # delete the image file
    imageio.mimwrite('plots/{}.gif'.format(str(filename)),
                     [image_list[i] for i in range(len(image_list))])  # compile image_list into GIF


def PlotTemporalCells(y, size):  # ellipse plotting module for cells (not final)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    ells = []
    for t in range(0, len(y)):
        x = y[t]
        # AG: REMOVED FACE COLOR, ADDED EDGE COLOR, goes from red to blue for the first cell as time goes by,
        # from orange to blue for the other one

        for i in range(np.size(x, axis=1)):
            ells.append(Ellipse((x[2, i], x[3, i]), 2 * x[0, i], 2 * x[1, i], 180 / np.pi * x[4, i], fc="none",
                                ec=(1.0 - 1.0 * float(t) / len(y), i / 2.0, 1.0 * float(t) / len(y))))

        for e in ells:
            ax.add_artist(e)

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)

    # AG: ADDED PLT.SHOW()
    plt.show()

    return np.size(ells)


def Translate(x, y, h, k):
    #   Translates vector (x, y) by vector (h, k)

    return x + h, y + k


def Rotate(x, y, phi):
    #   Rotates the vector (x, y) counter-clockwise arount the origin

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    xR = x * cosphi - y * sinphi
    yR = x * sinphi + y * cosphi

    return xR, yR


def mindist(x, i, j):
    p1 = x[2:4, i] + np.array(Rotate(x[1, i] / 2, 0, x[4, i]))
    p2 = x[2:4, j] + np.array(Rotate(x[1, j] / 2, 0, x[4, j]))

    q1 = x[2:4, i] - np.array(Rotate(x[1, i] / 2, 0, x[4, i]))
    q2 = x[2:4, j] - np.array(Rotate(x[1, j] / 2, 0, x[4, j]))
    
    eps = 1e-3

    d1 = q1 - p1
    d2 = q2 - p2
    d12 = p2 - p1

    D1 = np.dot(d1, d1)
    D2 = np.dot(d2, d2)
    R = np.dot(d1, d2)
    denom = D1*D2 - R**2

    if (D1<eps) != (D2<eps):
        S1 = np.dot(d1, d12)
        u = 0
        t = S1/D1

        if t<0:
            t = 0
        elif t>1:
            t = 1

        return np.sqrt(np.dot(t*d1 - d12, t*d1 - d12)), \
               np.vstack(((1-t)*p1 + t*q1, (1-u)*p2 + u*q2))

    elif (D1<eps) and (D2<eps):
        u = 0
        t = 0

        return np.sqrt(np.dot(d12, d12)), \
               np.vstack(((1-t)*p1 + t*q1, (1-u)*p2 + u*q2))


    elif abs(denom)<eps**2:
        S2 = np.dot(d2, d12)
        u = -S2/D2
        t = 0

        if u<0:
            u = 0
        elif u>1:
            u = 1

        return np.sqrt(np.dot(u*d2 + d12, u*d2 + d12)), \
               np.vstack(((1-t)*p1 + t*q1, (1-u)*p2 + u*q2))

    else:
        S1 = np.dot(d1, d12)
        S2 = np.dot(d2, d12)
        t = (S1*D2 - S2*R)/denom

        if t<0:
            t = 0
        elif t>1:
            t = 1

        u = (t*R - S2)/D2

        if u<0:
            u = 0
            t = S1/D1
            if t<0:
                t = 0
            elif t>1:
                t = 1

        elif u>1:
            u = 1
            t = (R + S1)/D1
            if t<0:
                t = 0
            elif t>1:
                t = 1    

        return np.sqrt(np.dot(t*d1 - u*d2 - d12, t*d1 - u*d2 - d12)), \
               np.vstack(((1-t)*p1 + t*q1, (1-u)*p2 + u*q2))


"""
def mindist_old(x, i, j):
    p1 = x[2:4, i] + np.array(Rotate(x[1, i] / 2, 0, x[4, i]))
    p2 = x[2:4, j] + np.array(Rotate(x[1, j] / 2, 0, x[4, j]))
    p = np.vstack((p1, p2))

    q1 = x[2:4, i] - np.array(Rotate(x[1, i] / 2, 0, x[4, i]))
    q2 = x[2:4, j] - np.array(Rotate(x[1, j] / 2, 0, x[4, j]))
    q = np.vstack((q1, q2))
    u = q - p

    p = np.vstack((p1 - p2, p))

    def f(t):
        return np.dot(t[0] * u[0, :] - t[1] * u[1, :] + p[0, :],
                      t[0] * u[0, :] - t[1] * u[1, :] + p[0, :])

    C = np.mod(np.abs(x[4, i] - x[4, j]), np.pi)

    if C > 1e-5:

        cross = np.cross(u[0, :], u[1, :])
        normC = np.linalg.norm(cross)
        s0 = np.dot(np.cross(p[0, :], u[1, :]), cross) / normC ** 2
        t0 = np.dot(np.cross(p[0, :], u[0, :]), cross) / normC ** 2

        if (0 <= s0 <= 1) and (0 <= t0 <= 1):

            d = np.sqrt(f((s0, t0)))

            a = np.add(p[1:3, :], np.vstack((s0 * u[0, :], t0 * u[1, :])))

        else:

            mini = np.empty((4, 3))
            # case 1: s=0, 0<=t<=1
            mini[0, :2] = [0, np.dot(p[0, :], u[1, :]) / np.linalg.norm(u[1, :]) ** 2]

            # case 2: s=1, 0<=t<=1
            mini[1, :2] = [1, np.dot((p[0, :] + u[0, :]), u[1, :]) /
                           np.linalg.norm(u[1, :]) ** 2]

            # case 3: 0<=s<=1, t=0
            mini[2, :2] = [-np.dot(p[0, :], u[0, :]) / np.linalg.norm(u[0, :]) ** 2, 0]

            # case 4: 0<=s<=1, t=1
            mini[3, :2] = [np.dot((u[1, :] - p[0, :]), u[0, :]) /
                           np.linalg.norm(u[0, :]) ** 2, 1]

            if 0 <= mini[0, 1] <= 1:
                mini[0, 2] = f((mini[0, 0], mini[0, 1]))
            else:
                mini[0, :] = (0, 0, f((0, 0)))
                mini = np.vstack((mini, (0, 1, f((0, 1)))))

            if 0 <= mini[1, 1] <= 1:
                mini[1, 2] = f((mini[1, 0], mini[1, 1]))
            else:
                mini[1, :] = (1, 0, f((1, 0)))
                mini = np.vstack((mini, (1, 1, f((1, 1)))))

            if 0 <= mini[2, 0] <= 1:
                mini[2, 2] = f((mini[2, 0], mini[2, 1]))
            else:
                mini[2, :] = (0, 0, f((0, 0)))
                mini = np.vstack((mini, (1, 0, f((1, 0)))))

            if 0 <= mini[3, 0] <= 1:
                mini[3, 2] = f((mini[3, 0], mini[3, 1]))
            else:
                mini[3, :] = (0, 1, f((0, 1)))
                mini = np.vstack((mini, (1, 1, f((1, 1)))))

            d2 = np.min(mini, axis=0)[2]

            t = mini[np.where(np.abs(mini[:, 2] - d2) < 1e-5)[0][0], :2]

            d = np.sqrt(d2)

            a = np.add(p[1:3, :], np.vstack((t[0] * u[0, :], t[1] * u[1, :])))


    else:

        uNorm = np.linalg.norm(u[1, :]) ** 2

        t = np.array([np.dot(p[0, :], u[1, :]) / uNorm,
                      np.dot(q[0, :] - p[2, :], u[1, :]) / uNorm])

        if not 0 <= t[0] <= 1:
            if np.linalg.norm(p[0, :]) < np.linalg.norm(u[1, :] - p[0, :]):
                t[0] = 0
            else:
                t[0] = 1
        if not 0 <= t[1] <= 1:
            if np.linalg.norm(p[2, :] - q[0, :]) < np.linalg.norm(u[1, :] +
                                                                  p[2, :] -
                                                                  q[0, :]):
                t[1] = 0
            else:
                t[1] = 1

        d2 = np.array([np.linalg.norm(p[2, :] + t[0] * u[1, :] - p[1, :]),
                       np.linalg.norm(p[2, :] + t[1] * u[1, :] - q[0, :])])

        d = np.min(d2)

        index = np.where(np.abs(d2 - d) < 1e-5)[0][0]

        a = np.vstack((p[1, :], q[0, :]))[index, :]

        a = np.vstack((a, p[2, :] + t[index] * u[1, :]))

    return d, a
"""


def BackgroundLattice(x, L, radius):
    # Produces a Lattice on which one can approximate cell positions

    # x is vector of cell metadata (size, position, orientation)
    # L is width of simulation square
    # radius is short axis of ellipse

    a = radius / (2*L)
    N = int(np.ceil(1 / a))
    BG = [[[] for i in range(N)] for j in range(N)]

    for k in range(np.size(x, axis=1)):
        Xk = int((x[2, k] + L) / radius)
        Yk = int((x[3, k] + L) / radius)

        BG[Xk][Yk].append(k)

    return BG


def pad(ulist, r=2):
    N = len(ulist)
    empty_rows = [[[]] * (N + 2 * r)] * r
    olist = empty_rows.copy()

    for i in range(N):
        olist += [[[]] * r + ulist[i] + [[]] * r]

    olist += empty_rows

    return olist


def Intersections(x, length, radius, L):
    r = int(np.ceil(length / radius)) + 1

    # initialize intersection index array
    n = np.size(x, axis=1)
    intersectingCells = [[] for k in range(n)]

    # set up coarse-grained background lattice (as a flattened array)
    BG = BackgroundLattice(x, L, radius)

    # pad the lattice array with extra empty rows and columns for grid searching
    BG = pad(BG, r)

    I = []

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):

            if (i != 0) or (j != 0):
                I.append((i, j))
            else:
                continue

    for i in range(np.size(x, axis=1)):
        Xi, Yi = int((x[2, i] + L) / radius), int((x[3, i] + L) / radius)

        for j in BG[Xi + r][Yi + r]:
            if i != j:

                d, a = mindist(x, i, j)

                intersectingCells[i].append([[j, a[0, :]], [d, a[1, :]]])

            else:
                continue

        for (k, l) in I:
            for j in BG[(Xi + r) + k][(Yi + r) + l]:

                if mindist(x, i, j)[0] >= x[0, i] + x[0, j]:
                    continue
                else:

                    d, a = mindist(x, i, j)

                    intersectingCells[i].append([[j, a[0, :]], [d, a[1, :]]])

    return intersectingCells
