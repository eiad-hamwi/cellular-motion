import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from fresh_attempt import InterPoints, Intersections, twoPTarea, Reproduce, Rotate


def GenerateCells(N, majorAxis, minorAxis, L, resolution=5):
    #   Generates a (5 x N) array of sizes (majorAxis, minorAxis), 2D-positions (x, y), and
    #   orientations (phi) in [-Pi,Pi) of all the cells
    pi = np.pi

    C = np.vstack((majorAxis, minorAxis, L * random.rand(), L * random.rand(), pi * (random.rand() - 1 / 2), 1))

    def mindist(a, C):
        d = []
        for j in range(np.size(C, axis=1)):
            d.append(np.sqrt((a[2] - C[2, j]) ** 2 + (a[3] - C[3, j]) ** 2))
        return min(d)

    n = 1
    while n < N:

        a = np.vstack((majorAxis, minorAxis, L * random.rand(), L * random.rand(), pi * (random.rand() - 1 / 2), 1))
        if mindist(a, C) < minorAxis:
            continue

        # Intersections func 
        Cprime = np.hstack((C, a))

        S = []
        intersectingCells = Intersections(Cprime, majorAxis, minorAxis, L)
        for j in intersectingCells[n]:
            k = np.size(InterPoints(a[:, 0], C[:, j]))
            if k > 4:
                S.append(j)

        if np.size(S) > 0:

            i = 0
            phi = a[4]
            I = np.zeros(resolution)

            while i < resolution:
                S = []
                intersectingCells = Intersections(Cprime, majorAxis, minorAxis, L)
                for j in intersectingCells[n]:
                    k = np.size(InterPoints(a[:, 0], C[:, j]))
                    if k > 4:
                        S.append(j)

                I[i] = np.size(S)
                a[4] = phi + i * pi / resolution
                Cprime = np.hstack((C, a))
                i += 1

            if min(I) == 0:
                i = random.choice(np.where(I == 0)[0])
                a[4] = phi + i * pi / resolution
                # =============================================================================
                #                 fits = np.where(I==0)[0]
                #                 minplus = np.min(fits)
                #                 minminus = resolution - 1 - np.max(fits)
                #                 if min(minplus, minminus) == minplus:
                #                     a[4] = phi + minplus * pi / resolution
                #                 else:
                #                     a[4] = phi - minminus * pi / resolution
                #                 
                # =============================================================================

                C = np.hstack((C, a))
                n += 1
            else:
                continue

        else:
            C = np.hstack((C, a))
            n += 1

    x = [C]

    return x


def dynamic_update_step(x, dt, majorAxis, minorAxis, L, rep=True, tau=10, elongationRate=1, sigma=1, mu=0.5):
    eps = 1e-5
    t = len(x) - 1
    N0 = np.size(x[t], axis=1)

    x.append(x[t])
    S = Intersections(x[t], majorAxis, minorAxis, L)

    for i in range(N0):

        for j in S[i]:
            Xint, Yint = InterPoints(x[t + 1][:, i], x[t + 1][:, j])
            n = np.size(Xint)
            A1, B1, x1, y1 = x[t + 1][0:4, i]
            A2, B2, x2, y2 = x[t + 1][0:4, j]
            delX = x1 - x2
            delY = y1 - y2
            dist = np.sqrt(delX ** 2 + delY ** 2)

            if n == 2:
                area = twoPTarea(x[t + 1][:, i], x[t + 1][:, j], Xint, Yint)
                radialX = np.average(Xint) - x1
                radialY = np.average(Yint) - y1

                if abs(Yint[1] - Yint[0]) > eps:
                    m = (Xint[0] - Xint[1]) / (Yint[1] - Yint[0])
                    a = np.cos(np.arctan(m))
                    b = np.sin(np.arctan(m))

                    if dist >= (x[t + 1][1, i] + x[t + 1][1, j]):
                        if a * radialX + b * radialY < 0:
                            sgn = 1
                        else:
                            sgn = -1
                    else:
                        if a * radialX + b * radialY > 0:
                            sgn = 1
                        else:
                            sgn = -1

                else:
                    m = (Yint[1] - Yint[0]) / (Xint[0] - Xint[1])
                    a = np.sin(np.arctan(m))
                    b = np.cos(np.arctan(m))

                    if dist > (x[t + 1][1, i] + x[t + 1][1, j]):
                        if a * radialX + b * radialY < 0:
                            sgn = 1
                        else:
                            sgn = -1

                    else:
                        if a * radialX + b * radialY > 0:
                            sgn = 1
                        else:
                            sgn = -1

                forceX = sgn * a * area
                forceY = sgn * b * area
                torque = radialX * forceY - radialY * forceX

            elif n == 4:
                print('4pt intersection ({0}, {1})'.format(i, j))
                forceX = 0
                forceY = 0
                torque = 0

            elif dist < (x[t + 1][1, i] + x[t + 1][1, j]):
                area = np.pi * min(x[t + 1][0, i] * x[t + 1][1, i], x[t + 1][0, j] * x[t + 1][1, j])
                forceX = area * delX / dist
                forceY = area * delY / dist
                torque = 0

            else:
                forceX = 0
                forceY = 0
                torque = 0

            x[t + 1][2, i] += mu * forceX * dt * majorAxis * minorAxis / A1 / B1
            x[t + 1][3, i] += mu * forceY * dt * majorAxis * minorAxis / A1 / B1
            x[t + 1][4, i] += 4 * mu * torque * dt / (A1 ** 2 + B1 ** 2)

            x[t + 1][2, j] -= mu * forceX * dt * majorAxis * minorAxis / A2 / B2
            x[t + 1][3, j] -= mu * forceY * dt * majorAxis * minorAxis / A2 / B2
            x[t + 1][4, j] -= 4 * mu * torque * dt / (A2 ** 2 + B2 ** 2)

        # growing the daughter cells in the G2 growth phase
        # (elongationRate) variable depends on the local concentration of nutrients

        if x[t + 1][5, i] == 0:

            if x[t + 1][0, i] < majorAxis:

                x[t + 1][0, i] += elongationRate * dt
                x[t + 1][1, i] += minorAxis / majorAxis * elongationRate * dt

            else:
                x[t + 1][5, i] = 1
                attachments[].delete()

    if rep:
        attachments = [[] for i in range(N0)]
        x, attachments = Reproduce(x, attachments, tau, dt)

    return x


def add_ellipse(x, majorAxis, minorAxis, X, Y, theta):
    return np.hstack((x, np.vstack((majorAxis, minorAxis, X, Y, theta, 1))))


def PlotCells(x, size):  # ellipse plotting module for cells (not final)

    ells = [Ellipse((x[2, i], x[3, i]), 2 * x[0, i], 2 * x[1, i], 180 / np.pi * x[4, i]) for i in
            range(np.size(x, axis=1))]

    fig = plt.figure(0)

    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)

    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)

    return np.size(ells)
