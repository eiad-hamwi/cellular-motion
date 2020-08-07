import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

import fresh_attempt


def GenerateCells(N, majorAxis, minorAxis, L, resolution=5):
    #   Generates a (7 x N) array of cell configurations [majorAxis, minorAxis, 2D-positions (x, y), orientations (phi) 
    #   in [-Pi,Pi), Reproduction Number, and time-to-budding] of all the N cells
    pi = np.pi

    Cells = np.vstack((majorAxis, minorAxis, L * random.rand(), L * random.rand(), pi * (random.rand() - 1 / 2), 1, 40 * random.rand()))

    # find minimum distance between new cell 'testCell' and other cells 'Cells
    def mindist(testCell, Cells):
        return min(np.sqrt((Cells[2, :] - testCell[2, 0]) ** 2 + (Cells[3, :] - testCell[3, 0]) ** 2))

    n = 1
    while n < N:

        testCell = np.vstack(
            (majorAxis, minorAxis, L * random.rand(), L * random.rand(), pi * (random.rand() - 1 / 2), 1, 40 * random.rand()))
        if mindist(testCell, Cells) < minorAxis:
            continue

        # Intersections func 
        Cprime = np.hstack((Cells, testCell))

        S = []
        intersectingCells = fresh_attempt.Intersections(Cprime, majorAxis, minorAxis, L)
        for j in intersectingCells[n]:
            k = np.size(fresh_attempt.InterPoints(testCell[:, 0], Cells[:, j]))
            if k > 4:
                S.append(j)

        if np.size(S) > 0:

            i = 0
            phi = testCell[4]
            possibleOrientations = np.zeros(resolution)

            while i < resolution:
                S = []
                intersectingCells = fresh_attempt.Intersections(Cprime, majorAxis, minorAxis, L)
                for j in intersectingCells[n]:
                    k = np.size(fresh_attempt.InterPoints(testCell[:, 0], Cells[:, j]))
                    if k > 4:
                        S.append(j)

                possibleOrientations[i] = np.size(S)
                testCell[4] = phi + i * pi / resolution
                Cprime = np.hstack((Cells, testCell))
                i += 1

            if min(possibleOrientations) == 0:
                i = random.choice(np.where(possibleOrientations == 0)[0])
                testCell[4] = phi + i * pi / resolution
                Cells = np.hstack((Cells, testCell))
                n += 1
            else:
                continue

        else:
            Cells = np.hstack((Cells, testCell))
            n += 1

    x = [Cells]

    return x


def GenerateCellsNonRandom(majorAxis, minorAxis, L, resolution=5):
    #   Generates a (7 x N) array of cell configurations [majorAxis, minorAxis, 2D-positions (x, y), orientations (phi)
    #   in [-Pi,Pi), Reproduction Number, and time-to-budding] of all the N cells
    pi = np.pi

    Cells = np.vstack((majorAxis, minorAxis, L*0.5, L*0.5, pi , 1, 40*random.rand()))

    Cells = np.hstack((Cells, np.vstack((majorAxis, minorAxis, 0.5*L + 5/3*majorAxis, L*0.5, pi , 1, 40*random.rand()))))

    x = [Cells]

    return x


def dynamic_update_step(x, attachments, dt, majorAxis, minorAxis, L, rep=True, tau=10, elongationRate=0.02, sigma=1,
                        mu=0.5):
    eps = 1e-5
    t = len(x) - 1
    N0 = np.size(x[t], axis=1)

    x.append(x[t].copy())
    S = fresh_attempt.Intersections(x[t], majorAxis, minorAxis, L)

    # loop over cells adult cells
    for i in range(N0):

#        print(S) # print S to figure out what's going on
#        x[t + 1][6, i] -= dt

        for j in S[i]:
            Xint, Yint = fresh_attempt.InterPoints(x[t + 1][:, i], x[t + 1][:, j], eps)
            n = np.size(Xint)
            A1, B1, x1, y1 = x[t + 1][0:4, i]
            A2, B2, x2, y2 = x[t + 1][0:4, j]
            delX = x1 - x2
            delY = y1 - y2
            dist = np.sqrt(delX ** 2 + delY ** 2)

            if np.size(Xint) != np.size(Yint) or n == 4:
                area = fresh_attempt.ShapelyArea(x[t + 1][:, i], x[t + 1][:, j])
                forceX = area * (x1 - x2)/np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                forceY = area * (y1 - y2) / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                torque = 0

            elif n == 2:
                area = fresh_attempt.twoPTarea(x[t + 1][:, i], x[t + 1][:, j], Xint, Yint)
                radialX = np.average(Xint) - x1
                radialY = np.average(Yint) - y1

                if abs(Yint[1] - Yint[0]) > eps:
                    m = (Xint[0] - Xint[1]) / (Yint[1] - Yint[0])
                    a = np.cos(np.arctan(m))
                    b = np.sin(np.arctan(m))
                else:
                    m = (Yint[0] - Yint[1]) / (Xint[1] - Xint[0])
                    a = np.sin(np.arctan(m))
                    b = np.cos(np.arctan(m))

                if a * radialX + b * radialY < 0:
                    sgn = 1
                else:
                    sgn = -1


                # if dist > (x[t + 1][1, i] + x[t + 1][1, j]):

                # else:
                #     if a * radialX + b * radialY > 0:
                #         sgn = 1
                #     else:
                #         sgn = -1

                forceX = sgn * a * area
                forceY = sgn * b * area
                torque = radialX * forceY - radialY * forceX


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

            # print variables to figure out what's going on
#            print('Time ')
#            print(t)
#            print(i)
#            print(forceX)
#            print(forceY)

#        if x[t + 1][5, i] > 0:

#            x[t + 1][6, i] -= dt

        # growing the daughter cells in the G2 growth phase
        # (elongationRate) variable depends on the local concentration of nutrients
        if x[t + 1][5, i] == 0:

            if x[t + 1][0, i] < majorAxis:

                x[t + 1][0, i] += elongationRate * dt
                x[t + 1][1, i] += minorAxis / majorAxis * elongationRate * dt

            # daughter cell finishes growing, detaches and becomes adult cell
            else:
                x[t + 1][5, i] = 1
                x[t + 1][6, i] = 40
                attachments[attachments[i][0]].remove(i)
                attachments[i].remove(attachments[i][0])

    if rep:
        attachments = [[] for i in range(N0)]
        x, attachments = fresh_attempt.Reproduce(x, attachments, tau, dt)

    return x


def add_ellipse(x, majorAxis, minorAxis, X, Y, theta):
    return np.hstack((x, np.vstack((majorAxis, minorAxis, X, Y, theta, 1))))


# IS THIS THE SAME AS THE ONE IN FRESH_ATTEMPT?
def PlootCells(x, size):  # ellipse plotting module for cells (not final)

    ells = [Ellipse((x[2, i], x[3, i]), 2 * x[0, i], 2 * x[1, i], 180 / np.pi * x[4, i]) for i in
            range(np.size(x, axis=1))]

    fig = plt.figure(0)

    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)

    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)

    return np.size(ells)



x = GenerateCellsNonRandom(1, 0.5, 10)
fresh_attempt.PlotCells(x, 0, 10)
while len(x)<200:
    x = dynamic_update_step(x, [], 0.02, 1.0, 0.5, 10, False)
fresh_attempt.PlotCells(x, 0, 10)
