import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from fresh_attempt import InterPoints, Intersections, twoPTarea, GenerateCells, Rotate


def dynamic_update_step(x, dt, A, B, L, rep=True, tau=10, grow=1, omega=1, mu=0.5):
    eps = 1e-5
    t = len(x) - 1

    S = Intersections(x[t], A, B, L)
    
    
    Sprime = []
    for i in S:
        for j in i:
            Sprime.append((i, j))
    

    x.append(x[t])

    for i in range(np.size(x[t], axis=1)):

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

            x[t + 1][2, i] += mu * forceX * dt * A * B / A1 / B1
            x[t + 1][3, i] += mu * forceY * dt * A * B / A1 / B1
            x[t + 1][4, i] += 4 * mu * torque * dt / (A1 ** 2 + B1 ** 2)

            x[t + 1][2, j] -= mu * forceX * dt * A * B / A2 / B2
            x[t + 1][3, j] -= mu * forceY * dt * A * B / A2 / B2
            x[t + 1][4, j] -= 4 * mu * torque * dt / (A2 ** 2 + B2 ** 2)
            

        # growing the daughter cells in the G2 growth phase

        if x[t + 1][5, i] == 0:
            if x[t + 1][0, i] < A:
                x[t + 1][0, i] += A * omega * grow * x[t + 1][0, i] * dt
                if x[t + 1][1, i] < B:
                    x[t + 1][1, i] += B * omega * grow * x[t + 1][1, i] * dt
            else:
                if x[t + 1][1, i] < B:
                    x[t + 1][1, i] += B * omega * grow * x[t + 1][1, i] * dt
                else:
                    x[t + 1][5, i] = 1

    if rep:
        x = Reproduce(x, tau, dt)

    return S


def Reproduce(x, attachments, tau, dt=1):
    #   Creates new cells to add to the list of existing cells occupying space
    #   tau is reproduction half-life per individual
    #   this step does NOT update the time i.e. cells do not move in this update step

    t = len(x) - 1
    N = np.size(x[t], axis=1)
    dn = np.random.poisson(N * dt / tau)
    S = np.random.choice(np.arange(N), dn, replace=False)
    phis = np.random.normal(0, np.pi / 6, dn)

    attachments.extend([[] for i in range(dn)])

    for i in range(dn):

        attachments[S[i]].append(N - 1 + i)

        k = random.uniform()
        if k < 0.5:
            xmid, ymid = Rotate(x[t][0, S[i]] * np.cos(phis[i]), x[t][1, S[i]] * np.sin(phis[i]), x[t][4, S[i]])
            xmid += x[t][2, S[i]]
            ymid += x[t][3, S[i]]
            theta = x[t][4, S[i]] + np.arctan(x[t][0, S[i]] / x[t][1, S[i]] * np.tan(phis[i]))
            r0 = x[t][1, S[i]] / 5
            xcen = xmid + r0 * np.cos(theta)
            ycen = ymid + r0 * np.sin(theta)
        else:
            xmid, ymid = Rotate(-x[t][0, S[i]] * np.cos(phis[i]), -x[t][1, S[i]] * np.sin(phis[i]), x[t][4, S[i]])
            xmid += x[t][2, S[i]]
            ymid += x[t][3, S[i]]
            theta = x[t][4, S[i]] + np.arctan(x[t][0, S[i]] / x[t][1, S[i]] * np.tan(phis[i]))
            r0 = x[t][1, S[i]] / 5
            xcen = xmid - r0 * np.cos(theta)
            ycen = ymid - r0 * np.sin(theta)

        x[t] = np.hstack((x[t], [[r0], [r0], [xcen], [ycen], [theta], [0]]))
        x[t][5, S[i]] += 1

    return x, attachments


def add_ellipse(x, A, B, X, Y, theta):
    return np.hstack((x, np.vstack((A, B, X, Y, theta, 1))))


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
