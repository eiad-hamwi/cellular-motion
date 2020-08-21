import numpy as np
import fresh_attempt
from numpy import random


def GenerateCells(N, radius, length, L, resolution=5):
    #   Generates a (10 x N) array of cell configurations 
    #   [radius, length, 2D positions and orientiation (x, y, phi), 
    #   2D linear and angular velocities (vx, vy, omega), 
    #   Reproduction Number, and time-to-budding] for each of the N cells
    pi = np.pi

    Cells = np.vstack((radius, 
                       length, 
                       L * random.rand(), L * random.rand(), 
                       pi * (random.rand() - 1 / 2), 0, 0, 0, 
                       1, 
                       40 * random.rand()
                       ))
    """
    for i in range(1,N):
        Cells = np.hstack((Cells, 
                           np.vstack((radius, 
                                      length, 
                                      L * random.rand(), L * random.rand(), 
                                      pi * (random.rand() - 1 / 2), 0, 0, 0, 
                                      1, 
                                      40 * random.rand()
                                      ))
                           ))
    """
    n = 1
 
    while n < N:
    

        testCell = np.vstack((radius, 
                       length, 
                       L * random.rand(), L * random.rand(), 
                       pi * (random.rand() - 1 / 2), 0, 0, 0, 
                       1, 
                       40 * random.rand()
                       ))

        # Intersections func 
        Cprime = np.hstack((Cells, testCell))

        S = []
        intersectingCells, d, a = fresh_attempt.Intersections(
            Cprime, length, radius, L)


        for j in intersectingCells[n]:
            if d[n, j] < radius / 2:
                S.append(j)

        if np.size(S) > 0:

            i = 0
            phi = testCell[4]
            possibleOrientations = np.zeros(resolution)

            while i < resolution:
                
                intersectingCells, d, a = fresh_attempt.Intersections(
                    Cprime, length, radius, L)
                

                possibleOrientations[i] = min(d[n, :])
                testCell[4] = phi + i * pi / resolution
                Cprime = np.hstack((Cells, testCell))
                i += 1

            if max(possibleOrientations) > radius / 2:
                i = random.choice(np.where(possibleOrientations > radius/2)[0])
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


def dynamic_update_step(x, attachments, dt, radius, length, L, rep=True, 
                        tau=10, elongationRate=0.02, kcc=1, gamma=0):
    eps = 3e-5
    t = len(x) - 1
    N0 = np.size(x[t], axis=1)

    x.append(x[t].copy())
    S, d, a = fresh_attempt.Intersections(x[t], length, radius, L)

    # loop over cells adult cells
    for i in range(N0):

#        x[t + 1][6, i] -= dt

        for j in S[i]:
            
            
            r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t][:8, i]
            r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, j]
            
            ncc = (a[i, j, 0, :] - a[i, j, 1, :]) / d[i, j]
            rcc = (a[i, j, 0, :] + a[i, j, 1, :]) / 2
            dcc = r1 + r2 - d[i, j]
            rprime1 = np.hstack((rcc - np.array([x1, y1]), 0))
            rprime2 = np.hstack((rcc - np.array([x2, y2]), 0))
            vcc = x[t][5:7, i] - x[t][5:7, j] + np.cross(
                np.array([0, 0, omega1]), rprime1)[:2] - np.cross(
                    np.array([0, 0, omega2]), rprime2)[:2]
                    
            tanComp = np.linalg.norm(vcc - np.dot(vcc, ncc) * ncc)
            if tanComp < 1e-5:
                tcc = np.roll(ncc, 1)
                tcc[0] *= -1
            else:
                tcc = (vcc - np.dot(vcc, ncc) * ncc) / tanComp
                
            VolI = np.pi * r1 ** 2 + 2 * r1 * l1
            VolJ = np.pi * r2 ** 2 + 2 * r2 * l2
            M = VolI * VolJ / (VolI + VolJ)
            
            force = (4/3 * kcc / np.sqrt(1/r1 + 1/r2) * np.sqrt(dcc) - 
                     gamma * M * np.dot(vcc, ncc)) * dcc * ncc
            
            torque = np.cross(rprime1, force)
            
            I = ((3*r1**2 + l1**2)/12 * l1*r1**2 + 
                 (2/5*r1**2 + l1**2/4) * 4/3*r1**2 )/(l1*r1**2 + 4/3*r1**2)*M
    

            

            x[t + 1][[2, 3], i] += force * dt / M
            x[t + 1][4, i] += torque[2] * dt / I


    return x
"""
            # print variables to figure out what's going on
            print('Time ')
            print(t)
            print(i)
            print(forceX)
            print(forceY)

        if x[t + 1][5, i] > 0:

            x[t + 1][6, i] -= dt

        # growing the daughter cells in the G2 growth phase
        # (elongationRate) variable depends on the local concentration of nutrients
        if x[t + 1][5, i] == 0:

            if x[t + 1][0, i] < radius:

                x[t + 1][0, i] += elongationRate * dt
                x[t + 1][1, i] += length / radius * elongationRate * dt

            # daughter cell finishes growing, detaches and becomes adult cell
            else:
                x[t + 1][5, i] = 1
                x[t + 1][6, i] = 40
                attachments[attachments[i][0]].remove(i)
                attachments[i].remove(attachments[i][0])

    if rep:
        attachments = [[] for i in range(N0)]
        x, attachments = fresh_attempt.Reproduce(x, attachments, tau, dt)
"""


def add_ellipse(x, majorAxis, minorAxis, X, Y, theta):
    return np.hstack((x, np.vstack((majorAxis, minorAxis, X, Y, theta, 1))))
