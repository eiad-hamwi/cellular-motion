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
                       L * (2 * random.rand() - 1), L * (2 * random.rand() - 1),
                       pi * (random.rand() - 1 / 2), 0, 0, 0,
                       1,
                       max(0, random.normal(40, 5))
                       ))
    """
    for i in range(1, N):
        Cells = np.hstack((Cells,
                           np.vstack((radius,
                                      length,
                                      L * (2 * random.rand() - 1), L * (2 * random.rand() - 1),
                                      pi * (random.rand() - 1 / 2), 0, 0, 0,
                                      1,
                                      max(0, random.normal(40, 5))
                                      ))
                           ))
    """
    n = 1

    while n < N:

        testCell = np.vstack((radius,
                              length,
                              L * (2 * random.rand() - 1), L * (2 * random.rand() - 1),
                              pi * (random.rand() - 1 / 2), 0, 0, 0,
                              1,
                              max(0, random.normal(40, 5))
                              ))

        # Intersections func 
        Cprime = np.hstack((Cells, testCell))

        S = []
        intersectingCells = fresh_attempt.Intersections(
            Cprime, length, radius, L)

        for j in range(len(intersectingCells[n])):
            if intersectingCells[n][j][1][0] == 0:
                S.append(intersectingCells[n][j][0][0])

        if np.size(S) > 0:

            i = 0
            phi = testCell[4]
            possibleOrientations = np.zeros(resolution)

            while i < resolution:

                intersectingCells = fresh_attempt.Intersections(
                    Cprime, length, radius, L)

                d = []
                for j in range(len(intersectingCells[n])):
                    d.append(intersectingCells[n][j][1][0])

                possibleOrientations[i] = max(d)
                testCell[4] = phi + i * pi / resolution
                Cprime = np.hstack((Cells, testCell))
                i += 1

            if max(possibleOrientations) > radius / 2:
                i = random.choice(np.where(possibleOrientations > radius / 2)[0])
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


def GenerateCellsNonRandom(radius, length, L):
    #   Generates a (7 x N) array of cell configurations [majorAxis, minorAxis, 2D-positions (x, y), orientations (phi)
    #   in [-Pi,Pi), Reproduction Number, and time-to-budding] of all the N cells
    pi = np.pi

    Cells = np.vstack((radius, length, L / 2, L / 2 - 0.001, 0,
                       0, 0, 0,
                       1, 40 * random.rand()
                       ))

    Cells = np.hstack((Cells,
                       np.vstack((radius, length, L / 2, L / 2 + length / 2, np.pi / 2,
                                  0, 0, 0,
                                  1, 40 * random.rand()))
                       ))

    x = [Cells]

    return x


def dynamic_update_step(x, attachments, dt, radius, length, L, C, rep=False, kcc=1, gamma=0):
    eps = 1e-5
    t = len(x) - 1
    N0 = np.size(x[t], axis=1)

    x.append(x[t].copy())
    S = fresh_attempt.Intersections(x[t], length, radius, L)

    TG2 = 60

    N = np.zeros(int(R_agar / dr))


    # loop over cells
    for i in range(N0):

        # adult cells with no attached daughter cells
        if (0 < x[t + 1][8, i] <= 35) and (len(attachments[i]) == 0):

            # mechanics of single cell collisions
            for j in range(len(S[i])):
                r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t][:8, i]
                r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, S[i][j][0][0]]
                VolI = np.pi * r1 ** 2 + 2 * r1 * l1
                VolJ = np.pi * r2 ** 2 + 2 * r2 * l2
                M = VolI * VolJ / (VolI + VolJ)

                if S[i][j][1][0] > 0:
                    ncc = (S[i][j][0][1] - S[i][j][1][1]) / S[i][j][1][0]
                else:
                    ncc1 = np.array([x2, y2]) + np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)]) - S[i][j][0][1]
                    ncc2 = np.array([x2, y2]) - np.array([l2 * np.cos(theta2), l2 * np.sin(theta2)]) - S[i][j][0][1]
                    dn1 = np.dot(ncc1, ncc1)
                    dn2 = np.dot(ncc2, ncc2)
                    if dn1 < dn2:
                        ncc = ncc1 / np.sqrt(dn1)
                    else:
                        ncc = ncc2 / np.sqrt(dn2)

                rcc = (S[i][j][0][1] + S[i][j][1][1]) / 2
                dcc = r1 + r2 - S[i][j][1][0]
                if dcc < 0:
                    dcc = 0
                rprime1 = np.hstack((rcc - np.array([x1, y1]), 0))
                rprime2 = np.hstack((rcc - np.array([x2, y2]), 0))
                vcc = x[t][5:7, i] - x[t][5:7, S[i][j][0][0]] + np.cross(
                    np.array([0, 0, omega1]), rprime1)[:2] - np.cross(
                    np.array([0, 0, omega2]), rprime2)[:2]

                tanComp = np.sqrt(np.dot(vcc - np.dot(vcc, ncc) * ncc, vcc - np.dot(vcc, ncc) * ncc))
                if tanComp < eps:
                    tcc = np.roll(ncc, 1)
                    tcc[0] *= -1
                else:
                    tcc = (vcc - np.dot(vcc, ncc) * ncc) / tanComp

                force = (4 / 3 * kcc / np.sqrt(1 / r1 + 1 / r2) * np.sqrt(dcc) -
                         gamma * M * np.dot(vcc, ncc)) * dcc * ncc
                torque = np.cross(rprime1, force)
                I = ((3 * r1 ** 2 + l1 ** 2) / 12 * l1 * r1 ** 2 +
                     (2 / 5 * r1 ** 2 + l1 ** 2 / 4) * 4 / 3 * r1 ** 2) / (l1 * r1 ** 2 + 4 / 3 * r1 ** 2) * VolI

                x[t + 1][[2, 3], i] += force * dt / VolI
                x[t + 1][4, i] += torque[2] * dt / I

            # countdown till budding
            if x[t + 1][9, i] > dt:
                x[t + 1][9, i] -= dt

            # budding
            else:
                x[t + 1][8, i] += 1
                x[t + 1][9, i] = max(0, random.normal(40, 5))
                # generate bud
                if rep:
                    k = random.uniform(low=-1, high=1)
                    phi = random.normal(loc=0, scale=np.pi / 6)

                    p = x[t + 1][2:4, i] + np.sign(k) * (np.array(
                        fresh_attempt.Rotate(x[t + 1][1, i] / 2, 0, x[t + 1][4, i])) +
                                                         np.array(fresh_attempt.Rotate(
                                                             x[t + 1][0, i] + (radius + length / 2) * dt / TG2, 0,
                                                             x[t + 1][4, i] + phi)))

                    babyCell = np.vstack((radius * dt / TG2,
                                          length * dt / TG2,
                                          p[0], p[1], x[t + 1][4, i] + phi,
                                          0, 0, 0,
                                          0,
                                          np.sign(k)
                                          ))

                    x[t + 1] = np.hstack((x[t + 1], babyCell))

                    attachments.extend([[i]])
                    attachments[i].append(np.size(x[t + 1], axis=1) - 1)

            if len(S[i]) == 0:
                r1, l1, x1, y1 = x[t][:4, i]
            Area = (np.pi * r1 + 2 * l1) * r1
            r_i = int(np.sqrt(x1 ** 2 + y1 ** 2) / dr)
            N[r_i] += Area / (np.pi * dr ** 2 * (2 * r_i + 1))


        # daughter cells (still attached)
        elif x[t + 1][8, i] == 0:

            r, l, _x, _y, _t = x[t][:5, i]

            Area = (np.pi * r + 2 * l) * r
            r_i = int(np.sqrt(_x ** 2 + _y ** 2) / dr)
            N[r_i] += Area / (np.pi * dr ** 2 * (2 * r_i + 1))

            elongationRate = gmax * C[0, r_i] / (C[0, r_i] + Ks)

            # G2 growth phase
            if r < radius:

                ds = (radius / 2 + length / 4) * dt / TG2 * x[t + 1][9, i] * \
                     np.array([np.cos(_t), np.sin(_t)])

                x[t + 1][0, i] += radius * dt * (elongationRate / gmax) / TG2
                x[t + 1][1, i] += length * dt * (elongationRate / gmax) / TG2
                x[t + 1][2, i] += ds[0]
                x[t + 1][3, i] += ds[1]
                x[t + 1][2, attachments[i][0]] -= ds[0]
                x[t + 1][3, attachments[i][0]] -= ds[1]


            # daughter cell finishes growing, detaches and becomes adult cell
            else:
                x[t + 1][8, i] = 1
                x[t + 1][9, i] = max(0, random.normal(40, 5))
                attachments[attachments[i][0]].remove(i)
                attachments[i].remove(attachments[i][0])

        # mechanics of multiple, attached cells (mother-daughter combinations)
        elif len(attachments[i]) > 0:

            force = []
            torque = []
            r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t + 1][:8, i]
            r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t + 1][:8, attachments[i][0]]
            xmid = (x1 * r1 * (np.pi * r1 + 2 * l1) + x2 * r2 * (np.pi * r2 + 2 * l2)
                    ) / (r1 * (np.pi * r1 + 2 * l1) + r2 * (np.pi * r2 + 2 * l2))
            ymid = (y1 * r1 * (np.pi * r1 + 2 * l1) + y2 * r2 * (np.pi * r2 + 2 * l2)
                    ) / (r1 * (np.pi * r1 + 2 * l1) + r2 * (np.pi * r2 + 2 * l2))

            # mechanics of mother cell
            for j in range(len(S[i])):
                r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, S[i][j][0][0]]

                ncc = (S[i][j][0][1] - S[i][j][1][1]) / S[i][j][1][0]
                rcc = (S[i][j][0][1] + S[i][j][1][1]) / 2
                dcc = r1 + r2 - S[i][j][1][0]
                if dcc < 0:
                    dcc = 0
                rprime1 = np.hstack((rcc - np.array([x1, y1]), 0))
                rprime2 = np.hstack((rcc - np.array([x2, y2]), 0))
                vcc = x[t][5:7, i] - x[t][5:7, S[i][j][0][0]] + np.cross(
                    np.array([0, 0, omega1]), rprime1)[:2] - np.cross(
                    np.array([0, 0, omega2]), rprime2)[:2]

                tanComp = np.linalg.norm(vcc - np.dot(vcc, ncc) * ncc)
                if tanComp < 1e-5:
                    tcc = np.roll(ncc, 1)
                    tcc[0] *= -1
                else:
                    tcc = (vcc - np.dot(vcc, ncc) * ncc) / tanComp

                VolI = (np.pi * r1 + 2 * l1) * r1
                VolJ = (np.pi * r2 + 2 * l2) * r2
                M = VolI * VolJ / (VolI + VolJ)
                F = (4 / 3 * kcc / np.sqrt(1 / r1 + 1 / r2) * np.sqrt(dcc) -
                     gamma * M * np.dot(vcc, ncc)) * dcc * ncc
                T = np.cross(np.hstack((rcc - np.array([xmid, ymid]), 0)), F)
                force.append(F)
                torque.append(T)

            # mechanics of attached, daughter cells
            for j in attachments[i]:
                r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, j]
                for k in range(len(S[j])):
                    r3, l3, x3, y3, theta3, vx3, vy3, omega3 = x[t][:8, S[j][k][0][0]]
                    ncc = (S[j][k][0][1] - S[j][k][1][1]) / S[j][k][1][0]
                    rcc = (S[j][k][0][1] + S[j][k][1][1]) / 2
                    dcc = r2 + r3 - S[j][k][1][0]
                    if dcc < 0:
                        dcc = 0
                    rprime2 = np.hstack((rcc - np.array([x2, y2]), 0))
                    rprime3 = np.hstack((rcc - np.array([x3, y3]), 0))
                    vcc = x[t][5:7, j] - x[t][5:7, S[j][k][0][0]] + np.cross(
                        np.array([0, 0, omega1]), rprime2)[:2] - np.cross(
                        np.array([0, 0, omega2]), rprime3)[:2]

                    tanComp = np.linalg.norm(vcc - np.dot(vcc, ncc) * ncc)
                    if tanComp < 1e-5:
                        tcc = np.roll(ncc, 1)
                        tcc[0] *= -1
                    else:
                        tcc = (vcc - np.dot(vcc, ncc) * ncc) / tanComp

                    VolJ = (np.pi * r2 + 2 * l2) * r2
                    VolK = (np.pi * r3 + 2 * l3) * r3
                    M = VolJ * VolK / (VolJ + VolK)
                    F = (4 / 3 * kcc / np.sqrt(1 / r2 + 1 / r3) * np.sqrt(dcc) -
                         gamma * M * np.dot(vcc, ncc)) * dcc * ncc
                    T = np.cross(np.hstack((rcc - np.array([xmid, ymid]), 0)), F)
                    force.append(F)
                    torque.append(T)

            # combine all forces and torques    (currently only works for 1 attachment)
            if (len(force) > 0) and (len(torque) > 0):

                r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t][:8, i]
                r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, attachments[i][0]]
                VolI = np.pi * r1 ** 2 + 2 * r1 * l1
                VolJ = np.pi * r2 ** 2 + 2 * r2 * l2
                M = VolI + VolJ
                d1 = np.sqrt((x1 - xmid) ** 2 + (y1 - ymid) ** 2)
                d2 = np.sqrt((x2 - xmid) ** 2 + (y2 - ymid) ** 2)
                I1 = (((3 * r1 ** 2 + l1 ** 2) / 12 * l1 * r1 ** 2 +
                       (2 / 5 * r1 ** 2 + l1 ** 2 / 4) * 4 / 3 * r1 ** 2
                       ) / (l1 * r1 ** 2 + 4 / 3 * r1 ** 2) + d1 ** 2) * VolI
                I2 = (((3 * r2 ** 2 + l2 ** 2) / 12 * l2 * r2 ** 2 +
                       (2 / 5 * r2 ** 2 + l2 ** 2 / 4) * 4 / 3 * r2 ** 2
                       ) / (l2 * r2 ** 2 + 4 / 3 * r2 ** 2) + d2 ** 2) * VolJ
                I = I1 + I2

                # dynamics
                for k in [i] + attachments[i]:
                    # rotation
                    dtheta = sum(torque)[2] * dt / I
                    x[t + 1][4, k] += dtheta
                    c = np.cos(dtheta)
                    s = np.sin(dtheta)

                    x[t + 1][[2, 3], k] -= np.array([xmid, ymid])

                    _x, _y = x[t + 1][[2, 3], k]
                    x[t + 1][[2, 3], k] = [_x * c - _y * s,
                                           _x * s + _y * c]

                    x[t + 1][[2, 3], k] += np.array([xmid, ymid])

                    # translation
                    x[t + 1][[2, 3], k] += sum(force) * dt / M

            if len(S[i]) == 0:
                VolI = (np.pi * r1 + 2 * l1) * r1
            r_i = int(np.sqrt(x1 ** 2 + y1 ** 2) / dr)
            N[r_i] += VolI / (np.pi * dr ** 2 * (2 * r_i + 1))


    C = continuousUpdate(C, N, dt, 0.000073)

    return x, attachments, C


# units of length are 1cm = 1e3 l(ength)

c0 = 2e-2               # intial concentration   ng/l^3
D = 400                 # diffusion constant     l^2/min
Y = 12.634              # cells/glucose yield    cells/ng
gmax = 5.243e-3         # max growth rate        1/min      corresponds to division time of 190mins
Ks = 2e-5               # growth constant        ng/l^3
H_agar = 20             # height of agar         l
R_agar = 100            # radius of agar         l
dz = 1 / 4              # height step            l      try:  10l = 0.01cm
dr = 1                  # radial step            l      try:  10l = 0.01cm
                        #                               try   dtC = 0.001 hr
C_0 = c0 * np.ones((int(H_agar / dz), int(R_agar / dr)))


def genN(x):
    # fixed step size
    N = np.zeros(int(R_agar / dr))

    for i in range(np.size(x, axis=1)):
        r, l = x[0:2, i]
        Area = np.pi * r ** 2 + 2 * r * l

        r_i = int(np.sqrt(x[2, i] ** 2 + x[3, i] ** 2) / dr)

        N[r_i] += Area / (np.pi * dr ** 2 * (2 * r_i + 1))

    return N


def updateC(C0, N, dt):
    C = C0

    C[1:-1, 1:-1] += dt * D * (
            np.tile(1 / np.arange(2 * dr, R_agar, dr), (int(H_agar / dz) - 2, 1)) * \
            (C0[1:-1, 2:] - C0[1:-1, :-2]) / (2 * dr) + \
            (C0[1:-1, :-2] - 2 * C0[1:-1, 1:-1] + C0[1:-1, 2:]) / (dr ** 2) + \
            (C0[:-2, 1:-1] - 2 * C0[1:-1, 1:-1] + C0[2:, 1:-1]) / (dz ** 2))

    C[:, 0] = C[:, 1]
    C[:, int(R_agar / dr) - 1] = C[:, int(R_agar / dr) - 2]
    C[int(H_agar / dz) - 1, :] = C[int(H_agar / dz) - 2, :]

    C[0, :] = -gmax * dz * N / (2 * Y * D) + (C[1, :] - Ks) / 2 + \
              np.sqrt(Ks * C[1, :] + (gmax * dz * N / (2 * Y * D) + (Ks - C[0, :]) / 2) ** 2)

    return C


def continuousUpdate(C, N, T, dt):
    for i in np.arange(0, T, dt):
        C = updateC(C, N, dt)
    return C
