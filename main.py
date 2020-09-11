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
        intersectingCells = fresh_attempt.Intersections(
            Cprime, length, radius, L)


        for j in range(len(intersectingCells[n])):
            if intersectingCells[n][j][1][0] < radius / 2:
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

                possibleOrientations[i] = min(d)
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


def dynamic_update_step(x, attachments, dt, radius, length, L, rep=False, 
                        tau=10, elongationRate=0.02, kcc=1, gamma=0):
    eps = 3e-5
    t = len(x) - 1
    N0 = np.size(x[t], axis=1)

    x.append(x[t].copy())
    S = fresh_attempt.Intersections(x[t], length, radius, L)
    
    TG2 = 60

    # loop over cells adult cells
    for i in range(N0):


        if (0 < x[t + 1][8, i] <= 35) and (len(attachments[i]) == 0):

            for j in range(len(S[i])):  
                r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t][:8, i]
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
                
                VolI = np.pi * r1 ** 2 + 2 * r1 * l1
                VolJ = np.pi * r2 ** 2 + 2 * r2 * l2
                M = VolI * VolJ / (VolI + VolJ)
                
                force = (4/3 * kcc / np.sqrt(1/r1 + 1/r2) * np.sqrt(dcc) - 
                         gamma * M * np.dot(vcc, ncc)) * dcc * ncc
                torque = np.cross(rprime1, force)
                I = ((3*r1**2 + l1**2)/12 * l1*r1**2 + 
                     (2/5*r1**2 + l1**2/4) * 4/3*r1**2 )/(l1*r1**2 + 4/3*r1**2)*VolI


                x[t + 1][[2, 3], i] += force * dt / VolI
                x[t + 1][4, i] += torque[2] * dt / I

  
            if x[t + 1][9, i] > dt:
                x[t + 1][9, i] -= dt
                    
            else:
                x[t + 1][8, i] += 1
                x[t + 1][9, i] = 40

                if rep:
                    
                    k = random.uniform(low=-1, high=1)
                    phi = random.normal(loc=0, scale=np.pi/6)
                    
                    p = x[t + 1][2:4, i] + np.sign(k) * (np.array(
                        fresh_attempt.Rotate(x[t + 1][1, i]/2, 0, x[t + 1][4, i])) + 
                        np.array(fresh_attempt.Rotate(
                            x[t + 1][0, i] + (radius + length/2) * dt / TG2, 0, 
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
                

# growing the daughter cells in the G2 growth phase
# (elongationRate) variable depends on the local concentration of nutrients
        elif x[t + 1][8, i] == 0:

            if x[t + 1][0, i] < radius:

                dr = (radius/2 + length/4) * dt / TG2 * x[t + 1][9, i] * np.array(
                    fresh_attempt.Rotate(1, 0, x[t + 1][4, i]))

                x[t + 1][0, i] += radius * dt / TG2
                x[t + 1][1, i] += length * dt / TG2
                x[t + 1][2, i] += dr[0]
                x[t + 1][3, i] += dr[1]
                x[t + 1][2, attachments[i][0]] -= dr[0]
                x[t + 1][3, attachments[i][0]] -= dr[1]
                
                

# daughter cell finishes growing, detaches and becomes adult cell
            else:
                x[t + 1][8, i] = 1
                x[t + 1][9, i] = 40
                attachments[attachments[i][0]].remove(i)
                attachments[i].remove(attachments[i][0])
                
        elif len(attachments[i]) > 0:
            
            force = []
            torque = []
            r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t + 1][:8, i]
            r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t + 1][:8, attachments[i][0]]
            xmid = (x1*r1*(np.pi*r1 + 2*l1) + x2*r2*(np.pi*r2 + 2*l2)
                    )/(r1*(np.pi*r1 + 2*l1) + r2*(np.pi*r2 + 2*l2))
            ymid = (y1*r1*(np.pi*r1 + 2*l1) + y2*r2*(np.pi*r2 + 2*l2)
                    )/(r1*(np.pi*r1 + 2*l1) + r2*(np.pi*r2 + 2*l2))

                          
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
                
                VolI = np.pi * r1 ** 2 + 2 * r1 * l1
                VolJ = np.pi * r2 ** 2 + 2 * r2 * l2
                M = VolI * VolJ / (VolI + VolJ)
                F = (4/3 * kcc / np.sqrt(1/r1 + 1/r2) * np.sqrt(dcc) - 
                     gamma * M * np.dot(vcc, ncc)) * dcc * ncc
                T = np.cross(np.hstack((rcc - np.array([xmid, ymid]), 0)), F)
                force.append(F)
                torque.append(T)
                
            for j in attachments[i]:
                r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t][:8, j]
                for k in range(len(S[j])):
                    r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, S[j][k][0][0]]
                    ncc = (S[j][k][0][1] - S[j][k][1][1]) / S[j][k][1][0]
                    rcc = (S[j][k][0][1] + S[j][k][1][1]) / 2
                    dcc = r1 + r2 - S[j][k][1][0]
                    if dcc < 0:
                        dcc = 0
                    rprime1 = np.hstack((rcc - np.array([x1, y1]), 0))
                    rprime2 = np.hstack((rcc - np.array([x2, y2]), 0))
                    vcc = x[t][5:7, j] - x[t][5:7, S[j][k][0][0]] + np.cross(
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
                    F = (4/3 * kcc / np.sqrt(1/r1 + 1/r2) * np.sqrt(dcc) - 
                         gamma * M * np.dot(vcc, ncc)) * dcc * ncc
                    T = np.cross(np.hstack((rcc-np.array([xmid, ymid]), 0)), F)
                    force.append(F)
                    torque.append(T)

            r1, l1, x1, y1, theta1, vx1, vy1, omega1 = x[t][:8, i]
            r2, l2, x2, y2, theta2, vx2, vy2, omega2 = x[t][:8, attachments[i][0]]
            VolI = np.pi * r1 ** 2 + 2 * r1 * l1
            VolJ = np.pi * r2 ** 2 + 2 * r2 * l2
            M = VolI + VolJ
            d1 = np.sqrt((x1-xmid)**2 + (y1-ymid)**2)
            d2 = np.sqrt((x2-xmid)**2 + (y2-ymid)**2)
            I1 = (((3*r1**2 + l1**2)/12 * l1*r1**2 + 
                     (2/5*r1**2 + l1**2/4) * 4/3*r1**2 
                     )/(l1*r1**2 + 4/3*r1**2) + d1**2)*VolI
            I2 = (((3*r2**2 + l2**2)/12 * l2*r2**2 + 
                     (2/5*r2**2 + l2**2/4) * 4/3*r2**2 
                     )/(l2*r2**2 + 4/3*r2**2) + d2**2)*VolJ
            I = I1 + I2
   
            if (len(force) > 0) and (len(torque) > 0):
                for k in [i, attachments[i][0]]:
                    x[t + 1][[2, 3], k] -= np.array([xmid, ymid])
                    x[t + 1][[2, 3], k] = np.array(fresh_attempt.Rotate(
                        x[t + 1][2, k], x[t + 1][3, k], 
                        sum(torque)[2] * dt / I))
                    x[t + 1][[2, 3], k] += np.array([xmid, ymid]
                                                    ) + sum(force) * dt / M
                    x[t + 1][4, k] += sum(torque)[2] * dt / I
                    
              
    return x, attachments
        

"""


def add_ellipse(x, majorAxis, minorAxis, X, Y, theta):
    return np.hstack((x, np.vstack((majorAxis, minorAxis, X, Y, theta, 1))))

"""
