import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def PlotCells(x, i, size):  # ellipse plotting module for cells (not final)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

    for j in range(np.size(x[i], axis=1)):
        ax.add_artist(
            Ellipse((x[i][2, j], x[i][3, j]), 2 * x[i][0, j] + x[i][1, j], 
                    2 * x[i][0, j], 180 / np.pi * x[i][4, j], fc="none",
                    ec="blue"))

    plt.show()

    return np.size(x[i], axis=1)


def PlotFewCells(x, t, J):  # ellipse plotting module for cells (not final)

    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')
    
    X = x[t][2, J]
    Y = x[t][3, J]
    A = max(x[t][0,0], x[t][0,1])
    
    
    ax.set_xlim(min(X)-2*A, max(X)+2*A)
    ax.set_ylim(min(Y)-2*A, max(Y)+2*A)

    for j in J:
        ax.add_artist(
            Ellipse((x[t][2, j], x[t][3, j]), 2 * x[i][0, j] + x[i][1, j], 
                    2 * x[i][0, j], 180 / np.pi * x[t][4, j], fc="none",
                    ec="blue"))
    
    return plt.show()


#   this initializes the Figure, Canvas, & Axes for the animation
def anim_init(size):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
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


#   this generates all the image files, writes the GIF animation, then deletes the image file
def animate(x, size, filename):
    image_list = []
    for i in range(len(x)):
        fig, canvas, ax = anim_init(size)                       # initialize Figure
        ax = ells(x, i, ax)                                     # add ellipses
        fig.savefig('plots/test.png')                           # save plot to temporary image file
        image_list.append(imageio.imread('plots/test.png'))     # transform image file into NumPy array
    os.remove('plots/test.png')                                 # delete the image file
    imageio.mimwrite('plots/{}.gif'.format(str(filename)), image_list)             # compile image_list into GIF


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
    
    p1 = x[2:4, i] + np.array(Rotate(x[1, i]/2, 0, x[4, i]))
    p2 = x[2:4, j] + np.array(Rotate(x[1, j]/2, 0, x[4, j]))
    p = np.vstack((p1, p2))
    
    q1 = x[2:4, i] - np.array(Rotate(x[1, i]/2, 0, x[4, i]))
    q2 = x[2:4, j] - np.array(Rotate(x[1, j]/2, 0, x[4, j]))
    q = np.vstack((q1, q2))  
    u = q - p
    
    p = np.vstack((p1 - p2, p))
    
    def f(t):
        return np.dot(t[0]*u[0,:] - t[1]*u[1,:] + p[0,:], 
                      t[0]*u[0,:] - t[1]*u[1,:] + p[0,:])
    
    C = np.mod(np.abs(x[4, i] - x[4, j]), np.pi)
    
    if C>1e-5:
        
        cross = np.cross(u[0,:], u[1,:])
        normC = np.linalg.norm(cross)
        s0 = np.dot(np.cross(p[0,:], u[1,:]), cross) / normC ** 2
        t0 = np.dot(np.cross(p[0,:], u[0,:]), cross) / normC ** 2
        
        if (0 <= s0 <= 1) and (0 <= t0 <= 1):
            
            d = np.sqrt(f((s0, t0)))
            
            a = np.add(p[1:3,:],  np.vstack((s0*u[0,:], t0*u[1,:])))
            
        else:
            
            mini = np.empty((4,3))
            # case 1: s=0, 0<=t<=1
            mini[0,:2] = [0, np.dot(p[0,:], u[1,:])/np.linalg.norm(u[1,:])**2]

                
            # case 2: s=1, 0<=t<=1
            mini[1,:2] = [1, np.dot((p[0,:] + u[0,:]), u[1,:])/
                          np.linalg.norm(u[1,:])**2]

            
            # case 3: 0<=s<=1, t=0
            mini[2,:2] = [-np.dot(p[0,:], u[0,:])/np.linalg.norm(u[0,:])**2, 0]

                
            # case 4: 0<=s<=1, t=1
            mini[3,:2] = [np.dot((u[1,:] - p[0,:]), u[0,:])/
                          np.linalg.norm(u[0,:])**2, 1]

            if 0 <= mini[0,1] <= 1:
                mini[0, 2] = f((mini[0,0], mini[0,1]))
            else:
                mini[0, :] = (0, 0, f((0, 0)))
                mini = np.vstack((mini, (0, 1, f((0, 1)))))
                
            if 0 <= mini[1,1] <= 1:
                mini[1, 2] = f((mini[1,0], mini[1,1]))
            else:
                mini[1, :] = (1, 0, f((1, 0)))
                mini = np.vstack((mini, (1, 1, f((1, 1)))))
                
            if 0 <= mini[2,0] <= 1:
                mini[2, 2] = f((mini[2,0], mini[2,1]))
            else:
                mini[2, :] = (0, 0, f((0, 0)))
                mini = np.vstack((mini, (1, 0, f((1, 0)))))
                
            if 0 <= mini[3,0] <= 1:
                mini[3, 2] = f((mini[3,0], mini[3,1]))
            else:
                mini[3, :] = (0, 1, f((0, 1)))
                mini = np.vstack((mini, (1, 1, f((1, 1)))))
                
                       
                
            d2 = np.min(mini, axis=0)[2]
            
            t = mini[np.where(np.abs(mini[:, 2]-d2)<1e-5)[0][0], :2]
            
            d = np.sqrt(d2)
            
            a = np.add(p[1:3, :], np.vstack((t[0]*u[0, :], t[1]*u[1, :])))
            
            
    else:
        
        uNorm = np.linalg.norm(u[1, :]) ** 2
        
        t = np.array([np.dot(p[0, :], u[1, :])/uNorm, 
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
            
        d2 = np.array([np.linalg.norm(p[2, :] + t[0]*u[1, :] - p[1, :]), 
                   np.linalg.norm(p[2, :] + t[1]*u[1, :] - q[0, :])])
        
        d = np.min(d2)
        
        index = np.where(np.abs(d2 - d) < 1e-5)[0][0]
        
        a = np.vstack((p[1, :], q[0, :]))[index, :]
        
        a = np.vstack((a, p[2, :] + t[index]*u[1, :]))
        
    return d, a
                

def BackgroundLattice(x, L, radius):
    # Produces a Lattice on which one can approximate cell positions

    # x is vector of cell metadata (size, position, orientation)
    # L is width of simulation square
    # radius is short axis of ellipse


    a = radius / L
    N = np.int(np.ceil(1 / a))
    BG = [[[] for i in range(N)] for j in range(N)]

    for k in range(np.size(x, axis=1)):
        Xk = int(np.floor(x[2, k] / radius))
        Yk = int(np.floor(x[3, k] / radius))

        BG[Xk][Yk].append(k)

    return BG


def pad(ulist, r=2):
    
    N = len(ulist)
    empty_rows=[[[]]*(N+2*r)]*r
    olist = empty_rows.copy()
    
    for i in range(N):
        olist += [[[]]*r + ulist[i] + [[]]*r]
    
    olist += empty_rows
    
    return olist


def Intersections(x, length, radius, L):
    r = int(np.ceil(length / radius)) + 1

    # initialize intersection index array
    n = np.size(x, axis=1)
    intersectingCells = [[] for k in range(n)]
    
    # initialize intersection data array
    d = np.full((n,n), np.nan)
    a = np.full((n,n,2,2), np.nan)

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
        Xi, Yi = int(np.floor(x[2, i] / radius)), int(np.floor(x[3, i] / 
                                                                  radius))

        for j in BG[Xi + r][Yi + r]:
            if i != j:
                intersectingCells[i].append(j)
                d[i, j], a[i, j] = mindist(x, i, j)
                d[j, i], a[j, i] = d[i, j], np.roll(a[i, j], 1, axis=0)
            else:
                continue

        for (k, l) in I:
            for j in BG[(Xi + r) + k][(Yi + r) + l]:
                
                if mindist(x, i, j)[0] >= x[0, i] + x[0, j]:
                    continue
                else:
                    intersectingCells[i].append(j)
                    d[i, j], a[i, j] = mindist(x, i, j)
                    d[j, i], a[j, i] = d[i, j], np.roll(a[i, j], 1, axis=0)


    return intersectingCells, d, a


def Reproduce(x, attachments, tau, dt=1):
    #   Creates new cells to add to the list of existing cells occupying space
    #   tau is reproduction half-life per individual
    #   this step does NOT update the time i.e. cells do not move in this update step

    t = len(x) - 1
    N = np.size(x[t], axis=1)
    dn = np.random.poisson(N * dt / tau)
    reproducingCells = np.random.choice(np.arange(N), dn, replace=False)
    phis = np.random.normal(0, np.pi / 6, dn)

    attachments.extend([[] for i in range(dn)])

    for i in range(dn):

        attachments[reproducingCells[i]].append(N - 1 + i)
        attachments[N - 1 + i].append(reproducingCells[i])

        # k gives a uniform chance of budding on either side of the mother cell
        k = random.uniform()
        if k < 0.5:

            xmid, ymid = Rotate(x[t][0, reproducingCells[i]] * np.cos(phis[i]),
                                x[t][1, reproducingCells[i]] * np.sin(phis[i]), x[t][4, reproducingCells[i]])
            xmid += x[t][2, reproducingCells[i]]
            ymid += x[t][3, reproducingCells[i]]

            theta = x[t][4, reproducingCells[i]] + np.arctan(
                x[t][0, reproducingCells[i]] / x[t][1, reproducingCells[i]] * np.tan(phis[i]))

            a0 = x[t][0, reproducingCells[i]] / 5
            b0 = x[t][1, reproducingCells[i]] / 5
            xcen = xmid + a0 * np.cos(theta)
            ycen = ymid + b0 * np.sin(theta)

        else:
            xmid, ymid = Rotate(-x[t][0, reproducingCells[i]] * np.cos(phis[i]),
                                -x[t][1, reproducingCells[i]] * np.sin(phis[i]), x[t][4, reproducingCells[i]])
            xmid += x[t][2, reproducingCells[i]]
            ymid += x[t][3, reproducingCells[i]]
            theta = x[t][4, reproducingCells[i]] + np.arctan(
                x[t][0, reproducingCells[i]] / x[t][1, reproducingCells[i]] * np.tan(phis[i]))
            a0 = x[t][0, reproducingCells[i]] / 5
            b0 = x[t][1, reproducingCells[i]] / 5
            xcen = xmid - a0 * np.cos(theta)
            ycen = ymid - b0 * np.sin(theta)

        x[t] = np.hstack((x[t], [[a0], [b0], [xcen], [ycen], [theta], [0], [0]]))
        x[t][5, reproducingCells[i]] += 1

    return x, attachments
