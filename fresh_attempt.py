import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely import affinity
from matplotlib.patches import Polygon


def PlotCells(x, size):  # ellipse plotting module for cells (not final)

    ells = [Ellipse((x[2, i], x[3, i]), 2 * x[0, i], 2 * x[1, i], 180 / np.pi * x[4, i]) for i in
            range(np.size(x, axis=1))]

    fig = plt.figure(0)

    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

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


# def Coeff(majorAxis, minorAxis, h, k, phi):
#     #   returns equation coefficients describing ellipse centered at (h, k) with orientation phi CCW relative to +x axis
#
#     sinphi = np.sin(phi)
#     cosphi = np.cos(phi)
#
#     AA = (cosphi / majorAxis) ** 2 + (sinphi / minorAxis) ** 2
#     BB = 2 * sinphi * cosphi / majorAxis ** 2 - 2 * sinphi * cosphi / minorAxis ** 2
#     CC = (sinphi / majorAxis) ** 2 + (cosphi / minorAxis) ** 2
#     DD = -2 * cosphi * (cosphi * h + sinphi * k) / majorAxis ** 2 + 2 * sinphi * (-sinphi * h + cosphi * k) / minorAxis ** 2
#     EE = -2 * sinphi * (cosphi * h + sinphi * k) / majorAxis ** 2 + 2 * cosphi * (sinphi * h - cosphi * k) / minorAxis ** 2
#     FF = ((cosphi * h + sinphi * k) / majorAxis) ** 2 + ((sinphi * h - cosphi * k) / minorAxis) ** 2 - 1
#
#     return AA, BB, CC, DD, EE, FF

def Coeff(v):
    #   returns equation coefficients describing ellipse centered at (h, k) with orientation phi CCW relative to +x axis

    majorAxis = v[0]
    minorAxis = v[1]
    h = v[2]
    k = v[3]
    phi = v[4]
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    AA = (cosphi / majorAxis) ** 2 + (sinphi / minorAxis) ** 2
    BB = 2 * sinphi * cosphi / majorAxis ** 2 - 2 * sinphi * cosphi / minorAxis ** 2
    CC = (sinphi / majorAxis) ** 2 + (cosphi / minorAxis) ** 2
    DD = -2 * cosphi * (cosphi * h + sinphi * k) / majorAxis ** 2 + 2 * sinphi * (
                -sinphi * h + cosphi * k) / minorAxis ** 2
    EE = -2 * sinphi * (cosphi * h + sinphi * k) / majorAxis ** 2 + 2 * cosphi * (
                sinphi * h - cosphi * k) / minorAxis ** 2
    FF = ((cosphi * h + sinphi * k) / majorAxis) ** 2 + ((sinphi * h - cosphi * k) / minorAxis) ** 2 - 1

    return AA, BB, CC, DD, EE, FF


def SolveX(y, V):
    #   solves for values of x for a point (x,y) on ellipse vector V

    (AA, BB, CC, DD, EE, FF) = Coeff(V)

    return np.roots([AA, BB * y + DD, CC * (y ** 2) + EE * y + FF])


def InterPoints(a, b, eps=1e-5):
    #   Finds the points of intersection between two ellipse vectors 'a' and 'b'

    #   a[0] = A1;            b[0] = A2;
    #   a[1] = B1;            b[1] = B2;
    #   a[2] = x1;            b[2] = x2;
    #   a[3] = y1;            b[3] = y2;
    #   a[4] = phi1;          b[4] = phi2;

    (AA1, BB1, CC1, DD1, EE1, FF1) = Coeff(a)
    (AA2, BB2, CC2, DD2, EE2, FF2) = Coeff(b)

    cy = np.zeros(5)
    cy[4] = (BB1 * CC2 - CC1 * BB2) * (AA1 * BB2 - BB1 * AA2) - (AA1 * CC2 - CC1 * AA2) ** 2
    cy[3] = 2 * (EE1 * AA2 - AA1 * EE2) * (AA1 * CC2 - CC1 * AA2) + (DD1 * CC2 + BB1 * EE2 - EE1 * BB2 - CC1 * DD2) * (
            AA1 * BB2 - BB1 * AA2) + (BB1 * CC2 - CC1 * BB2) * (AA1 * DD2 - DD1 * AA2)
    cy[2] = 2 * (FF1 * AA2 - AA1 * FF2) * (AA1 * CC2 - CC1 * AA2) + (DD1 * EE2 + BB1 * FF2 - FF1 * BB2 - EE1 * DD2) * (
            AA1 * BB2 - BB1 * AA2) + (DD1 * CC2 + BB1 * EE2 - EE1 * BB2 - CC1 * DD2) * (AA1 * DD2 - DD1 * AA2) - (
                    AA1 * EE2 - EE1 * AA2) ** 2
    cy[1] = 2 * (FF1 * AA2 - AA1 * FF2) * (AA1 * EE2 - EE1 * AA2) + (DD1 * EE2 + BB1 * FF2 - FF1 * BB2 - EE1 * DD2) * (
            AA1 * DD2 - DD1 * AA2) + (DD1 * FF2 - FF1 * DD2) * (AA1 * BB2 - BB1 * AA2)
    cy[0] = (DD1 * FF2 - FF1 * DD2) * (AA1 * DD2 - DD1 * AA2) - (AA1 * FF2 - FF1 * AA2) ** 2

    if abs(cy[4]) > 0:
        py = np.ones(5)
        for i in range(5):
            py[i] = cy[4 - i]
        r = np.roots(py)
        N = 4

    elif abs(cy[3]) > 0:
        py = np.ones(4)
        for i in range(4):
            py[i] = cy[3 - i]
        r = np.roots(py)
        N = 3

    elif abs(cy[2]) > 0:
        py = np.ones(3)
        for i in range(3):
            py[i] = cy[2 - i]
        r = np.roots(py)
        N = 2

    elif abs(cy[1]) > 0:
        r = np.array([-cy[0] / cy[1]])
        N = 1

    else:
        r = np.array([])
        N = 0

    X1 = np.array([])
    X2 = np.array([])
    Y = []

    for i in range(N):
        if abs(np.imag(r[i])) < eps:
            X1 = np.hstack((X1, SolveX(np.real(r[i]), a)))
            X2 = np.hstack((X2, SolveX(np.real(r[i]), b)))
            Y = np.hstack((Y, np.real(r[i])))

    X = X1[(np.abs(X2[:, None] - X1) < eps).any(0)]

    #if np.size(X) != np.size(Y):
    #    print(np.array([[X1, X2], [X, Y]]))

    return X, Y


def EllipseSegment(a, X, Y, test=False, b=None):
    #   Returns the area of an arc on the ellipse 'v' minus the area of the
    #   triangular segment of that arc

    #   'a' is a vector of (majorAxis, minorAxis, h, k, phi, ...)
    #   majorAxis is semi-major axis
    #   minorAxis is semi-minor axis
    #   (h,k) is the position of center of ellipse
    #   phi is the orientation of ellipse relative to x-axis

    #   X, Y are (ordered) vectors of the intersection points (X_i), (Y_i)
    #   X, Y are not relative to the ellipse, thus have no constraints on magnitude
    #   The arc is obtained by traversing the ellipse from (X1, Y1) to (X2, Y2)

    #   rotating and translating points in (X, Y) to the ellipse frame of reference

    x = (X - a[2]) * np.cos(a[4]) + (Y - a[3]) * np.sin(a[4])
    y = -(X - a[2]) * np.sin(a[4]) + (Y - a[3]) * np.cos(a[4])
    theta = np.zeros(2)

    for i in range(2):
        if abs(x[i]) > a[0]:
            x[i] = np.sign(x[i]) * a[0]
        if y[i] < 0:
            theta[i] = 2 * np.pi - np.arccos(x[i] / a[0])
        else:
            theta[i] = np.arccos(x[i] / a[0])

    #   rearrange set of points (X,Y) if midpoint is outside second ellipse
    #   this is to ensure correct direction of integration for the overlap area
    if test:
        if theta[0] > theta[1]:
            tmp = theta[0]
            theta[0] = theta[1]
            theta[1] = tmp

        #   find midpoint in ellipse reference frame
        xmid = a[0] * np.cos((theta[0] + theta[1]) / 2)
        ymid = a[1] * np.sin((theta[0] + theta[1]) / 2)

        #   transform to background frame
        xtr = xmid * np.cos(a[4]) - ymid * np.sin(a[4]) + a[2]
        ytr = xmid * np.sin(a[4]) + ymid * np.cos(a[4]) + a[3]

        #   finds polynomial coeffiecients for ellipse 'b'
        AA, BB, CC, DD, EE, FF = Coeff(b)

        #   tests whether midpoint is outside ellipse
        if AA * xtr ** 2 + BB * xtr * ytr + CC * ytr ** 2 + DD * xtr + EE * ytr + FF > 0:
            #   rearrange the order of points (x1, y1) and (x2, y2) since theta represents this
            tmp = theta[0]
            theta[0] = theta[1]
            theta[1] = tmp

    if theta[0] > theta[1]:
        theta[0] -= 2 * np.pi

    if (theta[1] - theta[0]) > np.pi:
        trsign = 1.0
    else:
        trsign = -1.0

    return 0.5 * (a[0] * a[1] * (theta[1] - theta[0]) + trsign * abs(x[0] * y[1] - x[1] * y[0]))


def InFrameEllipseSegment(v, x, y, theta):
    #   in this case (X, Y) ARE relative to the ellipse, thus x<majorAxis and y<minorAxis

    if theta[0] > theta[1]:
        theta[0] -= 2 * np.pi

    if (theta[1] - theta[0]) > np.pi:
        trsign = 1.0
    else:
        trsign = -1.0

    return 0.5 * (v[0] * v[1] * (theta[1] - theta[0]) + trsign * abs(x[0] * y[1] - x[1] * y[0]))


def twoPTarea(a, b, X, Y):
    area1 = EllipseSegment(a, X, Y, True, b)
    area2 = EllipseSegment(b, X, Y, True, a)

    return area1 + area2


def create_ellipse(center, lengths, angle=0):

    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)

    return ellr


def ShapelyArea(a, b):

    ellipse1 = create_ellipse((a[0],a[1]),(a[2],a[3]),a[4]*180/np.pi)
    ellipse2 = create_ellipse((b[0],b[1]),(b[2],b[3]),b[4]*180/np.pi)
    intersect = ellipse1.intersection(ellipse2)

    return intersect.area


# def fourPTarea(a, b, x, y):
#     # returns the overlap area with 4 intersection points (Xint, Yint)
#
#     # input values (Xint, Yint) relative to one ellipse
#
#     A1, B1 = a[0], a[1]
#     A2, B2 = b[0], b[1]
#     phi1, phi2 = a[4], b[4]
#     h1, k1, h2, k2 = a[2], a[3], b[2], b[3]
#     v=[A2, B2, h2 - h1, k2 - k1, phi2 - phi1]
#     AA, BB, CC, DD, EE, FF = Coeff(v)
#
#     xint = []
#     yint = []
#     xint_tr = []
#     yint_tr = []
#     theta = np.zeros(4)
#     theta_tr = np.zeros(4)
#
#     for i in range(4):
#         xa_t, ya_t = Translate(x[i], y[i], -h1, -k1)
#         xint, yint = np.hstack((xint, Rotate(xa_t, ya_t, -phi1)[0])), np.hstack((yint, Rotate(xa_t, ya_t, -phi1)[1]))
#         xb_t, yb_t = Translate(x[i], y[i], -h2, -k2)
#         xint_tr, yint_tr = np.hstack((xint_tr, Rotate(xb_t, yb_t, -phi2)[0])), np.hstack(
#             (yint_tr, Rotate(xb_t, yb_t, -phi2)[1]))
#
#     for i in range(4):
#         if yint[i] < 0:
#             theta[i] = 2 * np.pi - np.arccos(xint[i] / A1)
#         else:
#             theta[i] = np.arccos(xint[i] / A1)
#
#         if yint_tr[i] < 0:
#             theta_tr[i] = 2 * np.pi - np.arccos(xint_tr[i] / A2)
#         else:
#             theta_tr[i] = np.arccos(xint_tr[i] / A2)
#
#     #   re-arranging in counter-clockewise order
#
#     for i in range(1, 4):
#
#         tmp00 = theta[i]
#         tmp01 = theta_tr[i]
#         tmp10 = xint[i]
#         tmp11 = xint_tr[i]
#         tmp20 = yint[i]
#         tmp21 = yint_tr[i]
#
#         for k in range(i - 1, -1, -1):
#
#             if theta[k] > theta[k + 1]:
#                 theta[k + 1] = theta[k]
#                 theta_tr[k + 1] = theta_tr[k]
#                 xint[k + 1] = xint[k]
#                 xint_tr[k + 1] = xint_tr[k]
#                 yint[k + 1] = yint[k]
#                 yint_tr[k + 1] = yint_tr[k]
#                 theta[k] = tmp00
#                 theta_tr[k] = tmp01
#                 xint[k] = tmp10
#                 xint_tr[k] = tmp11
#                 yint[k] = tmp20
#                 yint_tr[k] = tmp21
#
#             else:
#                 break
#
#     area1 = 0.5 * abs((xint[2] - xint[0]) * (yint[3] - yint[1]) - (xint[3] - xint[1]) * (yint[2] - yint[0]))
#
#     xmid = A1 * np.cos((theta[0] + theta[1]) / 2)
#     ymid = B1 * np.sin((theta[0] + theta[1]) / 2)
#
#     if AA * xmid ** 2 + BB * xmid * ymid + CC * ymid ** 2 + DD * xmid + EE * ymid + FF < 0:
#         area2 = InFrameEllipseSegment(a, xint[[0, 1]], yint[[0, 1]], theta[[0, 1]])
#         area3 = InFrameEllipseSegment(a, xint[[2, 3]], yint[[2, 3]], theta[[2, 3]])
#         area4 = InFrameEllipseSegment(b, xint_tr[[1, 2]], yint_tr[[1, 2]], theta_tr[[1, 2]])
#         area5 = InFrameEllipseSegment(b, xint_tr[[3, 0]], yint_tr[[3, 0]], theta_tr[[3, 0]])
#     #   area2 = 0.5*(A1*B1*(theta[1] - theta[0]) - abs(xint[0]*yint[1] - xint[1]*yint[0]));
#     #   area3 = 0.5*(A1*B1*(theta[3] - theta[2]) - abs(xint[2]*yint[3] - xint[3]*yint[2]));
#     #   area4 = 0.5*(A2*B2*(theta_tr[2] - theta_tr[1]) - abs(xint_tr[1]*yint_tr[2] - xint_tr[2]*yint_tr[1]));
#     #   area5 = 0.5*(A2*B2*(theta_tr[0] - theta_tr[3] + 2*np.pi) - abs(xint_tr[3]*yint_tr[0] - xint_tr[0]*yint_tr[3]));
#
#     else:
#         area2 = InFrameEllipseSegment(a, xint[[1, 2]], yint[[1, 2]], theta[[1, 2]])
#         area3 = InFrameEllipseSegment(a, xint[[3, 0]], yint[[3, 0]], theta[[3, 0]])
#         area4 = InFrameEllipseSegment(b, xint_tr[[0, 1]], yint_tr[[0, 1]], theta_tr[[0, 1]])
#         area5 = InFrameEllipseSegment(b, xint_tr[[2, 3]], yint_tr[[2, 3]], theta_tr[[2, 3]])
#
#     #   area2 = 0.5*(A1*B1*(theta[2] - theta[1]) - abs(xint[1]*yint[2] - xint[2]*yint[1]));
#     #   area3 = 0.5*(A1*B1*(theta[0] - theta[3] + 2*np.pi) - abs(xint[3]*yint[0] - xint[0]*yint[3]));
#     #   area4 = 0.5*(A2*B2*(theta[1] - theta[0]) - abs(xint_tr[0]*yint_tr[1] - xint_tr[1]*yint_tr[0]));
#     #   area5 = 0.5*(A2*B2*(theta[3] - theta[2]) - abs(xint_tr[2]*yint_tr[3] - xint_tr[3]*yint_tr[2]));
#
#     return area1 + area2 + area3 + area4 + area5


def BackgroundLattice(x, L, minorAxis):
    # Produces a Lattice on which one can approximate cell positions

    # x is vector of cell metadata (size, position, orientation)
    # L is width of simulation square
    # minorAxis is short axis of ellipse

    def append_element(elements, x, y, N, value):
        # set (x,y) element in flattened 2D list
        elements[x + (y * (N + 1))].append(value)

    a = minorAxis / L
    N = np.int(np.ceil(1 / a))
    BG = [[] for k in range((N + 1) ** 2)]

    for i in range(np.size(x, axis=1)):
        Xi = int(np.floor(x[2, i] / minorAxis))
        Yi = int(np.floor(x[3, i] / minorAxis))

        append_element(BG, Xi, Yi, N, i)

    return BG


# Relative Position code:
# def minor(arr, i, j):
#     minor = np.delete(np.delete(arr, i, axis=0), j, axis=1)
#     return minor

# def RelPos(a, b):

#     AA1, BB1, CC1, DD1, EE1, FF1 = Coeff(a[0], a[1], a[2], a[3], a[4]);
#     AA2, BB2, CC2, DD2, EE2, FF2 = Coeff(b[0], b[1], b[2], b[3], b[4]);

#     d = AA1*(CC1*FF1 - EE1**2) - (CC1*(DD1**2)-2*BB1*DD1*EE1 + FF1*(BB1**2));
#     a = 1/d*(AA1*(CC1*FF2 - 2*EE1*EE2 + FF1*CC2) + 2*BB1*(EE1*DD2 - FF1*BB2 + DD1*EE2) + 2*DD1*(EE1*BB2 - CC1*DD2) - ((BB1**2)*FF2 + (DD1**2)*CC2 + (EE1**2)*AA2) + (CC1*FF1*AA2));
#     b = 1/d*(AA1*(CC2*FF2-EE2**2) + 2*BB1*(EE2*DD2 - FF2*BB2) + 2*DD1*(EE2*BB2 - CC2*DD2) + CC1*(AA2*FF2 - DD2**2) + 2*EE1*(BB2*DD2 - AA2*EE2) + FF1*(AA2*CC2 - BB2**2));
#     c = 1/d*(AA2*(CC2*FF2 - EE2**2) - ((BB2**2)*FF2 - 2*BB2*DD2*EE2 + (DD2**2)*CC2));


#     s4 = -27*(c**2) + 18*c*a*b + (a**2)*(b**2) - 4*(a**3)*c - 4*(b**3);

#     majorAxis = np.array([[AA1,BB1/2,DD1/2],[BB1/2,CC1,EE1/2],[DD1/2,EE1/2,FF1]]);
#     minorAxis = np.array([[AA2,BB2/2,DD2/2],[BB2/2,CC2,EE2/2],[DD2/2,EE2/2,FF2]]);

#     if (s4<0):

#         RelPos = 2;

#     elif (s4>0):
#         s1 = a;
#         s2 = a**2 - 3*b;
#         s3 = 3*a*c + b*(a**2) - 4*(b**2);

#         if ((s1>0) and (s2>0) and (s3>0)):

#             u = (-a - np.sqrt(s2))/3;
#             v = (-a + np.sqrt(s2))/3;

#             M = u*majorAxis + minorAxis;
#             N = v*majorAxis + minorAxis;

#             M11 = minor(M,0,0);
#             N11 = minor(N,0,0);

#         if (((M[1,1]*np.linalg.det(M)>0) and (np.linalg.det(M11)>0)) or ((N[1,1]*np.linalg.det(N)>0) and
#           (np.linalg.det(N11)>0))): RelPos = 4; else: RelPos = 1;

#         else:
#             RelPos = 3;

#     return RelPos, s4,s3,s2,s1


def Intersections(x, majorAxis, minorAxis, L):
    N = np.int(np.ceil(L / minorAxis))
    r = int(np.ceil(majorAxis / minorAxis))

    intersectingCells = [[] for k in range(np.size(x, axis=1))]

    # set up coarse-grained background lattice (as a flattened array)
    BG = BackgroundLattice(x, L, minorAxis)

    # pad the lattice array with extra empty rows and columns for grid searching
    BG[(N + 1) ** 2:(N + 1) ** 2] = [[] for i in range(r * (N + 2 * r + 1))]
    for i in range(N, -1, -1):
        BG[(i + 1) * (N + 1):(i + 1) * (N + 1)] = [[] for k in range(r)]
        BG[i * (N + 1):i * (N + 1)] = [[] for k in range(r)]
    BG[:0] = [[] for i in range(r * (N + 2 * r + 1))]

    # retrieval function from flattened array
    def get_element(elements, x, y, N):
        # get (x,y) element from flattened 2D list
        return elements[x + (y * (N + 1))]

    I = []

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):

            if (i != 0) or (j != 0):
                I.append((i, j))
            else:
                continue

    for i in range(np.size(x, axis=1)):
        Xi, Yi = int(np.floor(x[2, i] / minorAxis)), int(np.floor(x[3, i] / minorAxis))

        for j in get_element(BG, Xi + r, Yi + r, N + 2 * r):
            if i != j:
                intersectingCells[i].append(j)
            else:
                continue

        for (k, l) in I:
            for j in get_element(BG, (Xi + r) + k, (Yi + r) + l, N + 2 * r):
                dist = np.sqrt((x[2, i] - x[2, j]) ** 2 + (x[3, i] - x[3, j]) ** 2)

                if dist > x[0, i] + x[0, j]:
                    continue
                elif dist < x[1, i] + x[1, j]:
                    intersectingCells[i].append(j)
                else:
                    X, Y = InterPoints(x[:, i], x[:, j])
                    if np.size(X)>0 and                     
                    intersectingCells[i].append(j)
                else:
                    continue

    return intersectingCells


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
