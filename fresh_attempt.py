# -*- coding: utf-8 -*-
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import *
import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt



def GenerateCells(N, A, r, L):
    
#   Generates a (5 x N) array of sizes (A, B), 2D-positions (x, y), and 
#   orientations (phi) in [-Pi,Pi) of all the cells

    x = np.vstack((A*np.ones(N),A/r*np.ones(N),L*np.random.rand(2,N),np.pi*(np.random.rand(N)-1/2)));

    return x


def PlotCells(x,L,l,r=2): #ellipse plotting module for cells (not final)
    
    ells = [Ellipse((x[2,i],x[3,i]),2*l,l,180/np.pi*x[4,i]) for i in range(np.size(x,axis=1))]
    
    fig = plt.figure(0)

    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    
    return np.size(ells)


def Translate(x, y, h, k):
#   Translates vector (x, y) by vector (h, k)
    
    return (x+h, y+k)


def Rotate(x, y, phi):
#   Rotates the vector (x, y) counter-clockwise arount the origin
    
    sinphi = np.sin(phi);
    cosphi = np.cos(phi);
    
    xR = x*cosphi - y*sinphi;
    yR = x*sinphi + y*cosphi;
    
    return (xR, yR)


def Coeff(A, B, h, k, phi):
#returns equation coefficients describing ellipse 'j' relative to ellipse 'i'

    
    sinphi = np.sin(phi);
    cosphi = np.cos(phi);
    
    
    AA = (cosphi/A)**2 + (sinphi/B)**2;
    BB = 2*sinphi*cosphi/A**2 - 2*sinphi*cosphi/B**2;
    CC = (sinphi/A)**2 + (cosphi/B)**2;
    DD = -2*cosphi*(cosphi*h + sinphi*k)/A**2 + 2*sinphi*(-sinphi*h + cosphi*k)/B**2;
    EE = -2*sinphi*(cosphi*h + sinphi*k)/A**2 + 2*cosphi*(sinphi*h - cosphi*k)/B**2;
    FF = ((cosphi*h + sinphi*k)/A)**2 +  ((sinphi*h - cosphi*k)/B)**2 - 1;
    
    return AA, BB, CC, DD, EE, FF




    
def FindNeighbors(i, d, x):
    
    h = x[2,i];
    k = x[3,i];
    
    r = np.array([]); 
    
    
    for j in range(np.size(x, axis=1)):
        
        r = np.hstack((r,np.array([np.sqrt((x[2,j] - h)**2 + (x[3,j] - k)**2)])));
        
    S = np.where(r <= d);
    S = S[0];
    S = [S[i] for i in range(np.size(S))]
    
    return S




def SolveX(y, V):
#   solves for values of x for a point (x,y) on ellipse vector V
    
    (AA, BB, CC, DD, EE, FF) = Coeff(V[0], V[1], V[2], V[3], V[4]);
    
    return np.roots([AA, BB*y + DD, CC*(y**2) + EE*y + FF])


def InterPoints(a, b):
#   Finds the points of intersection between two ellipse vectors 'a' and 'b'
    
    
    
    eps = 1e-5;
#   a[0] = A1;            b[0] = A2;
#   a[1] = B1;            b[1] = B2;
#   a[2] = x1;            b[2] = x2;
#   a[3] = y1;            b[3] = y2;
#   a[4] = phi1;          b[4] = phi2;
    
    (AA1, BB1, CC1, DD1, EE1, FF1) = Coeff(a[0],a[1],a[2],a[3],a[4]);
    (AA2, BB2, CC2, DD2, EE2, FF2) = Coeff(b[0],b[1],b[2],b[3],b[4]);
    
    
    cy = np.zeros(5);
    cy[4] = (BB1*CC2 - CC1*BB2)*(AA1*BB2 - BB1*AA2) - (AA1*CC2 - CC1*AA2)**2;
    cy[3] = 2*(EE1*AA2 - AA1*EE2)*(AA1*CC2 - CC1*AA2) + (DD1*CC2 + BB1*EE2 - EE1*BB2 - CC1*DD2)*(AA1*BB2 - BB1*AA2) + (BB1*CC2 - CC1*BB2)*(AA1*DD2 - DD1*AA2);
    cy[2] = 2*(FF1*AA2 - AA1*FF2)*(AA1*CC2 - CC1*AA2) + (DD1*EE2 + BB1*FF2 - FF1*BB2 - EE1*DD2)*(AA1*BB2 - BB1*AA2) + (DD1*CC2 + BB1*EE2 - EE1*BB2 - CC1*DD2)*(AA1*DD2 - DD1*AA2) - (AA1*EE2 - EE1*AA2)**2;
    cy[1] = 2*(FF1*AA2 - AA1*FF2)*(AA1*EE2 - EE1*AA2) + (DD1*EE2 + BB1*FF2 - FF1*BB2 - EE1*DD2)*(AA1*DD2 - DD1*AA2) + (DD1*FF2 - FF1*DD2)*(AA1*BB2 - BB1*AA2);
    cy[0] = (DD1*FF2 - FF1*DD2)*(AA1*DD2 - DD1*AA2) - (AA1*FF2 - FF1*AA2)**2;
    
    if (abs(cy[4])>0):
        py = np.ones(5);
        for i in range(5):
            py[i]=cy[4-i];
        r = np.roots(py);
        N = 4;
        
    elif (abs(cy[3])>0):
        py = np.ones(4);
        for i in range(4):
            py[i]=cy[3-i];
        r = np.roots(py);
        N = 3;
        
    elif (abs(cy[2])>0):
        py = np.ones(3);
        for i in range(3):
            py[i]=cy[2-i];
        r = np.roots(py);
        N = 2;
        
    elif (abs(cy[1])>0):
        r = -cy[0]/cy[1];
        N = 1;
        
    else: 
        N = 0;
        return(NaN)
    
    #for i in range(N):
        #r[i] = r[i]*a[1];
    
    n = 0; #counting real intersections
    Y = [];
    X1 = [];
    X2 = [];
    
    for i in range(N):
        if (abs(np.imag(r[i]))<eps):
            n += 1
            Y = np.hstack((Y,np.real(r[i])));
            X1 = np.hstack((X1,SolveX(np.real(r[i]),a)));
            X2 = np.hstack((X2,SolveX(np.real(r[i]),b)));
            
    X = [];
    
    
    

    j0 = -1;
    for i in range(2*n):
        j=0;
        while (abs(X1[i]-X2[j])>eps):
#            print('Fail for i=%s, j=%s' % (i,j))
            if (j==2*n-1):
                break
            elif ((j+1==j0) and (j0!=2*n-1)):
                j += 2;
            else:
                j += 1;
        if (abs(X1[i]-X2[j])<eps):
            if (abs(np.imag(X1[i]))<eps):
                X = np.hstack((X,np.real(X1[i])));
                j0 = j;
            
                
 #           print('Success for i=%s, j=%s' % (i,j))
       
    
    
    return X, Y




def EllipseSegment(v, X, Y): 

#   Returns the area of an arc on the ellipse 'v' minus the area of the 
#   triangular segment of that arc

#   'v' is a vector of (A, B, h, k, phi)
#   A is semi-major axis
#   B is semi-minor axis
#   (h,k) is the position of center of ellipse
#   phi is the orientation of ellipse relative to x-axis

#   X, Y are (ordered) vectors of the intersection points (X_i), (Y_i) 
#   X, Y are not relative to the ellipse, thus have no constraints on magnitude

    
    
    x = np.zeros(np.size(X));
    y = np.zeros(np.size(Y));
    
#   rotating and translating points in (X, Y) to the ellipse frame of reference
    
    for i in range(np.size(X)):
        x_t, y_t = Translate(X[i], Y[i], -v[2], -v[3]);
        x[i], y[i] = Rotate(x_t, y_t, -v[4]);
                     

    if (y[0]<0):
        theta1=2*np.pi-np.arccos(x[0]/v[0]);
    else:
        theta1=np.arccos(x[0]/v[0]);
        
    if (y[1]<0):
        theta2=2*np.pi-np.arccos(x[1]/v[0]);
    else:
        theta2=np.arccos(x[1]/v[0]);
        
    if (theta1>theta2):
        theta1-=2*np.pi;
        
    if ((theta2-theta1)>np.pi):
        trsign=1.0;
    else:
        trsign=-1.0;
        
    return 0.5*(v[0]*v[1]*(theta2-theta1)+trsign*abs(x[0]*y[1]-x[1]*y[0]))



def InFrameEllipseSegment(v, x, y, theta): 


#   in this case (X, Y) ARE relative to the ellipse, thus x<A and y<B



    if (theta[0]>theta[1]):
        theta[0] -=2*np.pi;
        
    if ((theta[1]-theta[0])>np.pi):
        trsign=1.0;
    else:
        trsign=-1.0;
        
    return 0.5*(v[0]*v[1]*(theta[1]-theta[0])+trsign*abs(x[0]*y[1]-x[1]*y[0]))




def twoPTarea(a, b, X, Y):
    
    areaA1 = EllipseSegment(a, X, Y);
    areaA2 = EllipseSegment(a, np.roll(X,1), np.roll(Y,1));
    areaB1 = EllipseSegment(b, np.roll(X,1), np.roll(Y,1));
    areaB2 = EllipseSegment(b, X, Y);

    maxArea = min(np.pi*a[0]*a[1],np.pi*b[0]*b[1]);
    
    if ((areaA1 + areaB1<=maxArea) and (areaA2 + areaB2<=maxArea)):
        print('Interesting!')
        return -1
    

    elif (min(areaA1+areaB1,areaA2+areaB2)/maxArea<1.05):
            OverallArea = min(areaA1+areaB1,areaA2+areaB2)
    
    else:
        print('Something somewhere went quite wrong')
        return 0
        
        
    return OverallArea    


def cirArea(R,r,d):
    
    return r**2*np.arccos((d**2+r**2-R**2)/(2*d*r))+R**2*np.arccos((d**2+R**2-r**2)/(2*d*R))-np.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R))/2




def Testing2area(N, max, min):
    
    x = np.array([[],[],[]]);
    for i in range(N):
        x = np.hstack((x, [[(max-min)*np.random.rand() + min],[2*max*np.random.rand()],[2*max*np.random.rand()]]));

    circles = np.zeros((N,N));
    ellipses= np.zeros((N,N));
    
    test = np.zeros((N,N));

    for i in range(N):
        for j in range(N):
            circles[i,j] = cirArea(x[0,i],x[0,j],np.sqrt((x[1,i]-x[1,j])**2 + (x[2,i]-x[2,j])**2));
            
            X, Y = InterPoints([x[0,i], x[0,i], x[1,i], x[2,i], 0], [x[0,j], x[0,j], x[1,j], x[2,j], 0])
            ellipses[i,j]= twoPTarea([x[0,i], x[0,i], x[1,i], x[2,i], 0],[x[0,j], x[0,j], x[1,j], x[2,j], 0], X, Y)
    
            test[i,j] = ellipses[i,j]/circles[i,j]
            
    return np.average(test)
    

    
def fourPTarea(a, b, x, y):
# returns the overlap area with 4 intersection points (Xint, Yint)

# input values (Xint, Yint) relative to one ellipse

    A1, B1 = a[0], a[1];
    A2, B2 = b[0], b[1];
    phi1, phi2 = a[4], b[4];
    h1, k1, h2, k2 = a[2], a[3], b[2], b[3];
    AA, BB, CC, DD, EE, FF = Coeff(A2, B2, h2 - h1, k2 - k1, phi2 - phi1);
    
    xint = [];
    yint = [];
    xint_tr = [];
    yint_tr = [];
    theta = np.zeros(4);
    theta_tr = np.zeros(4);    
    
    for i in range(4):
        xa_t, ya_t = Translate(x[i], y[i], -h1, -k1)
        xint, yint = np.hstack((xint,Rotate(xa_t, ya_t, -phi1)[0])), np.hstack((yint, Rotate(xa_t, ya_t, -phi1)[1]));
        xb_t, yb_t = Translate(x[i], y[i], -h2, -k2)
        xint_tr, yint_tr = np.hstack((xint_tr,Rotate(xb_t, yb_t, -phi2)[0])), np.hstack((yint_tr, Rotate(xb_t, yb_t, -phi2)[1]));

    
    for i in range(4):
        if (yint[i]<0):
            theta[i] = 2*np.pi - np.arccos(xint[i]/A1);
        else:
            theta[i] = np.arccos(xint[i]/A1);
            
        if (yint_tr[i]<0):
            theta_tr[i] = 2*np.pi - np.arccos(xint_tr[i]/A2);
        else:
            theta_tr[i] = np.arccos(xint_tr[i]/A2);
            
    # so far, so good
            
    # ordering having troubles:
            
    for i in range(1,4):
        
        tmp00 = theta[i];               tmp01 = theta_tr[i];
        tmp10 = xint[i];                tmp11 = xint_tr[i];
        tmp20 = yint[i];                tmp21 = yint_tr[i];
        
        for k in range(i-1, -1, -1):
            
            if (theta[k]>theta[k+1]):
                theta[k+1] =  theta[k];     theta_tr[k+1] =  theta_tr[k];
                xint[k+1] = xint[k];        xint_tr[k+1] = xint_tr[k];
                yint[k+1] = yint[k];        yint_tr[k+1] = yint_tr[k];
                theta[k] = tmp00;             theta_tr[k] = tmp01;
                xint[k] = tmp10;              xint_tr[k] = tmp11;
                yint[k] = tmp20;              yint_tr[k] = tmp21;
                
            else:
                break
                

    area1 = 0.5*abs((xint[2] - xint[0])*(yint[3] - yint[1]) - (xint[3] - xint[1])*(yint[2] - yint[0]));
    


    xmid = A1*np.cos((theta[0]+theta[1])/2);
    ymid = B1*np.sin((theta[0]+theta[1])/2);

    if (AA*xmid**2 + BB*xmid*ymid + CC*ymid**2 + DD*xmid + EE*ymid + FF < 0):
        area2 = InFrameEllipseSegment(a, xint[[0,1]], yint[[0,1]], theta[[0,1]]);
        area3 = InFrameEllipseSegment(a, xint[[2,3]], yint[[2,3]], theta[[2,3]]);
        area4 = InFrameEllipseSegment(b, xint_tr[[1,2]], yint_tr[[1,2]], theta_tr[[1,2]])
        area5 = InFrameEllipseSegment(b, xint_tr[[3,0]], yint_tr[[3,0]], theta_tr[[3,0]])
#        area2 = 0.5*(A1*B1*(theta[1] - theta[0]) - abs(xint[0]*yint[1] - xint[1]*yint[0]));
#        area3 = 0.5*(A1*B1*(theta[3] - theta[2]) - abs(xint[2]*yint[3] - xint[3]*yint[2]));
#        area4 = 0.5*(A2*B2*(theta_tr[2] - theta_tr[1]) - abs(xint_tr[1]*yint_tr[2] - xint_tr[2]*yint_tr[1]));
#        area5 = 0.5*(A2*B2*(theta_tr[0] - theta_tr[3] + 2*np.pi) - abs(xint_tr[3]*yint_tr[0] - xint_tr[0]*yint_tr[3]));
        
    else:
        area2 = InFrameEllipseSegment(a, xint[[1,2]], yint[[1,2]], theta[[1,2]]);
        area3 = InFrameEllipseSegment(a, xint[[3,0]], yint[[3,0]], theta[[3,0]]);
        area4 = InFrameEllipseSegment(b, xint_tr[[0,1]], yint_tr[[0,1]], theta_tr[[0,1]])
        area5 = InFrameEllipseSegment(b, xint_tr[[2,3]], yint_tr[[2,3]], theta_tr[[2,3]])

#        area2 = 0.5*(A1*B1*(theta[2] - theta[1]) - abs(xint[1]*yint[2] - xint[2]*yint[1]));
#        area3 = 0.5*(A1*B1*(theta[0] - theta[3] + 2*np.pi) - abs(xint[3]*yint[0] - xint[0]*yint[3]));
#        area4 = 0.5*(A2*B2*(theta[1] - theta[0]) - abs(xint_tr[0]*yint_tr[1] - xint_tr[1]*yint_tr[0]));
#        area5 = 0.5*(A2*B2*(theta[3] - theta[2]) - abs(xint_tr[2]*yint_tr[3] - xint_tr[3]*yint_tr[2]));
        
    
    return area1 + area2 + area3 + area4 + area5






    
def get_element(elements, x, y, N):
    # get (x,y) element from flattened 2D list
    return elements[x + (y * (N+1))]

def append_element(elements, x, y, N, value):
    # set (x,y) element in flattened 2D list
    elements[x + (y * (N+1))].append(value)
    
    
def BackgroundLattice(x,L,B):
    # Produces a Lattice on which one can approximate cell positions
    
# x is vector of cell metadata (size, position, orientation)
# L is width of simulation square
# B is short axis of ellipse
    
    a = B/L;
    N = np.int(np.ceil(1/a));    
    BG = [[] for k in range((N+1)**2)];
    
    for i in range(np.size(x, axis=1)):
        
        x1 = int(np.floor(x[2,i]/B));
        y1 = int(np.floor(x[3,i]/B)); 
        
        append_element(BG, x1, y1, N, i)
        
    return BG

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
    
#     A = np.array([[AA1,BB1/2,DD1/2],[BB1/2,CC1,EE1/2],[DD1/2,EE1/2,FF1]]);
#     B = np.array([[AA2,BB2/2,DD2/2],[BB2/2,CC2,EE2/2],[DD2/2,EE2/2,FF2]]);

#     if (s4<0):
        
#         RelPos = 2;
        
#     elif (s4>0):
#         s1 = a;
#         s2 = a**2 - 3*b;
#         s3 = 3*a*c + b*(a**2) - 4*(b**2);
    
#         if ((s1>0) and (s2>0) and (s3>0)):
            
#             u = (-a - np.sqrt(s2))/3;
#             v = (-a + np.sqrt(s2))/3;
            
#             M = u*A + B;
#             N = v*A + B;
            
#             M11 = minor(M,0,0);
#             N11 = minor(N,0,0);
            
#             if (((M[1,1]*np.linalg.det(M)>0) and (np.linalg.det(M11)>0)) or ((N[1,1]*np.linalg.det(N)>0) and (np.linalg.det(N11)>0))): 
#                 RelPos = 4;
#             else: 
#                 RelPos = 1;
        
#         else:
#             RelPos = 3;

#     return RelPos, s4,s3,s2,s1
            
            
def Intersections(x, A, B, L):

    
    N = np.int(np.ceil(L/B));
    r = int(np.ceil(A/B));
    
    S = [[] for k in range(np.size(x, axis=1))]
    
    BG = BackgroundLattice(x, L, B)
    BG[(N+1)**2:(N+1)**2] = [[] for i in range(r*(N+2*r+1))]
    for i in range(N, -1, -1):
        BG[(i+1)*(N+1):(i+1)*(N+1)] = [[] for k in range(r)];
        BG[i*(N+1):i*(N+1)] = [[] for k in range(r)];
    BG[:0] = [[] for i in range(r*(N+2*r+1))]

    I = [];
    
    for i in range(-r,r+1):
        for j in range(-r,r+1):
        
            if ((i!=0) or (j!=0)):
                I.append((i,j));
            else:
                continue
    

    for i in range(np.size(x, axis=1)):
        Xi, Yi = int(np.floor(x[2,i]/B)), int(np.floor(x[3,i]/B))
        
        for j in get_element(BG, Xi+r, Yi+r, N+2*r):
            if (i!=j):
                S[i].append(j)
            else:
                continue
                
        for (k,l) in I:
            for j in get_element(BG, (Xi+r) + k, (Yi+r) + l, N+2*r):
                dist = np.sqrt((x[2,i] - x[2,j])**2 + (x[3,i] - x[3,j])**2)
                    
                if (dist>x[0,i]+x[0,j]):
                    continue
                elif (dist<x[1,i]+x[1,j]):
                    S[i].append(j);
                elif (np.size(InterPoints(x[:,i],x[:,j])) > 0):
                    S[i].append(j);
                else: 
                    continue

                    
    return S
                    

def TestIntersections(S):
    
    for i in range(np.size(S)):
        for j in S[i]:
            if i in S[j]:
                continue
            else:
                print('Error')
                return i,j
            
    return('Success!')
            
