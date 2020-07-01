# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:07:26 2020

@author: eiadh
"""


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import *
import numpy as np
from numpy import random
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


# In[68]:


def GenerateCells(N,L,l): # Generates uniformly dispersed cells in a box

    global t
    
    x=np.ones((4,N));
    t=0
    
    x[0]=l*x[0];
    x[1]=L*np.random.rand(N);
    x[2]=L*np.random.rand(N);
    x[3]=np.pi*np.random.rand(N)-np.pi/2;
    
    
    

    return x


# In[146]:


def IterateForward(gamma,l=1,dt=1,sigma=0.5): # Adds new cells to the list of existing cells occupying space
    
    global x
    N=np.size(x,axis=1)
    
    dT=np.max(sigma*np.random.randn()+dt,0)
    dn=np.random.poisson(N*dT/gamma)
    S=np.random.choice(range(N), dn, replace=False)
    
    n = np.vstack([x[0,S],x[1,S]+np.cos(x[3,S])*l/2,x[2,S]+np.sin(x[3,S])*l/2,x[3,S]])
    
    x = np.hstack((x,n))
    
    
    return x


# In[85]:


def PlotCells(x,L,l,r=2): #ellipse plotting module for cells (not final)
    
    ells = [Ellipse((x[1,i],x[2,i]),2*l,l,180/np.pi*x[3,i]) for i in range(np.size(x,axis=1))]
    
    fig = plt.figure(0)

    ax = fig.add_subplot(111, aspect='equal')
    for e in ells:
        ax.add_artist(e)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    
    return np.size(ells)



# In[564]:


def Coefficients(j, i, x, r=2):

#returns equation coefficients describing ellipse 'j' relative to ellipse 'i'

    
    delH = x[1,j] - x[1,i];
    delK = x[2,j] - x[2,i];
    delP = -x[3,i];
    cosP = np.cos(delP);
    sinP = np.sin(delP);
    
    A = x[0,j];
    B = A/r;
    h = delH*cosP - delK*sinP;
    k = delH*sinP + delK*cosP;
    phi = x[3,j] - x[3,i];
    
    sinphi = np.sin(phi);
    cosphi = np.cos(phi);
    
    AA = (cosphi/A)**2 + (np.sin(phi)/B)**2;
    BB = 2*sinphi*cosphi/A**2 - 2*sinphi*cosphi/B**2;
    CC = (sinphi/A)**2 + (cosphi/B)**2;
    DD = -2*cosphi*(cosphi*h + sinphi*k)/A**2 + 2*sinphi*(-sinphi*h + cosphi*k)/B**2;
    EE = -2*sinphi*(cosphi*h + sinphi*k)/A**2 + 2*cosphi*(sinphi*h - cosphi*k)/B**2;
    FF = ((cosphi*h + sinphi*k)/A)**2 +  ((sinphi*h - cosphi*k)/B)**2 - 1;
    
    return AA, BB, CC, DD, EE, FF

# In[564]:


def Coefficientss(h, k, phi, A, r=2):

#returns equation coefficients describing ellipse 'j' relative to ellipse 'i'


    B = A/r;
    
    sinphi = np.sin(phi);
    cosphi = np.cos(phi);
    
    
    AA = (cosphi/A)**2 + (np.sin(phi)/B)**2;
    BB = 2*sinphi*cosphi/A**2 - 2*sinphi*cosphi/B**2;
    CC = (sinphi/A)**2 + (cosphi/B)**2;
    DD = -2*cosphi*(cosphi*h + sinphi*k)/A**2 + 2*sinphi*(-sinphi*h + cosphi*k)/B**2;
    EE = -2*sinphi*(cosphi*h + sinphi*k)/A**2 + 2*cosphi*(sinphi*h - cosphi*k)/B**2;
    FF = ((cosphi*h + sinphi*k)/A)**2 +  ((sinphi*h - cosphi*k)/B)**2 - 1;
    
    return AA, BB, CC, DD, EE, FF



def InterSolver(A1, B1, h1, k1, phi1, A2, B2, h2, k2, phi2):
#setting up quartic equation coefficients
    
    (AA1, BB1, CC1, DD1, EE1, FF1) = Coefficientss(h1, k1, phi1, A1, A1/B1);
    (AA2, BB2, CC2, DD2, EE2, FF2) = Coefficientss(h2, k2, phi2, A2, A2/B2);
    
    cy = np.zeros(5);
    cy[4] = (BB1*CC2 - CC1*BB2)*(AA1*BB2 - BB1*AA2) - (AA1*CC2 - CC1*AA2)**2;
    cy[3] = 2*(EE1*AA2 - AA1*EE2)*(AA1*CC2 - CC1*AA2) + (DD1*CC2 + BB1*EE2 - EE1*BB2 - CC1*DD2)*(AA1*BB2 - BB1*AA2) + (BB1*CC2 - CC1*BB2)*(AA1*DD2 - DD1*AA2);
    cy[2] = 2*(FF1*AA2 - AA1*FF2)*(AA1*CC2 - CC1*AA2) + (DD1*EE2 + BB1*FF2 - FF1*BB2 - EE1*DD2)*(AA1*BB2 - BB1*AA2) + (DD1*CC2 + BB1*EE2 - EE1*BB2 - CC1*DD2)*(AA1*DD2 - DD1*AA2) - (AA1*EE2 - EE1*AA2)**2;
    cy[1] = 2*(FF1*AA2 - AA1*FF2)*(AA1*EE2 - EE1*AA2) + (DD1*EE2 + BB1*FF2 - FF1*BB2 - EE1*DD2)*(AA1*DD2 - DD1*AA2) + (DD1*FF2 - FF1*DD2)*(AA1*BB2 - BB1*AA2);
    cy[0] = (DD1*FF2 - FF1*DD2)*(AA1*DD2 - DD1*AA2) - (AA1*FF2 - FF1*AA2)**2;
    
    #solving quartic equation
    

    if (abs(cy[4])>0):
        py = np.ones(5);
        for i in range(4):
            py[4-i] = cy[i]/cy[4];
        r = np.roots(py)
        nroots = 4;
    elif (abs(cy[3])>0):
        py = np.ones(4);
        for i in range(3):
            py[3-i] = cy[i]/cy[3];
        r = np.roots(py)
        nroots = 3;
    elif (abs(cy[2])>0):
        py = np.ones(3);
        for i in range(2):
            py[2-i] = cy[i]/cy[2];
        r = np.roots(py)
        nroots = 2;
    elif (abs(cy[1])>0):
        py = np.ones(2);
        r = np.roots(py)
        nroots = 1;
    else:
        r=[];
        nroots = 0;
        
    nychk = 0;
    ychk = np.array([]);
    
    for i in range(nroots):
        if (np.imag(r[i])==0):
            nychk += 1;
            ychk = np.append(ychk,np.real(r[i])*B1);
            
    for j in range(1,nychk):
        tmp0 = ychk[j];
        for k in range(j - 1, -1, -1):
            if (ychk[k]==tmp0):
                break
            else: 
                ychk[k+1] = ychk[k];
        ychk[1]=tmp0;
    return (nroots, r, ychk)

    

# In[511]:


#Solves quartic intersection equation for parameter
    
#First ellipse with axes (A1,B1) is centered at the origin, and is 
#    aligned with the (x,y) axes
    
#Second ellipse has axes (A2,B2) and is rotated phi2r counter-clockwise 
#   relative to first ellipse. The second ellipse is also translated 
#   by (h2tr,k2tr). This indicates that the rotation is around the 
#   center of mass of the second ellipse, not around the origin of the first ellipse.
    



def IntersectionSolver(i, j, x, r=2):
#setting up quartic equation coefficients
    
    A1 = x[0,i];
    A2 = x[0,j];
    B1 = A1/r;
    B2 = A2/r;

    (AA1, BB1, CC1, DD1, EE1, FF1) = Coefficientss(x[1,i],x[2,i],x[3,i],x[0,i])
    (AA2, BB2, CC2, DD2, EE2, FF2) = Coefficientss(x[1,j],x[2,j],x[3,j],x[0,j])
    
    cy = np.zeros(5);
    cy[4] = (BB1*CC2 - CC1*BB2)*(AA1*BB2 - BB1*AA2) - (AA1*CC2 - CC1*AA2)**2;
    cy[3] = 2*(EE1*AA2 - AA1*EE2)*(AA1*CC2 - CC1*AA2) + (DD1*CC2 + BB1*EE2 - EE1*BB2 - CC1*DD2)*(AA1*BB2 - BB1*AA2) + (BB1*CC2 - CC1*BB2)*(AA1*DD2 - DD1*AA2);
    cy[2] = 2*(FF1*AA2 - AA1*FF2)*(AA1*CC2 - CC1*AA2) + (DD1*EE2 + BB1*FF2 - FF1*BB2 - EE1*DD2)*(AA1*BB2 - BB1*AA2) + (DD1*CC2 + BB1*EE2 - EE1*BB2 - CC1*DD2)*(AA1*DD2 - DD1*AA2) - (AA1*EE2 - EE1*AA2)**2;
    cy[1] = 2*(FF1*AA2 - AA1*FF2)*(AA1*EE2 - EE1*AA2) + (DD1*EE2 + BB1*FF2 - FF1*BB2 - EE1*DD2)*(AA1*DD2 - DD1*AA2) + (DD1*FF2 - FF1*DD2)*(AA1*BB2 - BB1*AA2);
    cy[0] = (DD1*FF2 - FF1*DD2)*(AA1*DD2 - DD1*AA2) - (AA1*FF2 - FF1*AA2)**2;
    
    #solving quartic equation
    

    if (abs(cy[4])>0):
        py = np.ones(5);
        for i in range(4):
            py[4-i] = cy[i]/cy[4];
        r = np.roots(py)
        nroots = 4;
    elif (abs(cy[3])>0):
        py = np.ones(4);
        for i in range(3):
            py[3-i] = cy[i]/cy[3];
        r = np.roots(py)
        nroots = 3;
    elif (abs(cy[2])>0):
        py = np.ones(3);
        for i in range(2):
            py[2-i] = cy[i]/cy[2];
        r = np.roots(py)
        nroots = 2;
    elif (abs(cy[1])>0):
        py = np.ones(2);
        r = np.roots(py)
        nroots = 1;
    else:
        r=[];
        nroots = 0;
        
    nychk = 0;
    ychk = np.array([]);
    
    for i in range(nroots):
        if (np.imag(r[i])==0):
            nychk += 1;
            ychk = np.append(ychk,np.real(r[i]));
            
    for j in range(1,nychk):
        tmp0 = ychk[j];
        for k in range(j - 1, -1, -1):
            if (ychk[k]==tmp0):
                break
            else: 
                ychk[k+1] = ychk[k];
        ychk[1]=tmp0;
    return (nroots, r, ychk)

    
#%%

#Returns the # number of intersections between two ellipses 1 & 2:
    
#   First ellipse has axes (A1,B1) , centerered at (h1, k1) and angle (phi1)
#  Second ellipse has axes (A2,B2) , centerered at (h2, k2) and angle (phi2) 

def EllipseCase(i, j, x, R=2):
        
    
    nroots, r, ychk = IntersectionSolver(i, j, x, R)
    
    Realroots = 0;
    
    for i in range(nroots):
        if (np.imag(r[i])==0):
            Realroots +=1;
        else:
            continue
    
    if (Realroots==0):
        return 0
    elif (Realroots==2):
        return 2
    elif (Realroots==4):
        return 4
    else: -1
    
    
def EllipseCAse(A1, B1, h1, k1, phi1, A2, B2, h2, k2, phi2):
        
    
    nroots, r, ychk = InterSolver(A1, B1, h1, k1, phi1, A2, B2, h2, k2, phi2)
    
    Realroots = 0;
    
    for i in range(nroots):
        if (np.imag(r[i])==0):
            Realroots +=1;
        else:
            continue
    
    if (Realroots==0):
        return 0
    elif (Realroots==2):
        return 2
    elif (Realroots==4):
        return 4
    else: -1


    
# In[565]:
    
# Produces a Lattice on which one can approximate cell positions
    
# x is vector of cell metadata (size, position, orientation)
# L is width of simulation square
# B is short axis of ellipse
    

def BackgroundLattice(x,L,B):
    
    
    a = B/L;
    N = np.int(np.ceil(1/a));    
    BG = np.zeros((N+1,N+1));
    
    for i in range(np.size(x, axis=1)):
        
        x1 = int(np.floor(x[1,i]/B));
        y1 = int(np.floor(x[2,i]/B)); 
        
        BG[x1,y1] += 1;
        
    return BG


#%%


def FindBGCell(x, x1, y1, B):

    
    Sxl = np.where(B*x1<=x[1,:]);
    Sxr = np.where(x[1,:]<B*(x1 + 1));
    Syl = np.where(B*y1<=x[2,:]);
    Syr = np.where(x[2,:]<B*(y1 + 1));
    
    S = np.intersect1d(Sxl,Sxr);
    S = np.intersect1d(S,Syl);
    S = np.intersect1d(S,Syr);

    return S


def FindBGCells(x, Xs, Ys, B):
    
    S = np.array([]);
    
    if (np.size(Xs)==np.size(Ys)):
        for i in range(np.size(Xs)):
            S = np.concatenate((S,FindBGCell(x,Xs[i],Ys[i],B)));
            
        return S
    
    else: 
        return -1


def FindNeighbors(i, d, x):
    
    h = x[1,i];
    k = x[2,i];
    
    r = np.array([]); 
    
    
    for j in range(np.size(x, axis=1)):
        
        r = np.hstack((r,np.array([np.sqrt((x[1,j] - h)**2 + (x[2,j] - k)**2)])));
        
    S = np.where(r <= d);
    S = S[0];
    S = [S[i] for i in range(np.size(S))]
    
    return S



# In[566]:



def InteractingLocations(BG, A, B):
    
    #Counts the number of potential intersections for each lattice point, within distance d = A/B
    
    d = int(np.ceil(A/B));
    
    sz = int(np.sqrt(np.size(BG)));
    
    xmin = 2*d;
    xmax = sz - 2*d - 2;
    ymin = 2*d;
    ymax = sz - 2*d - 2;
    
    
    Intersections = np.zeros((sz,sz))
    
    for i in range(1,d+1):
        
        Intersections += np.multiply(BG,np.pad(BG, [(i, 0), (0, 0)])[:-i,:]);
            
        Intersections += np.multiply(BG,np.pad(BG, [(0, i), (0, 0)])[i:i+sz,:]);
        
        Intersections += np.multiply(BG,np.pad(BG, [(0, 0), (i, 0)])[:,:-i]);
        
        Intersections += np.multiply(BG,np.pad(BG, [(0, 0), (0, i)])[:,i:i+sz]);
        
        for j in range(1,d+1):
                        
            Intersections += np.multiply(BG,np.pad(BG, [(i, 0), (j, 0)])[:-i,:-j]);
        
            Intersections += np.multiply(BG,np.pad(BG, [(i, 0), (0, j)])[:-i,j:j+sz]);
        
            Intersections += np.multiply(BG,np.pad(BG, [(0, i), (j, 0)])[i:i+sz,:-j]);
        
            Intersections += np.multiply(BG,np.pad(BG, [(0, i), (0, j)])[i:i+sz,j:j+sz]);
        
        
    (Xint, Yint) = np.asarray(Intersections>0).nonzero();
        
    NPInts = np.array([BG[Xint[i],Yint[i]] for i in range(np.size(Xint))]);
    
        
            
    return NPInts, Xint, Yint, Intersections
    


#%%
    

# Returns indices of intersecting cells
    
# Cell 'i' - primary cell
# tests intersection with cells within distance 'd'
    

def IntersectingNeighbors(i, d, x):
    
    S = FindNeighbors(i, d, x);
    n = int(np.size(S));
    
    m = np.array([]);
    for k in S:

        m = np.hstack((m,EllipseCase(i,k,x)));
          
         
    index = np.array([]);
    
    for i in np.where(m>0)[0]:
        index = np.concatenate((index,[S[i]]))

    return index
        

#%%
        
def InteractionPairs(x, A, B, L):
    
    BG = BackgroundLattice(x, L, B)
    
    (NPInts, Xint, Yint, Intersections) = InteractingLocations(BG, A, B); 
    
    d = A/B;
    
    pairs = np.array([[],[],[],[]]);
    pairs = pairs.T;
    
   # for i in FindBGCells(x, Xint, Yint):
        
   #     x1 = x[1,i];
   #     y1 = x[2,i];
        
   #     for j in range()
   #         pairs = np.hstack((pairs,))


#%%

def EllipseSegment(A,B,X1,Y1,X2,Y2): #returns the area of an arc on the ellipse minus the area of the triangular segment of arc
    eps = 1*10^(-1)                  #Ellipse has major axis A, minor axis B, and is centered at origin.
    if (A<=0) or (B<=0):             #The arc is bounded by points (x1,y1) and (x2,y2) on the ellipse (x/A)^2 + (y/B)^2 = 1
        return -1
    if abs(X1)/A>1:
        if abs(X1)-A>eps:
            return -1
        else:
            X1= -A
    if abs(X2)/A>1:
        if abs(X2)-A>eps:
            return -1
        else:
            X2= -A
    if (Y1<0):
        theta1=2*np.pi-np.arccos(X1/A);
    else:
        theta1=np.arccos(X1/A);
    if (Y2<0):
        theta2=2*np.pi-np.arccos(X2/A);
    else:
        theta2=np.arccos(X2/A);
    if (theta1>theta2):
        theta1-=2*np.pi;
    if ((theta2-theta1)>np.pi):
        trsign=1.0;
    else:
        trsign=-1.0;
    return 0.5*(A*B*(theta2-theta1)+trsign*abs(X1*Y2-X2*Y1))


# In[132]:


def EllipseLineOverlap(phi, h, k, A, B, X1, Y1, X2, Y2):
    
    
    
    #returns area of line intersecting ellipse, 
    #running counter-clockwise from (X1,Y1) to (X2,Y2)  
    
    #(X1,Y1) and (X2,Y2) do not have to be on ellipse, merely on the line

    # we will translate & rotate the ellipse so that it is at origin, and axes are (x,y)
    # such that it will become (x/A)^2 + (y/B)^2 = 1
    # 
    # First we translate to origin by subtracting the center vector (h,k)
    # Then we rotate by angle phi to fix coordinate axes
    
    #(h, k) = (x[1,i],x[2,i]);
    
    cosphi = np.cos(phi);
    sinphi = np.sin(phi);
    
    x10 = cosphi*(X1 - h) + sinphi*(Y1 - k);
    y10 = -sinphi*(X1 - h) + cosphi*(Y1 - k);
    x20 = cosphi*(X2 - h) + sinphi*(Y2 - k);
    y20 = -sinphi*(X2 - h) + cosphi*(Y2 - k);
    
    # to determine if they intersect, solve the following:
    eps = 1e-5
    
    if (abs(x20-x10)>eps):
        m = (y20 - y10)/(x20 - x10);
        a = (B**2 + (A*m)**2)/A**2;
        b = 2*(y10*m - x10*m**2);
        c = (y10**2 - 2*m*y10*x10 + (x10*m)**2 - B**2);
    elif (abs(y20-y10)>eps): 
        m = (x20 - x10)/(y20 - y10);
        a = (A**2 + (B*m)**2)/B**2;
        b = 2*(x10*m - y10*m**2);
        c = (x10**2 - 2*m*y10*x10 + (y10*m)**2 - A**2);
    else:
        return -1
    
    discrim = b**2 - 4*a*c;
    
    if (discrim<0):
        print("No intersection points.")
        return 0;
    elif (discrim>0):
        root1 = (- b - np.sqrt(discrim))/(2*a);
        root2 = (- b + np.sqrt(discrim))/(2*a);
    else:
        print("Line Tangent to Ellipse.")
        return 0
    
    # ordering the points in the direction of x10 -> x20
    
    if (abs(x20-x10)>eps):
        if (x10<x20):
            x1 = root1;
            x2 = root2;
        else:
            x1 = root2;
            x2 = root1;
            
        # the y-values can be found using linear equation
    
        y1 = y10 + m*(x1 - x10);
        y2 = y10 + m*(x2 - x10);
    
    else:
        # ordering the points in the direction of y10 -> y20
        if (y10<y20):
            y1 = root1;
            y2 = root2;
        else: 
            y1 = root2;
            y2 = root1;
    
        x1 = x10 + m*(y1 - y10);
        x2 = x10 + m*(y2 - y10);
        
    SegmentArea = EllipseSegment(A,B,x1,y1,x2,y2);
    
    return SegmentArea


#%%
    

def Rotate(x, y, phi):
#   Rotates the vector (x, y) counter-clockwise arount the origin
    
    sinphi = np.sin(phi);
    cosphi = np.cos(phi);
    
    xR = x*cosphi - y*sinphi;
    yR = x*sinphi + y*cosphi;
    
    return (xR, yR)


def Translate(x, y, h, k):
#   Translates vector (x, y) by vector (h, k)
    
    return (x+h, y+k)


#%%
    
def RelativeIntersection(i, j, x, r=2):
    eps = 1e-4
    r = 2; #aspect ratio
    

    
    n = EllipseCase(i, j, x, r);
    
    if ((n==0) or (n==-1)):
        print('No overlap')
        return 
    
    Niroots, yIroots, yI = IntersectionSolver(i, j, x);
    Njroots, yJroots, yJ = IntersectionSolver(j, i, x);
    
    
    AA1, BB1, CC1, DD1, EE1, FF1 = Coefficientss(x[1,i],x[2,i],x[3,i],x[0,i])
    AA2, BB2, CC2, DD2, EE2, FF2 = Coefficientss(x[1,j],x[2,j],x[3,j],x[0,j])
    
    
    xIs = [];
    xJs = [];
    for k in range(np.size(yI)):
        xIs = np.hstack((xIs,np.roots([AA1, BB1*yI[k] + DD1, CC1*yI[k]**2 + EE1*yI[k] + FF1])));
    for k in range(np.size(yJ)):
        xJs = np.hstack((xJs,np.roots([AA2, BB2*yJ[k] + DD2, CC2*yJ[k]**2 + EE2*yJ[k] + FF2])));


    xI = [];
    xJ = [];
    xItr = [];
    yItr = [];
    xJtr = [];
    yJtr = []; 

           
    for k in range(np.size(xIs)):
        for m in range(np.size(xJs)):
            if (abs(xIs[k]-xJs[m])<eps):
                xI = np.hstack((xI,xIs[k]));
                xJ = np.hstack((xJ,xJs[m]));
            else:
                continue
                
    
    for k in range(np.size(xI)):
        xIt, yIt = Translate(xI[k], yI[k], -x[1,i], -x[2,i]);
        xItr, yItr = np.hstack((xItr,Rotate(xIt, yIt, -x[3,i])[0])), np.hstack((yItr,Rotate(xIt, yIt, -x[3,i])[1]));
        
        xJt, yJt = Translate(xJ[k], yJ[k], -x[1,j], -x[2,j]);
        xJtr, yJtr = np.hstack((xJtr,Rotate(xJt ,yJt, -x[3,j])[0])), np.hstack((yJtr,Rotate(xJt, yJt, -x[3,j])[1]));
        
    return xItr, yItr, xJtr, yJtr


    
#%%
        
        
def TwoInt(i, j, x, xI, yI, xJ, yJ, r=2):
    
    AreaI1 = EllipseSegment(x[0,i], x[0,i]/r, xI[0], yI[0], xI[1], yI[1])
    AreaJ1 = EllipseSegment(x[0,j], x[0,j]/r, xJ[1], yJ[1], xJ[0], yJ[0])
    AreaI2 = EllipseSegment(x[0,i], x[0,i]/r, xI[1], yI[1], xI[0], yI[0])
    AreaJ2 = EllipseSegment(x[0,j], x[0,j]/r, xJ[0], yJ[0], xJ[1], yJ[1])

    maxArea = min(np.pi*x[0,i]**2/r,np.pi*x[0,j]**2/r);
    
    if ((AreaI1 + AreaJ1<=maxArea) and (AreaI2 + AreaJ2<=maxArea)):
        print('Interesting!')
        return -1
    
    elif ((AreaI1 + AreaJ1<=maxArea) and (AreaI2 + AreaJ2>maxArea)):
        
        OverallArea = AreaI1 + AreaJ1
    
    elif ((AreaI1 + AreaJ1>maxArea) and (AreaI2 + AreaJ2<=maxArea)):
        
        OverallArea = AreaI2 + AreaJ2
    
    else:
        if (min(AreaI1+AreaJ1,AreaI2+AreaJ2)/min(np.pi*x[0,i]**2/r,np.pi*x[0,j]**2/r)<1.05):
            OverallArea = min(AreaI1+AreaJ1,AreaI2+AreaJ2)
        else:
            print('Something somewhere went quite wrong')
            return 0
        
        
    return OverallArea

  
#%%
            
def FourInt(xint, yint, A1, B1, phi1, A2, B2, h2tr, k2tr, phi2, AA, BB, CC, DD, EE, FF):
    
# returns the overlap area with 4 intersection points (Xint, Yint)

# input values (Xint, Yint) relative to one ellipse
    
    xint_tr = np.zeros(4);
    yint_tr = np.zeros(4);
    theta = np.zeros(4);
    theta_tr = np.zeros(4);
    
    for i in range(4):
        if (yint[i]<0):
            theta[i] = 2*np.pi - np.arccos(xint[i]/A1);
        else:
            theta[i] = np.arccos(xint[i]/A1);
    
    for i in range(1,4):
        
        tmp0 = theta[i];
        tmp1 = xint[i];
        tmp2 = yint[i];
        
        for k in range(i-1, -1, -1):
            
            if (theta[k]<=tmp0):
                break
            
            theta[k+1] =  theta[k]
            xint[k+1] = xint[k]
            yint[k+1] = yint[k];
        
        theta[k+1] = tmp0;
        xint[k+1] = tmp1;
        yint[k+1] = tmp2;

    area1 = 0.5*abs((xint[2] - xint[0])*(yint[3] - yint[1]) - (xint[3] - xint[1])*(yint[2] - yint[0]));
    
    for i in range(4):
        
        xint_t, yint_t = Translate(xint[i], yint[i], -h2tr, -k2tr);
        xint_tr[i],yint_tr[i] = Rotate(xint_t, yint_t, phi1 - phi2);
        
        if (yint_tr[i]<0):
            theta_tr[i] = 2*np.pi - np.arccos(xint_tr[i]/A2);
        else:
            theta_tr[i] = np.arccos(xint_tr[i]/A2);
            
    xmid = A1*np.cos((theta[0]+theta[1])/2);
    ymid = B1*np.sin((theta[0]+theta[1])/2);

    if (AA*xmid**2 + BB*xmid*ymid + CC*ymid**2 + DD*xmid + EE*ymid + FF < 0):
        area2 = 0.5*(A1*B1*(theta[1] - theta[0]) - abs(xint[0]*yint[1] - xint[1]*yint[0]));
        area3 = 0.5*(A1*B1*(theta[3] - theta[2]) - abs(xint[2]*yint[3] - xint[3]*yint[2]));
        area4 = 0.5*(A2*B2*(theta_tr[2] - theta_tr[1]) - abs(xint_tr[1]*yint_tr[2] - xint_tr[2]*yint_tr[1]));
        area5 = 0.5*(A2*B2*(theta_tr[0] - theta_tr[3] + 2*np.pi) - abs(xint_tr[3]*yint_tr[0] - xint_tr[0]*yint_tr[3]));
        
    else:
        area2 = 0.5*(A1*B1*(theta[2] - theta[1]) - abs(xint[1]*yint[2] - xint[2]*yint[1]));
        area3 = 0.5*(A1*B1*(theta[0] - theta[3] + 2*np.pi) - abs(xint[3]*yint[0] - xint[0]*yint[3]));
        area4 = 0.5*(A2*B2*(theta[1] - theta[0]) - abs(xint_tr[0]*yint_tr[1] - xint_tr[1]*yint_tr[0]));
        area5 = 0.5*(A2*B2*(theta[3] - theta[2]) - abs(xint_tr[2]*yint_tr[3] - xint_tr[3]*yint_tr[2]));
        
    
    return area1 + area2 + area3 + area4 + area5
        


#%%
    
def returnArea(i, j, x, r=2):
    
    (xI, yI, xJ, yJ) = RelativeIntersection(i, j, x, r);
    
    if (EllipseCase(i, j, x, r)==0):
        return 0
    elif (EllipseCase(i, j, x, r)==2):
        return TwoInt(i, j, x, xI, yI, xJ, yJ)
    elif (EllipseCase(i, j, x)==4):
        (AA, BB, CC, DD, EE, FF) = Coefficients(j, i, x, r)
        return FourInt(xI, yI, x[0,i], x[0,i]/r, x[3,i], x[0,j], x[0,j]/r, x[1,j] - x[1,i], x[2,j] - x[2,i], x[3,j], AA, BB, CC, DD, EE, FF)
    else:
        return -1






#%%



def AreaTest(A1, B1, h1, k1, phi1, A2, B2, h2, k2, phi2):
    eps = 1e-4
    

    
    n = EllipseCAse(A1, B1, h1, k1, phi1, A2, B2, h2, k2, phi2);
    
    if (n==0):
        print('No overlap')
        return 
    
    Niroots, yIroots, yI = InterSolver(A1, B1, h1, k1, phi1, A2, B2, h2, k2, phi2);
    Njroots, yJroots, yJ = InterSolver(A2, B2, h2, k2, phi2, A1, B1, h1, k1, phi1);
    
    
    AA1, BB1, CC1, DD1, EE1, FF1 = Coefficientss(h1, k1, phi1, A1, A1/B1)
    AA2, BB2, CC2, DD2, EE2, FF2 = Coefficientss(h2, k2, phi2, A2, A2/B2)
    
    
    
        
         
    xIs = [];
    xJs = [];
    for k in range(np.size(yI)):
        xIs = np.hstack((xIs,np.roots([AA1, BB1*yI[k] + DD1, CC1*yI[k]**2 + EE1*yI[k] + FF1])));
    for k in range(np.size(yJ)):
        xJs = np.hstack((xJs,np.roots([AA2, BB2*yJ[k] + DD2, CC2*yJ[k]**2 + EE2*yJ[k] + FF2])));


    xI = [];
    xJ = [];
    xItr = [];
    yItr = [];
    xJtr = [];
    yJtr = []; 

           
    for k in range(np.size(xIs)):
        for m in range(np.size(xJs)):
            if (abs(xIs[k]-xJs[m])<eps):
                xI = np.hstack((xI,xIs[k]));
                xJ = np.hstack((xJ,xJs[m]));
            else:
                continue
                
    
    for k in range(np.size(xI)):
        xIt, yIt = Translate(xI[k], yI[k], -h1, -k1);
        xItr, yItr = np.hstack((xItr,Rotate(xIt, yIt, -phi1)[0])), np.hstack((yItr,Rotate(xIt, yIt, -phi1)[1]));
        
        xJt, yJt = Translate(xJ[k], yJ[k], -h2, -k2);
        xJtr, yJtr = np.hstack((xJtr,Rotate(xJt ,yJt, -phi2)[0])), np.hstack((yJtr,Rotate(xJt, yJt, -phi2)[1]));
        
        
    if (n==2):

        AreaI1 = EllipseSegment(A1, B1, xItr[0], yItr[0], xItr[1], yItr[1])
        AreaJ1 = EllipseSegment(A2, B2, xJtr[1], yJtr[1], xJtr[0], yJtr[0])
        AreaI2 = EllipseSegment(A1, B1, xItr[1], yItr[1], xItr[0], yItr[0])
        AreaJ2 = EllipseSegment(A2, B2, xJtr[0], yJtr[0], xJtr[1], yJtr[1])

        maxArea = min(np.pi*A1*B1,np.pi*A2*B2);
    
        if ((AreaI1 + AreaJ1<=maxArea) and (AreaI2 + AreaJ2<=maxArea)):
            print('Interesting!')
            return -1
    
        elif ((AreaI1 + AreaJ1<=maxArea) and (AreaI2 + AreaJ2>maxArea)):
        
            OverallArea = AreaI1 + AreaJ1
    
        elif ((AreaI1 + AreaJ1>maxArea) and (AreaI2 + AreaJ2<=maxArea)):
        
            OverallArea = AreaI2 + AreaJ2
    
        else:
            if (min(AreaI1+AreaJ1,AreaI2+AreaJ2)/maxArea<1.05):
                OverallArea = min(AreaI1+AreaJ1,AreaI2+AreaJ2)
            else:
                print('Something somewhere went quite wrong')
                return 0
        
        
        return OverallArea
    
    
    
    elif (n==4):
      
        xint_tr = np.zeros(4);
        yint_tr = np.zeros(4);
        theta = np.zeros(4);
        theta_tr = np.zeros(4);
        
        for i in range(4):
            if (abs(xItr[i])>A1):
                if (xItr[i]<0): 
                    xItr[i] = -A1;
                else:
                    xItr[i] = A1;
            if (yItr[i]<0):
                theta[i] = 2*np.pi - np.arccos(xItr[i]/A1);
            else:
                theta[i] = np.arccos(xItr[i]/A1);
    
    
        for i in range(1,4):
        
            tmp0 = theta[i];
            tmp1 = xItr[i];
            tmp2 = yItr[i];
        
            for k in range(i-1, -1, -1):
            
                if (theta[k]<=tmp0):
                    break
            
                theta[k+1] =  theta[k]
                xItr[k+1] = xItr[k]
                yItr[k+1] = yItr[k];
        
            theta[k+1] = tmp0;
            xItr[k+1] = tmp1;
            yItr[k+1] = tmp2;

    area1 = 0.5*abs((xItr[2] - xItr[0])*(yItr[3] - yItr[1]) - (xItr[3] - xItr[1])*(yItr[2] - yItr[0]));
    
    for i in range(4):
        
        xint_t, yint_t = Translate(xItr[i], yItr[i], h1 - h2, k1 - k2);
        xint_tr[i],yint_tr[i] = Rotate(xint_t, yint_t, phi1 - phi2);
        
        if (yint_tr[i]<0):
            theta_tr[i] = 2*np.pi - np.arccos(xint_tr[i]/A2);
        else:
            theta_tr[i] = np.arccos(xint_tr[i]/A2);
            
    xmid = A1*np.cos((theta[0]+theta[1])/2);
    ymid = B1*np.sin((theta[0]+theta[1])/2);

    if (AA2*xmid**2 + BB2*xmid*ymid + CC2*ymid**2 + DD2*xmid + EE2*ymid + FF2 < 0):
        area2 = 0.5*(A1*B1*(theta[1] - theta[0]) - abs(xItr[0]*yItr[1] - xItr[1]*yItr[0]));
        area3 = 0.5*(A1*B1*(theta[3] - theta[2]) - abs(xItr[2]*yItr[3] - xItr[3]*yItr[2]));
        area4 = 0.5*(A2*B2*(theta_tr[2] - theta_tr[1]) - abs(xint_tr[1]*yint_tr[2] - xint_tr[2]*yint_tr[1]));
        area5 = 0.5*(A2*B2*(theta_tr[0] - theta_tr[3] + 2*np.pi) - abs(xint_tr[3]*yint_tr[0] - xint_tr[0]*yint_tr[3]));
        
    else:
        area2 = 0.5*(A1*B1*(theta[2] - theta[1]) - abs(xItr[1]*yItr[2] - xItr[2]*yItr[1]));
        area3 = 0.5*(A1*B1*(theta[0] - theta[3] + 2*np.pi) - abs(xItr[3]*yItr[0] - xItr[0]*yItr[3]));
        area4 = 0.5*(A2*B2*(theta[1] - theta[0]) - abs(xint_tr[0]*yint_tr[1] - xint_tr[1]*yint_tr[0]));
        area5 = 0.5*(A2*B2*(theta[3] - theta[2]) - abs(xint_tr[2]*yint_tr[3] - xint_tr[3]*yint_tr[2]));
        
    
    return area1 + area2 + area3 + area4 + area5


    if (n==0):
        return 0
    elif (n==2):
        return TwoInt(i, j, x, xI, yI, xJ, yJ)
    elif (n==4):
        (AA, BB, CC, DD, EE, FF) = Coefficients(j, i, x)
        return FourInt(xI, yI, x[0,i], x[0,i]/r, x[3,i], x[0,j], x[0,j]/r, x[1,j] - x[1,i], x[2,j] - x[2,i], x[3,j], AA, BB, CC, DD, EE, FF)
    else:
        return -1







        