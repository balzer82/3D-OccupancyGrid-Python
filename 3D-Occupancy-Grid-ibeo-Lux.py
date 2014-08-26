# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# 3D Occupancy Grid with ibeo Lux Laserscanner

# <markdowncell>

# ![ibeo Lux](http://www.mechlab.de/wp-content/uploads/2012/02/ibeoLUX.jpg)

# <codecell>

import numpy as np
import time
import pandas as pd

# <codecell>

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from IPython.html.widgets import interact
from IPython.html import widgets
%matplotlib inline

# <headingcell level=3>

# Create Empty Grid

# <codecell>

l = 10.0 # Länge m
b = 10.0  # Breite m
h = 2.0  # Höhe m

r = 0.1 # Resolution m/gridcell

# <codecell>

print('%.1fmio Grid Cells' % ((l*b*h)/r**3/1e6))

# <headingcell level=2>

# Generate a LogOdds Grid

# <markdowncell>

# Why LogOdds? Numerically stable around $p=0$ and $p=1$ and reduces the mathematical efford to update the Grid (Bayes Rule) to just an addition!

# <codecell>

p = np.arange(0.01, 1.0, 0.01)
lo = np.log(p/(1-p))
plt.plot(p, lo)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Probability $p$')
plt.ylabel(r'Log Odds, $\log(\frac{p}{1-p})$')

# <markdowncell>

# So an initial uncertainty ($p=0.5$) is a zero in LogOdds. That's fine, because it is a very fast initialization of the grid!
# In order to store log(odds), we need negative values and decimal values. And `float32` ist fastest: http://stackoverflow.com/questions/15340781/python-numpy-data-types-performance
# 
# So let's use it!

# <codecell>

print "%ix%ix%i Grid" % (l/r, b/r, h/r)
startTime = time.time()

grid = np.zeros((l/r, b/r, h/r), dtype=np.float32) # Log Odds Grid must be initialized with zeros!

print "Stats: %.2fs, %.2fGB" % (time.time() - startTime, (grid.nbytes/1024.0**2))

# <headingcell level=3>

# 3D View

# <codecell>

def plot3Dgrid(grid, az, el):
    # plot the surface
    plt3d = plt.figure(figsize=(12, 6)).gca(projection='3d')

    # create x,y
    ll, bb = np.meshgrid(range(grid.shape[1]), range(grid.shape[0]))

    for z in range(grid.shape[2]):
        if not (np.max(grid[:,:,z])==np.min(grid[:,:,z])): # unberührte Ebenen nicht darstellen
            plt3d.contourf(ll, bb, grid[:,:,z], offset = z, alpha=0.3)

    plt3d.set_xlabel('B')
    plt3d.set_ylabel('L')
    plt3d.set_zlabel('H')
    plt3d.set_xlim3d(0, grid.shape[0])
    plt3d.set_ylim3d(0, grid.shape[1])
    plt3d.set_zlim3d(0, grid.shape[2])
    #plt3d.axis('equal')
    plt3d.view_init(az, el)
    return plt3d

# <codecell>

#plot3Dgrid(grid, 25, -30)

# <headingcell level=2>

# Integrate a measurement with BRESENHAM Algorithm

# <markdowncell>

# Amanatides, J., & Woo, A. (1987). A fast voxel traversal algorithm for ray tracing. Proceedings of EUROGRAPHICS, i. Retrieved from http://www.cse.yorku.ca/~amana/research/grid.pdf
# 
# Here is a Python Implementation: https://gist.github.com/salmonmoose/2760072

# <codecell>

def bresenham3D(startPoint, endPoint):
   # by Anton Fletcher
   # Thank you!
   path = [] 
    
   startPoint = [int(startPoint[0]),int(startPoint[1]),int(startPoint[2])]
   endPoint = [int(endPoint[0]),int(endPoint[1]),int(endPoint[2])]
 
   steepXY = (np.abs(endPoint[1] - startPoint[1]) > np.abs(endPoint[0] - startPoint[0]))
   if(steepXY):   
      startPoint[0], startPoint[1] = startPoint[1], startPoint[0]
      endPoint[0], endPoint[1] = endPoint[1], endPoint[0]
 
   steepXZ = (np.abs(endPoint[2] - startPoint[2]) > np.abs(endPoint[0] - startPoint[0]))
   if(steepXZ):
      startPoint[0], startPoint[2] = startPoint[2], startPoint[0]
      endPoint[0], endPoint[2] = endPoint[2], endPoint[0]
 
   delta = [np.abs(endPoint[0] - startPoint[0]), np.abs(endPoint[1] - startPoint[1]), np.abs(endPoint[2] - startPoint[2])]
 
   errorXY = delta[0] / 2
   errorXZ = delta[0] / 2
 
   step = [
      -1 if startPoint[0] > endPoint[0] else 1,
      -1 if startPoint[1] > endPoint[1] else 1,
      -1 if startPoint[2] > endPoint[2] else 1
   ]
 
   y = startPoint[1]
   z = startPoint[2]
 
   for x in range(startPoint[0], endPoint[0], step[0]):
      point = [x, y, z]
 
      if(steepXZ):
          point[0], point[2] = point[2], point[0]
      if(steepXY):
          point[0], point[1] = point[1], point[0]
 
      #print (point)

 
      errorXY -= delta[1]
      errorXZ -= delta[2]
 
      if(errorXY < 0):
          y += step[1]
          errorXY += delta[0]
 
      if(errorXZ < 0):
          z += step[2]
          errorXZ += delta[0]

      path.append(point)

   return path

# <headingcell level=3>

# Load some ibeo Lux Measurements

# <codecell>

def ibeo2XYZ(theta, dist, layer):
    '''
    Berechnet die kartesischen X,Y,Z-Koordinaten aus polaren Koordinaten des IBEO Lux Laserscanners unter der Annahme er ist perfekt horizontal ausgerichtet.
    
    Input:
        - theta: Horizontaler Winkel
        - dist : polarer Abstand
        - layer: Ebene
    '''
    # Ibeo Lux hat 3.2° bei 4 Ebenen vertikal
    oeffnungswinkel = 3.2
    ebenen = 4.0
    
    # aus Ebene den Vertikalwinkel berechnen
    phi = (layer * oeffnungswinkel/(ebenen-1) - oeffnungswinkel/2.0) * np.pi/180.0
    
    X = dist * np.cos(theta)
    Y = dist * np.sin(theta)
    Z = dist * np.sin(phi)
    
    return np.array([X, Y, Z])

# <codecell>

# or generate some values synthetically:
#angles = np.arange(-15, 15, 0.25)/180.0*np.pi
#distance = 5.0*np.ones(len(angles))
#layer = 3*np.ones(len(angles)) # Ebene {0,1,2,3}

# <codecell>

# some real ibeo lux measurements
data = pd.read_csv('Messung1.txt', delimiter='|')

# <codecell>

data.head(5)

# <headingcell level=4>

# Filter out an arbitrary measurement and bounded angle

# <codecell>

timestamp = 1341907053031
angles = data['<Winkel>'][(data['# <Zeitstempel>']==timestamp) & (data['<Winkel>']<0.5) & (data['<Winkel>']>-0.5)]
distance = data['<Radius>'][(data['# <Zeitstempel>']==timestamp) & (data['<Winkel>']<0.5) & (data['<Winkel>']>-0.5)]/100.0
layer = data['<Ebene>'][(data['# <Zeitstempel>']==timestamp) & (data['<Winkel>']<0.5) & (data['<Winkel>']>-0.5)]

# <codecell>

# Convert from spherical coordinates to cartesian
[xe, ye, ze] = ibeo2XYZ(angles.values, distance.values, layer.values)

# <headingcell level=3>

# Sensor Position

# <markdowncell>

# Rotation und Translation in homogenen Koordinaten, d.h. es kann alles über Matrizenmultiplikation gemacht werden.
# 
# $$\left[\begin{matrix}x \\ y \\ z \\ 1\end{matrix}\right]_\text{Endpoint} = \left[\begin{matrix} R_{3x3} & t_{1x3} \\ 0 & 1\end{matrix}\right] \cdot \left[\begin{matrix}x \\ y \\ z \\ 1\end{matrix}\right]_\text{Messpunkte}$$

# <codecell>

def Rypr(y, p, r):
    '''
    Rotationsmatrix für y=yaw, p=pitch, r=roll in degrees
    '''
    # from Degree to Radians
    y = y*np.pi/180.0
    p = p*np.pi/180.0
    r = r*np.pi/180.0
    
    Rr = np.matrix([[1.0, 0.0, 0.0],[0.0, np.cos(r), -np.sin(r)],[0.0, np.sin(r), np.cos(r)]])
    Rp = np.matrix([[np.cos(p), 0.0, np.sin(p)],[0.0, 1.0, 0.0],[-np.sin(p), 0.0, np.cos(p)]])
    Ry = np.matrix([[np.cos(y), -np.sin(y), 0.0],[np.sin(y), np.cos(y), 0.0],[0.0, 0.0, 1.0]])
    
    return Ry*Rp*Rr

# <codecell>

yaw   = 0.0 #  Gieren
pitch = 0.0 #  Nicken
roll  = 0.0 #  Wanken
dx= 0.0 #  Verschiebung in X
dy= 5.0 #  Verschiebung in Y
dz= 1.0 #  Verschiebung in Z

# <codecell>

RSensor = np.eye(4) # Einheitsmatrix erstellen

# Rotationsteil
RSensor[np.ix_([0,1,2],[0,1,2])] = Rypr(yaw, pitch, roll)

# Translationsteil
tsensor = np.array([[dx], [dy], [dz]]) 
RSensor[np.ix_([0,1,2],[3])] = tsensor

# <codecell>

RSensor

# <codecell>

[xe,ye,ze,w] = np.dot(RSensor, np.array((xe,ye,ze,np.ones(len(xe)))))

# <codecell>

plt3d = plt.figure(figsize=(12, 6)).gca(projection='3d')
plt3d.scatter(xe, ye, ze, c='r', label='Laserscanner Pointcloud')
plt3d.scatter(tsensor[0], tsensor[1], tsensor[2], c='k', s=200, label='ibeo Lux')
plt3d.view_init(45, -115)
plt3d.axis('equal')

# <headingcell level=2>

# Function which integrates the Measurement via Inverse Sensor Model

# <markdowncell>

# Values for hit and miss probabilities are taken from Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: an efficient probabilistic 3D mapping framework based on octrees. Autonomous Robots, 34(3), 189–206. doi:10.1007/s10514-012-9321-0

# <codecell>

# in LogOdds Notation!
loccupied = 0.85
lfree = -0.4

lmin = -2.0
lmax = 3.5

# <codecell>

def insertPointcloud(tSensor, xe,ye,ze):
    
    for i,val in enumerate(xe):
        
        # Insert Endpoints
        x=int(xe[i])
        y=int(ye[i])
        z=int(ze[i])

        grid[x,y,z] += loccupied # increase LogOdds Ratio

        if grid[x,y,z]>lmax: #clamping
            grid[x,y,z]=lmax

        
        # Grid cells in perceptual range of laserscanner
        for (x,y,z) in bresenham3D(tSensor, (xe[i], ye[i], ze[i])):

            grid[x,y,z] += lfree # decrease LogOdds Ratio

            if grid[x,y,z]<lmin: #clamping
                grid[x,y,z]=lmin
        

# <headingcell level=3>

# Sensor Origin

# <codecell>

RSensor = np.eye(3)  # Rotation Matrix
tSensor = tsensor/r  # Translation (shift from 0,0,0) in Grid Cell Numbers

# <codecell>

# integrate the measurement 5 times
for m in range(5):
    try:
        insertPointcloud(tSensor, xe/r,ye/r,ze/r)
    except:
        print('Fehler beim Einfügen der Messung. Grid zu klein gewählt?!')

# <headingcell level=3>

# 2D Plot of Grid Layer Z

# <codecell>

@interact
def plotmultivargauss(z = widgets.FloatSliderWidget(min=0, max=np.max(grid.shape[2])-1, step=1, value=10, description="")):
    plt.figure(figsize=(l/2, b/2))
    plt.contourf(grid[:,:,z])
    plt.axis('equal')

# <headingcell level=3>

# 3D Plot

# <codecell>

@interact
def plotmultivargauss(az = widgets.FloatSliderWidget(min=-90.0, max=90.0, step=1.0, value=65.0, description=""), \
                      el = widgets.FloatSliderWidget(min=-90.0, max=90.0, step=1.0, value=-20.0, description="")):

    plot3Dgrid(grid, az, el)

# <codecell>

print('Max Grid Value (Log Odds): %.2f' % np.max(grid))
print('Min Grid Value (Log Odds): %.2f' % np.min(grid))

# <headingcell level=2>

# From LogOdds Occupancy Grid to Probability Grid

# <markdowncell>

# The conversion from LogOdds notation to probabilities could be achieved by following formula:
# 
# $$P(l) = 1-\cfrac{1}{1+e^l}$$ with $l$=LogOdds Value

# <codecell>

def logOdds2Prob(grid):
    gridP = grid.copy()
    for z in range(grid.shape[2]):
        for y in range(grid.shape[1]):
            for x in range(grid.shape[0]):
                gridP[x,y,z] = 1.0-(1.0/(1.0+np.exp(grid[x,y,z])))
                
    return gridP

# <codecell>

gridP = logOdds2Prob(grid)

# <codecell>

plot3Dgrid(gridP, 65, -20)

# <codecell>

print('Max Grid Value (Probability): %.2f' % np.max(gridP))
print('Min Grid Value (Probability): %.2f' % np.min(gridP))

# <codecell>

print('Done.')

# <headingcell level=2>

# Convolve the Map for Path Planning

# <codecell>

from scipy.ndimage import gaussian_filter

# <codecell>

blurmap = gaussian_filter(gridP, 1)

# <codecell>

plot3Dgrid(blurmap, 65, -20)

# <codecell>


# <codecell>


