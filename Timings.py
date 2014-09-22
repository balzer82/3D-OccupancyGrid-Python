# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Timing Vergleich zwischen BRESENHAM und IncrementalDistance Raycasting Algorithmen

# <codecell>

import numpy as np

# <headingcell level=2>

# BRESENHAM Algorithm

# <markdowncell>

# Amanatides, J., & Woo, A. (1987). A fast voxel traversal algorithm for ray tracing. Proceedings of EUROGRAPHICS, i. Retrieved from http://www.cse.yorku.ca/~amana/research/grid.pdf
# 
# Here is a Python Implementation of BRESENHAM Algorithm: https://gist.github.com/salmonmoose/2760072

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

# <codecell>

goal = (5.5, 3.5, 0.0)

# <codecell>

%%timeit
cells = bresenham3D((0,0,0), (goal[0], goal[1], 0.0))

# <codecell>

cells

# <headingcell level=2>

# Incremental Distance Raycasting

# <codecell>

def ibeo2XYZ(theta, dist, layer, R, t):
    '''
    Berechnet die kartesischen X,Y,Z-Koordinaten aus polaren Koordinaten des IBEO Lux Laserscanners
   
    Input:
        - theta: Horizontaler Winkel
        - dist : polarer Abstand
        - layer: Ebene
        - R    : Euler Rotationsmatrix (Rotation Laserscanner)
        - t    : Translationsvektor (Position Laserscanner)
    
    if not R.shape == (3,3):
        raise ValueError('Rotationsmatrix muss 3x3 sein')
    if not t.shape == (3,1):
        raise ValueError('Translationsvektor muss 3x1 sein: [X],[Y],[Z]')
    '''
    
    # Ibeo Lux hat 3.2° bei 4 Ebenen vertikal
    oeffnungswinkel = 3.2
    ebenen = 4.0
    
    # aus Ebene den Vertikalwinkel berechnen
    phi = (layer * oeffnungswinkel/(ebenen-1) - oeffnungswinkel/2.0) * np.pi/180.0
    
    xe = dist * np.cos(theta)
    ye = dist * np.sin(theta)
    ze = dist * np.sin(phi)
    
    '''
    RSensor = np.eye(4) # Einheitsmatrix erstellen

    # Rotationsteil
    RSensor[np.ix_([0,1,2],[0,1,2])] = R

    # Translationsteil
    RSensor[np.ix_([0,1,2],[3])] = t
    
    Pointcloud = np.array((X,Y,Z,np.ones(np.size(X))))

    # Homogene Multiplikation von Punkten und Rotation+Translation
    [xe,ye,ze,w] = np.dot(RSensor, Pointcloud)
    '''
    return np.array([xe, ye, ze])

# <codecell>

def raycast(angle, distance, layer, R, t, dd):
    '''
    Calculates the Cells, which are crossed by a laser beam
    Input:
    '''
    dists = np.arange(0.0, distance, dd)
    CellHit = np.zeros([len(dists),3])

    for i,d in enumerate(dists):
        [X, Y, Z] = ibeo2XYZ(angle, d, layer, R, t)
        CellHit[i]= [int(X), int(Y), int(Z)]
        
    # Make visited Cells Unique
    # Thanks Joe Kingdon: http://stackoverflow.com/a/16971224
    uniq = np.unique(CellHit.view(CellHit.dtype.descr * CellHit.shape[1]))
    uniqCellHit = uniq.view(CellHit.dtype).reshape(-1, CellHit.shape[1])
    return uniqCellHit[:-1]

# <codecell>

angle = 0.55
dist = 6.5
layer= 1
R = np.matrix(np.eye(3))
t = np.zeros([3,1])

# <codecell>

%%timeit
cells = raycast(angle, dist, layer, R, t, 0.5)

# <codecell>

cells

# <headingcell level=1>

# Timing Vergleich zum Erstellen eines Grids

# <codecell>

l = 1000
b = 1000
h = 200

# <headingcell level=3>

# Python Nativ

# <codecell>

%timeit grid = [[[0 for x in range(l)] for y in range(b)] for z in range(h)]

# <headingcell level=3>

# Numpy Methods

# <headingcell level=4>

# Int

# <codecell>

%timeit grid = np.zeros((l, b, h), dtype=np.int)

# <codecell>

%timeit grid = np.ones((l, b, h), dtype=np.int)

# <codecell>

%timeit grid = -1 * np.ones((l, b, h), dtype=np.int)

# <codecell>

%%timeit
grid = np.empty((l,b,h))
grid.fill(-1.0)

# <headingcell level=4>

# Float32

# <codecell>

%timeit grid = np.zeros((l, b, h), dtype=np.float32)

# <codecell>

%timeit grid = np.ones((l, b, h), dtype=np.float32)

# <codecell>

%timeit grid = -1.0 * np.ones((l, b, h), dtype=np.float32)

# <codecell>


# <codecell>


