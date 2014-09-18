# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import string
import os
letters = string.lowercase

# <headingcell level=1>

# Insert a Laserscan into an Occupancy Grid

# <codecell>

%matplotlib inline

# <markdowncell>

# Rotationmatrix, see [3D Rotation mit Euler und Quaternion on Vimeo](http://vimeo.com/100209309)

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

# <markdowncell>

# ![ibeo Lux](http://www.mechlab.de/wp-content/uploads/2012/02/ibeoLUX.jpg)

# <markdowncell>

# Rotation und Translation in homogenen Koordinaten, d.h. es kann alles über Matrizenmultiplikation gemacht werden.
# 
# $$\left[\begin{matrix}x \\ y \\ z \\ 1\end{matrix}\right]_\text{Endpoint} = \left[\begin{matrix} R_{3x3} & t_{1x3} \\ 0 & 1\end{matrix}\right] \cdot \left[\begin{matrix}x \\ y \\ z \\ 1\end{matrix}\right]_\text{Messpunkte}$$

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
    '''
    if not R.shape == (3,3):
        raise ValueError('Rotationsmatrix muss 3x3 sein')
    if not t.shape == (3,1):
        raise ValueError('Translationsvektor muss 3x1 sein: [X],[Y],[Z]')
    
    
    # Ibeo Lux hat 3.2° bei 4 Ebenen vertikal
    oeffnungswinkel = 3.2
    ebenen = 4.0
    
    # aus Ebene den Vertikalwinkel berechnen
    phi = (layer * oeffnungswinkel/(ebenen-1) - oeffnungswinkel/2.0) * np.pi/180.0
    
    X = dist * np.cos(theta)
    Y = dist * np.sin(theta)
    Z = dist * np.sin(phi)
    
    
    RSensor = np.eye(4) # Einheitsmatrix erstellen

    # Rotationsteil
    RSensor[np.ix_([0,1,2],[0,1,2])] = R

    # Translationsteil
    RSensor[np.ix_([0,1,2],[3])] = t
    
    Pointcloud = np.array((X,Y,Z,np.ones(np.size(X))))

    # Homogene Multiplikation von Punkten und Rotation+Translation
    [xe,ye,ze,w] = np.dot(RSensor, Pointcloud)
    
    return np.array([xe, ye, ze])

# <codecell>

yaw   = 0.0 #  Gieren
pitch = 0.0 #  Nicken
roll  = 0.0 #  Wanken
dx= 0.0 #  Verschiebung in X
dy= 0.0 #  Verschiebung in Y
dz= 0.0 #  Verschiebung in Z

# <codecell>

# some real ibeo lux measurements
data = pd.read_csv('Messung1.txt', delimiter='|')

# <codecell>

timestamp = 1341907053031
f = (data['# <Zeitstempel>']==timestamp) & (data['<Winkel>']<0.5) & (data['<Winkel>']>-0.5)

angles = data['<Winkel>'][f]
distance = data['<Radius>'][f]/100.0
layer = data['<Ebene>'][f]

# <codecell>

# Convert from spherical coordinates to cartesian
R = Rypr(yaw, pitch, roll)
t = np.array([[dx], [dy], [dz]]) 
[xe, ye, ze] = ibeo2XYZ(angles.values, distance.values, layer.values, R, t)

# <codecell>


# <codecell>

plt3d = plt.figure(figsize=(12, 6)).gca(projection='3d', axisbg='w')
plt3d.scatter(xe, ye, ze, c='r', label='Laserscanner Pointcloud')
plt3d.scatter(dx, dy, dz, c='k', s=200, label='ibeo Lux')
plt3d.view_init(45, -115)
plt3d.axis('equal')

# <headingcell level=2>

# Raycasting

# <markdowncell>

# Raycasting with incrementally increasing distance

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

angle = 0.57
dist = 6.6
layer= 1
raycast(angle, dist, layer, R, t, 0.1)

# <headingcell level=3>

# Visualisierung

# <codecell>

dd = 1.3

plt.figure(figsize=(5.0,3.6))
plt.scatter(t[0], t[1], s=50, c='k')
plt.scatter(t[0]+np.cos(angle)*dist, t[1]+np.sin(angle)*dist, s=50, c='r')
plt.plot((t[0], t[0]+np.cos(angle)*dist), (t[1], t[1]+np.sin(angle)*dist), c='k', alpha=0.5)
plt.axis('equal');
plt.xlim(0, 6)
plt.ylim(0, 4)
plt.xlabel('X')
plt.ylabel('Y')


dists = np.arange(0.0, dist, dd)
for d in dists:
    [X, Y, Z] = ibeo2XYZ(angle, d, layer, R, t)
    plt.scatter(X,Y, c='r', alpha=1.0, marker='*', s=40)

# Annotations
cells = raycast(angle, dist, layer, R, t, dd)

for i, cell in enumerate(cells):
    plt.text(cell[0]+0.5, cell[1]+0.5, letters[i], ha='center', va='center')

plt.title(r'$\Delta d$=%.1f' % dd)
plt.savefig('IncrementalDistance-Raycasting-d%.1f.png' % (dd), dpi=150)

# <codecell>

#os.system('convert -delay 50 Incremental*.png IncrementalDistance-Raycasting.gif')
#os.system('rm Incremental*.png')

# <codecell>


# <codecell>


