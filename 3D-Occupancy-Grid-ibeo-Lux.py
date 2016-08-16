
# coding: utf-8

# # 3D Occupancy Grid with ibeo Lux Laserscanner

# ![ibeo Lux](http://www.mechlab.de/wp-content/uploads/2012/02/ibeoLUX.jpg)

# In[1]:

import numpy as np
import time
import pandas as pd
import pickle


# In[2]:

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from IPython.html.widgets import interact
from IPython.html import widgets
get_ipython().magic(u'matplotlib inline')


# ### Create Empty Grid

# In[3]:

l = 10.0 # Länge m
b = 10.0  # Breite m
h = 2.0  # Höhe m

r = 0.1 # Resolution m/gridcell


# In[4]:

print('%.1fmio Grid Cells' % ((l*b*h)/r**3/1e6))


# ## Generate a LogOdds Grid

# Why LogOdds? Numerically stable around $p=0$ and $p=1$ and reduces the mathematical efford to update the Grid (Bayes Rule) to just an addition!

# In[5]:

p = np.arange(0.01, 1.0, 0.01)
lo = np.log(p/(1-p))
plt.plot(p, lo)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Probability $p$')
plt.ylabel(r'Log Odds, $\log(\frac{p}{1-p})$')


# So an initial uncertainty ($p=0.5$) is a zero in LogOdds. That's fine, because it is a very fast initialization of the grid!
# In order to store log(odds), we need negative values and decimal values. And `float32` ist fastest: http://stackoverflow.com/questions/15340781/python-numpy-data-types-performance
# 
# So let's use it!

# In[6]:

print "%ix%ix%i Grid" % (l/r, b/r, h/r)
startTime = time.time()

grid = np.zeros((l/r, b/r, h/r), dtype=np.float32) # Log Odds Grid must be initialized with zeros!

print "Stats: %.2fs, %.2fGB" % (time.time() - startTime, (grid.nbytes/1024.0**2))


# ### 3D View

# In[7]:

def plot3Dgrid(grid, az, el):
    # plot the surface
    plt3d = plt.figure(figsize=(12, 6)).gca(projection='3d', axisbg='w')

    # create x,y
    ll, bb = np.meshgrid(range(grid.shape[1]), range(grid.shape[0]))

    for z in range(grid.shape[2]):
        if not (np.max(grid[:,:,z])==np.min(grid[:,:,z])): # unberührte Ebenen nicht darstellen
            cp = plt3d.contourf(ll, bb, grid[:,:,z], offset = z, alpha=0.3, cmap=cm.Greens)

    cbar = plt.colorbar(cp, shrink=0.7, aspect=20)
    cbar.ax.set_ylabel('$P(m|z,x)$')
    
    plt3d.set_xlabel('X')
    plt3d.set_ylabel('Y')
    plt3d.set_zlabel('Z')
    plt3d.set_xlim3d(0, grid.shape[0])
    plt3d.set_ylim3d(0, grid.shape[1])
    plt3d.set_zlim3d(0, grid.shape[2])
    #plt3d.axis('equal')
    plt3d.view_init(az, el)
    return plt3d


# In[8]:

#plot3Dgrid(grid, 25, -30)


# ## Integrate a measurement with BRESENHAM Algorithm

# Amanatides, J., & Woo, A. (1987). A fast voxel traversal algorithm for ray tracing. Proceedings of EUROGRAPHICS, i. Retrieved from http://www.cse.yorku.ca/~amana/research/grid.pdf
# 
# Here is a Python Implementation of BRESENHAM Algorithm: https://gist.github.com/salmonmoose/2760072

# In[9]:

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


# In[10]:

import string
letters = string.lowercase


# In[11]:

goal = (5.5, 3.5, 0.0)

plt.figure(figsize=(5.0,3.6))
plt.scatter(goal[0], goal[1], s=50, c='r')
plt.plot((0, goal[0]), (0, goal[1]), c='k', alpha=0.5)
plt.axis('equal');
plt.xlim(0, 6)
plt.ylim(0, 4)
plt.xlabel('X')
plt.ylabel('Y')

# Annotations
#cells = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (2.5, 1.5), (3.5, 1.5), (3.5, 2.5), (4.5, 2.5)]
cells = bresenham3D((0,0,0), (goal[0], goal[1], 0.0))

for i, cell in enumerate(cells):
    plt.text(cell[0]+0.5, cell[1]+0.5, letters[i], ha='center', va='center')

plt.savefig('BRESENHAM-Raycasting.png', dpi=150)


# Does not hit all traversed grid cells

# ### Sensor Position and Orientation

# Rotation und Translation in homogenen Koordinaten, d.h. es kann alles über Matrizenmultiplikation gemacht werden.
# 
# $$\left[\begin{matrix}x \\ y \\ z \\ 1\end{matrix}\right]_\text{Endpoint} = \left[\begin{matrix} R_{3x3} & t_{1x3} \\ 0 & 1\end{matrix}\right] \cdot \left[\begin{matrix}x \\ y \\ z \\ 1\end{matrix}\right]_\text{Messpunkte}$$
# 
# wobei $R$ die Rotationsmatrix ist und $t$ der Verschiebungsvektor

# In[12]:

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


# In[13]:

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


# In[14]:

# or generate some values synthetically:
#angles = np.arange(-15, 15, 0.25)/180.0*np.pi
#distance = 5.0*np.ones(len(angles))
#layer = 3*np.ones(len(angles)) # Ebene {0,1,2,3}


# ### Load some ibeo Lux Measurements

# In[15]:

# some real ibeo lux measurements
data = pd.read_csv('Messung1.txt', delimiter='|')


# In[16]:

data.head(5)


# #### Filter out an arbitrary measurement and bounded angle

# In[17]:

timestamp = 1341907053031
f = (data['# <Zeitstempel>']==timestamp) & (data['<Winkel>']<0.5) & (data['<Winkel>']>-0.5)

angles = data['<Winkel>'][f]
distance = data['<Radius>'][f]/100.0
layer = data['<Ebene>'][f]


# In[18]:

yaw   = 0.0 #  Gieren in Grad
pitch = 0.0 #  Nicken in Grad
roll  = 0.0 #  Wanken in Grad
dx= 0.0 #  Verschiebung in X in Meter
dy= 5.0 #  Verschiebung in Y in Meter
dz= 1.0 #  Verschiebung in Z in Meter


# In[19]:

# Convert from spherical coordinates to cartesian
R = Rypr(yaw, pitch, roll)
t = np.array([[dx], [dy], [dz]]) 
[xe, ye, ze] = ibeo2XYZ(angles.values, distance.values, layer.values, R, t)


# In[20]:

plt3d = plt.figure(figsize=(12, 6)).gca(projection='3d', axisbg='w')
plt3d.scatter(xe, ye, ze, c='r', label='Laserscanner Pointcloud')
plt3d.scatter(t[0], t[1], t[2], c='k', s=200, label='ibeo Lux')
plt3d.view_init(45, -115)
plt3d.axis('equal')
plt3d.set_xlabel('X')
plt3d.set_ylabel('Y')


# ## Function which integrates the Measurement via Inverse Sensor Model

# Values for hit and miss probabilities are taken from Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: an efficient probabilistic 3D mapping framework based on octrees. Autonomous Robots, 34(3), 189–206. doi:10.1007/s10514-012-9321-0

# In[21]:

# in LogOdds Notation!
loccupied = 0.85
lfree = -0.4

lmin = -2.0
lmax = 3.5


# In[22]:

def insertPointcloudBRESENHAM(tSensor, xe,ye,ze):
    
    for i,val in enumerate(xe):
        
        # Insert Endpoints
        y=int(xe[i])
        x=int(ye[i]) # !!! Koordinatenswitch zwischen X & Y
        z=int(ze[i])

        # Inverse Sensor Model
        grid[x,y,z] += loccupied # increase LogOdds Ratio

        if grid[x,y,z]>lmax: #clamping
            grid[x,y,z]=lmax

        
        # Grid cells in perceptual range of laserscanner
        for (y,x,z) in bresenham3D(tSensor, (xe[i], ye[i], ze[i])): # !!! Koordinatenswitch zwischen X & Y

            grid[x,y,z] += lfree # decrease LogOdds Ratio

            if grid[x,y,z]<lmin: #clamping
                grid[x,y,z]=lmin        


# ### Sensor Origin

# In[23]:

tSensor = t/r  # Translation (shift from 0,0,0) in Grid Cell Numbers
tSensor


# In[24]:

# integrate the measurement 5 times
for m in range(5):
    try:
        insertPointcloudBRESENHAM(tSensor, xe/r,ye/r,ze/r)
    except:
        print('Fehler beim Einfügen der Messung. Grid zu klein gewählt?!')


# ### 2D Plot of Grid Layer Z

# In[25]:

@interact
def plotmultivargauss(z = widgets.FloatSliderWidget(min=0, max=np.max(grid.shape[2])-1, step=1, value=10, description="")):
    plt.figure(figsize=(l/2, b/2))
    plt.contourf(grid[:,:,z], cmap=cm.Greens)
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')


# ### 3D Plot

# In[26]:

@interact
def plotmultivargauss(az = widgets.FloatSliderWidget(min=-90.0, max=90.0, step=1.0, value=45.0, description=""),                       el = widgets.FloatSliderWidget(min=-180.0, max=180.0, step=1.0, value=-115.0, description="")):

    plot3Dgrid(grid, az, el)


# In[27]:

print('Max Grid Value (Log Odds): %.2f' % np.max(grid))
print('Min Grid Value (Log Odds): %.2f' % np.min(grid))


# #### Dump the Occupancy Grid to file

# In[28]:

pklfile = open('occupancy-grid-LogOdds.pkl', 'wb')
pickle.dump(grid, pklfile)
pklfile.close()


# ## From LogOdds Occupancy Grid to Probability Grid

# The conversion from LogOdds notation to probabilities could be achieved by following formula:
# 
# $$P(l) = 1-\cfrac{1}{1+e^{lo}}$$ with $lo$=LogOdds Value

# In[29]:

gridP = np.asarray([1.0-(1.0/(1.0+np.exp(lo))) for lo in grid])


# In[30]:

plot3Dgrid(gridP, 45, -115)
plt.savefig('3D-Occupancy-Grid.png')


# In[31]:

print('Max Grid Value (Probability): %.2f' % np.max(gridP))
print('Min Grid Value (Probability): %.2f' % np.min(gridP))


# In[32]:

print('Done.')


# ## Convolve the Map for Path Planning

# In[33]:

from scipy.ndimage import gaussian_filter


# In[34]:

blurmap = gaussian_filter(gridP, 0.4)


# In[35]:

plot3Dgrid(blurmap, 45, -115)


# In[36]:

print('Max Grid Value (Probability): %.2f' % np.max(blurmap))
print('Min Grid Value (Probability): %.2f' % np.min(blurmap))


# #### Dump the convolved map

# In[37]:

pklfile = open('occupancy-grid-Blur.pkl', 'wb')
pickle.dump(blurmap, pklfile)
pklfile.close()


# In[37]:



