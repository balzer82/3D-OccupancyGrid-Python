# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab

from IPython.html.widgets import interact
from IPython.html import widgets

%matplotlib inline

# <headingcell level=1>

# Inverse Sensor Models

# <markdowncell>

# Gives the probability $P$ of how a sensor reading $z$ might be, with a given sensor pose $x$ and an environment $m$.
# 
# $P(z\mid x,m)$

# <headingcell level=2>

# 2D Mixture Density

# <markdowncell>

# Good tutorial video:

# <codecell>

from IPython.display import HTML
HTML("""
<video width="500" height="260" controls>
  <source src="http://ais.informatik.uni-freiburg.de/teaching/ss13/robotics/recordings/rob1-07-sensor-models.mp4" type="video/mp4">
</video>
""")

# <markdowncell>

# $P(z\mid x,m) = \left(\begin{matrix}P_{hit}(z\mid x,m) \\ P_{unexp}(z \mid x,m) \\ P_{max}(z \mid x,m) \\ P_{rand}(z \mid x,m) \end{matrix}\right)$

# <codecell>

zs = np.arange(0, 10.01, 0.01)   # Entfernungen

# <headingcell level=4>

# Measurement Noise

# <markdowncell>

# Sensor readings are normal distributed around the true value with standard deviation $\sigma_z$ and mean $z_{exp}$.
# 
# $P_{hit}(z\mid x,m) = \frac{1}{\sqrt{2 \pi \sigma_z^2}} \cdot \exp{\left(-\frac{1}{2} \cdot \frac{(z-z_{exp})^2}{\sigma_z^2}\right)}$

# <codecell>

varz = 0.05 # Variance
zexp = 5.0  # here is the obstacle

Pzx_hit = 1.0/np.sqrt(2*np.pi*varz) * np.exp(-1/2.0*(zs-zexp)**2/varz)

# <headingcell level=4>

# Unexpected Obstacles

# <markdowncell>

# There might be measurements before the beam hits the real obstacle.
# 
# $P_{unexp}(z \mid x,m) = \lambda \cdot \exp{\left(-\lambda \cdot z\right)}$

# <codecell>

lamb = 0.5

Pzx_unexp = lamb * np.exp(-lamb * zs)
Pzx_unexp[zs>zexp] = 0.0

# <headingcell level=4>

# Random Measurements

# <markdowncell>

# There is a random distribution, that the measurement is somewhere.
# 
# $P_{rand}(z \mid x,m) = \frac{1}{z_{max}}$

# <codecell>

Pzx_rand = np.ones(len(zs)) * 0.01

# <headingcell level=4>

# Max Range Model

# <markdowncell>

# The beam can get lost and is responding with the maximum value.
# 
# $P_{max}(z \mid x,m) = \delta(z_{max})$

# <codecell>

Pzx_maxrange = np.zeros(len(zs))
Pzx_maxrange[-1:] = 1.5

# <headingcell level=3>

# Resulting Mixture Density

# <codecell>

Pzx = Pzx_hit + Pzx_unexp + Pzx_rand + Pzx_maxrange

# <codecell>

plt.figure(figsize=(12,6))
ax1 = plt.subplot2grid((4,4), (0,0), rowspan = 2)
plt.plot(zs, Pzx_hit)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('Measurement Noise')
plt.ylabel(r'$P(z \mid x,m)$')

ax2 = plt.subplot2grid((4,4), (0,1), rowspan = 2, sharey=ax1)
plt.plot(zs, Pzx_unexp)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('Unexcpected Obstacles')

ax3 = plt.subplot2grid((4,4), (0,2), rowspan = 2, sharey=ax1)
plt.plot(zs, Pzx_rand)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('Random Measurement')

ax4 = plt.subplot2grid((4,4), (0,3), rowspan = 2, sharey=ax1)
plt.plot(zs, Pzx_maxrange)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('Max Range Measurement')

ax5 = plt.subplot2grid((4,4), (2,0), colspan=4, rowspan = 3)
plt.plot(zs, Pzx)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('= Mixture Density')
plt.ylabel(r'$P(z|x,m)$')
plt.xlabel('z [$m$]')

for ax in [ax2, ax3, ax4]:
    plt.setp(ax.get_yticklabels(), visible=False)
    # The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
    #ax.set_yticks(ax.get_yticks()[1:])
    ax.text(-1.5,np.max(Pzx_hit)/2,'+', fontsize=20)

plt.tight_layout()
plt.savefig('InverseSensorModel-MixtureDensity.png', dpi=150)

# <headingcell level=2>

# Fit the model with real data

# <markdowncell>

# ![ibeo Lux](http://www.mechlab.de/wp-content/uploads/2012/02/ibeoLUX.jpg)
# 
# Maximize likelihood of the data:
# $P(z \mid z_{exp})$

# <codecell>

import pandas as pd

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

data = pd.read_csv('kalibrierungtisch2.txt', delimiter='|')

# <codecell>

data.head(5)

# <headingcell level=4>

# Let's see how they are collected, line by line

# <codecell>

@interact
def integratemeasurement(t = widgets.IntSliderWidget(min=1, max=5000, step=1, value=40, description="")):
    #print t
    f = (data['<Ebene>']==3) & (data['<Winkel>']>-0.015) & (data['<Winkel>']<0.015)
    angle = data['<Winkel>'][f].iloc[:t]
    distance = data['<Radius>'][f].iloc[:t] /100.0
    layer = data['<Ebene>'][f].iloc[:t]

    plt.scatter(distance,angle, s=50, alpha=0.1)
    
    plt.axis('equal')
    plt.xlabel('z [$m$]')
    plt.ylabel('$\phi$ [$rad$]')

# <codecell>

timestamps = data['# <Zeitstempel>'].unique()

# <codecell>

x=[]
y=[]
for i, timestamp in enumerate(timestamps):
    
    #f = (data['# <Zeitstempel>']==timestamp) & (data['<Ebene>']==3)
    #f = (data['# <Zeitstempel>']==timestamp) & ((data.index%467.0)==0)
    f = (data['# <Zeitstempel>']==timestamp) & (data['<Ebene>']==3) & (data['<Winkel>']>-0.015) & (data['<Winkel>']<0.01)
    
    angle = data['<Winkel>'][f]
    distance = data['<Radius>'][f] /100.0
    layer = data['<Ebene>'][f]
    
    # 350. Punkt jedes Zeitstempels nehmen
    [xe, ye, ze] = ibeo2XYZ(angle, distance, layer)
    
    x.extend(xe)
    y.extend(ye)

# <codecell>

plt.scatter(x,y)
plt.axis('equal')

# <codecell>


# <codecell>

plt.figure(figsize=(6,3))
plt.hist(x, bins=5, align='mid');
#plt.axvline(np.mean(x), alpha=0.6, c='k')
plt.xlim(0, 5)
#plt.title('Histogram of real ibeo Lux Sensor measurement')
plt.xlabel('z [$m$]')

plt.tight_layout()
plt.savefig('Histogram-ibeoLux-InverseSensorModel.png', dpi=150)

# <markdowncell>

# Actually, the sensor is so good, that all assumptions are not necessary. It basically just detects the obstacle with a very narrow normal distribution. In case of a deterministic grid, you even do not have to assume a normal distribution, because the discrete cells can't determine the difference.
# 
# Like so, it is implemented in Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: an efficient probabilistic 3D mapping framework based on octrees. Autonomous Robots, 34(3), 189–206. doi:10.1007/s10514-012-9321-0

# <headingcell level=2>

# Multivariante Gaussian for Sensor

# <markdowncell>

# Sensors scan a 3D environment, so we have to take a look at a multivariant gaussian sensor.

# <headingcell level=4>

# Range for Sensor Readings

# <codecell>

zs = np.arange(0, 10.1, 0.01)   # Entfernungen
ts = np.arange(-1, 1.1, 0.01)   # Winkel

# <headingcell level=4>

# Standard Deviations for Angle and Distance

# <codecell>

def calcmultivargauss(sigmaz, sigmat, dxy, txy):
    P = np.zeros((len(zs), len(ts)))
    x = np.zeros((len(zs), len(ts)))
    y = np.zeros((len(zs), len(ts)))
    for i,z in enumerate(zs):
        for j,t in enumerate(ts):
            x[i,j] = z * np.cos(t)
            y[i,j] = z * np.sin(t)
            P[i,j] = 1.0/(2.0*np.pi*sigmat*sigmaz) * np.exp(-0.5*((z-dxy)**2/sigmaz**2 + (t-txy)**2/sigmat**2))
            
    return x,y,P

# <codecell>

@interact
def plotmultivargauss(sigmaz = widgets.FloatSliderWidget(min=0.01, max=5.0, step=0.01, value=0.3, description=""), \
                      sigmat = widgets.FloatSliderWidget(min=0.01, max=1.0, step=0.01, value=0.2, description=""), \
                      dxy = widgets.FloatSliderWidget(min=0.0, max=9.0, step=0.1, value=4.0, description=""), \
                      txy = widgets.FloatSliderWidget(min=-1.0, max=1.0, step=0.1, value=0.0, description="")):
    
    x,y,P = calcmultivargauss(sigmaz, sigmat, dxy, txy)
    
    plt.contourf(x, y, P, cmap=cm.gray_r)
    plt.scatter(0, 0, s=250, c='k')
    plt.plot([0, 10*np.cos(np.max(ts))],[0, 10*np.sin(np.max(ts))], '--k')
    plt.plot([0, 10*np.cos(np.min(ts))],[0, 10*np.sin(np.min(ts))], '--k')
    plt.xlabel('X');
    plt.ylabel('Y');
    plt.xlim([0, 8]);
    plt.ylim([-4, 4]);
    
    return plt

# <codecell>


# <codecell>

sigmaz = 0.3  # Entfernung
sigmat = 0.2  # Winkel

dxy = 8.0  # Sensor Reading Entfernung
txy = 0.0  # Sensor Reading Winkel

x, y, P = calcmultivargauss(sigmaz, sigmat, dxy, txy)

# <headingcell level=4>

# 3D Plot

# <codecell>

fig = plt.figure(figsize=(12,6))
ax = fig.gca(projection='3d', axisbg='w')

ax.scatter(0,0, s=100, c='k')
ax.plot_wireframe(x, y, P, rstride=50, cstride=5)

ax.set_xlabel(r'x [$m$]')
#ax.set_xlim(0, 8)
ax.set_ylabel('y [$m$]')
#ax.set_ylim(-4, 4)
ax.set_zlabel('P')
#ax.set_zlim(-100, 100)
ax.view_init(elev=45., azim=180.)
plt.savefig('InverseSensorModel-3D.png', dpi=150)

# <headingcell level=2>

# KONOLIGE Ansatz

# <markdowncell>

# Is ja alles ganz nett, aber lässt sich schlecht implementieren. Deshalb Ansatz von Konolige, K. (1997). Improved occupancy grids for map building. Autonomous Robots, 367, 351–367. Retrieved from http://cs.krisbeevers.com/research/research_qual/05-konolige97.pdf
# 
# 1. The range error becomes proportionally larger at increasing range.
# 2. The probability of detection becomes smaller at larger ranges.
# 
# A mathematical model for target reflection in the 1D case is:
# 
# $$p(z=D|C) = \frac{\alpha(z_i)}{\sqrt{2\pi} \delta(z_i)} \cdot \exp{\frac{-(D-z_i)^2}{2\delta(z_i)^2}}$$
# 
# where the target is at distance $z_i$ from the transducer. In this model, $\alpha(z_i)$ is the attenuation of detection with distance, $\delta(z_i)$ is the range variance (increasing with distance).
# 
# Example: For a Polaroid Ultrasonic $d(z) = 0.01 + 0.015\cdot z$ and $\alpha = 0.6(1-\min(1; 0.25r))$

# <codecell>

zs = np.arange(0, 10.01, 0.01)   # Entfernungen
ts = np.arange(-1, 1.01, 0.01)    # Winkel

# <codecell>

D = 5.0 # Sensor Measurement

# <codecell>

d = 0.01+0.15*zs
a = 0.1*(1-np.min((np.ones(len(zs)), 1/10.0*zs), axis=0))

# <codecell>

plt.figure(figsize=(9,3))
plt.subplot(121)
plt.plot(zs, d)
plt.xlabel('Distance in m')
plt.ylabel(r'$\delta(z)$')
plt.title('Range Error')

plt.subplot(122)
plt.plot(zs, a)
plt.xlabel('Distance in m')
plt.ylabel(r'$\alpha(z)$')
plt.title('Detection Attenuation')

plt.tight_layout()

# <codecell>

# The effect of F is to make the no change for cells everywhere but in the vicinity of the range reading r = D
F = 0.0001

# <codecell>

# probability Density
p = a*np.exp(-(zs-D)**2/(2.0*d**2)) + F

p = p/np.max(p)

plt.plot(zs, p)
plt.axvline(D, c='k')
plt.ylabel(r'$P(z|x,m)$')
plt.xlabel('z [$m$]')

# <codecell>


# <markdowncell>

# Probability that no range reading was received at a distance less than D:
# 
# $P = 1- \int_0^D p(r=x|C)\mathrm{d}x$

# <codecell>

# probability that no range reading was received at a distance less than D
P = 1.0-np.cumsum(p)/np.sum(p)
P[zs>D] = 0.0

plt.plot(zs, P)
plt.axvline(D, c='k', alpha=0.2)
plt.ylabel(r'$P(\neg z|x,m)$')
plt.xlabel('z [$m$]')

# <codecell>

# sensor probability density
spd = P*p

spd = spd/np.max(spd)

plt.plot(zs, spd)
plt.axvline(D, c='k', alpha=0.5)
#plt.ylim(0, 0.01)
plt.ylabel(r'$P(z|x,m)$')
plt.xlabel('z [$m$]')

# <codecell>

plt.figure(figsize=(8,5))
ax1 = plt.subplot2grid((4,4), (0,0), rowspan = 2, colspan = 2)
plt.plot(zs, p)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('Measurement Noise')
plt.ylabel(r'$P(z \mid x,m)$')

ax2 = plt.subplot2grid((4,4), (0,2), rowspan = 2, colspan = 2, sharey=ax1)
plt.plot(zs, P)
plt.ylabel(r'$P(\neg z|x,m)$')
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('No Range Reading')

ax3 = plt.subplot2grid((4,4), (2,0), colspan=4, rowspan = 3)
plt.plot(zs, spd)
plt.axvline(zexp, c='k', alpha=0.5)
plt.title('= KONOLIGE Inverse Sensor Model')
plt.ylabel(r'$P(z|x,m)$')
plt.xlabel('z [$m$]')

for ax in [ax2]:
    plt.setp(ax.get_yticklabels(), visible=False)
    # The y-ticks will overlap with "hspace=0", so we'll hide the bottom tick
    #ax.set_yticks(ax.get_yticks()[1:])
    ax.text(-1.5,np.max(P)/2,'+', fontsize=20)

plt.tight_layout()
plt.savefig('InverseSensorModel-KONOLIGE.png', dpi=150)

# <headingcell level=2>

# LANGERWISCH Error Bounding Box

# <markdowncell>

# Langerwisch, M., & Wagner, B. (2013). Building variable resolution occupancy maps assuming unknown but bounded sensor errors. Intelligent Robots and Systems ( …, 4687–4693. Retrieved from http://www.rts.uni-hannover.de/images/2/24/Langerwisch13-IROS.pdf

# <headingcell level=4>

# Simple Model

# <markdowncell>

# The error afflicted dis- tance will be the interval [d], having w([d]) as the maximum assumed error bound. The laser beam is assumed to spread in width with an angle of ϕb. Together with the angular position of the scanning mirror ϕm, the beam angle results in [ϕ] = [ϕm − ϕb/2 ,ϕm + ϕb/2 ]. Now, an outer boundary of
# the possible locations of the real object reflection can be calculated in the sensor coordinate frame:

# <markdowncell>

# $[p'] = \left[\begin{matrix}d \cdot \cos(\phi) \\ d \cdot \sin(\phi)\end{matrix}\right] $

# <codecell>

wd = 0.04 # maximum assumed error bound for distance measurement
phib = 0.01 # rad

d = 1.0     # m
phi = 0.8   # rad

# <codecell>

@interact
def plotmultivargauss(wd = widgets.FloatSliderWidget(min=0.005, max=1.0, step=0.005, value=0.05, description=""), \
                      phib = widgets.FloatSliderWidget(min=0.01, max=1.0, step=0.01, value=0.01, description=""), \
                      d = widgets.FloatSliderWidget(min=0.0, max=10.0, step=0.1, value=1.0, description=""), \
                      phi = widgets.FloatSliderWidget(min=-1.0, max=1.0, step=0.1, value=0.6, description="")):

    p = {}
    p['p1'] = ((d-wd) * np.cos(phi-phib), (d-wd) * np.sin(phi-phib))
    p['p2'] = ((d+wd) * np.cos(phi-phib), (d+wd) * np.sin(phi-phib))
    p['p3'] = ((d+wd) * np.cos(phi+phib), (d+wd) * np.sin(phi+phib))
    p['p4'] = ((d-wd) * np.cos(phi+phib), (d-wd) * np.sin(phi+phib))
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    [plt.scatter(v[0], v[1], s=10) for k,v in p.iteritems()]
    plt.plot((0, d*np.cos(phi)), (0, d*np.sin(phi)), c='r', alpha=0.5)
    plt.axis('equal')
    plt.xlabel('x [$m$]')
    plt.ylabel('y [$m$]')
    plt.title('Error Bounding Box')
    plt.scatter(0,0, s=50, c='k', label='Sensor')

    plt.subplot(122)
    [plt.scatter(v[0], v[1], s=100) for k,v in p.iteritems()]
    plt.axis('equal')
    axlims = plt.axis()
    plt.plot((0, d*np.cos(phi)), (0, d*np.sin(phi)), c='r', alpha=0.5)
    plt.axis(axlims)
    plt.title('Error Bounding Box (Zoom In)')

    plt.tight_layout()
    #plt.savefig('InverseSensorModel-Langerwisch-BoundingBox.png', dpi=150)
    return plt

# <headingcell level=4>

# Including Beam Width

# <markdowncell>

# Moreover, the width of the laser beam has to be considered, because it actually never starts with beam width 0.
# Therefore, we displace the origin of the beam virtually to behind. The distance [d] is extended by

# <codecell>

@interact
def plotmultivargauss(wd = widgets.FloatSliderWidget(min=0.005, max=1.0, step=0.005, value=0.05, description=""), \
                      phib = widgets.FloatSliderWidget(min=0.01, max=1.0, step=0.01, value=0.01, description=""), \
                      d = widgets.FloatSliderWidget(min=0.0, max=10.0, step=0.1, value=1.0, description=""), \
                      phi = widgets.FloatSliderWidget(min=-1.0, max=1.0, step=0.1, value=0.6, description=""), \
                      wb = widgets.FloatSliderWidget(min=0.01, max=1.0, step=0.01, value=0.01, description="")):

    dd = 1.0/np.tan(phib/2.0) * wb/2
    
    p = {}
    p['p1'] = (((d+dd-wd) * np.cos(phi-phib)) - dd*np.cos(phi), ((d+dd-wd) * np.sin(phi-phib)) - dd*np.sin(phi))
    p['p2'] = (((d+dd+wd) * np.cos(phi-phib)) - dd*np.cos(phi), ((d+dd+wd) * np.sin(phi-phib)) - dd*np.sin(phi))
    p['p3'] = (((d+dd-wd) * np.cos(phi+phib)) - dd*np.cos(phi), ((d+dd-wd) * np.sin(phi+phib)) - dd*np.sin(phi))
    p['p4'] = (((d+dd+wd) * np.cos(phi+phib)) - dd*np.cos(phi), ((d+dd+wd) * np.sin(phi+phib)) - dd*np.sin(phi))
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    [plt.scatter(v[0], v[1], s=10) for k,v in p.iteritems()]
    plt.plot((0, d*np.cos(phi)), (0, d*np.sin(phi)), c='r', alpha=0.5)
    plt.axis('equal')
    plt.title('Error Bounding Box')
    plt.scatter(0,0, s=50, c='k', label='Sensor')

    plt.subplot(122)
    [plt.scatter(v[0], v[1], s=100) for k,v in p.iteritems()]
    plt.axis('equal')
    axlims = plt.axis()
    plt.plot((0, d*np.cos(phi)), (0, d*np.sin(phi)), c='r', alpha=0.5)
    plt.axis(axlims)
    plt.title('Error Bounding Box (Zoom In)')

    plt.tight_layout()
    
    return plt

