'''
  [Event display producer]
  Implements an ECAL/HCAL tower
'''

import numpy as np
from numpy import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def NewTower(x, y, D, Off=0, dth=0.02, color=[0.3, 0.2, 1], thoffset=0):
  R = sqrt(x*x + y*y)
  th = np.arccos(x/R)*( (-1) if y<0 else 1   ) + thoffset
  R = R+Off # Add offset
  x1 = R*cos(th+dth)
  y1 = R*sin(th+dth)
  x2 = R*cos(th-dth)
  y2 = R*sin(th-dth)
  x3 = (R+D)*cos(th-dth)
  y3 = (R+D)*sin(th-dth)
  x4 = (R+D)*cos(th+dth)
  y4 = (R+D)*sin(th+dth)
  return Polygon([[x1, y1], [x2,y2], [x3,y3], [x4, y4]], color=color)


class Tower:

  def __init__(self, x, y, D, offset=0, angle=0.02, color=[0.3, 0.2, 1], thoffset=0):
    self.x = x
    self.y = y
    self.D = D
    self.offset = offset
    self.angle = angle
    self.color = color
    self.thoffset = thoffset
    self.Dmin = 0.1
    self.cluster = None

  def Get(self):
    self.poli = NewTower(self.x, self.y, self.D, self.offset, self.angle, self.color, self.thoffset)
    return self.poli

  def GetCluster(self, Dmax=0.8, angle_s=0.02, n=3):
    dangle = np.random.normal(0., angle_s, n)
    D = np.random.uniform(self.Dmin, Dmax, n)
    cluster = [NewTower(self.x, self.y, D[i], self.offset, self.angle, self.color, dangle[i]) for i in range(n)]
    self.cluster = cluster
    return cluster

  def Draw(self, ax):
    if self.cluster is None:
      ax.add_patch(self.Get())
    else:
      for p in self.cluster:
        ax.add_patch(p)
