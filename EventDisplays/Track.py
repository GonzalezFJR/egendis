'''
  [Event Display Producer]
  Implements tracker tracks
'''

import numpy as np
from numpy import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

npoints = 1000

def Circle(R, x0=0, y0=0, amin=0, amax=2*pi):
  ''' Draw a circle '''
  t = np.linspace(amin, amax, npoints)
  x = x0 + R*cos(t)
  y = y0 + R*sin(t)
  return x, y

def NewTrack(R, Rmax = 1, sign=None, theta=None, x0=None, y0=None):
    ''' Get x,y points for a track with a curvature R inside a circle of radius Rmax '''
    if x0 is None or y0 is None:
      if theta is None: 
        theta = np.random.uniform(0, 2*pi)
      x0 = R*cos(theta)
      y0 = R*sin(theta)
    if sign is None: sign = np.random.choice([-1, 1])
    x,y = Circle(R, x0, y0, theta, theta-pi*sign)
    is_in = sqrt(x*x + y*y) < Rmax
    x = x[is_in]
    y = y[is_in]
    return x, y

def GetTrackForAngle(R, angle=None, curvature=None, rnom=1., smear=0.05, isRad=False):
  ''' R > rnom... curvature +/- 1 for clockwise/anticlockwise '''
  if angle is None: 
    return NewTrack(R, rnom)
  if curvature is None: 
    curvature = np.random.choice([-1, 1])
  if smear is None:
    smear = 0.05
  alpha = angle/180*pi if not isRad else angle
  while alpha <= 0: alpha += 2*pi
  alpha = np.random.normal(alpha, smear/alpha)
  th = -1.*curvature*np.arccos(1/(2*R)) + alpha # 1 --> rnom
  #beta = 2*alpha - th
  return NewTrack(R, rnom, -1.*curvature if curvature is not None else None, th)

def GetTrackPosMax(x, y):
  imax = np.argmax(x*x + y*y)
  return x[imax], y[imax]

def GetXYforAngle(th, r=1):
  x = r*cos(th)
  y = r*sin(th)
  return x, y

class Track:
  ''' A set of points to draw a track with a color '''
  def __init__(self, R, Rmax=1, color=[1,1,1], width=1, angle=None, sign=None, smear=0.05):
    x, y = GetTrackForAngle(R, angle=angle, rnom=Rmax, curvature=sign, smear=smear)
    self.x = x
    self.y = y
    self.color = color
    self.width = width

    xmax, ymax = GetTrackPosMax(x, y)
    self.angle = angle if angle is not None else np.arctan2(ymax,xmax)
    while self.angle < 0: self.angle += 2*pi
    self.xmax = Rmax*cos(self.angle)
    self.ymax = Rmax*sin(self.angle)

  def GetTrack(self):
    return self.x, self.y

  def GetTrackPosMax(self):
    return self.xmax, self.ymax

  def GetColor(self):
    return self.color

  def SetColor(self, color):
    self.color = color

  def GetMaxPoint(self):
    return self.xmax, self.ymax

  def Draw(self, ax):
    ax.plot(self.x, self.y, color=self.color, linewidth=self.width)
