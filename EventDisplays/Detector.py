'''
  [Event display producer]
  Implements a simple detector
'''

import numpy as np
from numpy import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Objects import *

class Subdetector:
  def __init__(self, color=[0,0,0], width=1, x=None, y=None, poly=None):
    self.color = color
    self.width = width
    self.x = x
    self.y = y
    self.poly = poly

  def Draw(self, ax):
    if self.poly is not None:
      ax.add_patch(self.poly)
    else:
      ax.plot(self.x, self.y, color=self.color, linewidth=self.width)

class Detector:
  def __init__(self, Rtracker=1, DEcal=0.5, DHcal=0.6, Dmuon=0.1):
      self.Rtracker = Rtracker
      self.DEcal = DEcal
      self.DHcal = DHcal
      self.Dmuon = Dmuon
      self.REcal = self.Rtracker + self.DEcal
      self.RHcal = self.REcal + self.DHcal
      self.Rmuon = self.RHcal + self.Dmuon
      self.ColorMuon = [1,1,1]
      self.ColorTracker = [1,1,1]
      self.ColorECAL = [1,1,1]
      self.ColorHCAL = [1,1,1]
      self.detector = []

  def Tracker(self):
    x, y = Circle(self.Rtracker, 0, 0)
    det = Subdetector(x=x, y=y, color=self.ColorTracker)
    self.detector.append(det)

  def ECAL(self):
    x, y = Circle(self.REcal, 0, 0)
    det = Subdetector(x=x, y=y, color=self.ColorECAL)
    self.detector.append(det)

  def HCAL(self):
    x, y = Circle(self.RHcal, 0, 0)
    det = Subdetector(x=x, y=y, color=self.ColorHCAL)
    self.detector.append(det)

  def Muons(self):
    a0 = 2*pi/24
    ang = np.array([a0+(2*pi)/12*i for i in range(12)])
    x = self.Rmuon*cos(ang)
    y = self.Rmuon*sin(ang)
    vertices = np.transpose(np.array([x, y]))
    pol = Polygon(vertices, color=self.ColorMuon, closed=True)
    pol.set(fill=False)
    det = Subdetector(poly=pol)
    self.detector.append(det)
    
  def Draw(self, ax):
    for d in self.detector:
      d.Draw(ax)
