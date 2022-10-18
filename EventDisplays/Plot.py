import numpy as np
from numpy import cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.patches import Polygon
from Objects import *
from Detector import Detector

Rtracker = 1
ECALmax = 0.7
HCALmax = 0.6

O = Object(Rtracker, ECALmax, HCALmax)

def GetEventDisplay(command='', outname='event.png', nPU=None):
  if not nPU is None: nPU = O.nPU
  O.Empty()
  O.PU(nPU)

  if command == '':
    ''' Send random collision '''
    processes = ['H', 'ttH', 'tt', 'W', 'Z', 'WW', 'WZ', 'ZZ', 'WWW', 'WWZ', 'ZZZ', 'tZq', 'tW', 'pp', 'ttf', 'Zf', 'Wf', 'tWf', 'tttt', 'ttbb']
    command = np.random.choice(processes)
  report = O.Interpret(command)

  D = Detector()
  D.Tracker()
  D.ECAL()
  #D.HCAL()
  D.Muons()


  ### Printing
  fsize = 7
  fig = plt.figure(figsize=(fsize,fsize))
  ax = fig.add_subplot(1, 1, 1)

  # Background
  bkg = Polygon([[-2.2, -2.2], [-2.2, 2.2], [2.2, 2.2], [2.2, -2.2]], color='black')
  ax.add_patch(bkg)

  #for p in towers: axdep.add_patch(p)

  D.Draw(ax)
  O.Draw(ax)
  #ax.plot(xmain, ymain, '-k')

  ax.set_xlim([-2.2, 2.2])
  ax.set_ylim([-2.2, 2.2])
  ax.set_axis_off()
  ax.margins(x=0.0, y=0.0)
  plt.tight_layout()
  if not outname.endswith('.png'): outname += '.png'
  fig.savefig(outname, facecolor='black')
  plt.close()
  return report
