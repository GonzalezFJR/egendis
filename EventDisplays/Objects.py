from Track import *
from Tower import *

class Object:
  ''' Object is a set of track + ECAL + HCAL with colors '''
  # Electron, muon, jet, photon, PU

  def __init__(self, Rtracker=1, ECALmax=0.7, HCALmax=0.6, HCALmin=None, ECALsize=0.02, ECALspread=0.02, HCALsize=0.025, HCALspread=0.02):
    self.Rtracker = Rtracker
    self.RtrackMax = 8*Rtracker
    self.PUcolor = [1, 0.45, 0]
    self.ElecColor = [0.2, 0.8, 1]#[0., 0.7, 0.15]
    self.MuonColor = [0.9, 0.1, 0.1]
    self.JetColor = [0.5, 0.5, 0.5]
    self.ECALmaxTowers = 6
    self.ECALmin = 0 
    self.ECALmax = ECALmax
    self.ECALangle_size = ECALsize
    self.ECALangle_spread = ECALspread
    self.ECALcolor = self.ElecColor#[0.2, 0.8, 1]
    self.HCALmaxTowers = 5
    self.HCALmin = HCALmin
    self.HCALmax = HCALmax
    self.HCALangle_size = HCALsize
    self.HCALangle_spread = HCALspread
    self.HCALcolor = self.PUcolor#[0.8, 0.4, 0.2]
    self.nPU = 50
    if self.HCALmin is None: self.HCALmin = ECALmax-0.1 # small overlap between ECAL and HCAL
    self.objects = []
    self.tags = []
    self.decays = []
    self.process = []
    self.status = ''

  def Empty(self):
    self.objects = []
    self.tags = []
    self.decays = []
    self.process = []
    self.status = ''

  def AddTag(self, tag):
    self.tags.append(tag)

  def AddDecay(self, dec):
    self.decays.append(dec)

  def AddPart(self, part):
    self.process.append(part)

  def GetReport(self):
    prstring = 'Production process: \n '
    for pr in self.process:
      prstring += pr
    decstring = 'Desintegraciones: '
    for d in self.decays:
     decstring += '\n ' + d
    fstring = 'Estado final:\n'
    for t in self.tags:
      fstring += t + ' '
    s = prstring + '\n'
    if len(self.decays) != 0: s += decstring + '\n'
    s += fstring
    return s

  def GetRfromPt(self, pt=None):
    if pt is None:
      R = np.random.uniform(1, self.RtrackMax)
    else:
      pt = pt/100 * self.RtrackMax
      R = np.random.normal(pt, 2)
      if R > self.RtrackMax: R = self.RtrackMax
      elif R < 1: R = 1.01
    return R

  def GetNtowersFromPt(self, pt=None, nTowersMax=6):
    if pt is None:
      nTowers = int(1 + np.random.normal(2, 1.5))
    else:
      pt = pt/100 * nTowersMax
      nTowers = int(1+np.random.normal(pt, 1.5))
    if nTowers<1: nTowers = 1
    if nTowers > nTowersMax: nTowers = nTowersMax
    return nTowers

  def AddTrack(self, R, angle=None, sign=None, color=[1,1,1], width=1, smear=0.02, Rmax=None):
    if Rmax is None: Rmax = self.Rtracker
    track = Track(R, Rmax, color, width, angle=angle, sign=sign, smear=smear)
    self.objects.append(track)
    return track.angle

  def AddEcalTower(self, x, y, nTowers=3, angle=None, dangle=None, color=None, offset=None, Dmax=None):
    if offset is None: offset = self.ECALmin
    if Dmax   is None: Dmax   = self.ECALmax
    if angle  is None: angle  = self.ECALangle_size
    if dangle is None: dangle = self.ECALangle_spread
    if color  is None: color  = self.ECALcolor
    ecal = Tower(x, y, Dmax, offset, angle, color, dangle)
    ecal.GetCluster(Dmax, dangle, nTowers)
    self.objects.append(ecal)

  def AddHcalTower(self, x, y, nTowers=3, angle=None, dangle=None, color=None, offset=None, Dmax=None):
    if offset is None: offset = self.HCALmin
    if Dmax   is None: Dmax   = self.HCALmax
    if angle  is None: angle  = self.HCALangle_size
    if dangle is None: dangle = self.HCALangle_spread
    if color  is None: color  = self.HCALcolor
    hcal = Tower(x, y, Dmax, offset, angle, color, dangle)
    hcal.GetCluster(Dmax, dangle, nTowers)
    self.objects.append(hcal)

  def Electron(self, pt=None, angle=None, sign=None, smear=None, deltaECAL=None):
    ''' Track + ECAL tower '''
    color = self.ElecColor; width = 2
    track = Track( self.GetRfromPt(pt), self.Rtracker, color, width, angle=angle, sign=sign, smear=smear)
    x0, y0 = track.GetTrackPosMax()
    self.objects.append(track)
    self.AddEcalTower(x0, y0, self.GetNtowersFromPt(pt, self.ECALmaxTowers))#, color=color)
    self.AddTag('e')

  def Muon(self, pt=None, angle=None, sign=None, smear=None):
    ''' Large track '''
    color = self.MuonColor
    width = 2.5
    self.AddTrack(self.GetRfromPt(pt), angle=angle, sign=None, color=color, width=width, Rmax=self.RtrackMax)
    self.AddTag('µ')

  def Jet(self, nTracks=3, pt=None, angle=None, smear=None, deltaECAL=None, deltaHCAL=None):
    ''' Several tracks + small ECAL tower + large HCAL tower '''
    color = self.JetColor
    if angle is None: angle = np.random.uniform(0, 360) # Fix angle first to get a correlation!!!
    if smear is None: smear = 0.05
    for i in range(nTracks):
      self.AddTrack( self.GetRfromPt(pt), angle=angle, color=color, width=1, smear=smear)
    x0, y0 = GetXYforAngle(angle/180*pi, self.Rtracker)
    self.AddEcalTower(x0, y0, self.GetNtowersFromPt(pt, self.ECALmaxTowers), dangle=self.ECALangle_spread*((1.2+pt/100) if pt is not None else 1.5) )
    self.AddHcalTower(x0, y0, self.GetNtowersFromPt(pt, self.HCALmaxTowers), dangle=self.HCALangle_spread*((1.2+pt/100) if pt is not None else 1.5) )
    self.AddTag('jet')

  def Photon(self, pt=None, angle=None, smear=None, deltaECAL=None):
    ''' No track + large ECAL tower '''
    track = Track( self.GetRfromPt(pt), self.Rtracker, angle=angle, smear=smear)
    x0, y0 = track.GetTrackPosMax()
    self.AddEcalTower(x0, y0, self.GetNtowersFromPt(pt, self.ECALmaxTowers), dangle=deltaECAL)
    self.AddTag('γ')

  def PU(self, ntracks=None, Rmin=0.6, Rmax=8, Rtracker=1, color=None, width=1):
    ''' Multiple tracks, small random deposits '''
    if color is None: color = self.PUcolor
    if ntracks is None: ntracks = self.nPU
    R = []
    while len(R) < ntracks:
      r = np.random.normal(1.5, 2)
      if (r > Rmin) and (r < Rmax):
        R.append(r)
    for r in R:
      angle = self.AddTrack(r, color=color, width=width)
      x0, y0 = GetXYforAngle(angle, self.Rtracker)
      # Add small 
      if np.random.uniform(0, 1) < 0.1:
        self.AddEcalTower(x0, y0, 1, Dmax=self.ECALmax/2)
        if np.random.uniform(0, 1) < 0.3:
          self.AddHcalTower(x0, y0, 1, Dmax=self.HCALmax/2)
      if np.random.uniform(0, 1) < 0.1:
        self.AddHcalTower(x0, y0, 1, Dmax=self.HCALmax/2)

  ################################################################################################################################################
  ################################################################################################################################################
  def Tau(self, angle=None, mode=''): # e, m, h
    if mode == 'e':
      self.AddDecay('Tau -> eν')
      self.Electron(angle=angle)
    elif mode == 'm':
      self.AddDecay('Tau -> µν')
      self.Muon(angle=angle)
    elif mode == 'h':
      self.AddDecay('Tau -> hadrones')
      self.Jet(angle=angle)
    else:
      self.Tau(mode=np.random.choice(['e', 'm', 'h']))
    
  def W(self, mode=''): # h, e, m, y
    mode = mode.lower()
    if mode == 'e':
      self.AddDecay('W -> eν')
      self.Electron()
    elif mode == 'm':
      self.AddDecay('W -> µν')
      self.Muon()
    elif mode == 'y':
      self.AddDecay('W -> τν')
      self.Tau()
    elif mode == 'h': # Jets back to back
      self.AddDecay('W -> hadrones (2 jets)')
      angle = np.random.uniform(0, 360)
      angle2 = np.random.normal(angle+pi, 20)
      while angle2 > (360): angle2 -= 360
      self.Jet(angle=angle)
      self.Jet(angle=angle2)
    else:
      self.W(mode=np.random.choice(['e', 'm', 'y', 'h']))

  def Z(self, mode=''): # vv, ee, mm, yy, qq
    mode = mode.lower()
    angle = np.random.uniform(0, 360)
    angle2 = np.random.normal(angle+180, 20)
    while angle2 > 360: angle2 -= 360
    if mode in ['e', 'ee']:
      self.AddDecay('Z -> ee')
      self.Electron(angle=angle)
      self.Electron(angle=angle2)
    elif mode in ['m', 'mm']:
      self.AddDecay('Z -> µµ')
      self.Muon(angle=angle)
      self.Muon(angle=angle2)
    elif mode in ['y', 'yy']:
      self.AddDecay('Z -> ττ')
      self.Tau(angle=angle)
      self.Tau(angle=angle2)
    elif mode in ['v', 'vv']:
      self.AddDecay('Z -> neutrinos')
    elif mode in ['q', 'h', 'hh', 'qq']:
      self.AddDecay('Z -> hadrones (2 jets)')
      self.Jet(angle=angle)
      self.Jet(angle=angle2)
    else:
      self.Z(mode=np.random.choice(['m', 'm', 'e', 'e', 'q', 'q', 'y', 'y', 'v']))
      
  def Top(self, mode=''): # WWbb
    mode = mode.lower()
    self.AddDecay('Quark top -> W+b')
    self.Jet(nTracks=4)
    self.W()

  def Higgs(self, mode=''): # ff, ZZ, WW, yy, bb
    mode = mode.lower()
    angle = np.random.uniform(0, 360)
    angle2 = np.random.normal(angle+180, 20)
    while angle2 > (360): angle2 -= 360
    if mode == 'ff':
      self.AddDecay('H -> γγ')
      self.Photon(angle=angle)
      self.Photon(angle=angle2)
    elif mode == 'ZZ':
      self.AddDecay('H -> ZZ')
      self.Z()
      self.Z()
    elif mode == '4m':
      self.AddDecay('H -> ZZ')
      self.Z(mode='mm')
      self.Z(mode='mm')
    elif mode == '4e':
      self.AddDecay('H -> ZZ')
      self.Z(mode='ee')
      self.Z(mode='ee')
    elif mode == '4l':
      self.AddDecay('H -> ZZ')
      self.Z(mode=np.random.choice(['ee', 'mm']))
      self.Z(mode=np.random.choice(['ee', 'mm']))
    elif mode == 'WW':
      self.AddDecay('H -> WW')
      self.W()
      self.W()
    elif mode == 'bb':
      self.AddDecay('H -> bb')
      self.Jet(angle=angle,  nTracks=4)
      self.Jet(angle=angle2, nTracks=4)
    elif mode == 'yy':
      self.AddDecay('H -> ττ')
      self.Tau(angle=angle)
      self.Tau(angle=angle2)
    else:
      self.Higgs(mode=np.random.choice(['ZZ', 'ZZ', 'ff', 'ff', '4l', '4e', '4m', 'WW', 'WW', 'bb', 'yy']))

  def Proton(self):
    self.Jet(nTracks=1)

  def Neutron(self):
    th = np.random.uniform(0, 360)
    self.AddHcalTower(self.Rtracker*cos(th), self.Rtracker*sin(th))

  def Interpret(self, obj):
    if isinstance(obj, str) and ',' in obj:
      obj = obj.replace(' ', '').split(',')
    elif isinstance(obj, str):
      obj = obj.replace(' ', '').replace(',', '')
      obj = [o for o in obj]
    part = ''
    for o in obj:
      part = o
      o = o.lower()
      if o in ['e', 'elec', 'electron']:
        self.Electron()
      elif o in ['µ', 'm', 'mu', 'muon']:
        self.Muon()
        path = 'µ'
      elif o in ['j', 'jet']:
        self.Jet()
        part = 'jet'
      elif o in ['f', 'foton', 'fot', 'photon', 'γ']:
        self.Photon()
        part = 'γ'
      elif o in ['u', 'd', 's', 'c', 'b','g', 'q', 'quark', 'gluon']:
        if o == 'g': part = 'gluon'
        if o == 'q': part = 'quark'
        self.Jet()
      elif o in ['t', 'top']:
        self.Top()
      elif o in ['ν', 'v', 'nu', 'neutrino']:
        part = 'ν'
        pass
      elif o in ['W', 'w', 'bosonw', 'boson w']:
        part = 'W'
        self.W()
      elif o in ['Z', 'z', 'bosonz', 'boson z']:
        part = 'Z'
        self.Z()
      elif o in ['H', 'h', 'higgs']:
        part = 'H'
        self.Higgs()
      elif o in ['p', 'proton']: # proton
        part = 'protón'
        self.Proton()
      elif o in ['n', 'neutron']: # neutron
        part = 'Neutrón'
        self.Neutron()
      elif o in ['τ', 'y', 'tau']: # tau
        part = 'τ'
        self.Tau()
      else:
        self.status = 'Comando no entendido: "%s"'%o
        print(self.status)
      if self.status == '':
        self.AddPart(part)
    return self.GetReport()
  
  def Draw(self, ax):
    ''' Draw everything on a given ax '''
    for o in self.objects:
      o.Draw(ax)
