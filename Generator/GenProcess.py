'''
  Event generator
'''

import numpy as np
import vector
import awkward as ak
import pickle

class Particle:
    def __init__(self, label=None, p4=None, mass=None, charge=None, energy=0, name=None, verbose=False, id=None):
        self.name = name
        self.label = label
        self.p4 = p4
        self.verbose = verbose
        self.id = id if id is not None else GetParticleId(self.label)
        if mass is None:
            self.mass = GetMassPartName(label)
        if charge is None:
            self.charge = GetChargePartName(label)
        if p4 is None:
            self.energy = energy
            self.p4 = vector.obj(pt=0., eta=0., phi=0., mass=self.mass)
        elif isinstance(p4, list) or isinstance(p4, tuple) and len(p4) == 4:
          # Assume pt, eta, phi, mass
          pt, eta, phi, mass = p4
          self.p4 = vector.obj(pt=pt, eta=eta, phi=phi, mass=mass)
        else:
            self.energy = p4.energy

    # Define the properties of the class
    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label
    
    @property
    def p4(self):
        return self._p4
    
    # Define the setters of the class
    @name.setter
    def name(self, name):
        self._name = name

    @label.setter
    def label(self, label):
        self._label = label

    @p4.setter
    def p4(self, p4):
        self._p4 = p4

    def __str__(self):
        return f"{self.label}) --> {self.p4}, id = {self.id}"

    def decay(self):
        decayList = GetDecays(self, self.charge)
        listOfDaughters = TwoBodyDecay(self, decayList, verbose=self.verbose)
        return listOfDaughters
    

def GetMassPartName(label):
    mass = 0.
    label = label.lower()
    if 't' in label:
        mass = 173.21
    elif 'w' in label:
        mass = 80.379
    elif 'z' in label:
        mass = 91.1876
    elif 'h' in label:
        mass = 125.09
    elif 'b' in label:
        mass = 4.18
    elif 'c' in label:
        mass = 1.275
    elif 'mu' in label or 'µ' in label:
        mass = 0.105658
    elif 'q' in label:
        mass = 0.05
    elif 'nu' in label or 'ν' in label:
        mass = 0.0
    elif 'gamma' in label or 'γ' in label or 'g' in label:
        mass = 0.0
    elif 'tau' in label or 'τ' in label:
        mass = 1.77686
    elif 'e' in label or 'e-' in label or 'e+' in label:
        mass = 0.000511

    return mass

def GetChargePartName(label):
    charge = 0
    label = label.lower()
    if '+' in label:
        charge = 1
    elif '-' in label:
        charge = -1
    elif 't' in label or 'b' in label or 'c' in label or 'mu' in label or 'µ' in label or 'w' in label or 'tau' in label or 'τ' in label:
        charge = 1
    elif 'z' in label or 'h' in label or 'gamma' in label or 'nu' in label or 'ν' in label or 'γ' in label:
        charge = 0
    return charge

def GetDecays(part, charge=None):
    # Get the decay products of a particle
    # part is a Particle object, return a list of particle objects
    if isinstance(part, Particle):
        label  = part.label
        charge = part.charge
    else:
        label = part
    posdecays = []
    if label.lower() in ['t', 'tbar', 't+', 't-']:
        if charge == 1:
            posdecays = [['W+', 'b']]
        else:
            posdecays = [['W-', 'b']]
    elif label.lower() in ['w', 'w+', 'w-']:
        if charge == 1:
            posdecays = [['mu+', 'nu'], ['e+', 'nu'], ['tau+', 'nu'], ['q', 'q']]
        else:
            posdecays = [['mu-', 'nu'], ['e-', 'nu'], ['tau-', 'nu'], ['q', 'q']]
    elif label.lower() == 'z':
        posdecays = [['mu+', 'mu-'], ['e+', 'e-'], ['tau+', 'tau-'], ['q', 'q'], ['nu', 'nu']]
    elif label.lower() == 'h':
        posdecays = [['Z', 'Z'], ['W+', 'W-'], ['gamma', 'gamma'], ['b', 'b'], ['tau+', 'tau-']]
    elif label.lower() in ['tau', 'tau+', 'tau-', 'τ', 'τ+', 'τ-']: #### Single neutrino in tau decays to avoid 3-body decays !!!
        if charge == 1:
            posdecays = [['q', 'q'], ['mu+', 'nu']]
        else:
            posdecays = [['q','q'], ['mu-', 'nu']]
    if len(posdecays) == 0:
        print ("ERROR: No decays found for particle", label)
    decays = posdecays[np.random.choice(range(len(posdecays)))]
    return NewParticles(CheckPartLabel(decays), verbose=part.verbose if isinstance(part, Particle) else False)

def CheckPartLabel(label):
    if isinstance(label, str):
        label = label.replace('ttbar', 't+,t-').replace('WW', 'W+W-').replace('tt', 't+,t-')
    if isinstance(label, str) and ',' in label:
        label = label.replace(' ', '').split(',')
    if isinstance(label, str):
        if "tau" in label: label = label.replace("tau", "τ")
        if "gamma" in label: label = label.replace("gamma", "γ")
        if "mu" in label: label = label.replace("mu", "µ")
        if "nu" in label: label = label.replace("nu", "ν")
        label = list(label)
    elif isinstance(label, list):
        for i, lab in enumerate(label):
            label[i] = lab.replace('ttbar', 't+,t-').replace('WW', 'W+W-').replace('tt', 't+,t-')
            if "tau" in lab: label[i] = lab.replace("tau", "τ")
            if "gamma" in lab: label[i] = lab.replace("gamma", "γ")
            if "mu" in lab: label[i] = lab.replace("mu", "µ")
            if "nu" in lab: label[i] = lab.replace("nu", "ν")
    return label

def GetParticleId(label):
  if isinstance(label, list):
    return [GetParticleId(x) for x in label]
  if isinstance(label, list) and len(label) == 1: 
    label = label[0]
  label = label.lower()
  if label.endswith('+') or label.endswith('-'): label = label[:-1]
  if   label == 'e': return 11
  elif label == 'µ': return 13
  elif label == 'τ': return 15
  elif label == 'γ': return 22
  elif label == 'ν': return 12
  elif label == 'W': return 23
  elif label == 'Z': return 24
  elif label == 'b': return 5
  elif label == 'c': return 4
  elif label == 't': return 6
  return 1
  

def NewParticles(label, verbose=False):
    label = CheckPartLabel(label)
    particles = []
    for lab in label:
        particles.append(Particle(lab, verbose=verbose))
    return particles

#def GenProcess(p, s=-1, s_sigma=None, verbose=False):
def GenProcess(p, s=1300, verbose=False):
    # p is a list of particles (names)
    # s is the EFFECTIVE center of mass energy -- by defaiut is the energy of the process
    # s_sigma is the width of the center of mass energy -- by default it is 10% of s
    # s = 1000 GeV, s_sigma = 300 GeV
    particles = NewParticles(p, verbose)

    # Generate the center of mass energy
    totMass = sum([p.mass for p in particles])
    q = totMass # Transferred energy
    s_sigma = np.sqrt(2*s/np.sqrt(q)) # This formula works well to add a boost that depends on s and q !!
    q += np.abs(np.random.exponential(s_sigma)) # That random distribution works well!!

    # Generate the center-of-mass energy compatible with the sum of masses
    e_left = q - totMass
    
    # Let's distribute the energy
    particles = np.random.permutation(particles) # First suffle the elements
    for i in range(len(particles)):
        if i == len(p) - 1:
          particles[i].energy = (particles[i].mass + e_left)
        else:
          extra_e = np.random.uniform(0, e_left)
          particles[i].energy = particles[i].mass + extra_e
          e_left -= extra_e

    # Now, generate the angles... let's assume isotropic
    npart = len(particles)
    phi = np.random.uniform(0, 2 * np.pi, npart)
    theta = np.arccos(np.random.uniform(-1, 1, npart))
    for i in range(npart):
        momentum = np.sqrt(particles[i].energy**2 - particles[i].mass**2)
        pt = momentum * np.sin(theta[i])
        eta = -np.log(np.tan(theta[i] / 2))
        particles[i].p4 = vector.obj(pt=pt, mass=particles[i].mass, phi=phi[i], eta=eta)

    return particles


# Using scikit-hep vectors: https://github.com/scikit-hep/vector
def TwoBodyDecay(parent, listOfDaughters=None, verbose=False):
    if listOfDaughters is None or len(listOfDaughters) == 0:
        return [parent]
    parentvec = parent.p4 if isinstance(parent, Particle) else parent

    isOffset = parentvec.mass < sum([d.mass for d in listOfDaughters])
    if isOffset: 
        # Change the mass of the second daughter to compensate for the offset
        listOfDaughters[1].mass = parentvec.mass - listOfDaughters[0].mass
        #print(f"Offset decay of parent {parent.label} to daughters: {[p.label for p in listOfDaughters]}")
        #parentvec.mass = sum([d.mass for d in listOfDaughters])+0.0001

    # Generate the daughter particles
    daughter1, daughter2 = [p.p4 for p in listOfDaughters]
    mass1, mass2 = [p.mass for p in listOfDaughters]
    # Generate the random angles, back to back in the parent rest frame
    theta1 = np.arccos(np.random.uniform(-1, 1))
    phi1 = np.random.uniform(0, 2*np.pi)
    theta2 = np.pi - theta1
    phi2 = phi1 + np.pi
    eta1 = -np.log(np.tan(theta1 / 2))
    eta2 = -np.log(np.tan(theta2 / 2))

    # Get the momenta energy of the decay producs in the rest of fram of the parent
    p1 = np.sqrt((parentvec.mass**2 - (mass1 + mass2)**2) * (parentvec.mass**2 - (mass1 - mass2)**2) / (4 * parentvec.mass**2))
    p2 = np.sqrt(parentvec.mass**2 - p1**2)
    # Boost the momenta to the lab frame
    daughter1 = vector.obj(pt=p1*np.sin(theta1), mass=mass1, phi=phi1, eta=eta1)
    daughter2 = vector.obj(pt=p2*np.sin(theta2), mass=mass2, phi=phi2, eta=eta2)

    # Boost the daughters to the parent rest frame
    daughter1 = daughter1.boost_p4(parentvec)
    daughter2 = daughter2.boost_p4(parentvec)
    pt1 = daughter1.pt; phi1 = daughter1.phi; mass1 = daughter1.mass
    pt2 = daughter2.pt; phi2 = daughter2.phi; mass2 = daughter2.mass
    theta1 = np.arccos(daughter1.z / daughter1.p)
    eta1 = -np.log(np.tan(theta1 / 2))
    theta2 = np.arccos(daughter2.z / daughter2.p)
    eta2 = -np.log(np.tan(theta2 / 2))
    daughter1 = vector.obj(pt=pt1, mass=mass1, phi=phi1, eta=eta1)
    daughter2 = vector.obj(pt=pt2, mass=mass2, phi=phi2, eta=eta2)

    # Return the daughter particles
    listOfDaughters[0].p4 = daughter1
    listOfDaughters[1].p4 = daughter2
    if verbose:
        print (f"Decay of [{parent.label:{2}}] [pt = {parentvec.pt:.2f}, eta = {parentvec.eta:.2f}, phi = {parentvec.phi:.2f}, mass = {parentvec.mass:.2f}]")
        print (f"Daughter [{listOfDaughters[0].label:{2}}] [pt = {daughter1.pt:.2f}, eta = {daughter1.eta:.2f}, phi = {daughter1.phi:.2f}, mass = {daughter1.mass:.2f}]")
        print (f"Daughter [{listOfDaughters[1].label:{2}}] [pt = {daughter2.pt:.2f}, eta = {daughter2.eta:.2f}, phi = {daughter2.phi:.2f}, mass = {daughter2.mass:.2f}]")
    return listOfDaughters

decayingParticles = ['t', 'τ', 'τ-', 'τ+', 'w', 'z', 'h', 'w+', 'w-', 't+', 't-']
def DecayListOfParticles(listOfParticles):
    global decayingParticles
    labels =  [p.label.lower() for p in listOfParticles]
    while IsThereDecayingPart(labels):
        for p in listOfParticles:
          if p.label.lower() in decayingParticles:
            dacays = p.decay()
            listOfParticles = np.append(listOfParticles, dacays)
            listOfParticles = np.delete(listOfParticles, np.where(listOfParticles == p))
        labels =  [p.label.lower() for p in listOfParticles] # Update the labels
    return listOfParticles

# Check in any of the elements in list1 is in list2
def IsThereDecayingPart(labels):
    global decayingParticles
    for l in decayingParticles:
        if l in labels:
            return True
    return False


class Event:
  def __init__(self, particles=None, process='', nPU=-1):
    self.nPU = nPU
    self.process = process
    self.cme = 13000

    if particles is not None:
      self.SetEventPart(particles)
    elif process != '':
      self.GenEvent()

  def SetEventPart(self, parts):
    ''' Get the event structure from a set of particles in an event '''
    self.InitStruct()
    self.particles = parts
    met = vector.obj(pt=0, eta=0, phi=0, mass=0)
    for p in parts:
      id = np.abs(p.id)
      if id == 22:
        self.Photon_pt.append(p.p4.pt)
        self.Photon_eta.append(p.p4.eta)
        self.Photon_phi.append(p.p4.phi)
        self.Photon_mass.append(p.p4.mass)
      elif id == 11:
        self.Electron_pt.append(p.p4.pt)
        self.Electron_eta.append(p.p4.eta)
        self.Electron_phi.append(p.p4.phi)
        self.Electron_mass.append(p.p4.mass)
        self.Electron_charge.append(p.charge)
      elif id == 13:
        self.Muon_pt.append(p.p4.pt)
        self.Muon_eta.append(p.p4.eta)
        self.Muon_phi.append(p.p4.phi)
        self.Muon_mass.append(p.p4.mass)
        self.Muon_charge.append(p.charge)
      elif id <= 5:
        self.Jet_pt.append(p.p4.pt)
        self.Jet_eta.append(p.p4.eta)
        self.Jet_phi.append(p.p4.phi)
        self.Jet_mass.append(p.p4.mass)
        self.Jet_partId.append(p.id)
      if id != 12 and id != 14 and id != 16:
        pp = p.p4.to_xyzt()
        met = met + pp
    met = -met
    self.MET_pt = met.pt
    self.MET_phi = met.phi
      
  def GenEvent(self, process=None):
    ''' Given a process, generate an event and store the event structure '''
    if process is None: process = self.process
    particles = GenProcess(process, self.cme, verbose=False)
    decays = DecayListOfParticles(particles)
    self.SetEventPart(decays)

  def InitStruct(self):
    ''' The recorded events consists of a set of muons, electrons, jets and photons, MET, number of PU particles '''
    self.nMuon = []
    self.nElectron = []
    self.nJet = []
    self.nPhoton = []
    self.Muon_pt = []
    self.Muon_eta = []
    self.Muon_phi = []
    self.Muon_mass = []
    self.Muon_charge = []
    self.Electron_pt = []
    self.Electron_eta = []
    self.Electron_phi = []
    self.Electron_mass = []
    self.Electron_charge = []
    self.Jet_pt = []
    self.Jet_eta = []
    self.Jet_phi = []
    self.Jet_mass = []
    self.Jet_partId = []
    self.Photon_pt = []
    self.Photon_eta = []
    self.Photon_phi = []
    self.Photon_mass = []
    self.MET_pt = None
    self.MET_phi = None

  def __str__(self):
    return f"Event: nLep = {len(self.Electron) + len(self.Muon)}, nJets = {len(self.Jet)}, nPhotons = {len(self.Photon)}, nPU = {self.nPU}"




class EventGenerator(Event):

    def __init__(self, process, outname=None):
      self.process = process
      self.outname = outname
      self.event = Event(process=process)
      self.InitStruct()
      self.InitArrays()
      self.filter = None

    def SetFilter(self, f):
      self.filter = f

    def InitArrays(self):
      self.nMuon = []
      self.nElectron = []
      self.nJet = []
      self.nPhoton = []
      self.MET_pt = [] #ak.Array([])
      self.MET_phi = [] #ak.Array([])

    def SetCounters(self, event):
      nMuon = len(self.event.Muon_pt)
      nElec = len(self.event.Electron_pt)
      nJet  = len(self.event.Jet_pt)
      nPhot = len(self.event.Photon_pt)
      self.nMuon_val = nMuon
      self.nElectron_val = nElec
      self.nJet_val = nJet
      self.nPhoton_val = nPhot
      MET_pt  = self.event.MET_pt
      MET_phi = self.event.MET_phi
      self.MET_pt_val = (MET_pt)
      self.MET_phi_val = (MET_phi)

    def add(self, event):
        self.Muon_pt.append(self.event.Muon_pt)
        self.Muon_eta.append(self.event.Muon_eta)
        self.Muon_phi.append(self.event.Muon_phi)
        self.Muon_mass.append(self.event.Muon_mass)
        self.Muon_charge.append(self.event.Muon_charge)
        self.Electron_pt.append(self.event.Electron_pt)
        self.Electron_eta.append(self.event.Electron_eta)
        self.Electron_phi.append(self.event.Electron_phi)
        self.Electron_mass.append(self.event.Electron_mass)
        self.Electron_charge.append(self.event.Electron_charge)
        self.Jet_pt.append(self.event.Jet_pt)
        self.Jet_eta.append(self.event.Jet_eta)
        self.Jet_phi.append(self.event.Jet_phi)
        self.Jet_mass.append(self.event.Jet_mass)
        self.Jet_partId.append(self.event.Jet_partId)
        self.Photon_pt.append(self.event.Photon_pt)
        self.Photon_eta.append(self.event.Photon_eta)
        self.Photon_phi.append(self.event.Photon_phi)
        self.Photon_mass.append(self.event.Photon_mass)

        self.MET_pt.append(self.MET_pt_val)
        self.MET_phi.append(self.MET_phi_val)
        self.nMuon.append(self.nMuon_val)
        self.nElectron.append(self.nElectron_val)
        self.nJet.append(self.nJet_val)
        self.nPhoton.append(self.nPhoton_val)

    def Generate(self, n):
      i = 0
      while i < n:
        if (i+1)%100 == 0: print(' || Progress --> [%i/%i]'%(i+1, n))
        self.event.GenEvent()
        self.SetCounters(self.event)
        nMuon = self.nMuon_val; nElec = self.nElectron_val; nJet = self.nJet_val; nPhot = self.nPhoton_val
        if self.filter is not None and (not eval(self.filter)):
          continue
        i += 1
        self.add(self.event)
      # Create a dict to store the arrays
      self.events = self.GetEventsDict()
      if self.outname is not None:
        self.SaveEvents(self.outname)

    def GetEventsDict(self):
        events = {}
        events['nMuon'] = self.nMuon
        events['nElectron'] = self.nElectron
        events['nJet'] = self.nJet
        events['nPhoton'] = self.nPhoton
        events['Muon_pt'] = self.Muon_pt
        events['Muon_eta'] = self.Muon_eta
        events['Muon_phi'] = self.Muon_phi
        events['Muon_mass'] = self.Muon_mass
        events['Muon_charge'] = self.Muon_charge
        events['Electron_pt'] = self.Electron_pt
        events['Electron_eta'] = self.Electron_eta
        events['Electron_phi'] = self.Electron_phi
        events['Electron_mass'] = self.Electron_mass
        events['Electron_charge'] = self.Electron_charge
        events['Jet_pt'] = self.Jet_pt
        events['Jet_eta'] = self.Jet_eta
        events['Jet_phi'] = self.Jet_phi
        events['Jet_mass'] = self.Jet_mass
        events['Jet_partId'] = self.Jet_partId
        events['Photon_pt'] = self.Photon_pt
        events['Photon_eta'] = self.Photon_eta
        events['Photon_phi'] = self.Photon_phi
        events['Photon_mass'] = self.Photon_mass
        events['MET_pt'] = self.MET_pt
        events['MET_phi'] = self.MET_phi
        return events

    def __str__(self):
      s = '\n'.join([
      (' -- %i events -- '),
      (' Muons    '+ str(self.nMuon)),
      (' Electron '+ str(self.nElectron)),
      (' Jets     '+ str(self.nJet)),
      (' Photons  '+ str(self.nPhoton)),
      (' MET pt   '+ str(self.MET_pt)),
      (' MET phi  '+ str(self.MET_phi))])
      return s

    def SaveEvents(self, outname):
      # Save into a pkl file
      with open(outname, 'wb') as f:
        pickle.dump(self.events, f)


from coffea.nanoevents.methods.vector import behavior
def GetEventStruct(eventsDict):
  Muon = ak.zip({"pt": eventsDict["Muon_pt"], "eta": eventsDict["Muon_eta"], "phi": eventsDict["Muon_phi"], "mass": eventsDict["Muon_mass"], "charge": eventsDict["Muon_charge"]}, with_name = "PtEtaPhiMLorentzVector" ,behavior=behavior)
  Electron = ak.zip({"pt": eventsDict["Electron_pt"], "eta": eventsDict["Electron_eta"], "phi": eventsDict["Electron_phi"], "mass": eventsDict["Electron_mass"], "charge": eventsDict["Electron_charge"]}, with_name = "PtEtaPhiMLorentzVector" ,behavior=behavior)
  Photon = ak.zip({"pt": eventsDict["Photon_pt"], "eta": eventsDict["Photon_eta"], "phi": eventsDict["Photon_phi"], "mass": eventsDict["Photon_mass"]}, with_name = "PtEtaPhiMLorentzVector" ,behavior=behavior)
  Jet = ak.zip({"pt": eventsDict["Jet_pt"], "eta": eventsDict["Jet_eta"], "phi": eventsDict["Jet_phi"], "mass": eventsDict["Jet_mass"], "partId": eventsDict["Jet_partId"]}, with_name = "PtEtaPhiMLorentzVector" ,behavior=behavior)
  MET = ak.zip({"pt": eventsDict["MET_pt"], "eta": np.zeros_like(eventsDict["MET_pt"], dtype=float), "phi": eventsDict["MET_phi"], "mass": np.zeros_like(eventsDict["MET_pt"], dtype=float)}, with_name = "PtEtaPhiMLorentzVector" ,behavior=behavior)
  events = {'Muon': Muon, 'Electron': Electron, 'Photon': Photon, 'Jet': Jet, 'MET': MET}
  return events

'''
# TODO Generate events with multiprocess

def GenerateN(process, N, filter=None):
  generator = EventGenerator(process, outname=None)
  generator.SetFilter(filter)
  generator.Generate(N)
  return generator.events

import multiprocessing as mp
def GenerateEvents(process, N, filter=None, outname=None, nSlots=1):
    if nSlots == 1:
        events = GenerateN(process, N, filter=filter)
    else:
        pool = mp.Pool(processes=nSlots)
        results = [pool.apply_async(GenerateN, args=(process, N//nSlots, filter)) for i in range(nSlots)]
        events = [p.get() for p in results]
        events = {k: np.concatenate([e[k] for e in events]) for k in events[0].keys()}
    if outname is not None:
        with open(outname, 'wb') as f:
          pickle.dump(events, f)
    return events

#GenerateEvents('ttW', 1000, filter='nMuon==1 and nElec==1 and nJet>=1', outname='ttW.pkl', nSlots=1)
'''

