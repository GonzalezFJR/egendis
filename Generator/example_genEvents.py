from GenProcess import EventGenerator

g = EventGenerator('ttW', outname='ttW.pkl')
g.SetFilter('nMuon>=0 and nElec>=1 and nJet>=1')
g.Generate(100)
