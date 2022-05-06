from SignalProc import *
import matplotlib.pyplot as plt

Sig = SignalProc(6,102400,8192)
Sig.MakeSig(128,1)
Sig.MakeNoise(1.0,0)
#Sig.Decimate(4)
Sig.WindowType ='blackmanharris'
Sig.NFFT       = 1024. 
Sig.PSDOn      = 1
Sig.NFFT        = 2*Sig.Fs
Sig.Segment     = 100
Sig.Cummulative = 1
Sig.FcumBegin   = 2
Sig.FcumEnd     = Sig.Fs/2;
Sig.Name[0] = 'a$_x$';Sig.Unit[0]='m/$s^2$'
Sig.Name[1] = 'a$_y$';Sig.Unit[1]='m/$s^2$'
Sig.Spectrum([0,1])
plt.figure(1)
plt.clf()
Sig.plot([0,1])
plt.figure(2)
plt.clf()
Sig.plotSpectrum([0,1])
