"""

This module contains the Signal processing class for estimating
non-parametric models

"""
import math
from numpy.linalg import norm
from numpy.fft import fft
from numpy import cos
from numpy import pi
import numpy as np
from scipy import signal
import time

import matplotlib.pyplot as plt
class SignalProc(object):
    """
    
    A class for performing non-parametric model fitting

    .. note::   Ported to Python v3.x.x
                and therfore not more downwards compatible with Python v 2.x.x
          
    Parameters
    ----------
    Nchnl : int
        The number of signals.
    length : int
        The size of the FFT segment.
    Fs : float
        The sampling frequency of the acquired time series
    
    Methods
    -------
    MakeSig(self,F0):
        Generate a sin(2*pi*F0/Fs*time)
    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}
    Spectrum()
        Caluculate the Power Density Spectrum
        
    Returns
    -------
    An Instance of the class
    
    Examples
    --------    
    >>> ss = SignalProc(6,10240,2048)
    >>> ss.MakeSig(128)
    >>> ss.WindowType ='hanning'
    >>> ss.NFFT       = 1024 
    >>> ss.PSDOn      = 1
    >>> ss.Spectrum(12)
    >>> ss.plotSpectrum()    

    Copyright
    ---------  
    E.J.J. Doppenberg
    IPOS.vision
    12-05-2016
        
    """
     
    Counter = 0;
    def __init__(self,*args):
        self.Fs         = 1024
        self.length     = 102400 
        self.Nchnl      = 4 
        nargin          = len(args)
        if nargin >= 0:
            self.Nchnl  = args[0] 
        if nargin >= 1:
            self.length = args[1]  
        if nargin >= 2:
            self.Fs      = args[2]  
        self.refr       = 1 
        self.resp       = 0
        self.F0         = 63.
        self.norm       = 1.
        self.NFFT       = 4096
        self.Segment    = 100
        self.PSDOn      = 0
        self.OnePlot    = 0
        self.Cummulative= 0
 
        self.FcumBegin  = 0
        self.FcumEnd    = self.Fs/2
           
        self.WindowType  = 'hanning'
        self.window      = np.hanning(self.NFFT)

        self.CalData     = np.zeros((self.length,self.Nchnl))
        self.Data        = np.zeros((self.length,self.Nchnl))
        self.freq        = self.MakeFreqSpan()
        self.time        = self.MakeTimeSpan()
        self.PSD         = np.ones((int(self.NFFT/2),self.Nchnl), dtype=float)
        self.Sx          = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sy          = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sxy         = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Txy         = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sxx         = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Syy         = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Cxy         = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.gg          = np.ones((self.Nchnl), dtype=float)
        self.Gain        = np.ones((self.Nchnl), dtype=float)
        self.Sens        = np.ones((self.Nchnl), dtype=float)
        self.Unit        = ['V']
        self.Name        = ['P']        
        self.GraphLabel  = ['Exp']
        self.Title       = ['Graph']
        for nchl in range(1,self.Nchnl):
            self.Unit       += ['V']
            self.Name       += ['P']            
            self.GraphLabel += ['Exp']
            self.Title      += ['Graph']
 
             
        SignalProc.Counter +=1
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$       
    def __call__(self,*args):
 
        self.Fs         = 4096
        self.Nchnl      = 4
        self.length     = 102400
        nargin          = len(args)
        if nargin >= 0:
            self.Nchnl  = args[0] 
        if nargin >= 1:
            self.length = args[1]  
        if nargin >= 2:
            self.Fs      = args[2] 
 
        self.FcumBegin  = 0
        self.FcumEnd    = self.Fs/2
        self.WindowType = 'hanning'
        self.window     = self.MakeWindow()
        self.NFFT       = 4096
        self.Segment    = 100
        self.PSDOn      = 0
        self.OnePlot    = 0
        self.Cummulative= 0        
        self.CalData    = np.zeros(self.length,self.Nchnl)
        self.Data       = np.zeros(self.length,self.Nchnl)
        self.freq       = self.MakeFreqSpan()
        self.time       = self.MakeTimeSpan()
        self.PSD        = np.ones((self.NFFT/2,self.Nchnl), dtype=float)
        self.Sx         = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sy         = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sxy        = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Txy        = np.zeros((self.NFFT,self.Nchnl), dtype=complex)
        self.Sxx        = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Syy        = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Cxy        = np.zeros((self.NFFT,self.Nchnl), dtype=float)
        self.Gain        = np.ones((self.Nchnl), dtype=float)
        self.Sens        = np.ones((self.Nchnl), dtype=float)
        self.Unit        = ['V']
        self.Name        = ['P']
        self.GraphLabel  = ['Exp']
        self.Title       = ['Graph']
        for nchl in range(1,self.Nchnl):
            self.Unit       += ['V']
            self.Name       += ['P']
            self.GraphLabel += ['Exp']
            self.Title      += ['Graph']
 
        return
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$     
    def __repr__(self):
        Rspace = 30;
        str3  ="Fs:".rjust(Rspace)+" %d\n".ljust(0) %self.Fs
        str3 += "NFFT:".rjust(Rspace)+" %d\n".ljust(0) %self.NFFT
        str3 += "Segment:".rjust(Rspace)+" %d\n".ljust(0) %self.Segment
        str3 += "PSDOn:".rjust(Rspace)+" %d\n".ljust(0) %self.PSDOn
        str3 += "OnePlot:".rjust(Rspace)+" %d\n".ljust(0) %self.OnePlot
        str3 += "Cummulative:".rjust(Rspace)+" %d\n".ljust(0) %self.Cummulative
        str3 += "length:".rjust(Rspace)+" %d\n".ljust(0) %self.length
        str3 += "Nchnl:".rjust(Rspace)+" %d\n".ljust(0) %self.Nchnl
        str3 += "FCumBegin:".rjust(Rspace)+" %d\n".ljust(0) %self.FcumBegin
        str3 += "FCumEnd:".rjust(Rspace)+" %d\n".ljust(0) %self.FcumEnd
        str3 += "refr:".rjust(Rspace)+" %d\n".ljust(0) %self.refr
        str3 += "resp:".rjust(Rspace)+" %d\n".ljust(0) %self.resp
        str3 += "WindowType:".rjust(Rspace)+" '%s'\n".ljust(0) %self.WindowType
        str3 += "window:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.window)
        str3 += "CalData:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.CalData))
        str3 += "Data:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Data))
        str3 += "time:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.time)
        str3 += "freq:".rjust(Rspace)+" [%d]\n".ljust(0) % np.size(self.freq)
        str3 += "PSD:".rjust(Rspace)+" [%s]\n".ljust(0) %str(np.shape(self.PSD))
        str3 += "Sx:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sx))
        str3 += "Sy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sy))
        str3 += "Sxx:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sx))
        str3 += "Syy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sy)) 
        str3 += "Sxy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sx))
        str3 += "Txy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sy))
        str3 += "Cxy:".rjust(Rspace)+" [%s]\n".ljust(0) % str(np.shape(self.Sx))
        str3 += "GraphLabel:".rjust(Rspace)+" [%s]\n".ljust(0) % self.GraphLabel
        str3 += "Unit:".rjust(Rspace)+" [%s]\n".ljust(0) % self.Unit
        str3 += "Name:".rjust(Rspace)+" [%s]\n".ljust(0) % self.Name
        str3 += "Title:".rjust(Rspace)+" [%s]\n".ljust(0) % self.Title
        str3 += "Gain:".rjust(Rspace)+"[%s]\n".ljust(0) % str(self.Gain)
        str3 += "Sens:".rjust(Rspace)+"[%s]\n".ljust(0) % str(self.Sens)

        return str3;        
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    def Calibrate(self):
#      self.CalData = np.copy(self.Data) 
      self.gg           = 1./(self.Gain*self.Sens)
      self.CalData = np.multiply(self.gg,self.Data)
      return self.CalData       
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        
    def plot(self,*args):
        nargin     = len(args)
        rChnl = np.arange(0,self.Nchnl)
        if nargin > 0: rChnl = args[0]

        self.Calibrate()  
        self.length = np.size(self.CalData,0)
        self.MakeTimeSpan()
        hgcf = plt.gcf().number
        if self.OnePlot == 0:
          nPlot    = len(rChnl)
          d_ln1=np.arange(0,nPlot,dtype=object)
          self.d_ln1=np.arange(0,nPlot,dtype=object)
          plt.figure(num=hgcf,clear=True)
          d_fig, d_axs= plt.subplots(nPlot, 1, num=hgcf)
          n_chl = 0
          for Chnl in rChnl:
              d_ln1[n_chl] = d_axs[n_chl].plot(self.time,(self.CalData[:,Chnl]))
              stdev = (np.std(self.CalData[:,n_chl]))    
              str = "    $\sigma$ = %6.4e %s    " % (stdev,self.Unit[Chnl]);
              d_axs[n_chl].set_title(str)
              str = "%s -> [%s] " %(self.Name[Chnl],self.Unit[Chnl]);
              d_axs[n_chl].set_ylabel(str)
              d_axs[n_chl].autoscale(tight = True)
              d_axs[n_chl].grid()
              d_axs[n_chl].grid('on','minor')
              n_chl += 1
          d_axs[n_chl-1].set_xlabel('time ->[s]');
        else:
          d_ln1=np.arange(0,1,dtype=object)
          plt.figure(num=hgcf,clear=True)
          d_fig, d_axs= plt.subplots(1, 1, num=hgcf, clear =True)
          n_chl = 0
          for Chnl in rChnl:
              d_ln1[n_chl] = d_axs.plot(self.time,(self.CalData[:,Chnl]))
          stdev = (np.std(self.CalData[:,n_chl]))    
          str = "$\sigma$ = %6.4e %s" % (stdev,self.Unit[n_chl]);
          d_axs.set_title(str)
          str = "%s -> [%s] " %(self.Name[n_chl],self.Unit[n_chl]);
          d_axs.set_ylabel(str) ; plt.autoscale(tight = True)
          d_axs.grid()
          d_axs.grid('on','minor')
          d_axs.set_xlabel('time ->[s]');

        self.d_fig=d_fig
        self.d_ln1=d_ln1
        self.d_axs=d_axs
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  
    def determin_limits(self,jj):
        # sens        = (256*2**23)/2.5
        # sens = 1.0
        mn = (min(self.CalData[:,jj]))
        mx = max(self.CalData[:,jj])
        if mn == mx: mn -= 1e-3
        if mn < 0: mn *= 1.1 
        else: mn /= 1.1
        if mx < 0: mx /= 1.1 
        else: mx *= 1.1
        self.d_axs[jj].set_ylim(mn, mx) 
   #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
   
    def determin_PSD_limits(self,jj):
        # sens        = (256*2**23)/2.5
        # sens = 1.0
        dF       = float(self.Fs)/float(self.NFFT)
        idxB     = int(self.FcumBegin/dF)
        idxE     = int(self.FcumEnd/dF) 
        data = self.PSD[:,jj]
        if self.Cummulative == 1:
            data = self.PSD[idxB:idxE,jj].cumsum();
        mn = 10.*np.log10(min(self.PSD[:,jj]))
        mx = 10.*np.log10(max(data))
        if mn == mx: mn -= 1e-3
        if mn < 0: mn *= 1.1 
        else: mn /= 1.1
        if mx < 0: mx /= 1.1 
        else: mx *= 1.1
        self.f_axs[jj].set_ylim(mn, mx) 
   #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
   
    def update_plot_data(self):
        try:
            self.Calibrate()  
            for idx,ll in np.ndenumerate(self.d_ln1):
                stdev = (np.std(self.CalData[:,idx]))    
                str = "std = %6.4e %s" % (stdev,"V")
                # str = "std 12. V"
                self.d_axs[idx].set_title(str)
                ll[0].set_ydata(self.CalData[:,idx])
                ll[0].figure.canvas.draw_idle() 
                self.determin_limits(idx)
                ll[0].figure.canvas.flush_events()
                # print("subplot: ",idx)
        except:
            self.plot()
            time.sleep(2)
            print("err SignalProc.plot_data") 
   #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$                    
       
    def update_plot_spectrum(self):
        try:
            dF       = float(self.Fs)/float(self.NFFT)
            idxB     = int(self.FcumBegin/dF)
            idxE     = int(self.FcumEnd/dF)   
            nSubplot = 0 
            idx = 0
            line = 0
            while(line < len(self.f_ln1)):
                self.f_axs[nSubplot].set_visible(True)
                ll = self.f_ln1[line]
                ll[0].set_data(self.freq,10.*np.log10(abs(self.PSD[:,idx])))
                cummulative = self.PSD[idxB:idxE,idx].cumsum();
                std = 10*np.log10(cummulative[-1]); #According to Parseval
                if self.Cummulative == 1:
                    line += 1;ll = self.f_ln1[line]
                    ll[0].set_data(self.freq[idxB:idxE],10.*np.log10(abs(self.PSD[idxB:idxE,idx].cumsum())))
                    line += 1;ll = self.f_ln1[line]
                    ll[0].set_data(self.freq[idxE:idxB-1:-1],10.*np.log10(abs(self.PSD[idxE:idxB-1:-1,idx].cumsum())))   
                    line += 1
                str = "$\sigma$ = %6.4f dB re %s" % (std,self.Unit[idx]);
                self.f_axs[nSubplot].set_title(str)
                str = "%s -> [dB re %s] " %(self.Name[idx],self.Unit[idx]);
                if self.PSDOn == 1:str = "%s -> [dB re %s/$\sqrt{Hz}$]" % (self.Name[idx],self.Unit[idx]);
                self.f_axs[nSubplot].set_ylabel(str) ;
                self.determin_PSD_limits(idx)
                ll[0].figure.canvas.draw_idle() 
                ll[0].figure.canvas.flush_events()
                nSubplot += 1 
                idx += 1
                # print("subplot: ",idx)
        except:
         print("err; SignalProc.update_plot_spectrum") 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$                    
          
    def MakeSig(self,F0,*args):
        self.MakeTimeSpan()
        self.F0           = F0
        nargin     = len(args)
        vecChnl = np.arange(0,self.Nchnl)
        if nargin > 1: vecChnl = args[0]
        for nChnl in vecChnl:
            self.Data[:,nChnl] = np.sin(2*pi*self.F0*self.time)
        
    def MakeNoise(self,std,*args):
        nargin     = len(args)
        vecChnl = np.arange(0,self.Nchnl)
        if nargin > 1: vecChnl = args[0]
        for nChnl in vecChnl:
            self.Data[:,nChnl]  = std*np.random.randn(self.length)      
        
    def Spectrum(self,*args):
        self.Calibrate()
        vecChnl = np.arange(0,self.Nchnl)
        nargin     = len(args)
        vecChnl = np.arange(0,self.Nchnl)
        if nargin > 0: vecChnl = args[0]
        if self.PSDOn == 1: self.NFFT = self.Fs      
        self.PSD        = np.zeros((int(self.NFFT/2),self.Nchnl), dtype=float)
        self.MakeWindow()
        self.MakeFreqSpan()
        self.length     = np.size(self.CalData,0)
        self.Segment    = int(self.length/self.NFFT)
        wnd             = self.window
        self.norm       = norm(wnd)**2
        double2single   = 2.0                
        for nChnl in vecChnl:
            for span in range(0,self.Segment):
                Bg                 = int(span*self.NFFT)
                end                = int(Bg+self.NFFT)
                # mn                 = np.mean(self.CalData[int(Bg):int(end),np.int(nChnl)])
                yw                 = wnd*(signal.detrend(self.CalData[int(Bg):int(end),np.int(nChnl)]))
                a                  = fft((yw),self.NFFT)
                ac                 = np.conj(a)
                pxx                = np.abs(a*ac)
                self.PSD[:,nChnl] +=  double2single*pxx[0:int(self.NFFT/2)]
#            self.Segment        = int(span/self.NFFT)+1   
            self.PSD[:,nChnl]  /= (float(self.Segment)*self.NFFT*self.norm)      
        return self.PSD
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        
    def TransferFunction(self, *args):
        self.Calibrate()
        nargin          = len(args)
        self.refr       = 0
        self.resp       = 1
        if nargin >= 1:
            self.refr = args[0]
        if nargin >= 2:
            self.resp = args[1]          
        if self.PSDOn == 1: self.NFFT = self.Fs      
        self.MakeWindow()
        self.MakeFreqSpan()
        
        self.length     = np.size(self.CalData,0)
        self.Segment    = round(float(self.length)/float(self.NFFT))
        wnd             = self.window
        self.norm       = norm(wnd)**2
        Resp            = self.CalData[:,self.resp]
        Ref             = self.CalData[:,self.refr]      
        nsamps          = self.NFFT
        start_index     = 0
        stop_index      = self.NFFT+start_index
        num_periods     = int(math.floor((len(Ref)-start_index)/self.NFFT))
        self.Sxx        = np.zeros((nsamps), dtype=float)
        self.Syy        = np.zeros((nsamps), dtype=float)
        self.Sxy        = np.zeros((nsamps), dtype=complex)
        
        for i in range(num_periods):
           # mnref        = np.mean(Ref[i*self.NFFT+start_index:i*self.NFFT+stop_index])
           # mnresp       = np.mean(Resp[i*self.NFFT+start_index:i*self.NFFT+stop_index])
           respCalData  = wnd*(signal.detrend(Resp[i*self.NFFT+start_index:i*self.NFFT+stop_index]))
           refCalData   = wnd*(signal.detrend(Ref[i*self.NFFT+start_index:i*self.NFFT+stop_index]))

           self.Sx      = (2./self.NFFT)*np.fft.fft(refCalData)
           self.Sy      = (2./self.NFFT)*np.fft.fft(respCalData)
           self.Syy    += np.absolute(self.Sy)**2
           self.Sxx    += np.absolute(self.Sx)**2
           self.Sxy    += self.Sy*np.conjugate(self.Sx)
        
        WinNorm     = num_periods*self.norm    #Normalizing scale factor for window type
    #average the CalData
        self.Sxx   /= WinNorm
        self.Syy   /= WinNorm
        self.Sxy   /= WinNorm
        self.Txy    = self.Sxy/(self.Sxx+1e-40)
        self.Cxy    = np.absolute((self.Sxy)**2/(self.Sxx*self.Syy))
        self.Cxy[0] = 0 # DC -component has no useful info
        return self.Txy, self.Cxy
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    def PlotBode(self):
        hgcf = plt.gcf().number
        nPlot    = 3
        b_ln1=np.arange(0,nPlot,dtype=object)
        plt.figure(num=hgcf,clear=True)
        b_fig, b_axs= plt.subplots(nPlot, 1, num=hgcf)
        
        Lrng     = int(len(self.Txy)/2)
        rng      = np.arange(Lrng,dtype=int)
        freqs    = np.fft.fftfreq(self.Txy.shape[0], d=(1./self.Fs))
        LowLim   = (float(self.Fs)/float(self.NFFT))
        UpperLim = max(freqs)

        b_ln1[0] = b_axs[0].semilogx(freqs[rng],20.*np.log10(np.absolute(self.Txy[rng])))
        b_axs[0].minorticks_on()
        b_axs[0].grid('on',which='both',axis='x')
        b_axs[0].grid('on',which='major',axis='y')
        b_axs[0].set_xlim(LowLim,UpperLim)
        b_axs[0].set_ylabel('Txy -> dB re []')
    
        # plt.subplot(312)
        b_ln1[1] = b_axs[1].semilogx(freqs[rng],180.*((np.angle(self.Txy[rng])))/np.pi)
        b_axs[1].grid('on',which='both',axis='x')
        b_axs[1].grid('on',which='major',axis='y')
        b_axs[1].set_xlim(LowLim,UpperLim)
        b_axs[1].set_ylabel('Phase ->  [grd]') 
        
        # plt.subplot(313)
        b_ln1[2] = b_axs[2].semilogx(freqs[rng],self.Cxy[rng])
        b_axs[2].grid('on',which='both',axis='x')
        b_axs[2].grid('on',which='major',axis='y')
        b_axs[2].set_xlim(LowLim,UpperLim)
        b_axs[2].set_ylabel('Coherence ->  []')
        b_axs[2].set_xlabel('Frequency -> [Hz]') 
        self.b_fig=b_fig
        self.b_ln1=b_ln1
        self.b_axs=b_axs
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$       
    def plotSpectrum(self,*args):
      nargin     = len(args)
      rChnl = np.arange(0,self.Nchnl)
      if nargin > 0: rChnl = args[0]
      hgcf = plt.gcf().number
      nPlot    = len(rChnl)
      nSubplot = 0
      dF       = float(self.Fs)/float(self.NFFT)
      idxB     = int(self.FcumBegin/dF)
      idxE     = int(self.FcumEnd/dF) 
      f_ln1=np.arange(0,nPlot,dtype=object)
      plt.figure(num=hgcf,clear=True)
      f_fig, f_axs = plt.subplots(nPlot, 1, num=hgcf)
      if self.OnePlot == 0:
          f_ln1 = []
          for nChnl in rChnl:
              cummulative = self.PSD[idxB:idxE,nChnl].cumsum();
              std = 10*np.log10(cummulative[-1]); #According to Parseval
              line = f_axs[nSubplot].plot(self.freq,10.*np.log10(abs(self.PSD[:,nChnl])))
              f_ln1.append(line)
              str = "$\sigma$ = %6.4f dB re %s" % (std,self.Unit[nChnl]);
              f_axs[nSubplot].set_title(str)
              str = "%s -> [dB re %s] " %(self.Name[nChnl],self.Unit[nChnl]);
              if self.PSDOn == 1:str = "%s -> [dB re %s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
              f_axs[nSubplot].set_ylabel(str) ;
              # plt.autoscale(tight = True)
              f_axs[nSubplot].grid()
              f_axs[nSubplot].grid('on','minor')

              if self.Cummulative == 1:
                  line = f_axs[nSubplot].plot(self.freq[idxB:idxE],10.*np.log10(abs(cummulative)))
                  f_ln1.append(line)
                  line = f_axs[nSubplot].plot(self.freq[idxE:idxB-1:-1],10.*np.log10(abs(self.PSD[idxE:idxB-1:-1,nChnl].cumsum())))     
                  f_ln1.append(line)
              nSubplot += 1
          f_axs[nSubplot-1].set_xlabel('Frequency ->[Hz]');   
      else:
          f_ln1=np.arange(0,1,dtype=object)
          plt.figure(num=hgcf,clear=True)
          f_fig, f_axs= plt.subplots(1, 1, num=hgcf)
          for nChnl in rChnl:
              f_ln1 = f_axs.plot(self.freq,np.sqrt(abs(self.PSD[:,rChnl])))
          nChnl  = rChnl[0]
          str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
          if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
          f_axs.set_ylabel(str) ;
          # f_axs.autoscale(tight = True)
          f_axs.grid()
          f_axs.grid('on','minor')
          f_axs.set_xlabel('Frequency ->[Hz]');
      self.f_fig=f_fig
      self.f_ln1=f_ln1
      # self.fc_ln1=fc_ln1
      self.f_axs=f_axs
 
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$       
    def plotSpectrumLinY(self,rChnl):
      nPlot    = len(rChnl)
      nSubplot = 1
      dF       = float(self.Fs)/float(self.NFFT)
      idxB     = int(self.FcumBegin/dF)
      idxE     = int(self.FcumEnd/dF) 
      if self.OnePlot == 0:             
          for nChnl in (rChnl):
              plt.subplot(nPlot,1,nSubplot)
              nSubplot += 1
              cummulative = self.PSD[idxB:idxE,nChnl].cumsum();
              std = np.sqrt(cummulative[-1]); #According to Parseval
              plt.plot(self.freq,np.sqrt(abs(self.PSD[:,nChnl])))
              if self.Cummulative == 1:
                  plt.plot(self.freq[idxB:idxE],np.sqrt(abs(cummulative)))
                  plt.plot(self.freq[idxE:idxB-1:-1],np.sqrt(abs(self.PSD[idxE:idxB-1:-1,nChnl].cumsum())))  
              str = "$\sigma$ = %6.4f %s" % (std,self.Unit[nChnl]);
              plt.title(str)
#        plt.title(r'$\alpha > \beta$')

              str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
              if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
              plt.ylabel(str) ; plt.autoscale(tight = True)
              plt.grid()
              plt.grid('on','minor')
              plt.show()
        
          plt.subplot(nPlot,1,nSubplot-1)
          plt.xlabel('Frequency ->[Hz]'); 
          plt.show()
      else:
          plt.plot(self.freq,np.sqrt(abs(self.PSD[:,rChnl])))
          nChnl  = rChnl[0]
          str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
          if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
          plt.ylabel(str) ; plt.autoscale(tight = True)
#          plt.legend(self.Name[0])
#          plt.legend(self.Name[7])
          plt.grid()
          plt.grid('on','minor')
          plt.xlabel('Frequency ->[Hz]'); 
          plt.show()
    
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$       
    def plotSpectrumSingleLinY(self,nChnl):
      nSubplot = 1
      dF       = float(self.Fs)/float(self.NFFT)
      idxB     = int(self.FcumBegin/dF)
      idxE     = int(self.FcumEnd/dF) 
      if self.OnePlot == 0:             

          nSubplot += 1
          cummulative = self.PSD[idxB:idxE,nChnl].cumsum();
          std = np.sqrt(cummulative[-1]); #According to Parseval
          plt.plot(self.freq,np.sqrt(abs(self.PSD[:,nChnl])))
          if self.Cummulative == 1:
              plt.plot(self.freq[idxB:idxE],np.sqrt(abs(cummulative)))
              plt.plot(self.freq[idxE:idxB-1:-1],np.sqrt(abs(self.PSD[idxE:idxB-1:-1,nChnl].cumsum())))  
          str = "$\sigma$ = %6.4f %s" % (std,self.Unit[nChnl]);
          plt.title(str)
#        plt.title(r'$\alpha > \beta$')

          str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
          if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
          plt.ylabel(str) ; plt.autoscale(tight = True)
          plt.grid()
          plt.grid('on','minor')
          plt.show()
#    
#          plt.subplot(nPlot,1,nSubplot-1)
          plt.xlabel('Frequency ->[Hz]'); 
          plt.show()
      else:
          plt.plot(self.freq,np.sqrt(abs(self.PSD[:,nChnl])))
#          nChnl  = nChnl[0]
          str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
          if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
          plt.ylabel(str) ; plt.autoscale(tight = True)
#          plt.legend(self.Name[0])
#          plt.legend(self.Name[7])
          plt.grid()
          plt.grid('on','minor')
          plt.xlabel('Frequency ->[Hz]'); 
          plt.show()
    
      
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    def plotSpectrumLogLog(self,rChnl):
      nPlot    = len(rChnl)
      nSubplot = 1
      dF       = float(self.Fs)/float(self.NFFT)
      idxB     = int(self.FcumBegin/dF)
      idxE     = int(self.FcumEnd/dF) 
      if self.OnePlot == 0:             
          for nChnl in rChnl:
              plt.subplot(nPlot,1,nSubplot)
              nSubplot += 1
              cummulative = self.PSD[idxB:idxE,nChnl].cumsum();
              std = np.sqrt(cummulative[-1]); #According to Parseval
              plt.loglog(self.freq,np.sqrt(abs(self.PSD[:,nChnl])))
              if self.Cummulative == 1:
                  plt.loglog(self.freq[idxB:idxE],np.sqrt(abs(cummulative)))
                  plt.loglog(self.freq[idxE:idxB-1:-1],np.sqrt(abs(self.PSD[idxE:idxB-1:-1,nChnl].cumsum())))  
              str = "$\sigma$ = %6.4f %s" % (std,self.Unit[nChnl]);
              plt.title(str)
#        plt.title(r'$\alpha > \beta$')

              str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
              if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
              plt.ylabel(str) ; plt.autoscale(tight = True)
              plt.grid()
              plt.grid('on','minor')
              plt.show()
        
          plt.subplot(nPlot,1,nSubplot-1)
          plt.xlabel('Frequency ->[Hz]'); 
          plt.show()
      else:
          plt.loglog(self.freq,np.sqrt(abs(self.PSD[:,rChnl])))
          nChnl  = rChnl[0]
          str = "%s -> [%s] " %(self.Name[nChnl],self.Unit[nChnl]);
          if self.PSDOn == 1:str = "%s ->  [%s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
          plt.ylabel(str) ; plt.autoscale(tight = True)
#          plt.legend(self.Name[0])
#          plt.legend(self.Name[7])
          plt.grid()
          plt.grid('on','minor')
          plt.xlabel('Frequency ->[Hz]'); 
          plt.show()
      
      
 #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$      
      
    def plotSpectrumLog(self,rChnl):
      nPlot    = len(rChnl)
      nSubplot = 1
      dF       = float(self.Fs)/float(self.NFFT)
      idxB     = int(self.FcumBegin/dF)
      idxE     = int(self.FcumEnd/dF)
     
      for nChnl in rChnl:
        plt.subplot(nPlot,1,nSubplot)
        nSubplot += 1
        
        cummulative = self.PSD[idxB:idxE,nChnl].cumsum();
        std = 10*np.log10(cummulative[-1]); #According to Parseval
        plt.semilogx(self.freq,10.*np.log10(abs(self.PSD[:,nChnl])))
        if self.Cummulative == 1:
            plt.semilogx(self.freq[idxB:idxE],10.*np.log10(abs(cummulative)))
            plt.semilogx(self.freq[idxE:idxB-1:-1],10.*np.log10(abs(self.PSD[idxE:idxB-1:-1,nChnl].cumsum())))
        str = "$\sigma$ = %6.4f dB re %s" % (std,self.Unit[nChnl]);
        plt.title(str)
        str = "%s -> [dB re %s] " %(self.Name[nChnl],self.Unit[nChnl]);
        if self.PSDOn == 1:str = "%s -> [dB re %s/$\sqrt{Hz}$]" % (self.Name[nChnl],self.Unit[nChnl]);
        plt.ylabel(str) ; plt.autoscale(tight = True)
        plt.grid()
        plt.grid('on','minor')
        plt.show()
      plt.subplot(nPlot,1,nSubplot-1)
      plt.xlabel('Frequency ->[Hz]'); 
      plt.show()
  #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
    def SetCumSumRange(self,Fbegin,Fend):
       if Fbegin > Fend:
           tmp    = Fbegin
           Fbegin = Fend
           Fend   = tmp
   
       if Fend > (self.NFFT/2):Fend = self.NFFT/2
       if Fbegin < 0:Fbegin = 0
       self.FcumBegin = Fbegin
       self.FcumEnd   = Fend
   #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 
   
    def MakeWindow(self):
        WnTp   = self.WindowType.lower()
        if WnTp == 'rect':
            self.window = np.ones(self.NFFT)
        elif WnTp == 'hanning':
            self.window   = np.hanning(self.NFFT)
        elif WnTp == 'hamming':
            self.window   = np.hamming(self.NFFT)                
        elif WnTp == 'blackmanharris':
            self.window   = self.SpecialBlackmanHarris7T()            
        else:
            self.window   = np.hanning(self.NFFT)    

        return self.window
        
    def SetWindowType(self,type):
        self.WindowType = type
        return self.WindowType
        
    def SpecialBlackmanHarris7T(self):
    # coeff. 4-term window 5-term window 6-term window 7-term window
    # A0 3.635819267707608e-001 3.232153788877343e-001 2.935578950102797e-001 2.712203605850388e-001
    # A1 4.891774371450171e-001 4.714921439576260e-001 4.519357723474506e-001 4.334446123274422e-001
    # A2 1.365995139786921e-001 1.755341299601972e-001 2.014164714263962e-001 2.180041228929303e-001
    # A3 1.064112210553003e-002 2.849699010614994e-002 4.792610922105837e-002 6.578534329560609e-002
    # A4 1.261357088292677e-003 5.026196426859393e-003 1.076186730534183e-002
    # A5 1.375555679558877e-004 7.700127105808265e-004
    # A6 1.368088305992921e-005

        A0 =2.712203605850388e-001
        A1 =4.334446123274422e-001
        A2 =2.180041228929303e-001
        A3 =6.578534329560609e-002
        A4 =1.076186730534183e-002
        A5 =7.700127105808265e-004
        A6 =1.368088305992921e-005
        N  = int(self.NFFT)
        n  = 2.*(np.linspace(0,N-1,N))/(N)
        self.window = A0-A1*cos(pi*n)+A2*cos(2*pi*n)-A3*cos(3*pi*n)+A4*cos(4*pi*n)-A5*cos(5*pi*n)+A6*cos(6*pi*n)
        return self.window    
        
            
    def MakeFreqSpan(self):
        self.freq     = np.linspace(0,int(self.NFFT/2)-1,int(self.NFFT/2))*float(self.Fs)/float(self.NFFT)
        return self.freq
        
    def MakeTimeSpan(self):
        self.time     = np.arange(0,self.length,1)/self.Fs
        return self.time
        
    def Decimate(self,Factor):
#        FNyq        = 0.5*self.Fs/Factor       #New Nyquist Freq
#        F0          = 0.95*FNyq               #New Bandpass Freq
        tmp         = signal.decimate(self.Data,Factor,n=None, ftype='fir', axis=0)
        self.Data   = tmp
        self.length = len(tmp)
        self.Fs    /= Factor 
        
    def ShiftLeftSample(self,rChnl,No):
        
        for nChnl in rChnl:
            self.Data[:,nChnl] = np.roll(self.Data[:,nChnl],No)
        