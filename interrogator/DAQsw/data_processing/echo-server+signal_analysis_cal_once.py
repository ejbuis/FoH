#!/usr/bin/env python3
import os
try:
    # Change the current working Directory    
    os.chdir("C:\\Users\\ommensv\\KM3NeT\\SignalProc")
    print("Directory changed")
except OSError:
    print("Can't change the Current Working Directory")

import socket
import h5py
import threading
from threading import Event
import logging
from time import time
# import ctypes, time, os, sys, platform, tempfile, re, urllib
from scipy.signal import butter, lfilter, freqz, detrend
from numpy import *
from SignalProc import *
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
plt.ion()

NOFCHANNELS  = 4
INTEGERNOBYTES = 4 
TRANSFERSIZE = (24000)
ARRAYSIZE    = int(TRANSFERSIZE/INTEGERNOBYTES)
FSAMPLE      = 105469
NTRACE       = 10
th = np.linspace(0, 2*np.pi, TRANSFERSIZE)


sem      = threading.Semaphore()
plotting = threading.Semaphore()
array    = np.sin(th)
HOST = "192.168.0.100"  # Standard loopback interface address (localhost)
# HOST = "127.0.0.1"
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

#%%
folder = "/home/gebruiker/Km3Net/new_dataset"  
os.chdir(folder)
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
print("Deleted files in folder")
f = h5py.File('dataset.hdf5', 'a')
new_data = np.zeros(shape=(0,4))
f.create_dataset('data',data=new_data,compression=("gzip"),chunks=True,maxshape=(None,4))
print("Succesfully created dataset")

#%%

class KM3Net(SignalProc):
    _instance = None # The singleton instance of the library.
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = SignalProc(NOFCHANNELS)
        return cls._instance    
    Counter = 0;
    def __init__(self,*args):
        super().__init__(NOFCHANNELS, TRANSFERSIZE, FSAMPLE)
        self.ntraces    = NTRACE;
        self.Ndata      = int(self.length)
        self.Nplot      = 1024
        # self.Fs         = FSAMPLE
        self.DataRaw       = ones(self.length,dtype=int32)
        # self.Data     = ones((self.length,self.Nchnl),dtype=float)
        self.HOST = "192.168.0.100"  # Standard loopback interface address (localhost)
        self.PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
        self.count = 0
        self.toggle = 0
        self.recv_cnt = 0
        self.max_cnt = 0
        self.min_cnt = 0
        self.deltatime = 0
        self.result = ones((self.length,self.Nchnl),dtype=int32)
        self.CumData1 = ones(self.length,dtype=int32)
        self.CumData2 = ones(self.length,dtype=int32)
        #self.f = h5py.File('dataset.hdf5', 'a')
        # self.Sig      = SignalProc()
 
        KM3Net.Counter +=1
    def __call__(self):
        self.ntraces    = NTRACE;
        self.Ndata      = int(self.length)
        self.Nplot      = 1024
        # self.Fs         = 1024.
        self.DataRaw       = ones(self.length,dtype=int32)
        # self.Data     = ones((self.length,self.Nchnl),dtype=float)
        self.HOST = "192.168.0.100"  
        self.PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
        self.count = 0
        self.toggle = 0
        self.recv_cnt = 0
        self.max_cnt = 0
        self.min_cnt = 0
        self.deltatime = 0
        self.result = ones((self.length,self.Nchnl),dtype=int32)
        self.CumData1 = ones(self.length,dtype=int32)
        self.CumData2 = ones(self.length,dtype=int32)
        #self.f = h5py.File('dataset.hdf5', 'a')
        # self.Sig      = SignalProc()
        return
     
    def __repr__(self):
        Rspace = 30;
        str3  = "Ndata:".rjust(Rspace)+" %d\n".ljust(0) %self.Ndata
        str3 += "Nplot:".rjust(Rspace)+" %d\n".ljust(0) %self.Nplot 
        str3 += "Nchnl:".rjust(Rspace)+" %d\n".ljust(0) %self.Nchnl 
        str3 += "ntraces:".rjust(Rspace)+" %d\n".ljust(0) %self.ntraces        
        str3 += "refr:".rjust(Rspace)+" %d\n".ljust(0) %self.refr
        str3 += "resp:".rjust(Rspace)+" %d\n".ljust(0) %self.resp
        str3 += "DataRaw:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.DataRaw)
        str3 += "CumData1:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.CumData1)
        str3 += "CumData2:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.CumData2)    
        str3 += "Host:".rjust(Rspace)+" %s\n".ljust(0) % self.HOST
        str3 += "Port:".rjust(Rspace)+ " %d\n".ljust(0) % self.PORT
        str3 += "count:".rjust(Rspace)+ " %d\n".ljust(0) % self.count
        str3 += "toggle:".rjust(Rspace)+ " %d\n".ljust(0) % self.toggle
        str3 += "recv_cnt:".rjust(Rspace)+ " %d\n".ljust(0) % self.recv_cnt
        str3 += "max_cnt:".rjust(Rspace)+ " %d\n".ljust(0) % self.max_cnt
        str3 += "min_cnt:".rjust(Rspace)+ " %d\n".ljust(0) % self.min_cnt        
        str3 += "deltatime:".rjust(Rspace)+ " %f\n".ljust(0) % self.deltatime
        str3 += "result:".rjust(Rspace)+ " [%dx%d]\n".ljust(0) % shape(self.result)
        str3 += super().__repr__()
        return str3;
    
    def data2matrix(self):
        self.Ndata  = size(self.DataRaw)
        self.length = int(self.Ndata/self.Nchnl)
        self.Nplot  = self.length
        self.Data = self.DataRaw.astype(float)
        self.Data = self.Data.reshape((self.length,self.Nchnl))
        self.vector = self.DataRaw.astype(float)
        self.DataRaw = np.zeros_like(self.CumData1)
        
    def butter_lowpass_filter(self):
        order = 5
        cutoff = self.Fs*0.8
        normal_cutoff = cutoff / self.Fs
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        self.Data[:,0] = lfilter(b, a, self.Data[:,0])        
        
    def plotSpect(self,*args):
        nargin          = len(args)
        if nargin >= 1:
            self.PSDOn = args[0] 
            dF = 1
        if nargin >= 2:
            dF = args[1]  
        if nargin >= 3:
            self.Fs      = args[2]  

        self.Name[0] = 'a$_x$';self.Unit[0]='m/$s^2$'
        self.Name[1] = 'a$_y$';self.Unit[1]='m/$s^2$'
         
        self.Name[0] = 'Chnl$_1$';self.Unit[0]='V'
        self.Name[1] = 'Chnl$_2$';self.Unit[1]='V'
        self.Name[2] = 'Chnl$_3$';self.Unit[2]='V'
        self.Name[3] = 'Chnl$_4$';self.Unit[3]='V'
        self.Spectrum([0,1,2,3])
        plt.figure(1)
        plt.clf()
        self.plot([0,1,2,3])
        plt.figure(2)
        plt.clf()
        self.plotSpectrumLog([0,1,2,3])

    def plot_raw(self):
        # self.data2matrix()
        self.MakeTimeSpan()
        figure(1)
        plot(self.time,self.Data)
        stdev = std(self.Data,axis=0) 
        str   = ([ "std={:0.2f}".format(x) for x in stdev ])
        title(str)
        xlabel('time ->[s]'); ylabel('S ->[V]') ; autoscale(tight = True)
        # show() 
        
    def plot_raw_raw(self):
        # self.data2matrix()
        self.MakeTimeSpan()
        figure(1)
        plot(self.vector)
    
        # show()

event = Event()

def isOpen(ip, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.settimeout(timeout)
        try:
                s.connect((ip, int(port)))
                s.shutdown(socket.SHUT_RDWR)
                return True
        except:
                return False
        finally:
                s.close()


def collect_socket_data(km3net):
    km3net.count = 0;
    km3net.initplot = 0;
    t = threading.currentThread()
    data = km3net.DataRaw;km3net.CumData1 = data;km3net.CumData2 = data
    while getattr(t, "do_run", True):
        while 1:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    
                    if isOpen(HOST, PORT) is False:
                        print("Binding socket")
                        s.bind((HOST, PORT))
                    s.listen()
                    conn, addr = s.accept()
                    with conn:
                        print("Connected by", addr)            
                        km3net.toggle  = 0
                        cnt            = 0
                        km3net.count   = 0
                        km3net.max_cnt = 0
                        km3net.min_cnt = 1e6
                        while getattr(t, "do_run", True):
                            try:                                                     
                                t1 = time.process_time() 
                                next_offset = 0
                                # buf = bytearray(toread)
                                # view = memoryview(buf)
                                # while toread:
                                #     nbytes = sock.recv_into(view, toread)
                                #     view = view[nbytes:] # slicing views is cheap
                                #     toread -= nbytes
                                while TRANSFERSIZE - next_offset > 0:
                                    recv_size = conn.recv_into(data, TRANSFERSIZE - next_offset,socket.MSG_WAITALL)
                                    next_offset += recv_size
                                view = memoryview(data)    
                                # km3net.recv_cnt = conn.recv_into(data,TRANSFERSIZE)
                                km3net.recv_cnt = next_offset
                                cnt += (time.process_time()-t1)
                                next_offset = km3net.recv_cnt 
                                if next_offset > km3net.max_cnt: km3net.max_cnt  = (next_offset)
                                if next_offset < km3net.min_cnt: km3net.min_cnt  = (next_offset)
                                if (km3net.count%km3net.ntraces == 0 and km3net.count >0): 
                                    if km3net.toggle == 0:
                                        km3net.CumData1 = np.append(km3net.CumData1,view[:ARRAYSIZE])
                                        km3net.DataRaw   = km3net.CumData1
                                        km3net.CumData1 = []
                                        km3net.CumData2 = []
                                        km3net.data     = []
                                        km3net.toggle = 1                                      
                                    else:
                                        km3net.CumData2 = np.append(km3net.CumData2,view[:ARRAYSIZE])
                                        km3net.DataRaw = km3net.CumData2                                      
                                        km3net.CumData2 = []
                                        km3net.CumData1 = []
                                        km3net.data     = []
                                        km3net.toggle = 0
                                    km3net.deltatime = cnt
                                    cnt = 0
                                    sem.release()                             
                                else:
                                    if km3net.toggle == 0:
                                        km3net.CumData1 = np.append(km3net.CumData1,view[:ARRAYSIZE])
                                    else:
                                        km3net.CumData2 = np.append(km3net.CumData2,view[:ARRAYSIZE])
                                km3net.count += 1
                            except KeyboardInterrupt:
                                s.close()
                                print("caught keyboard interrupt, exiting")
                            if event.is_set():
                                print("event interrupt, exiting")
                                s.close()
                                break
            except OSError:
                print("$$$$$$$$$$$$$$Exception received")
                s.close()

            
        
def convert_data(km3net,f):
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        while sem.acquire(blocking=True):
            #begin_time = datetime.now()
            #formatted_time = begin_time.strftime('%M:%S.%f')
            #print(formatted_time)
            km3net.data2matrix()
            print("data2matrix")
            f['data'].resize((f['data'].shape[0] + km3net.Data.shape[0]), axis=0)
            f['data'][-km3net.Data.shape[0]:] = km3net.Data
            print(f['data'].shape)
            if f['data'].shape >= ((2100000),4):
                f.close()
                f = h5py.File('dataset' + str(km.count) + '.hdf5', 'a')
                f.create_dataset('data',data=np.zeros(shape=(0,4)),compression=("gzip"),chunks=True,maxshape=(None,4))
                print("Succesfully created dataset" + str(km.count) + ".hdf5")
            else:
                pass
            plotting.release()
            if event.is_set():
                break
        #curr_time = datetime.now()
        #formatted_time_2 = curr_time.strftime('%M:%S.%f')
        #print(formatted_time_2)
        
            
def plot_data_thread(km):
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        while plotting.acquire(blocking=True):
            print("count=",km.count)
            print("Toggle",km.toggle)
            # kmkmkm.update_plot_data()
 
                       
def test(km,f):
    km.Sens=ones((km.Nchnl), dtype=float)*(256*2**23)/(2.5)# one shift in SAI protocol

    # km.Sens=ones((km.Nchnl), dtype=float)*(2**31)/1.0# Calibration for simulation debug mode
    km.Cummulative=1
    #km.ntraces = 2178
    #km.ntraces = 363
    km.ntraces = 40
    #km.ntraces = 2143
    km.plot()
    threads = list()
    event.clear()
    # km = KM3Net()
      
    x = threading.Thread(target=convert_data,args=(km,f))
    threads.append(x)
    x.start()
    x.do_run =True
    y = threading.Thread(target=collect_socket_data, args=(km,))
    threads.append(y)
    y.start()
    y.do_run =True
    z = threading.Thread(target=plot_data_thread, args=(km,))
    threads.append(z)
    z.start()
    z.do_run =True    
    # for index, thread in enumerate(threads):
    #     thread.join()
    #     x.do_run =True # dummy operation
    #     y.do_run =True # dummy operation
    #     count = 0
    # while 1:
    #     try:
    #         aa=2*3  
    #     except KeyboardInterrupt:
    #         event.set()
    #         print("KeyboardInterrupt")
    #         # y.do_run = False
    #         # x.do_run = False
    #         # z.do_run = False
    #         # time.sleep(1)
    #         break
    #         event.set()
    return x,y
 
            
    
def main():
    if "--test" in sys.argv:
        km = KM3Net()
        test(km)
        event.set()        
    
if __name__ == "main":
      main()
      
#%%
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import os
import h5py

#filename = 'Sine_10kHz_1keerwegschrijven_met_ref.hdf5'

#filename = 'dataset88937_sine_measurement_compare_DAQs.hdf5'
#filename = 'dataset23599-calibration.hdf5'
#filename = 'data_compare_DAQs_calibration_test_calibrated.csv'
#filename = 'data_compare_DAQs_sine_10kHz.csv'

#filename = 'dataset21782-Compare-DAQS-calibration-ntraces=363.hdf5'
#filename = 'dataset1841-Compare-DAQs-calibration-ntraces=40.hdf5'
#filename = 'dataset3641-Compare-DAQs-calibration-10.hdf5'
#filename = 'dataset922-Compare-DAQs-ntraces=40-52.7kHz.hdf5'
#filename = 'dataset2148-ntraces=2143-105kHz.hdf5'
#filename = 'dataset94300-calibration-Compare-DAQs-105kHz-ntaces=2143.hdf5'

#filename = 'dataset1842-Compare-DAQs-calibration-13(101).hdf5'
#filename = 'dataset1843-test-Compare-DAQs-sine-10kHz.hdf5'
#filename = 'dataset1842-calibration-sine-2V-10kHz-slow-cal.hdf5'



#%%

filename = 'dataset12322-Calibration-109-105469-ntraces-40.hdf5'
#filename = 'dataset1402-sine-0.5kHz.hdf5'
#filename = '.hdf5'


class SignalAnalysis(SignalProc):
    _instance = None # The singleton instance of the library.
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = SignalProc(NOFCHANNELS)
        return cls._instance
    #Counter = 0;
    def __init__(self):
        super().__init__(NOFCHANNELS, TRANSFERSIZE, FSAMPLE)
        self.filename = ""
        self.f_s = FSAMPLE
        self.Sensitivity = (256*2**23)/(2.5)
        self.Phase_uncal = []
        self.data_uncal = []
        self.data_cal = []
        self.Gain_ = []
        self.Offset = []
        self.A0 = 0
        self.B0 = 0
        self.A1 = 0
        self.B1 = 0
        self.A2 = 0
        self.B2 = 0
        self.phi1 = 0
        self.phi2 = 0
        self.phi3 = 0
        self.PSDOn = 1
        self.OnePlot = 1
 
        #KM3Net.Counter +=1
    def __call__(self):
        self.filename = ""
        self.f_s = FSAMPLE
        self.Sensitivity = (256*2**23)/(2.5)
        self.Phase_uncal = []
        self.data_uncal = []
        self.data_cal = []
        self.Gain_ = []
        self.Offset = []
        self.A0 = 0
        self.B0 = 0
        self.A1 = 0
        self.B1 = 0
        self.A2 = 0
        self.B2 = 0
        self.phi1 = 0
        self.phi2 = 0
        self.phi3 = 0
        self.PSDOn = 1
        self.OnePlot = 1
        return
    
    def __repr__(self):
        Rspace = 30;
        str3 = "filename:".rjust(Rspace)+" %s\n".ljust(0) % self.filename
        str3 += "fsample:".rjust(Rspace)+" %d\n".ljust(0) %self.f_s
        str3 += "Sensitivity:".rjust(Rspace)+" %d\n".ljust(0) %self.Sensitivity
        str3 += "Raw phase:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.Phase_uncal)
        str3 += "Uncalibrated data:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.data_uncal)
        str3 += "Calibrated data:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.data_cal)
        str3 += "Gain:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.Gain_)
        str3 += "Offset:".rjust(Rspace)+" [%d]\n".ljust(0) % size(self.Offset)
        str3 += "Mean I1:".rjust(Rspace)+" %d\n".ljust(0) %self.A0
        str3 += "Amplitifcation I1:".rjust(Rspace)+" %d\n".ljust(0) %self.B0
        str3 += "Mean I2:".rjust(Rspace)+" %d\n".ljust(0) %self.A1
        str3 += "Amplitifcation I2:".rjust(Rspace)+" %d\n".ljust(0) %self.B1
        str3 += "Mean I3:".rjust(Rspace)+" %d\n".ljust(0) %self.A2
        str3 += "Amplitifcation I3:".rjust(Rspace)+" %d\n".ljust(0) %self.B2
        str3 += "Angle I1:".rjust(Rspace)+" %d\n".ljust(0) %self.phi1
        str3 += "Angle I2:".rjust(Rspace)+" %d\n".ljust(0) %self.phi2
        str3 += "Angle I3:".rjust(Rspace)+" %d\n".ljust(0) %self.phi3
        str3 += "PSD on:".rjust(Rspace)+" %d\n".ljust(0) %self.PSDOn
        str3 += "One plot:".rjust(Rspace)+" %d\n".ljust(0) %self.OnePlot
        str3 += super().__repr__()
        return str3;
    
    def readhdf5file(self,filename):
        self.filename = filename
        try:
            os.chdir("C:\\Users\\ommensv\\Documents\\Km3Net\\The_measurements\\Data_hdf5_files")
            #os.chdir("/home/gebruiker/Km3Net/Compare_DAQS_first_test")
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")  
        f = h5py.File(filename, 'r')
        #print(list(f.keys()))
        dset = f['data']
        my_data = dset[:]
        #t1 = 6.9
        #t2 = 7.5
        #Lng1 = int(np.round(t1*self.f_s))
        #Lng2 = int(np.round(t2*self.f_s-1))
        #my_data = my_data[Lng1:Lng2,:]
        #my_data = my_data[self.Fs:len(my_data),:]
        self.data_uncal = my_data/(self.Sensitivity)
    
    def readmatfile(self,filename):
        self.filename = filename
        try:
            # Change the current working Directory    
            os.chdir("C:\\Users\\ommensv\\Documents\\Km3Net\\The_measurements")
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")  
        
        import scipy.io
        mat = scipy.io.loadmat(self.filename)
        m = mat.get('DataMat')    
        self.data_uncal = m
    
    #m = readmatfile('Time_FL_738_akoestisch-geisoleerd-02-gainoffset.mat')
    #t = m[:,0] 
    #theta_e = m[:,1]
    #I1 = m[:,2]
    #I2 = m[:,3]
    #I3 = m[:,4]
    
    def readcsvfile(self,filename):
        self.filename = filename
        try:
            # Change the current working Directory    
            os.chdir("C:\\Users\\ommensv\\Documents\\Km3Net\\The_measurements\\Data_csv_files")
            print("Directory changed")
        except OSError:
            print("Can't change the Current Working Directory")  
        
        from numpy import genfromtxt
        my_data = genfromtxt(self.filename, delimiter=',')    
        self.data_uncal = my_data
        # Reading in data with: np.savetxt("data.csv", my_data, delimiter=",")
        
    def normal_round(self,n, decimals=0):
        expoN = n * 10 ** decimals
        if abs(expoN) - abs(math.floor(expoN)) < 0.5:
            return math.floor(expoN) / 10 ** decimals
        return math.ceil(expoN) / 10 ** decimals
    
    def GainOffset(self,m,t1,t2):
        #f_s = 105469
        #t1 = 6 
        #t2 = 8
        Lng1 = int(np.round(t1*self.f_s))
        Lng2 = int(np.round(t2*self.f_s-1))
        
        MaxI = np.zeros(3)
        MinI = np.zeros(3)
        MeanI = np.zeros(3)
        AmpI = np.zeros(3)
        
        MaxI[0] = max(m[Lng1:Lng2,0])
        MinI[0] = min(m[Lng1:Lng2,0])
        MeanI[0] = (MaxI[0] + MinI[0])/2
        AmpI[0] = (MaxI[0] - MinI[0])/2
        
        MaxI[1] = max(m[Lng1:Lng2,1])
        MinI[1] = min(m[Lng1:Lng2,1])
        MeanI[1] = (MaxI[1] + MinI[1])/2
        AmpI[1] = (MaxI[1] - MinI[1])/2
        
        MaxI[2] = max(m[Lng1:Lng2,2])
        MinI[2] = min(m[Lng1:Lng2,2])
        MeanI[2] = (MaxI[2] + MinI[2])/2
        AmpI[2] = (MaxI[2] - MinI[2])/2
        
        Gain = 1/(AmpI/AmpI[1])
        NewMean = MeanI*Gain
        Offset = NewMean-NewMean[1]
        
        I1_k = m[:,0]*Gain[0]-Offset[0]
        I2_k = m[:,1]*Gain[1]-Offset[1]
        I3_k = m[:,2]*Gain[2]-Offset[2]
    
        num = math.sqrt(3)*(np.squeeze(I2_k)-np.squeeze(I3_k))
        denom = 2*np.squeeze(I1_k)-(np.squeeze(I2_k)+np.squeeze(I3_k))
        Phase_ = np.arctan2(num,denom)
        Phase_e = np.unwrap(Phase_)
        
        #self.Plot_I_and_Phase(I1_k,I2_k,I3_k,Phase_e)
        
        return I1_k,I2_k,I3_k,Phase_e
    
    def PhaseGainOffset(self,t1,t2):
        #f_s = 105469
        #t1 = 6 #4.65
        #t2 = 8 #4.8
        Lng1 = int(np.round(t1*self.f_s))
        Lng2 = int(np.round(t2*self.f_s-1))
        #obj = deltaphase
        
        MaxI = np.zeros(3)
        MinI = np.zeros(3)
        MeanI = np.zeros(3)
        AmpI = np.zeros(3)
        
        MaxI[0] = max(np.squeeze(self.data_uncal[Lng1:Lng2,0]))
        MinI[0] = min(np.squeeze(self.data_uncal[Lng1:Lng2,0]))
        MeanI[0] = (MaxI[0] + MinI[0])/2
        AmpI[0] = (MaxI[0] - MinI[0])/2
         
        MaxI[1] = max(np.squeeze(self.data_uncal[Lng1:Lng2,1]))
        MinI[1] = min(np.squeeze(self.data_uncal[Lng1:Lng2,1]))
        MeanI[1] = (MaxI[1] + MinI[1])/2
        AmpI[1] = (MaxI[1] - MinI[1])/2
        
        MaxI[2] = max(np.squeeze(self.data_uncal[Lng1:Lng2,2]))
        MinI[2] = min(np.squeeze(self.data_uncal[Lng1:Lng2,2]))
        MeanI[2] = (MaxI[2] + MinI[2])/2
        AmpI[2] = (MaxI[2] - MinI[2])/2
        
        #indx = [0,1,2] + 3*(1-1)
        self.Gain_ = 1/(AmpI/AmpI[1])
        NewMean = MeanI*self.Gain_
        self.Offset = NewMean-NewMean[1]
    
        #VectorThetaEstTodd
        # 9 parameters for calibration:
        self.A0 = MeanI[0] #Interrogator Mean I1
        self.B0 = AmpI[0] #Interrogator Amplification I1
        self.A1 = MeanI[1] #Interrogator Mean I2
        self.B1 = AmpI[1] #Interrogator Amplification I2
        self.A2 = MeanI[2] #Interrogator Mean I3
        self.B2 = AmpI[2] #Interrogator Amplification I3
        
        self.phi1 = 0.0 #Interrogator Angle I1
        #phi2 = 2.235210383459345
        #phi3 = -2.011549576063332
        self.phi2,self.phi3 = self.deltaphase(t1,t2)
        
    def Phase_and_I_PGO(self):
        I1_k = np.squeeze(self.data_uncal[:,0])*self.Gain_[0]-self.Offset[0]
        I2_k = np.squeeze(self.data_uncal[:,1])*self.Gain_[1]-self.Offset[1]
        I3_k = np.squeeze(self.data_uncal[:,2])*self.Gain_[2]-self.Offset[2]
        
        Eps = 1e-7
        A0_Eps = self.A0 + Eps
        B0_Eps= self.B0 + Eps
        A1_Eps = self.A1 + Eps
        B1_Eps = self.B1 + Eps
        A2_Eps = self.A2 + Eps
        B2_Eps = self.B2 + Eps
        
        #A0_Eps = self.normal_round(self.A0 + Eps,15)
        #B0_Eps = self.normal_round(self.B0 + Eps,15)
        #A1_Eps = self.normal_round(self.A1 + Eps,15)
        #B1_Eps = self.normal_round(self.B1 + Eps,15)
        #A2_Eps = self.normal_round(self.A2 + Eps,15)
        #B2_Eps = self.normal_round(self.B2 + Eps,15)
        
        Aa0 = A0_Eps
        Aa1 = A1_Eps
        Aa2 = A2_Eps
        Bb0 = B0_Eps/B0_Eps
        Bb1 = B1_Eps/B0_Eps
        Bb2 = B2_Eps/B0_Eps
        
        I1_k_n = self.data_uncal[:,0]/Aa0
        I2_k_n = self.data_uncal[:,1]
        I3_k_n = self.data_uncal[:,2]
        
        mu2 = (Bb1*np.cos(self.phi2))/Aa1
        mu3 = (Bb2*np.cos(self.phi3))/Aa2
        gm2 = (Bb1*np.sin(self.phi2))/Aa1
        gm3 = (Bb2*np.sin(self.phi3))/Aa2
        
        #print(Aa1*Aa2*(mu2-mu3))
        #print(Aa2*(mu3-Bb0/Aa0))
        
        #I1_n = Aa1*IAa2*(mu2-mu3)*I1_k_n
        #I2_n = Aa2*(mu2-Bb0/Aa0)*I2_k_n
        #I3_n = Aa1*(Bb0/Aa0-mu2)*I3_k_n
        
        num = (Aa1*Aa2*(mu2-mu3))*I1_k_n + (Aa2*(mu3-Bb0/Aa0))*I2_k_n + (Aa1*(Bb0/Aa0-mu2))*I3_k_n
        denom = (Aa1*Aa2*(gm2-gm3))*I1_k_n + Aa2*gm3*I2_k_n + Aa1*(-gm2)*I3_k_n
    
        Phase_ = np.arctan2(num,denom)
        Phase_e = np.unwrap(Phase_)
        
        data_cal = I1_k,I2_k,I3_k,Phase_e
        self.data_cal = np.transpose(data_cal)
    
        #return I1_k,I2_k,I3_k,Phase_e
    
    def findMax(self,Isd):
        #f_s = 105469
        y = Isd
        indx = np.argmax(y)
        #indx = np.sort(indx)
        ndev = 3
        if indx <= ndev:
            indx = ndev + 1
        Leny = len(y)
        if (indx + ndev) >= Leny:
            indx = Leny - ndev - 1
        delta = 0.05
        x = np.arange(-ndev,ndev+1,1) + indx
        xx = np.arange(-ndev,ndev+1,delta) + indx
        f = CubicSpline(x,y[x])
        #Yint_max = max(y)
        f_x = f(xx)
        peaks, _ = find_peaks(f_x, height=0)
        Yint_max = max(f_x[peaks])
        return Yint_max
    
    def findMin(self,Isd):
        #f_s = 105469
        y = Isd
        indx = np.argmin(y)
        #indx = np.sort(indx)
        ndev = 3
        if indx <= ndev:
            indx = ndev + 1
        Leny = len(y)
        if (indx + ndev) >= Leny:
            indx = Leny - ndev - 1
        delta = 0.05
        x = np.arange(-ndev,ndev+1,1) + indx
        xx = np.arange(-ndev,ndev+1,delta) + indx
        f = CubicSpline(x,y[x])
        f_x = f(xx)
        peaks, _ = find_peaks(-f_x, height=0)
        Yint_min = min(f_x[peaks])
        return Yint_min
    
    def deltaphase(self,t1,t2):
        #f_s = 105469 #Hz
        #Ts = 1/fs
        Lng1 = int(np.round(t1*self.f_s))
        Lng2 = int(np.round(t2*self.f_s-1))
        Imax = np.array([max(self.data_uncal[Lng1:Lng2,0]),max(self.data_uncal[Lng1:Lng2,1]),max(self.data_uncal[Lng1:Lng2,2])])
        Imin = np.array([min(self.data_uncal[Lng1:Lng2,0]),min(self.data_uncal[Lng1:Lng2,1]),min(self.data_uncal[Lng1:Lng2,2])])
        amp = (Imax - Imin)/2
        mid = (Imax + Imin)/2
        
        Iz1 = (self.data_uncal[Lng1:Lng2,0]-mid[0])/amp[0]
        Iz2 = (self.data_uncal[Lng1:Lng2,1]-mid[1])/amp[1]
        Iz3 = (self.data_uncal[Lng1:Lng2,2]-mid[2])/amp[2]
        
        Is12 = 0.5*(Iz1+Iz2)
        Id12 = 0.5*(Iz1-Iz2)
        
        Yint_max = self.findMax(Is12)
        Imaxsd12_1 = Yint_max
        Yint_min = self.findMin(Is12)
        Iminsd12_1 = Yint_min
        Yint_max = self.findMax(Id12)
        Imaxsd12_2 = Yint_max
        Yint_min = self.findMin(Id12)
        Iminsd12_2 = Yint_min
        
        ampsd12 = (Imaxsd12_1 - Iminsd12_1)/2
        phi12 = 2*math.acos(ampsd12)
        
        Is13 = 0.5*(Iz1+Iz3)
        Id13 = 0.5*(Iz1-Iz3)
        
        Yint_max = self.findMax(Is13)
        Imaxsd13_1 = Yint_max
        Yint_min = self.findMin(Is13)
        Iminsd13_1 = Yint_min
        Yint_max = self.findMax(Id13)
        Imaxsd13_2 = Yint_max
        Yint_min = self.findMin(Id13)
        Iminsd13_2 = Yint_min
            
        ampsd13 = (Imaxsd13_1 - Iminsd13_1)/2
        phi13 = -2*math.acos(ampsd13)
        #phi13 = (-115.253300989759/180)*math.pi
        
        #Imaxsd12 = np.array([Imaxsd12_1, Imaxsd12_2])
        #Iminsd12 = np.array([Iminsd12_1, Iminsd12_2])
        #ampsd12 = (Imaxsd12 - Iminsd12)/2
        #phi2 = 2*math.acos(ampsd12[0])
        
        return phi12,phi13
    
    def Plot_raw_data(self):
        t = np.linspace(0,(len(self.data_uncal)/self.f_s),len(self.data_uncal))
        plt.figure(1)
        plt.clf()
        plt.plot(t,self.data_uncal[:,0],label="I\u2080",color = 'darkblue',linewidth=0.2, linestyle='-')
        plt.plot(t,self.data_uncal[:,1],label="I\u2081",color = 'red',linewidth=0.2, linestyle='-')
        plt.plot(t,self.data_uncal[:,2],label="I\u2082",color = 'gold',linewidth=0.2, linestyle='-')
        #plt.title('Amplitude of three uncalibrated signals in time')
        plt.xlabel('Time [s]')
        plt.ylabel('Light intensity [V]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        num = math.sqrt(3)*(np.squeeze(self.data_uncal[:,1])-np.squeeze(self.data_uncal[:,2]))
        denom = 2*np.squeeze(self.data_uncal[:,0])-(np.squeeze(self.data_uncal[:,1])+np.squeeze(self.data_uncal[:,2]))
        Phase_ = np.arctan2(num,denom)
        self.Phase_uncal = np.unwrap(Phase_)
        plt.figure(2)
        plt.clf()
        plt.plot(t,self.Phase_uncal,color = 'red',linewidth=0.5, linestyle='-')
        #plt.plot(t,self.Phase_uncal,label="theta_e",color = 'red',linewidth=0.5, linestyle='-')
        #plt.title('Amplitude of reference signal in time (channel 4)')
        plt.xlabel('Time [s]')
        plt.ylabel('Phase [rad]')
        plt.grid()
    
    def Plot_calibrated_data(self):
        t = np.linspace(0,(len(self.data_cal)/self.f_s),len(self.data_cal))
        plt.figure(3)
        plt.clf()
        plt.plot(t,self.data_cal[:,0],label="I\u2080",color = 'darkblue',linewidth=0.2, linestyle='-')
        plt.plot(t,self.data_cal[:,1],label="I\u2081",color = 'red',linewidth=0.2, linestyle='-')
        plt.plot(t,self.data_cal[:,2],label="I\u2082",color = 'gold',linewidth=0.2, linestyle='-')
        #plt.title('Amplitude of three Python calibrated signals in time')
        plt.xlabel('Time [s]')
        plt.ylabel('Light intensity [V]')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.figure(4)
        plt.clf()
        plt.plot(t,self.data_cal[:,3],color = 'red',linewidth=0.5, linestyle='-')
        #plt.title('Phase of Python calibrated signals in time')
        plt.xlabel('Time [s]')
        plt.ylabel('Phase [rad]')
        plt.grid()

    def plotSpect(self,*args):
        nargin          = len(args)
        if nargin >= 1:
            self.PSDOn = args[0] 
            dF = 1
        if nargin >= 2:
            dF = args[1]  
        if nargin >= 3:
            self.Fs      = args[2]  

        self.Name[0] = 'a$_x$';self.Unit[0]='m/$s^2$'
        self.Name[1] = 'a$_y$';self.Unit[1]='m/$s^2$'
        
        self.Name[0] = 'I\u2080';self.Unit[0]='V'
        self.Name[1] = 'I\u2081';self.Unit[1]='V'
        self.Name[2] = 'I\u2082';self.Unit[2]='V'
        self.Name[3] = '\u03B8';self.Unit[3]='rad'
        self.Spectrum([0,1,2,3])
        plt.figure(1)
        plt.clf()
        self.plot([0,1,2,3])
        plt.figure(2)
        plt.clf()
        self.plotSpectrumLog([0,1,2,3])
        
    def plotSpect_Phase(self,*args):
        nargin          = len(args)
        if nargin >= 1:
            self.PSDOn = args[0] 
            dF = 1
        if nargin >= 2:
            dF = args[1]  
        if nargin >= 3:
            self.Fs      = args[2]  
        
        #self.Nchnl = 1
        self.Name[0] = 'a$_x$';self.Unit[0]='m/$s^2$'
        self.Name[1] = 'a$_y$';self.Unit[1]='m/$s^2$'
        
        self.Name[0] = 'I\u2080';self.Unit[0]='V'
        self.Name[1] = 'I\u2081';self.Unit[1]='V'
        self.Name[2] = 'I\u2082';self.Unit[2]='V'
        self.Name[3] = '\u03B8';self.Unit[3]='rad'
        self.Spectrum([0,1,2,3])
        plt.figure(1)
        plt.clf()
        self.plot([0,])
        plt.figure(2)
        plt.clf()
        self.plotSpectrumLog([0,])
            
#%%
sig.Data = np.zeros((2100000,4))
t = np.linspace(0,(len(sig.data_cal)/sig.f_s),len(sig.data_cal))
sig.Data[:,0] = sig.data_cal[:,3]
sig.Data[:,1] = sig.data_cal[:,3]
sig.Data[:,2] = sig.data_cal[:,3]
sig.Data[:,3] = sig.data_cal[:,3]
#%%
plt.plot(t,sig.Data[:,0])