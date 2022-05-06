#!/usr/bin/env python3

import socket
import h5py
import os
from datetime import datetime
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
                                        km3net.DataRaw   = km3net.CumData1
                                        km3net.CumData1 = []
                                        km3net.CumData2 = []
                                        km3net.data     = []
                                        km3net.toggle = 1                                      
                                    else:
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
            begin_time = datetime.now()
            formatted_time = begin_time.strftime('%M:%S.%f')
            print(formatted_time)
            km3net.data2matrix()
            print("data2matrix")
            f['data'].resize((f['data'].shape[0] + km3net.Data.shape[0]), axis=0)
            f['data'][-km3net.Data.shape[0]:] = km3net.Data
            print(f['data'].shape)
            if f['data'].shape >= (km3net.length,4):
                f.close()
                f = h5py.File('dataset' + str(km.count) + '.hdf5', 'a')
                f.create_dataset('data',data=np.zeros(shape=(0,4)),compression=("gzip"),chunks=True,maxshape=(None,4))
                print("Succesfully created new dataset")
            else:
                pass
            plotting.release()
            if event.is_set():
                break
        curr_time = datetime.now()
        formatted_time_2 = curr_time.strftime('%M:%S.%f')
        print(formatted_time_2)
        
            
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
    km.ntraces = 2178
    #km.ntraces = 40
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
             
#%% Writing to a hdf5 file
def writehdf5file(km):
    try:
        n = 10
        f = h5py.File('test.hdf5', 'w')
        dataset = np.zeros((km.length*n,5))
        #while km.toggle = 1
        # append
        for i in range(n): 
            if km.toggle == 1:
                #dset = f.create_dataset('data', data = [km.Data,km.time])
                dataset[km.length*i:km.length*(i+1),0:4] = km.Data
                #dataset[km.length*i:km.length*(i+1),4] = km.time
                time.sleep(km.length/km.Fs)
            else:
                time.sleep(km.length/km.Fs)
        dataset[:,4] = np.linspace(0,(km.length/km.Fs)*n,km.length*n)
        dset = f.create_dataset('data', data = dataset)
        f.close()
        print("Write to hdf5 file successful")
    except OSError:
        print("Can't write to hdf5 file")


    

