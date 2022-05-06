#%% Importing packages
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
#from scipy.interpolate import UnivariateSpline
import os
import h5py

#%% Reading in hdf5 file
def readhdf5file(filename):
    # Change the current working Directory    
    os.chdir("/home/gebruiker/Km3Net/new_dataset")
    #print("Directory changed")
    f = h5py.File(filename, 'r')
    #print(list(f.keys()))
    dset = f['data']
    my_data = dset[:]
    return my_data

#%% Reading in mat file
def readmatfile(filename):
    try:
        # Change the current working Directory    
        os.chdir("/home/gebruiker/Km3Net")
        print("Directory changed")
    except OSError:
        print("Can't change the Current Working Directory")  
    
    import scipy.io
    mat = scipy.io.loadmat(filename)
    m = mat.get('DataMat')    
    return m

#m = readmatfile('Time_FL_738_akoestisch-geisoleerd-02-gainoffset.mat')
#t = m[:,0] 
#theta_e = m[:,1]
#I1 = m[:,2]
#I2 = m[:,3]
#I3 = m[:,4]

#%% Reading in csv file
def readcsvfile(filename):
    try:
        # Change the current working Directory    
        os.chdir("C:/Users/ommensv/.spyder-py3")
        print("Directory changed")
    except OSError:
        print("Can't change the Current Working Directory")  
    
    from numpy import genfromtxt
    my_data = genfromtxt(filename, delimiter=',')    
    return my_data
    # Reading in data with: np.savetxt("data.csv", my_data, delimiter=",")
    
#%%
def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals

#%% Calibration - GainOffset
def GainOffset(m,t1,t2):
    f_s = 105469 
    #t1 = 6 
    #t2 = 8
    Lng1 = int(np.round(t1*f_s))
    Lng2 = int(np.round(t2*f_s-1))
    
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

    return I1_k,I2_k,I3_k,Phase_e


#%% Calibration - PhaseGainOffset
def PhaseGainOffset(m,t1,t2):
    f_s = 105469
    #t1 = 6 #4.65
    #t2 = 8 #4.8
    Lng1 = int(np.round(t1*f_s))
    Lng2 = int(np.round(t2*f_s-1))
    #obj = deltaphase
    
    MaxI = np.zeros(3)
    MinI = np.zeros(3)
    MeanI = np.zeros(3)
    AmpI = np.zeros(3)
    
    MaxI[0] = max(np.squeeze(m[Lng1:Lng2,0]))
    MinI[0] = min(np.squeeze(m[Lng1:Lng2,0]))
    MeanI[0] = (MaxI[0] + MinI[0])/2
    AmpI[0] = (MaxI[0] - MinI[0])/2
     
    MaxI[1] = max(np.squeeze(m[Lng1:Lng2,1]))
    MinI[1] = min(np.squeeze(m[Lng1:Lng2,1]))
    MeanI[1] = (MaxI[1] + MinI[1])/2
    AmpI[1] = (MaxI[1] - MinI[1])/2
    
    MaxI[2] = max(np.squeeze(m[Lng1:Lng2,2]))
    MinI[2] = min(np.squeeze(m[Lng1:Lng2,2]))
    MeanI[2] = (MaxI[2] + MinI[2])/2
    AmpI[2] = (MaxI[2] - MinI[2])/2
    
    #indx = [0,1,2] + 3*(1-1)
    Gain = 1/(AmpI/AmpI[1])
    NewMean = MeanI*Gain
    Offset = NewMean-NewMean[1]
    
    I1_k = np.squeeze(m[:,0])*Gain[0]-Offset[0]
    I2_k = np.squeeze(m[:,1])*Gain[1]-Offset[1]
    I3_k = np.squeeze(m[:,2])*Gain[2]-Offset[2]

    #VectorThetaEstTodd
    Eps = 1e-7
    # 9 parameters for calibration:
    A0 = MeanI[0] #Interrogator Mean I1
    B0 = AmpI[0] #Interrogator Amplification I1
    A1 = MeanI[1] #Interrogator Mean I2
    B1 = AmpI[1] #Interrogator Amplification I2
    A2 = MeanI[2] #Interrogator Mean I3
    B2 = AmpI[2] #Interrogator Amplification I3
    
    phi2,phi3 = deltaphase(I1,I2,I3,t1,t2)
    
    #phi1 = 0.0 #Interrogator Angle I1
    #phi2 = 2.143817129719248 #Interrogator Angle I2
    #phi2 = 2.142695197656572
    #phi3 = -2.099637795394208 #Interrogator Angle I3
    #phi3 = -2.0991257096480838
    
    A0 = normal_round(A0 + Eps,15)
    B0 = normal_round(B0 + Eps,15)
    A1 = normal_round(A1 + Eps,15)
    B1 = normal_round(B1 + Eps,15)
    A2 = normal_round(A2 + Eps,15)
    B2 = normal_round(B2 + Eps,15)
    
    Aa0 = A0
    Aa1 = A1
    Aa2 = A2
    Bb0 = B0/B0
    Bb1 = B1/B0
    Bb2 = B2/B0
    
    I1_k_n = m[:,0]/Aa0
    I2_k_n = m[:,1]
    I3_k_n = m[:,2]
    
    mu2 = (Bb1*np.cos(phi2))/Aa1
    mu3 = (Bb2*np.cos(phi3))/Aa2
    gm2 = (Bb1*np.sin(phi2))/Aa1
    gm3 = (Bb2*np.sin(phi3))/Aa2
    
    #print(Aa1*Aa2*(mu2-mu3))
    #print(Aa2*(mu3-Bb0/Aa0))
    
    #I1_n = Aa1*IAa2*(mu2-mu3)*I1_k_n
    #I2_n = Aa2*(mu2-Bb0/Aa0)*I2_k_n
    #I3_n = Aa1*(Bb0/Aa0-mu2)*I3_k_n
    
    num = (Aa1*Aa2*(mu2-mu3))*I1_k_n + (Aa2*(mu3-Bb0/Aa0))*I2_k_n + (Aa1*(Bb0/Aa0-mu2))*I3_k_n
    denom = (Aa1*Aa2*(gm2-gm3))*I1_k_n + Aa2*gm3*I2_k_n + Aa1*(-gm2)*I3_k_n

    Phase_ = np.arctan2(num,denom)
    Phase_e = np.unwrap(Phase_)

    return I1_k,I2_k,I3_k,Phase_e

#%% Calibration - PhaseGainOffset - find max value
def findMax(Isd,t1,t2):
    #f_s = 1e5
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
    Yint_max = max(f(xx))
    return Yint_max

#%% Calibration - PhaseGainOffset - find min value
def findMin(Isd,t1,t2):
    #f_s = 1e5
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
    Yint_min = min(f(xx))
    return Yint_min

#%% Calibration - PhaseGainOffset - deltaphase
def deltaphase(I1,I2,I3,t1,t2):
    f_s = 105469 #Hz
    #Ts = 1/fs
    Lng1 = int(np.round(t1*f_s))
    Lng2 = int(np.round(t2*f_s-1))
    Imax = np.array([max(I1[Lng1:Lng2]),max(I2[Lng1:Lng2]),max(I3[Lng1:Lng2])])
    Imin = np.array([min(I1[Lng1:Lng2]),min(I2[Lng1:Lng2]),min(I3[Lng1:Lng2])])
    amp = (Imax - Imin)/2
    mid = (Imax + Imin)/2
    
    Iz1 = (I1[Lng1:Lng2]-mid[0])/amp[0]
    Iz2 = (I2[Lng1:Lng2]-mid[1])/amp[1]
    Iz3 = (I3[Lng1:Lng2]-mid[2])/amp[2]
    
    Is12 = 0.5*(Iz1+Iz2)
    Id12 = 0.5*(Iz1-Iz2)
    
    Yint_max = findMax(Is12,t1,t2)
    Imaxsd12_1 = Yint_max
    Yint_min = findMin(Is12,t1,t2)
    Iminsd12_1 = Yint_min
    Yint_max = findMax(Id12,t1,t2)
    Imaxsd12_2 = Yint_max
    Yint_min = findMin(Id12,t1,t2)
    Iminsd12_2 = Yint_min
    
    ampsd12 = (Imaxsd12_1 - Iminsd12_1)/2
    phi12 = 2*math.acos(ampsd12)
    
    Is13 = 0.5*(Iz1+Iz3)
    Id13 = 0.5*(Iz1-Iz3)
    
    Yint_max = findMax(Is13,t1,t2)
    Imaxsd13_1 = Yint_max
    Yint_min = findMin(Is13,t1,t2)
    Iminsd13_1 = Yint_min
    Yint_max = findMax(Id13,t1,t2)
    Imaxsd13_2 = Yint_max
    Yint_min = findMin(Id13,t1,t2)
    Iminsd13_2 = Yint_min
        
    ampsd13 = (Imaxsd13_1 - Iminsd13_1)/2 #1.305297436365872
    #phi13 = 2.137346947927385
    phi13 = -2*math.acos(ampsd13)
    
    #Imaxsd12 = np.array([Imaxsd12_1, Imaxsd12_2])
    #Iminsd12 = np.array([Iminsd12_1, Iminsd12_2])
    #ampsd12 = (Imaxsd12 - Iminsd12)/2
    #phi2 = 2*math.acos(ampsd12[0])
    
    return phi12,phi13

#%% Plotting calibrated IF signals and phase
def Plot_I_and_Phase(t,I1,I2,I3,Phase):
    plt.figure(1)
    plt.clf()
    plt.plot(t,I1,label="I_0",color = 'darkblue',linewidth=0.2, linestyle='-')
    plt.plot(t,I2,label="I_1",color = 'red',linewidth=0.2, linestyle='-')
    plt.plot(t,I3,label="I_2",color = 'gold',linewidth=0.2, linestyle='-')
    plt.title('Amplitude of three Python calibrated signals in time')
    plt.xlabel('time [s]')
    plt.ylabel('I [V]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.figure(2)
    plt.clf()
    plt.plot(t,Phase,label="theta_e",color = 'red',linewidth=0.5, linestyle='-')
    plt.title('Phase of Python calibrated signals in time')
    plt.xlabel('time [s]')
    plt.ylabel('Phase [rad]') 
    
#%% Error in calibrated phase in time
def Plot_error_of_I_and_Phase(t,I1_c,I1_k,I2_c,I2_k,I3_c,I3_k,Phase_e,theta_e):
    Error_I1 = I1_c - I1_k
    Error_I2 = I2_c - I2_k
    Error_I3 = I3_c - I3_k
    plt.figure(3)
    plt.clf()
    plt.plot(t,Error_I1,label="Error_I1",color = 'darkblue',linewidth=0.2, linestyle='-')
    plt.plot(t,Error_I2,label="Error_I2",color = 'red',linewidth=0.2, linestyle='-')
    plt.plot(t,Error_I3,label="Error_I3",color = 'gold',linewidth=0.2, linestyle='-')
    plt.title('Error of three intensity signals in time')
    plt.xlabel('time [s]')
    plt.ylabel('Amplitude [V]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    Error_Phase_e = abs(Phase_e) - abs(theta_e)
    plt.figure(4)
    plt.clf()
    plt.plot(t,Error_Phase_e,color = 'darkblue',linewidth=0.2, linestyle='-')
    plt.title('Error of calibrated phase in time')
    plt.xlabel('time [s]')
    plt.ylabel('Phase [rad]')

#%% Calculating with the functions

# Reading in csv file with time, calibrated/uncalibrated signals and phase
#my_data = readcsvfile('data_76_with_precision18.csv') 

#Extracting data from csv file
#m = my_data[:,0:3] #raw 3 intensities
#I1 = my_data[:,0] #I0
#I2 = my_data[:,1] #I1
#I3 = my_data[:,2] #I2
#theta_ = my_data[:,3] #calibrated theta wrapped
#t = my_data[:,4] #time
#I1_c = my_data[:,5] #I0 calibrated
#I2_c = my_data[:,6] #I1 calibrated
#I3_c = my_data[:,7] #I2 calibrated
#theta_e = my_data[:,8] #calibrated theta unwrapped

#%%
#Calibrating raw IF signals with PhaseGainOffset
#t1 = 6 #sec
#t2 = 8 #sec
#calibrated_data = PhaseGainOffset(m,t1,t2) 
#I1_k = calibrated_data[0]
#I2_k = calibrated_data[1]
#I3_k = calibrated_data[2]
#Phase_e = calibrated_data[3]


#%%

#Plotting calibrated IF signals and phase
#Plot_I_and_Phase(t,I1_k,I2_k,I3_k,Phase_e) #Python calibrated IF signals and phase
#Plot_I_and_Phase(t,I1_c,I2_c,I3_c,theta_e) #Matlab calibrated IF signals and phase

#Plotting error of calibrated phase in time (relative from data from Matlab)
#Plot_error_of_I_and_Phase(t,I1_c,I1_k,I2_c,I2_k,I3_c,I3_k,Phase_e,theta_e)


#%% Reading in hdf5 file with raw (uncalibrated) signals and time
#filename = 'test-1sec.hdf5'
#filename = 'dataset131408.hdf5'
filename = 'Sine_10kHz_1keerwegschrijven_met_ref.hdf5'
my_data = readhdf5file(filename)
#Sensitivity = 2.14748365e+09
#km.Sens = array([2.14748365e+09, 2.14748365e+09, 2.14748365e+09, 2.14748365e+09])
Sensitivity = (256*2**23)/(2.5)
my_data = my_data/Sensitivity

#Extracting data from hdf5 file
m = my_data[:,0:4] #raw 3 intensities
I1 = my_data[:,0] #I0
I2 = my_data[:,1] #I1
I3 = my_data[:,2] #I2
Iref = my_data[:,3] # Reference signal
Fs = 105469
t = np.linspace(0,(len(I1)/Fs),len(I1))


#%%
num = math.sqrt(3)*(np.squeeze(I2)-np.squeeze(I3))
denom = 2*np.squeeze(I1)-(np.squeeze(I2)+np.squeeze(I3))
Phase_ = np.arctan2(num,denom)
Phase_uncal = np.unwrap(Phase_)

#%%

Plot_I_and_Phase(t,I1,I2,I3,-Phase_uncal) #Python calibrated IF signals and phase

#%%Calibrating raw IF signals with PhaseGainOffset

t1 = 1.0 #sec
t2 = 2.0 #sec
calibrated_data = PhaseGainOffset(m,t1,t2) 
I1_k = calibrated_data[0]
I2_k = calibrated_data[1]
I3_k = calibrated_data[2]
Phase_e = calibrated_data[3]

#%% Plotting calibrated IF signals and phase
Plot_I_and_Phase(t,I1,I2,I3,Phase_e) #Python calibrated IF signals and phase

#%%
#fig, axs = plt.subplots(4)
#fig.suptitle('lesgo')
#axs[0].plot(t,I1_k)
#axs[1].plot(t,I2_k)
#axs[2].plot(t,I3_k)
#axs[3].plot(t,Phase_e)


