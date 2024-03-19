# -*- coding: utf-8 -*-
"""
filter module
This file is located here:
C:\\Users\aborst\.spyder2\blindschleiche.py
"""
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import math
from scipy import ndimage
from scipy.optimize import curve_fit
# Import Bessel function.
from scipy.special import jn
from mpl_toolkits.mplot3d import Axes3D
# Import colormaps.
from matplotlib import cm
# Import lighting object for shading surface plots.
from matplotlib.colors import LightSource

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import itertools

# --------------- GENERAl FUNCTIONS -----------------------------

# Lowpass and highpass functions work with n-dimensional inputs
# Always filters along the last axis
    
def lowpass(x,tau):
    
    # swaps input dimension such as last dim becomes first
    
    x=x.transpose(np.roll(np.arange(x.ndim),1))  
    n=x.shape[0]
    result=np.zeros_like(x)
    
    if tau<1:
        result=x
    if tau>=1:
        result[0]=x[0]
        for i in range(0,n-1):
            result[i+1]=1.0/tau*(x[i]-result[i])+result[i]
            
    # swaps output dimension such as first dimension becomes last again
            
    result=result.transpose(np.roll(np.arange(result.ndim),-1))
                
    return result
    
def highpass(x,tau): 
    result=x-lowpass(x,tau) 
    return result
    
def bandpass(x,tauhp,taulp):
    result=highpass(x,tauhp)
    result=lowpass(result,taulp)
    return result
    
def normalize(x):
    mymax=np.nanmax(x)
    mymin=np.nanmin(x)
    if np.abs(mymax)>np.abs(mymin):
        absmax=np.abs(mymax)
    else:
        absmax=np.abs(mymin)
    result=x/absmax
    if mymax==mymin:
        result=x*0.0
    return result
    
def equalize(image):
    image=image.astype(int)
    image=ceil(image,254)  
    hist,bins=np.histogram(image,bins=255,range=[0,254])
    hist=blurr(1.0*hist/np.sum(hist),5)
    cdf=integrate(hist)*255.0
    eqimage=cdf[image]
    return eqimage
    
def lowpass_amp_spectrum(tf,tau):
    w=2*np.pi*tf
    result=1.0/np.sqrt(1.0+(tau*w)**2)
    return result/np.max(result)
    
def highpass_amp_spectrum(tf,tau):
    w=2*np.pi*tf
    result=tau*w/np.sqrt(1.0+(tau*w)**2)
    return result/np.max(result)
    
def bandpass_amp_spectrum(tf,tauhp,taulp):
    result=lowpass_amp_spectrum(tf,taulp)*highpass_amp_spectrum(tf,tauhp)
    return result/np.max(result)
    
def lowpass_phase_spectrum(tf,tau):
    w=2*np.pi*tf
    result=-np.arctan(tau*w)
    return result
    
def highpass_phase_spectrum(tf,tau):
    w=2*np.pi*tf
    result=np.arctan(1.0/(tau*w))
    return result
    
def bandpass_phase_spectrum(tf,tauhp,taulp):
    result=lowpass_phase_spectrum(tf,taulp)+highpass_phase_spectrum(tf,tauhp)
    return result  
    
# returns the integral (along the last axis) of the input function
# if along another axis, use np.transpose before
    
def integrate(x):
    if x.ndim==1:
        result=np.cumsum(x)
    if x.ndim==2:
        result=np.cumsum(x,axis=1)
    if x.ndim==3:
        result=np.cumsum(x,axis=2)
    return result

# returns the differential (along the last axis) of the input function
# if along another axis, use np.transpose before

def differentiate(x):
    xshape=x.shape
    if x.ndim==1:
        interim=x-np.roll(x,1)
        result=interim[1:xshape[0]]
    if x.ndim==2:
        interim=x-np.roll(x,1,axis=1)
        result=interim[:,1:xshape[1]]
    if x.ndim==3:
        interim=x-np.roll(x,1,axis=2)
        result=interim[:,:,1:xshape[2]]
    return result

def Gauss1D(FWHM,RFsize):
    myrange=RFsize/2
    sigma=FWHM/(2.0*np.sqrt(2*np.log(2)))
    x=np.arange(-myrange,(myrange+1),1)*1.0
    z=np.exp(-x**2/(2*(sigma**2)))
    z=z/np.sum(z)
    return z
    
def Gauss2D(FWHM,RFsize):
    myrange=RFsize/2
    sigma=FWHM/(2.0*np.sqrt(2*np.log(2)))
    x=np.arange(-myrange,(myrange+1),1)
    y=np.arange(-myrange,(myrange+1),1)
    x,y=np.meshgrid(x,y)
    r=np.sqrt(x**2+y**2)
    z=np.exp(-r**2/(2*(sigma**2)))
    z=z/np.sum(z)
    return z
    
# blurr calculates the convolution of an image with a Gaussian
# The cross section FWHM is the full width at half-maximum
# filter normalized so that integral of filter = 1.0

def blurr(inp_image,FWHM):
    if inp_image.ndim==1: z=Gauss1D(FWHM,4*FWHM)
    if inp_image.ndim==2: z=Gauss2D(FWHM,4*FWHM)
    result=scipy.ndimage.convolve(inp_image,z)
    return result

# calculates a rebinned array of input x
# all new dims must be integer fractions or multiples of input dims

def rebin(x,f0,f1=0,f2=0):
    
    mydim=x.ndim
    n=x.shape
    
    if mydim==1:
        
        result=np.zeros((f0))
        if f0 <=  n[0]:
            result=x[0:n[0]:int(n[0]/f0)]
        if f0 >  n[0]:
            result=np.repeat(x,int(f0/n[0]))
            
    if mydim==2:
        
        result=np.zeros((f0,f1))
        interim=np.zeros((f0,n[1]))
        
        #handling 1st dim
        
        if f0 <=  n[0]:
            interim=x[0:n[0]:int(n[0]/f0),:]
        if f0 >  n[0]:
            interim=np.repeat(x,int(f0/n[0]),axis=0)
            
        #handling 2nd dim
        
        if f1 <=  n[1]:
            result=interim[:,0:n[1]:int(n[1]/f1)]
        if f1 >  n[1]:
            result=np.repeat(interim,int(f1/n[1]),axis=1)
            
    if mydim==3:
        
        result=np.zeros((f0,f1,f2))
        interim1=np.zeros((f0,n[1],n[2]))
        interim2=np.zeros((f0,f1,n[2]))
        
        #handling 1st dim
        
        if f0 <=  n[0]:
            interim1=x[0:n[0]:int(n[0]/f0),:,:]
        if f0 >  n[0]:
            interim1=np.repeat(x,int(f0/n[0]),axis=0)
            
        #handling 2nd dim
        
        if f1 <=  n[1]:
            interim2=interim1[:,0:n[1]:int(n[1]/f1),:]
        if f1 >  n[1]:
            interim2=np.repeat(interim1,int(f1/n[1]),axis=1)
            
        #handling 3rd dim
        
        if f2 <=  n[2]:
            result=interim2[:,:,0:n[2]:int(n[2]/f2)]
        if f2 >  n[2]:
            result=np.repeat(interim2,int(f2/n[2]),axis=2)

    return result.copy()
             
# Calculates the rectilinear fct of x: x=x if x > thrld, and x=thrld otherwise

def rect(x,thrld):
    result=x-thrld
    result=result*(result>0)
    result=result+thrld
    return result
    
# Calculates the ceiled fct of x: x=x if x < thrld, and x=thrld otherwise
    
def ceil(x,thrld):
    result=x-thrld
    result=result*(result<0)
    result=result+thrld
    return result
    
def limit(x,lowertrld,uppertrld):
    result=rect(x,lowertrld)
    result=ceil(result,uppertrld)
    return result
    
def binomial(x,y):
    if y == x:
        result=1
    elif y == 1:         
        result=x
    elif y > x:          
        result=0
    else:                
        a = math.factorial(x)
        b = math.factorial(y)
        c = math.factorial(x-y)  
        div = a / (b * c)
        result=div
    return 1.0*result
    
def binomial_distribution(n,p):
    probab=np.zeros(n+1)
    for i in range(n+1):
        probab[i]=binomial(n,i)*(p**i)*((1-p)**(n-i))
    return probab
    
def draw_from_p(inp_p):
    p=inp_p/np.sum(inp_p)
    dim=p.shape[0]
    rand_num=np.random.choice(np.arange(dim),p=p)
    return rand_num

def sigmoidal(x,center,slope,maximum):
    output=maximum/(1.0+np.exp(-(x-center)*slope))
    return output
    
def saturate(x,max_fac,sat_fac):
    if sat_fac>7:
        sat_fac=7
    sat_strength=np.array([100,5,2,1,0.5,0.2,0.1,0.02])
    myx=np.linspace(0,100,101)*0.01
    transferfct=myx/(myx+sat_strength[sat_fac])
    transferfct=transferfct/np.max(transferfct)
    x=x*100.0/max_fac
    x=x*(x>0) 
    x=x-100
    x=x*(x<0)+100
    result=max_fac*transferfct[x.astype(int)]
    return result
    
# Calculates the Pearson Correlation Coefficient
 
def calc_corr(a,b):
    a=a-np.mean(a)
    b=b-np.mean(b)
    cov=np.sum(a*b)
    sigma_a=np.sum(a*a)
    sigma_b=np.sum(b*b)
    corr=cov/np.sqrt(sigma_a*sigma_b) 
    return corr

# Calculates the Cross Covariance Fct
   
def calc_cc(a,b,dt):
    a=a-np.mean(a)
    b=b-np.mean(b)
    if a.ndim==1:
        cc=np.zeros(dt*2)
        for i in range(dt*2):
            k=i-dt
            cc[i]=np.mean(a*np.roll(b,-k))
    if a.ndim==2:
        cc=np.zeros((dt*2,dt*2))
        for i in range(dt*2):
            k=i-dt
            print(k)
            yroll=np.roll(b,-k,axis=0)
            for j in range(dt*2):
                l=j-dt
                print(l)
                xroll=np.roll(yroll,-l,axis=1)
                cc[i,j]=np.mean(a*xroll)
    return cc
    
# Calculates the mean and SEM along the first axis
    
def calc_meanSEM(data):

    n=data.shape
    n=n[0]
    
    mean=np.mean(data,axis=0)
    var=mean*0.0
    
    for i in range(n):
        var+=(mean-data[i])**2
        
    SEM=np.sqrt(var/(n*(n-1)))
    
    return mean, SEM
    
def calc_Ldir(data,plot_switch=0):
    
    # assumes first column to hold the angle in deg, and the second one the length
    
    # tranform to radian
    angle = np.pi*2.0/360.*data[:,0]
    
    # calc cartesian coords
    xcood = np.cos(angle)*data[:,1]
    ycood = np.sin(angle)*data[:,1]
    
    xcsum = np.sum(xcood)
    ycsum = np.sum(ycood)
    numer = np.sqrt(xcsum**2+ycsum**2)
    denom = np.abs(np.sum(data[:,1]))
    Ldir  = numer/denom
    
    if plot_switch==1:
        
        plt.polar(angle,data[:,1],linewidth=2)

    return Ldir    
        
def calc_amp_phase(image):
    a=np.fft.fft2(image)
    amp=np.abs(a)
    phase=np.arctan2(a.imag,a.real)
    amp=1.0*amp/(1.0*image.size)
    plt.matshow(amp,cmap='gray',vmax=0.01)
    plt.matshow(phase,cmap='gray',vmin=-np.pi,vmax=np.pi)
    return amp, phase
    
# Calculates the Membrane Potential
# gleak=1.0
    
def calc_Vm(gexc,ginh,gleak):
    gexc=rect(gexc,0)
    ginh=rect(ginh,0)
    Eexc=+50.0
    Einh=-30.0
    Vm=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    return Vm

def calc_CaInd(Vm):   
    
    n= Vm.size 
    ca=np.zeros(n)
    caind=np.zeros(n)
    
    dt=0.001
    
    gamma=50.0
    kf=0.01
    kb=10.0
    kd=kb/kf # nMol
    ymax=100.0 # initial indicator conc in nMol
    ivacc=rect(Vm,0)*1000.0
    
    print('Kd    = ',int(kd), ' nMol')
    print('[Ind] = ',int(ymax), ' nMol')
    
    x0=50.0             # initial free Ca at rest
    y0=ymax*x0/(x0+kd)  # Indicator-bound Ca at rest
    
    ca[0]=x0 
    caind[0]=y0
    
    for i in range(n-1):
        
        t=i+1
        
        icapump=(gamma*ca[t-1])
        indforward=kf*ca[t-1]*(ymax-caind[t-1])
        indbackward=kb*caind[t-1]
        
        ca[t]=((ivacc[t]-icapump-indforward+indbackward)*dt+ca[t-1])
        caind[t]=(indforward-indbackward)*dt+caind[t-1]
        
    return caind
         
# Calculates HH model for current input
# gleak=1.0
    
def calc_HH(current):
    
    # spike threshold at current = 0.18  
    
    noft_points=current.shape[0]
     
    deltat=0.0005

    Vm=np.zeros(noft_points)
    gNa=np.zeros(noft_points)
    gK=np.zeros(noft_points)
    Vm[0]=-0.07
    
    m=0
    n=0
    h=0
    
    Eleak=-0.07     # =  -70 mV
    gleak=1.0
    memcap=0.01

    ENa=+0.020      # = + 20 mV
    EK =-0.100      # = -100 mV
         
    gNa=np.zeros(noft_points)
    gK =np.zeros(noft_points)

    gNamax=200.0
    gKmax =100.0
    
    mmidv=-55.0
    mslope=0.35
    mtau=0.001
    
    hmidv=-65.0
    hslope=-0.15
    htau=0.003
    
    nmidv=-55.0
    nslope=0.15
    ntau=0.004

    myx=np.linspace(0,200,201)-100
    mss=1.0/(1.0+np.exp((mmidv-myx)*mslope))
    hss=1.0/(1.0+np.exp((hmidv-myx)*hslope))
    nss=1.0/(1.0+np.exp((nmidv-myx)*nslope))
    
    for i in range(noft_points-1):
        t=i+1
        Vindex=int(1000.0*Vm[t-1])+100
        Vindex=limit(Vindex,-99,99)
        m=deltat/mtau*(mss[Vindex]-m)+m
        n=deltat/ntau*(nss[Vindex]-n)+n
        h=deltat/htau*(hss[Vindex]-h)+h
        gNa[t]=gNamax*(m**3)*h
        gK[t] =gKmax *(n**4)
        Vm[t]=(current[t]+ENa*gNa[t]+EK*gK[t]+Eleak*gleak+Vm[t-1]*memcap/deltat)/(gleak+gNa[t]+gK[t]+memcap/deltat)
        
    return Vm*1000.0
    
# calculates the gradient of a scalar fct of 2 variables
# returns the first derivatives along x and y respectively
   
def gradient(myf):
    myshape=myf.shape
    if myf.ndim == 2:
        xshape=myshape[1]
        yshape=myshape[0]
        gradx=myf-np.roll(myf,1,axis=1)
        grady=myf-np.roll(myf,1,axis=0)
        gradx=gradx[1:yshape,1:xshape]
        grady=grady[1:yshape,1:xshape]
        result=gradx,grady
    else:
        print('not a 2-dim array')
        result=0
    return result
    
# calculates the divergence of a vector field
# returns the sum of the first derivatives 
    
def divergence(gradx,grady):
    myshape=gradx.shape
    xshape=myshape[1]
    yshape=myshape[0]
    divx=gradx-np.roll(gradx,1,axis=1)
    divy=grady-np.roll(grady,1,axis=0)
    result=divx+divy
    result=result[1:yshape,1:xshape]
    return result
    
# calculates the laplacian of a scalar fct of 2 variables 
    
def laplace(myf):
    if myf.ndim == 2:
        u,v=gradient(myf)
        result=divergence(u,v)
    else:
        print('not a 2-dim array')
        result=0
    return result
       
# --------------- MODELS OF MOTION DETECTORS -----------------------------

def add_photonnoise(signal,meanlum):
    noisysignal=np.random.poisson(signal*meanlum)/(1.0*meanlum)
    noiselevel=np.sqrt(np.mean((noisysignal-signal)**2))/np.sqrt(np.mean((signal)**2))
    print('noiselevel=', noiselevel)
    
    return noisysignal

# calculates the output of a 2dim array of EMDs
# deltat determines temporal resolution in ms
    
def calc_4QHR(R16,deltat):
    
    noff=40
    
    lp=lowpass(R16,250/deltat)
    hp=highpass(R16,250/deltat)
    
    Txa=lp[:,0:noff-1,:]*hp[:,1:noff,:]
    Txb=lp[:,1:noff,:]*hp[:,0:noff-1,:]
    
    mdout=Txa-Txb
    HS=np.mean(np.mean(mdout,axis=0),axis=0)
    
    return HS
    
def calc_2QHR(lp,hp):
    
    noff=40
    
    Txa=rect(lp[:,0:noff-1,:]*hp[:,1:noff,:],0)
    Txb=rect(lp[:,1:noff,:]*hp[:,0:noff-1,:],0)
    
    return Txa, Txb

def calc_HRBL(lp,hp):
    
    noff=40
    DC=0.02
    
    A=rect(lp[:,0:noff-2,:],0)
    B=rect(hp[:,1:noff-1,:],0)
    C=rect(lp[:,2:noff-0,:],0)
    
    Txa=rect(A*B/(DC+C),DC)
    
    A=rect(lp[:,2:noff-0,:],0)
    B=rect(hp[:,1:noff-1,:],0)
    C=rect(lp[:,0:noff-2,:],0)
    
    Txb=rect(A*B/(DC+C),DC)
    
    return Txa, Txb

def calc_HR(lp,hp):
    
    noff=40
    DC=0.02
    
    A=rect(lp[:,0:noff-2,:],0)
    B=rect(hp[:,1:noff-1,:],0)
    
    Txa=rect(A*B/DC,DC)
    
    A=rect(lp[:,2:noff-0,:],0)
    B=rect(hp[:,1:noff-1,:],0)
    
    Txb=rect(A*B/DC,DC)
    
    return Txa, Txb

def calc_BL(lp,hp):
    
    noff=40
    DC=0.02
    
    B=rect(hp[:,1:noff-1,:],0)
    C=rect(lp[:,2:noff-0,:],0)
    
    Txa=rect(B/(DC+C),DC)
    
    B=rect(hp[:,1:noff-1,:],0)
    C=rect(lp[:,0:noff-2,:],0)
    
    Txb=rect(B/(DC+C),DC)
    
    return Txa, Txb
    
def calc_cb_HRBL(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0
    
    Mi9=rect(1.0-lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,2:noff-0,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9+Mi4,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi9=rect(1.0-lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,0:noff-2,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9+Mi4,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_HR(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0
    
    Mi9=rect(1.0-lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi9=rect(1.0-lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi9,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_BL(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0

    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,2:noff-0,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi4,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,0:noff-2,:],0)
    
    gexc=rect(Mi1,0)
    ginh=rect(Mi4,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_eeiHRBL(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    Einh=-20.0
    gleak=1.0
    
    Mi9=rect(lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,2:noff-0,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)
    ginh=rect(Mi4,0)

    Txa=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    Mi9=rect(lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    Mi4=rect(lp[:,0:noff-2,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)
    ginh=rect(Mi4,0)
    
    Txb=(Eexc*gexc+Einh*ginh)/(gexc+ginh+gleak)
    
    return Txa, Txb
    
def calc_cb_eeHR(lp,hp):
    
    noff=40
    
    Eexc=+50.0
    gleak=1.0
    
    Mi9=rect(lp[:,0:noff-2,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)

    Txa=(Eexc*gexc)/(gexc+gleak)
    
    Mi9=rect(lp[:,2:noff-0,:],0)
    Mi1=rect(hp[:,1:noff-1,:],0)
    
    gexc=rect(Mi1,0)+rect(Mi9,0)
    
    Txb=(Eexc*gexc)/(gexc+gleak)
    
    return Txa, Txb
    
def NewEMD(stimulus,deltat,det_switch,ret_switch,noisefac=0, resting_factor=0):
    
    n=stimulus.shape
    maxtime=n[2]
    noff=40
        
    ON_lptau=50.0/deltat
    OFF_lptau=50.0/deltat

    L1gain=1.0
    L2gain=1.0
    ONlpgain=1.0
    ONhpgain=1.0
    OFFlpgain=1.0
    OFFhpgain=1.0
    T4gain=1.0
    T5gain=1.0
    LPigain=1.0

    R16=rebin(stimulus,noff,noff,maxtime)
    
    # add noise 
    
    if noisefac!=0: 
        stimulus=add_photonnoise(stimulus,noisefac)
    
    # tilt the image slightly
    
    for i in range(3):
        j=10*i
        k=10*(i+1)
        R16[k:k+10,:,:]=np.roll(R16[j:j+10:,:],1,axis=1)
    
    if det_switch==0:
        
        print('4QD HR')
        HS=calc_4QHR(R16,deltat)
        result=HS
        print('HS cell')
        
    if det_switch > 0:
        
        interim=highpass(R16,250/deltat)
        interim=interim+0.1*R16
        L1=L1gain*rect(interim,0)
        L2=L2gain*rect(-(interim-0.05),0)
        
        ONlp=ONlpgain*lowpass(R16,ON_lptau)
        ONhp=ONhpgain*L1
        OFFlp=OFFlpgain*lowpass(1.0-R16,OFF_lptau)
        OFFhp=OFFhpgain*L2
        
        if det_switch==1:
            print('a*b/c HRBL')
            T4a,T4b=calc_HRBL(ONlp,ONhp)
            T5a,T5b=calc_HRBL(OFFlp,OFFhp)
            
        if det_switch==2:
            print('a*b   HR')
            T4a,T4b=calc_HR(ONlp,ONhp)
            T5a,T5b=calc_HR(OFFlp,OFFhp)
            
        if det_switch==3:
            print('b/c   BL')
            T4a,T4b=calc_BL(ONlp,ONhp)
            T5a,T5b=calc_BL(OFFlp,OFFhp)
        
        if det_switch==4:
            print('conduct based HRBL')
            T4a,T4b=calc_cb_HRBL(ONlp,ONhp)
            T5a,T5b=calc_cb_HRBL(OFFlp,OFFhp)
        
        if det_switch==5:
            print('conduct based HR')
            T4a,T4b=calc_cb_HR(ONlp,ONhp)
            T5a,T5b=calc_cb_HR(OFFlp,OFFhp)
                             
        if det_switch==6:
            print('conduct based BL')
            T4a,T4b=calc_cb_BL(ONlp,ONhp)
            T5a,T5b=calc_cb_BL(OFFlp,OFFhp)
            
        if det_switch==7:
            print('conduct based exc-exc HRBL')
            T4a,T4b=calc_cb_eeiHRBL(ONlp,ONhp)
            T5a,T5b=calc_cb_eeiHRBL(OFFlp,OFFhp)
            
        if det_switch==8:
            print('conduct based exc-exc HR')
            T4a,T4b=calc_cb_eeHR(ONlp,ONhp)
            T5a,T5b=calc_cb_eeHR(OFFlp,OFFhp)
            
        T4rest=np.mean(T4a[:,:,0:50])
        
        #print 'T4 resting potential: ', T4rest
    
        T4a=T4gain*rect(T4a,T4rest*resting_factor)
        T4b=T4gain*rect(T4b,T4rest*resting_factor)
        T5a=T5gain*rect(T5a,T4rest*resting_factor)
        T5b=T5gain*rect(T5b,T4rest*resting_factor)
        
        T4a_mean=np.mean(np.mean(T4a,axis=0),axis=0)
        T4b_mean=np.mean(np.mean(T4b,axis=0),axis=0)
        T5a_mean=np.mean(np.mean(T5a,axis=0),axis=0)
        T5b_mean=np.mean(np.mean(T5b,axis=0),axis=0)

        synamp=0.05
        
        gexc=(T4a_mean+T5a_mean)*synamp
        ginh=(T4b_mean+T5b_mean)*synamp*LPigain
        
        HS=(40.0*gexc-20.0*ginh)/(gexc+ginh+1.0)
        
        #HS=T4a_mean+T5a_mean-LPigain*(T4b_mean+T5b_mean)   
        
        detpos=19
        
        print('detector position =', detpos)
    
        if ret_switch == 0: 
            result=T4a_mean
            print('T4a_mean')
        if ret_switch == 1: 
            result=T4b_mean
            print('T4b_mean')
        if ret_switch == 2: 
            result=HS
            print('HS cell') 
        if ret_switch == 3: 
            result=T4a[detpos,detpos,:]
            print('T4a')
        if ret_switch == 4: 
            result=T5a[detpos,detpos,:]
            print('T5a')
        if ret_switch == 5: 
            result=ONlp[detpos,detpos,:]
            print('ONlp')
        if ret_switch == 6: 
            result=ONhp[detpos,detpos,:]
            print('ONhp')
        
    return result
    
# --------------- STIMULUS GENERATION -----------------------------
    
# calculates sine grating movie
# velo is velocity function and determines length of movie
# img_rot in degree determines the angle at which the grating is moving
# spat_freq is the spatial frequency - a value of 5 = 40 deg wavelength
    
def calc_singlewave(img_size,new_spat_freq,x,y):
    sinewave=np.sin((np.linspace(0,img_size-1,img_size)-x+y)/img_size*2.0*np.pi*new_spat_freq)
    return sinewave

def calc_sinegrating(tf, img_rot, spat_freq, contrast):
    print()
    print('rot angle =', img_rot)
    n=tf.shape
    maxtime=n[0]
    img_size=200
    movie=np.zeros((img_size,img_size,maxtime))
    img_rot_rad=img_rot/360.0*2*np.pi
    new_spat_freq=spat_freq*np.cos(img_rot_rad)
    spat_wlength = img_size/spat_freq
    velo=tf*spat_wlength/100.0
    print('Spatial Wavelength =', spat_wlength)
    if np.min(velo)==0:
        print('Velocity  (dt=10 msec) =', np.max(velo)*100, 'deg/s')
        print('Temp Freq (dt=10 msec) =', np.max(tf), 'Hz')
    if np.max(velo)==0:
        print('Velocity  (dt=10 msec) =', np.min(velo)*100, 'deg/s')   
        print('Temp Freq (dt=10 msec) =', np.min(tf), 'Hz')
    yshift=np.tan(img_rot_rad)
    new_velo=velo/np.cos(img_rot_rad)
    if img_rot==0:
        image=np.zeros((img_size,img_size))
        interim=calc_singlewave(img_size,spat_freq,0,0)
        for i in range(img_size):
            image[i,0::]=interim 
        for i in range(maxtime):
            movie[:,:,i]=np.roll(image,int(sum(velo[0:i])),axis=1)      
    else:        
        for i in range(maxtime):
                for y in range(img_size):
                    movie[y,:,i]=calc_singlewave(200,new_spat_freq,int(sum(new_velo[0:i])),int(yshift*y))  
    movie=movie*contrast*0.5+0.5
    return movie
     
def calc_1Dsinegrating(tf, spat_freq, contrast):
    n=tf.shape
    maxtime=n[0]
    img_size=200
    movie=np.zeros((img_size,maxtime))
    image=np.sin(np.linspace(0,img_size-1,img_size)/img_size*2.0*np.pi*spat_freq)
    image=image/2.0*contrast+0.5
    spat_wlength = img_size/spat_freq
    velo=tf*spat_wlength/100.0
    print('Spat Wlength =', spat_wlength)
    print('Velocity  (dt=10 msec) =', np.max(velo)*100, 'deg/s')
    print('Temp Freq (dt=10 msec) =', np.max(tf), 'Hz')
    for i in range(maxtime):
        int_image=np.roll(image,int(sum(velo[0:i])),axis=0)
        movie[:,i]=int_image
    return movie

# calculates motion noise movie
    
def calc_motion_noise(maxtime,direction,perccoher,blurrindex):
    
    print()
    print('coherence [%]', perccoher)
    print()
    
    resol=100
    mostart=50
    mostop=maxtime-50
    movie=np.zeros((resol,resol,maxtime))
    nofdots=500
    seq=np.random.permutation(np.linspace(0,nofdots-1,nofdots))
    xpos=np.random.randint(resol, size=nofdots)
    ypos=np.random.randint(resol, size=nofdots)
    xvec=np.random.randint(-2,3, size=nofdots)
    yvec=np.random.randint(-2,3, size=nofdots)
    for i in range(maxtime):   
        movie[xpos,ypos,i]=0
        if np.remainder(i,10) == 0:          
                # ---------- new -----------
                xpos=np.random.randint(resol, size=nofdots)
                ypos=np.random.randint(resol, size=nofdots)      
                # --------------------------            
                xvec=np.random.randint(-2,3, size=nofdots)
                yvec=np.random.randint(-2,3, size=nofdots)
                if mostart<i<mostop:
                    seq=np.random.permutation(np.linspace(0,nofdots-1,nofdots))
                    xvec[seq.astype(int)[0:nofdots/100*perccoher]]=direction
                    yvec[seq.astype(int)[0:nofdots/100*perccoher]]=0
        xpos=np.remainder(xpos+xvec,resol)
        ypos=np.remainder(ypos+yvec,resol)
        movie[ypos,xpos,i]=10.0
    movie=rebin(movie,200,200,maxtime)
    if blurrindex==1:
        for i in range(maxtime):
            print(i, end=' ')
            movie[:,:,i]=blurr(movie[:,:,i],5)
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    return movie
    
def calc_motion_noise_new(maxtime,direction,velo,perccoher,blurrindex):
    
    print()
    print('coherence [%]', perccoher)
    print()
    
    resol=100
    nofdots=5000 # total number of dots in whole movie
    mean_lifetime=20 # frames
    print('nof dots per frame', nofdots*mean_lifetime/maxtime)
    
    movie=np.zeros((resol,resol,maxtime))
    
    xpos=np.random.randint(0, resol, size=nofdots)
    ypos=np.random.randint(0, resol, size=nofdots)
    lifetime=np.random.randint(mean_lifetime-10,mean_lifetime+10, size=nofdots)
    tstart=np.random.randint(0, maxtime, size=nofdots)
    tstop=ceil(tstart+lifetime,maxtime)
    
    xdir=np.random.randint(-1,2, size=nofdots)
    ydir=np.random.randint(-1,2, size=nofdots)
    
    nofcoher_dots=perccoher/100.0*nofdots
    
    # correction
    # nofcoher_dots=(9*nofcoher_dots-nofdots)/8.0
    
    print(nofcoher_dots)
    
    xdir[0:nofcoher_dots]=direction
    ydir[0:nofcoher_dots]=0
    
    for i in range(nofdots):
        counter=0
        for t in range(tstart[i],tstop[i]):
            counter+=1
            velofac=int(counter*velo)
            myxpos=np.remainder(xpos[i]+velofac*xdir[i],resol)
            myypos=np.remainder(ypos[i]+velofac*ydir[i],resol)
            movie[myypos,myxpos,t]=10
            
    movie=rebin(movie,200,200,maxtime)
    
    if blurrindex==1:
        for i in range(maxtime):
            print(i, end=' ')
            movie[:,:,i]=blurr(movie[:,:,i],5)
            
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    
    return movie
    
def calc_PDND_motion_noise(maxtime,velo,perccoher,blurrindex):
    
    mystim1=calc_motion_noise_new(maxtime/2,1,velo,perccoher,blurrindex)
    mystim2=calc_motion_noise_new(maxtime/2,-1,velo,perccoher,blurrindex)
    output=np.concatenate((mystim1,mystim2),axis=2)
    
    return output
    
# calculates motion field
# switch = 1: coherent dot motion rightward
# switch = 2: coherent dot motion leftward
# switch = 3: coherent dot motion upward
# switch = 4: coherent dot motion downward
# switch = 5: expanding dot motion
# switch = 6: contracting dot motion
# switch = 7: cw rotating dot motion
# switch = 8: ccw rotating dot motion

def calc_motion_field(maxtime,switch):    
    movie=np.zeros((200,200,maxtime))
    speed=2.0
    movie=np.zeros((200,200,maxtime))
    nofdots=500
    print('number of dots:', nofdots)
    xpos=np.random.random_sample(nofdots)*200
    ypos=np.random.random_sample(nofdots)*200
    for i in range(maxtime):
        print(i, end=' ')
        movie[ypos.astype(int),xpos.astype(int),i]=0
        for j in range(nofdots):
            if switch==1:
                xpos[j]=np.remainder(xpos[j]+speed,200)
            if switch==2:
                xpos[j]=np.remainder(xpos[j]-speed,200)
            if switch==3:
                ypos[j]=np.remainder(ypos[j]-speed,200)
            if switch==4:
                ypos[j]=np.remainder(ypos[j]+speed,200)
            if switch==5:
                alpha=np.arctan2(ypos[j]-100,xpos[j]-100)
                xvec=speed*np.cos(alpha)
                yvec=speed*np.sin(alpha)
                xpos[j]=xpos[j]+xvec
                ypos[j]=ypos[j]+yvec
                if xpos[j] > 199 or xpos[j] < 0:
                    xpos[j]=np.random.random_sample()*50+75
                if ypos[j] > 199 or ypos[j] < 0:
                    ypos[j]=np.random.random_sample()*50+75
            if switch==6:
                alpha=np.arctan2(ypos[j]-100,xpos[j]-100)
                xvec=speed*np.cos(alpha)
                yvec=speed*np.sin(alpha)
                xpos[j]=xpos[j]-xvec
                ypos[j]=ypos[j]-yvec
                if (xpos[j]>97 and xpos[j]<103) and (ypos[j]>97 and ypos[j]<103):
                    xpos[j]=np.random.random_sample()*200
                    ypos[j]=np.random.random_sample()*200
            if switch==7:
                xvec=speed*0.05*(ypos[j]-100)
                yvec=speed*0.05*(xpos[j]-100)
                xpos[j]=xpos[j]-xvec
                ypos[j]=ypos[j]+yvec
                if (xpos[j]>199 or xpos[j]<0):
                    xpos[j]=np.random.random_sample()*200
                if (ypos[j]>199 or ypos[j]<0):
                    ypos[j]=np.random.random_sample()*200
            if switch==8:
                xvec=speed*0.05*(ypos[j]-100)
                yvec=speed*0.05*(xpos[j]-100)
                xpos[j]=xpos[j]+xvec
                ypos[j]=ypos[j]-yvec
                if (xpos[j]>199 or xpos[j]<0):
                    xpos[j]=np.random.random_sample()*200
                if (ypos[j]>199 or ypos[j]<0):
                    ypos[j]=np.random.random_sample()*200
        movie[ypos.astype(int),xpos.astype(int),i]=10.0
    for i in range(100):
        movie[:,:,i]=movie[:,:,100]
        movie[:,:,maxtime-100+i]=movie[:,:,maxtime-101]
    for i in range(maxtime):
        print(i, end=' ')
        movie[:,:,i]=blurr(movie[:,:,i],5)
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    return movie
    
# calculates transparent motion
    
def calc_transparent_motion(maxtime):
    rightward = calc_motion_field(maxtime,1)
    leftward  = calc_motion_field(maxtime,2)
    movie=rightward+leftward
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    return movie

# Calculates Phi Motion
#
#    switch=1: phi motion
#    switch=2: rev phi
#    switch=3: reiser phi

def calc_phi_motion(switch,tstep,xstep,maxtime,wavel=50):
    movie=np.zeros((200,200,maxtime))
    image=np.zeros((200,200))
    counter=0
    if wavel==50:
        for i in range(4):
            image[:,0+i*50:25+i*50]=1.0
    if wavel==100:
        for i in range(2):
            image[:,0+i*100:50+i*100]=1.0
    if wavel==200:
        for i in range(1):
            image[:,0+i*200:100+i*200]=1.0
    image=np.roll(image,10)
    #image=blurr(image,5)
    image=np.roll(image,-10)
    movie[:,:,0]=image
    for i in range(maxtime-1):
        t=i+1
        movie[:,:,t]=movie[:,:,t-1]
        if np.remainder(t,tstep)==1:
            counter+=1
            movie[:,:,t]=np.roll(image,xstep*counter,axis=1)
            if np.remainder(counter,2)==0:
                if switch == 2: 
                    movie[:,:,t]=1-movie[:,:,t]
                if switch == 3: 
                    movie[:,:,t]=(-1)*movie[:,:,t]
    if switch==3: movie=0.5*(movie+1.0)
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    return movie

# calculates correlated motion
# rule=1: no correlation
# rule=2: Fourier motion
# rule=3: 3-P correlation (converging)
# rule=4: 3-P correlation (diverging)
 
def calc_glider_motion(rule,parity,maxtime):
    speed=2
    movie=np.zeros((200,40,maxtime/speed))-1
    if rule==1:
        for t in range(maxtime/speed):
            for x in range(39):
                if np.random.randn()>0:
                    movie[:,x,t]=+1  
    if rule==2:  
        for x in range(40):
            if np.random.randn()>0:
                movie[:,x,0]=+1       
        for t in range(maxtime/speed-1):
            if np.random.randn()>0:
                movie[:,0,t+1]=+1
            for x in range(39):
                movie[:,x+1,t+1]=movie[:,x,t]*parity         
    if rule==3:  
        for x in range(40):
            if np.random.randn()>0:
                movie[:,x,0]=+1      
        for t in range(maxtime/speed-1):
            if np.random.randn()>0:
                movie[:,0,t+1]=+1
            for x in range(39):
                movie[:,x+1,t+1]=movie[:,x,t]*movie[:,x+1,t]*parity
    if rule==4:  
        for x in range(40):
            if np.random.randn()>0:
                movie[:,x,0]=+1      
        for t in range(maxtime/speed-1):
            if np.random.randn()>0:
                movie[:,0,t+1]=+1
            for x in range(39):
                movie[:,x+1,t+1]=movie[:,x,t]*movie[:,x,t+1]*parity
    movie=np.repeat(movie,5,axis=1)
    movie=np.repeat(movie,speed,axis=2)
    for t in range(50):
        movie[:,:,t]=movie[:,:,50]
        movie[:,:,maxtime-50+t]=movie[:,:,maxtime-51]
    movie=(movie+1.0)*0.5
    for t in range(maxtime):
        print(t, end=' ')
        movie[:,:,t]=blurr(movie[:,:,t],5)            
    plt.imshow(np.transpose(movie[0,:,:]), cmap='gray')
    return movie
    
# calculates apparent motion movie   
    
def calc_app_motion(dphi,dt):    
    maxtime=100
    movie=np.zeros((200,200,maxtime))
    movie[:,50:60,10:maxtime-1]=1.0
    movie[:,50+dphi:60+dphi,10+dt:maxtime-1]=1.0
    for i in range(maxtime):
        movie[:,:,i]=blurr(movie[:,:,i],5)
    return movie
    
def calc_looming_square(maxtime,speed):
    movie=np.zeros((200,200,maxtime))+1.0
    size=40.0
    for t in range(maxtime):
        alpha=np.arctan(size/(1.0*(maxtime/speed+10-t/speed)))/(2.0*np.pi)*360.0
        print(alpha)        
        lower=100-round(alpha)
        upper=100+round(alpha)
        movie[lower:upper,lower:upper,t]=0
    for t in range(100):
        movie[:,:,t]=movie[:,:,100]
        movie[:,:,maxtime-100+t]=movie[:,:,maxtime-101]    
    for t in range(maxtime):
        print(t)
        movie[:,:,t]=blurr(movie[:,:,t],5)
    return movie

# calculates sawtooth movie
# velo is velocity function and determines length of movie
# polarity 1/2 defines a positively or negatively going sawtooth
# spat_freq is the spatial frequency - a value of 5 = 40 deg wavelength
 
def calc_sawtooth(velo,polarity,spat_freq):
    n=velo.shape
    maxtime=n[0]
    img_size=200
    movie=np.zeros((img_size,img_size,maxtime))
    image=np.zeros((img_size,img_size))
    interim=np.remainder(np.linspace(0,img_size-1,img_size),img_size/spat_freq)
    interim=interim/np.max(interim)
    if polarity == 2:
            interim=interim[::-1]
    for i in range(img_size):
        image[i,0::]=interim 
    spat_wlength = img_size/spat_freq
    print('Spat Wlength =', spat_wlength)
    temp_freq = np.max(velo)*1000.0/spat_wlength
    print('Temp Freq =', temp_freq)
    for i in range(maxtime):
        movie[:,:,i]=np.roll(image,int(sum(velo[0:i])),axis=1)
    return movie

# calculates ON or OFF edge movie
# switch=1: ON edge, switch =2, OFF edge
# velo = deg_per_frame
    
def calc_edge(velo,onoffswitch):
    maxtime=1000
    img_size=200
    movie=np.zeros((img_size,img_size,maxtime))
    for i in range(maxtime):
        movie[:,0:(i*velo),i]=1
    if onoffswitch == 2:
        movie=1-movie
    return movie
    
# calculates ON or OFF bar movie
# switch=1: ON edge, switch =2, OFF edge
    
def calc_bar(switch,width):
    if width>50:
        width=50
    maxtime=300
    img_size=200
    movie=np.zeros((img_size,img_size,maxtime))
    for i in range(img_size):
        movie[:,i,50+i:50+width+i]=1.0
    if switch == 2:
        movie=1.0-movie
    for i in range(maxtime):
        movie[:,:,i]=blurr(movie[:,:,i],5)
    return movie
    
# calculates hopping ON or OFF bar 
# switch=1: ON edge, switch =2, OFF edge
    
def calc_hopp_bar(switch):
    maxtime=200
    movie=np.zeros((200,200,maxtime))
    for i in range(40):
        movie[:,5*i:5*i+5,5*i:5*i+5]=1.0
    if switch == 2:
        movie=1.0-movie
    #for i in range(maxtime):
    #movie[:,:,i]=blurr(movie[:,:,i],5)
    return movie
    
# --------------- test motion detector models -------------------------------
    
def calc_dphi_dependence(EMDswitch,celltype,resting_factor=0):
        
    response=np.zeros((50))
    
    for i in range(50):
        print(i)
        mymovie=calc_app_motion(i,10)
        result=NewEMD(stimulus=mymovie,deltat=10,det_switch=EMDswitch,ret_switch=celltype,resting_factor=resting_factor)
        response[i]=np.sum(result)
    return response   

# calculates orientation tuning of EMD
    
def calc_orientation_tuning(EMDswitch,celltype,spat_freq,plotit=1,resting_factor=0):
    
    response=np.zeros(13)
    myvelo=np.zeros(300)
    myvelo[50:300]=1.0
    
    for w in range(13):
        print()
        img_rot=w*30
        print('angle=', img_rot)
        movie=calc_sinegrating(myvelo,img_rot,spat_freq,1.0)
        output=NewEMD(stimulus=movie,deltat=10,det_switch=EMDswitch,ret_switch=celltype,resting_factor=resting_factor)
        response[w]=np.mean(output[200:300])-np.mean(output[0:50])
    response=normalize(response)
    
    if plotit==1:
        
        myalpha=np.linspace(90,450,13)
        plt.polar(myalpha/360.0*2*np.pi,rect(response,0),linewidth=4,color='black')
        plt.ylim(0,1.2)
        plt.yticks(np.arange(3)*0.5)
        locs, labels=plt.yticks()
        plt.yticks(locs, ('','',''))
    
    return response
    
def calc_tf_dependence(EMDswitch,celltype,spat_freq=5,contrast=1,plotit=1,resting_factor=0):
    
    grating_switch=0
    ss_switch=1
    
    if EMDswitch==0: 
        mylabel='4QD'
    if EMDswitch==1: 
        mylabel='2QD'
    if EMDswitch==2: 
        mylabel='ab/c HRBL'
    if EMDswitch==3: 
        mylabel='cb HRBL'
    if EMDswitch==4: 
        mylabel='cb HR'
    if EMDswitch==5: 
        mylabel='cb BL'
    
    if celltype==0: mylabel=mylabel+' T4a'
    if celltype==1: mylabel=mylabel+' T4b'
    if celltype==2: mylabel=mylabel+' HS' 
    
    maxvelo=np.array([0.1,0.2,0.5,1.0,2.0,5.0,10.0])
    velofct=np.zeros(300)
    velofct[50:250]=1.0
    resp=np.zeros((2,7))

    for j in range(2):
        k=j*2-1
        for i in range(7):
            print()
            movie=calc_sinegrating(k*velofct*maxvelo[i],0,spat_freq,contrast)
            if grating_switch==1:
                movie=movie>0.5
            output=NewEMD(stimulus=movie,deltat=10,det_switch=EMDswitch,ret_switch=celltype,resting_factor=resting_factor)
            if ss_switch==0:
                resp[j,i]=np.mean(output[50:250])-np.mean(output[0:50])
            if ss_switch==1:
                resp[j,i]=np.mean(output[150:250])-np.mean(output[0:50])
                
    if plotit==1:
        plt.figure()
        plt.plot(maxvelo, resp[1,:]/np.max(resp), linewidth=3, color='blue',label=mylabel+'PD')
        plt.plot(maxvelo, resp[0,:]/np.max(resp), linewidth=3, color='red', label=mylabel+'ND')
        plt.plot(maxvelo,maxvelo*0, color='black')
        plt.xlabel('temp. frequency [Hz]',fontsize=8)
        plt.ylabel('response',fontsize=8)
        plt.xscale('log')
        plt.xlim(0.1,10)
        plt.legend(fontsize=8,loc=4, frameon=False)
        plt.ylim(-1.1,1.1)
    
    return resp
    
def calc_edgevelo_dependence(EMDswitch,celltype,onoffswitch,plotit=1,resting_factor=0):
    
    if onoffswitch == 1: print('ON  edge')
    if onoffswitch == 2: print('OFF edge')
    
    if EMDswitch==0: 
        mylabel='4QD'
    if EMDswitch==1: 
        mylabel='2QD'
    if EMDswitch==2: 
        mylabel='ab/c HRBL'
    if EMDswitch==3: 
        mylabel='cb HRBL'
    if EMDswitch==4: 
        mylabel='cb HR'
    if EMDswitch==5: 
        mylabel='cb BL'
    
    if celltype==0: mylabel=mylabel+' T4a'
    if celltype==1: mylabel=mylabel+' T4b'
    if celltype==2: mylabel=mylabel+' HS'
    
    maxvelo=np.array([0.1,0.2,0.5,1.0,2.0,5.0,10.0])
    resp=np.zeros((2,7))

    for j in range(2):
        for i in range(7):
            stimperiod=200.0/maxvelo[i]
            print('Velo=',maxvelo[i],'[deg/frame]')
            print('stimperiod =', stimperiod, '[frames]')
            movie=calc_edge(maxvelo[i],onoffswitch)
            if j == 0: movie=movie[:,::-1,:]
            output=NewEMD(stimulus=movie,deltat=10,det_switch=EMDswitch,ret_switch=celltype,resting_factor=resting_factor)
            resp[j,i]=np.mean(output[0:int(stimperiod)])
    
    if plotit==1:
        plt.figure()
        plt.plot(maxvelo*100,resp[1,:]/np.max(resp), linewidth=3, color='blue', label=mylabel+' PD')
        plt.plot(maxvelo*100,resp[0,:]/np.max(resp), linewidth=3, color='red', label=mylabel+' ND')
        plt.plot(maxvelo*100,maxvelo*0, color='black')
        plt.xlabel('edge velocity [deg/sec]')
        plt.ylabel('response') 
        plt.xscale('log')
        plt.legend(fontsize=10, loc=4, frameon=False)
        plt.ylim(-1.1,1.1)
        
    return resp
    
def calc_contrast_dependence(EMDswitch,celltype,spat_freq, PDNDswitch,plotit=1,resting_factor=0):
    
    if EMDswitch==0: 
        mylabel='4QD'
    if EMDswitch==1: 
        mylabel='2QD'
    if EMDswitch==2: 
        mylabel='ab/c HRBL'
    if EMDswitch==3: 
        mylabel='cb HRBL'
    if EMDswitch==4: 
        mylabel='cb HR'
    if EMDswitch==5: 
        mylabel='cb BL'
    
    if celltype==0: mylabel=mylabel+' T4a'
    if celltype==1: mylabel=mylabel+' T4b'
    if celltype==2: mylabel=mylabel+' HS'
        
    contrast=np.linspace(0,1,6)
    velo=np.zeros(500)
    velo[50:500]=1.0*PDNDswitch
    resp=np.zeros(6)

    for i in range(6):
        print()
        print(contrast[i])
        movie=calc_sinegrating(velo,0,spat_freq,contrast[i])
        output=NewEMD(stimulus=movie,deltat=10,det_switch=EMDswitch,ret_switch=celltype,resting_factor=resting_factor)
        output=output-np.mean(output[0:50])
        resp[i]=np.mean(output[200:500])
        
    if plotit==1:

        plt.plot(contrast, resp, linewidth=3, color='black', label=mylabel)
        plt.xlabel('contrast')
        plt.ylabel('response')
        plt.legend(fontsize=10, loc=4, frameon=False)
    
    return resp

#---------------------------------------------------------------------------------
    
def fit_bandpass(stim,mydata,nofsteps,stepsize,wstart,wstop):
    myerror=np.zeros((nofsteps,nofsteps))
    for i in range(nofsteps):
        hptc=(i)*stepsize
        hp=highpass(stim,hptc)
        for j in range(nofsteps):
            lptc=(j+1)*stepsize
            output=lowpass(hp,lptc)
            output=normalize(output)
            myerror[i,j]=np.nanmean((mydata[wstart:wstop]-output[wstart:wstop])**2)
    
    opti=np.argmin(myerror)
    indices=np.unravel_index(opti,(nofsteps,nofsteps))
    hptc=(indices[0])*stepsize
    lptc=(indices[1]+1)*stepsize
    
    return np.array([hptc,lptc])
    
# calculates temp freq response of LP detector

def calc_LPresp(taulp,om):
    denom=(1.0+(taulp*om)**2)
    output=(taulp*om)/denom
    return output
    
# calculates temp freq response of HP-LP or HP-In detector

def calc_HPLPresp(taulp,tauhp,om,switch):
    denom=(1.0+(taulp*om)**2)*(1.0+(tauhp*om)**2)
    output=tauhp*om*(switch*1.0+tauhp*taulp*om**2)/denom
    return output

# determines temp freq optimum of HP-LP or HP-In detector
# between between 0.1 and 10 Hz
# as a function of taulp (from 1 to 200 ms) and tauhp (from 1 to 200 ms)

def calc_tfopt(switch):
    if switch==0: print('HP-In Detector')
    if switch==1: print('LP-HP Detector')
    tf=np.linspace(0,99,100)*0.1
    om=tf*2.0*np.pi
    myrange=1000
    opt=np.zeros((myrange,myrange))
    for i in range(myrange):
        taulp=(i+1)*0.001
        for j in range(myrange):
            tauhp=(j+1)*0.001
            resp=calc_HPLPresp(taulp,tauhp,om,switch)
            opt[i,j]=np.argmax(resp)*0.1
    
    plt.imshow(np.log10(opt), cmap='hot',vmin=-0.5,vmax=1)
    return opt
    
# Same as above, however explicit calculation
    
def calc_numtfopt(taulp, tauhp):
    L=taulp**2
    H=tauhp**2
    x=np.sqrt(1.0/(L*H)*(L+H+np.sqrt((L+H)**2+12*H*L)))
    tfopt=x/(2.0*np.pi)
    return tfopt

# writes an mp4 movie to the disk

def make_movie(movie, name="my_movie.mp4", fps=30):
    
    import matplotlib.animation as animation

    n=movie.shape        
    fig = plt.figure()    
    im = plt.imshow(movie[:,:,0],cmap='gray',vmin=0,vmax=1)
    
    def animate(i):
        
        im.set_data(movie[:,:,i])
        return im,

    a = animation.FuncAnimation(fig, animate, frames=n[2])
    mywriter = animation.FFMpegWriter(fps=fps,bitrate=1000)
    a.save(name, writer=mywriter)

# removes motion artefact from a movie

def remove_motion(images):
    n=images.shape
    output=np.zeros((n[0],n[1],n[2]))
    output[:,:,0]=images[:,:,0]
    for i in range(n[2]-1):
        myframe=i+1
        myshift=cv2.phaseCorrelate(output[:,:,0],images[:,:,myframe])
        print(myshift)
        output[:,:,myframe]=np.roll(images[:,:,myframe],-np.int(np.round(myshift[1])), axis=0)
        output[:,:,myframe]=np.roll(output[:,:,myframe],-np.int(np.round(myshift[0])), axis=1)
    
    return output

# plots a flow-field
# switch=1: translation downward
# switch=2: expansion
# switch=1: rotation
# switch=4: random
    
def plot_flowfield(res,ffswitch,colorswitch):
    n=res
    myscale=8
    x,y=np.mgrid[-n:n+1,-n:n+1]
    
    if ffswitch==1:
        u,v=0,-res
    if ffswitch==2:
        u,v=x,y
    if ffswitch==3:
        u,v=-y,x
    if ffswitch==4:
        u=(np.random.random_sample((2*n+1,2*n+1))-0.5)*res*2
        v=(np.random.random_sample((2*n+1,2*n+1))-0.5)*res*2

    if colorswitch==0:
        plt.quiver(x,y,u,v,color='black',scale=myscale, scale_units='xy')
        
    if colorswitch==1:
        z=-v
        plt.quiver(x,y,u,v,z,cmap='bwr',scale=myscale, scale_units='xy')
        
    plt.xlim(-n-1,n+1)
    plt.ylim(-n-1,n+1)
    
def nice2Dfct(switch):
    
    if switch==1:
        points = np.linspace(-10, 10, 101)
        X,Y = np.meshgrid(points, points)
        R = np.sqrt(X**2 + Y**2)
        Z = jn(0,R)
        
    if switch==2:       
        X,Y = np.mgrid[-4:2:100j, -4:2:100j]
        Z = 10 * np.cos(X**2 - Y**2)
        
    if switch==3:             
        X,Y = np.mgrid[-4:2:100j, -4:2:100j]
        Z = 10 * np.cos(X**2 + Y**2)
        
    return Z
    
def setmyaxes(myxpos,myypos,myxsize,myysize):
    
    ax=plt.axes([myxpos,myypos,myxsize,myysize])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
# plots either a 3D grid or 3D surface of f(x,y)
    
def plot3D(data,switch=1,color_switch=2):
    
    fig = plt.figure(figsize=(9,6))
    ax = fig.gca(projection='3d')
    
    n=data.shape
    X = np.linspace(1, n[1], n[1])
    Y = np.linspace(1, n[0], n[0])
    X, Y = np.meshgrid(X, Y)
    Z = data
    
    # Create a light source with 180 deg azimuth, 45 deg elevation.
    
    light = LightSource(180,45)
    
    if color_switch==1:
        illuminated_surface = light.shade(Z, cmap=cm.copper)
        mycolor='copper'
    if color_switch==2:
        illuminated_surface = light.shade(Z, cmap=cm.coolwarm)
        mycolor='coolwarm'
        
    # Set view parameters with 45 deg azimuth, 60 deg elevation.   
        
    ax.view_init(20,45)
    
    if switch==1:
        ax.plot_surface(X, Y, Z, cmap=mycolor,rstride=1, cstride=1,
                linewidth=0, antialiased=True,facecolors=illuminated_surface)
    if switch==2:
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.show()
    
def plot3Dline(x,y,z):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x,y,z)

    
def plotShade(data):

    n=data.shape
    X = np.linspace(1, n[1], n[1])
    Y = np.linspace(1, n[0], n[0])
    X, Y = np.meshgrid(X, Y)
    Z = data
    
    cmap = plt.cm.copper
    ls = LightSource(315, 45)
    rgb = ls.shade(Z, cmap)

    fig, ax = plt.subplots()
    ax.imshow(rgb)

    # Use a proxy artist for the colorbar...
    im = ax.imshow(Z, cmap=cmap)
    im.remove()
    fig.colorbar(im)

    plt.show()
    
def plot3DCube(data):
    
    data=1.0*rebin(data,50,50,50)
    
    # add black edges
    
    data[:,0,0]=0
    data[:,49,0]=0
    data[:,49,49]=0
    data[:,0,49]=0
    
    data[0,:,0]=0
    data[49,:,0]=0
    data[49,:,49]=0
    data[0,:,49]=0

    data[0,0,:]=0
    data[49,0,:]=0
    data[49,49,:]=0
    data[0,49,:]=0
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
  
    xx, yy = np.meshgrid(np.arange(50), np.arange(50))
    zz = np.arange(50)*0+50
    roofdata=np.transpose(data[49,:,:])
    ax.plot_surface(xx,yy,zz, rstride=1, cstride=1, facecolors=plt.cm.gray(roofdata), shade=False)
    
    yy, zz = np.meshgrid(np.arange(50), np.arange(50))
    xx = np.arange(50)*0+50
    sidedata=data[:,49,:]
    ax.plot_surface(xx,yy,zz, rstride=1, cstride=1, facecolors=plt.cm.gray(sidedata), shade=False)
    
    xx, zz = np.meshgrid(np.arange(50), np.arange(50))
    yy = np.arange(50)*0
    frontdata=data[:,:,0]
    ax.plot_surface(xx,yy,zz, rstride=1, cstride=1, facecolors=plt.cm.gray(frontdata), shade=False)
    
    plt.axis('off')
    plt.show()
    
def plot_polar(data,mycolor='black',mylabel='Cntrl',overplot=0):
    
    if overplot==0:
        plt.figure()
        
    nofdim=data.ndim
        
    if nofdim==1:
    
        mean=1.0*data/np.max(data)
        
    else:
        
        mean=data[0]/np.max(data[0])
        sem=data[1]/np.max(data[0])
        
    n=mean.shape[0]
    
    angle=np.arange(n)
    
    plt.polar(angle/(1.0*n)*2*np.pi,mean,linewidth=3,color=mycolor,label=mylabel)
    plt.yticks(np.arange(3)*0.5)
    locs, labels=plt.yticks()
    plt.yticks(locs, ('','',''))
    plt.ylim(0,1.2)
    plt.legend(loc=6,frameon=False)
    
# ---- color stuff --------------------------
    
def pickcolor(ccode,i):
    
    mycolor=np.array(['11','33','55','77','99','BB','DD','FF'])
    myclrar=np.array(['#FF0000','#FF8800','#888800','#008800','#00FF88','#00FFFF','#0088FF','#0000FF'])
    if ccode==1: thecolor='#'+mycolor[i]+mycolor[i]+'00'
    if ccode==2: thecolor='#'+mycolor[i]+'00'+mycolor[i]
    if ccode==3: thecolor='#'+'00'+mycolor[i]+mycolor[i]
    if ccode==4: thecolor=myclrar[i]
    return thecolor
    
def getRGBvals(ccode,shift=0,view=0):
    
    rgbvals=np.zeros((256,3))
    
    if ccode<5:
        for i in range(256):
            if ccode==1: interim=plt.cm.hot(i)
            if ccode==2: interim=plt.cm.rainbow(i)
            if ccode==3: interim=plt.cm.bwr(i)
            if ccode==4: interim=plt.cm.hsv(i)
            for j in range(3):
                rgbvals[i,j]=interim[j]
            
    if ccode==5:
        interim=(np.sin(np.linspace(0,2.0*np.pi,256))+1.0)*127.0
        rgbvals[:,0]=np.roll(interim,+64)
        rgbvals[:,1]=np.roll(interim,-64)
        rgbvals[:,2]=np.roll(interim,128)
        
    rgbvals=np.roll(rgbvals,shift,axis=0)
        
    if view==1:
        
        plt.figure()
        plt.plot(rgbvals[:,0], linewidth=3, color='red')
        plt.plot(rgbvals[:,1], linewidth=3, color='green')
        plt.plot(rgbvals[:,2], linewidth=3, color='blue')
        
        if ccode==1: plt.title('CMap HOT')
        if ccode==2: plt.title('CMap Rainbow')
        if ccode==3: plt.title('CMap BWR')
        if ccode==4: plt.title('CMap HSV')
        if ccode==5: plt.title('CMap Circular')
            
    return rgbvals
    
def convim(image,ccode,shift=0,view=0):
    n=image.shape
    outim=np.zeros((n[0],n[1],3))
    myimage=normalize(1.0*image)*255
    myrgbvals=getRGBvals(ccode)
    for i in range(3):
        outim[:,:,i]=myrgbvals[myimage.astype(int),i]
    outim=normalize(outim)
    if view==1:
        plt.figure()
        plt.imshow(outim)
    return outim
    
def create_colorwheel(ccode, satfac=0.3, shift=0, discr=0):
    
    phase=np.zeros((256,256))
    amp=np.zeros((256,256))
    outim=np.zeros((256,256,3))
    
    x,y=np.mgrid[-128:128,-128:128]
    x=-1.0*x
    amp=np.sqrt(x**2+y**2)
    phase=np.arctan2(x,y)+np.pi
                  
    myrgbvals=getRGBvals(ccode,shift)
    phase=normalize(phase)*255
    if discr != 0:
        interim=(phase+32)/64
        phase=interim.astype(int)*64
        phase=phase*(phase<200)
    amp=normalize(amp)
    amp=amp/(satfac+amp)
    
    for i in range(3):
        outim[:,:,i]=myrgbvals[phase.astype(int),i]*amp
        
    outim=normalize(outim)
    outim=outim[:,::-1,:]
    plt.imshow(outim)
    
def createhuesat(satfac=10.0):
    
    outim=np.zeros((256,256,3))
    phase=np.zeros((256,256))
    
    for i in range(256):
        phase[i,:]=i   
    amp=np.transpose(phase)
    
    myrgbvals=getRGBvals(4)+1
    for i in range(3):
        outim[:,:,i]=myrgbvals[phase.astype(int),i]*amp      
    outim=ceil(outim,satfac)
    
    outim=normalize(outim)
    plt.imshow(outim, origin='lower')
    
def showdirmap(amp,phase,ccode=4,satfac=0.1,shift=0):
    
    outim=np.zeros((256,256,3))
    
    myrgbvals=getRGBvals(ccode,shift)
    phase=normalize(phase)*255
    amp=normalize(amp)
    amp=amp/(satfac+amp)
    
    for i in range(3):
        outim[:,:,i]=myrgbvals[phase.astype(int),i]*amp
        
    outim=normalize(outim)
    plt.imshow(outim)
    
def create_overlay(greyim,actim,thrld,ccode):
    
    gshape=greyim.shape
    ashape=actim.shape
    outputim=np.zeros((gshape[0],gshape[1],3))
    
    if gshape != ashape: print('images must have same shape')
    if greyim.ndim !=2 or actim.ndim !=2: print('images must be 2D!')
    
    convactim=convim(actim,ccode)
    
    convactim=normalize(convactim)
    greyim=normalize(greyim)
    
    mask=actim>thrld 
    
    for k in range(3):
        outputim[:,:,k]=(1-mask)*greyim+mask*convactim[:,:,k]

    plt.imshow(outputim)
    
    return outputim

# ------Compartmental Modeling ------------------------------
# ----- artificial test neurons -----------------------------

def build_cable(nofcomps,complength,compradius):
    
    cable=np.zeros((nofcomps,7))
    
    for i in range(nofcomps):
        
        cable[i,0]=i+1
        cable[i,2]=i*complength
        cable[i,5]=compradius
        cable[i,6]=i
        
    cable[0,6]=-1
    
    return cable
    
def build_tapered_cable(nofcomps,complength,compradius):
    
    cable=np.zeros((nofcomps,7))
    
    for i in range(nofcomps):
        
        cable[i,0]=i+1
        cable[i,2]=i*complength
        cable[i,5]=compradius*i
        cable[i,6]=i
        
    cable[0,6]=-1
    
    return cable
    
def build_tree(nofcomps,complength,compradius):
    
    tree=np.zeros((nofcomps,7))
    
    for i in range(nofcomps/2):
        
        tree[i,0]=i+1
        tree[i,2]=i*complength
        tree[i,5]=compradius
        tree[i,6]=i
        
    for i in range(nofcomps/2,nofcomps):
        
        tree[i,0]=i+1
        tree[i,2]=nofcomps/4*complength
        tree[i,3]=(i+2-nofcomps/2)*complength
        tree[i,5]=compradius*0.5
        tree[i,6]=i
        
    tree[nofcomps/2,6]=nofcomps/4+1
    tree[0,6]=-1
    
    return tree
    
# --------calculate Adjancy and Conductance Matrices ---

def calc_AdjM(mycell):
    
    interim=mycell.shape
    nofcomps=interim[0]
    
    IdenM=np.identity(nofcomps)
    AdjM=np.zeros((nofcomps,nofcomps))
    
    for i in range(nofcomps):
        if mycell[i,6]!=-1:
            AdjM[i,mycell[i,6]-1]=1
            
    AdjM[:,:]=AdjM[:,:]+np.transpose(AdjM[:,:])+IdenM  
    
    return AdjM
    
def calc_Conductance_M(mycell, Rm=16000.0, Ra=200.0, Cm=0.6, deltat=0.001):
       
    # Rm=16000.0 # Ohm cm^2
    # Ra=0200.0  # Ohm cm
    # Cm=0.6     # microFarad/(cm**2)
    # deltat= 0.001 # = 1 msec
    
    # --- define matrix M --------
    
    interim=mycell.shape
    nofcomps=interim[0]
            
    M=np.zeros((nofcomps,nofcomps))
    
    # -------------------------------------
    
    compdiam=mycell[:,5]*2.0    # in micrometer
    complength=np.zeros(nofcomps)
    
    # complength defined backwards
    
    for i in range(1,nofcomps,1):
            
            aind=int(mycell[i,0]-1)
            bind=int(mycell[i,6]-1)
            axyz=mycell[aind,2:5]
            bxyz=mycell[bind,2:5]
            
            complength[i]=np.sqrt(np.sum((axyz-bxyz)**2)) # in micrometer
            
            meandiam=(compdiam[aind]+compdiam[bind])*0.5
            area=meandiam**2.0/4.0*np.pi
            M[bind,aind]=-area/complength[aind]/Ra*10**(-4)
            M[aind,bind]=M[bind,aind]
            
    complength[0]=complength[1]
    
    gleak=(compdiam*np.pi*complength)/(Rm*10**8)
    memcap=(compdiam*np.pi*complength)*Cm*(10**-6)/(10**8)
    
    for i in range(nofcomps):
        M[i,i]=gleak[i]-np.sum(M[i])
    
    M=sparse.csr_matrix(M)
        
    return M,memcap,gleak
    
def calc_Vol_Surf(swcfile):
    
    #swcfile='T4a_swc_deposit/85.swc'   
    mycell=np.loadtxt(swcfile)
    mycell[:,2:6]=0.01*mycell[:,2:6]   
    interim=mycell.shape
    nofcomps=interim[0]
    
    # -------------------------------------
    
    compdiam=mycell[:,5]*2.0    # in micrometer
    complength=np.zeros(nofcomps)
    compvol=np.zeros(nofcomps)
    compsurf=np.zeros(nofcomps)
    
    # complength defined backwards
    
    for i in range(1,nofcomps,1):
            
            aind=int(mycell[i,0]-1)
            bind=int(mycell[i,6]-1)
            axyz=mycell[aind,2:5]
            bxyz=mycell[bind,2:5]
            
            complength[i]=np.sqrt(np.sum((axyz-bxyz)**2)) # in micrometer
            
            meandiam=(compdiam[aind]+compdiam[bind])*0.5
            area=meandiam**2.0/4.0*np.pi
            
            compvol[i]=area*complength[i]
            compsurf[i]=compdiam[i]*np.pi*complength[i]
            
    total_vol=np.sum(compvol)
    total_surf=np.sum(compsurf)
    
    print('Volume  [micrometer^3]:', format(total_vol,'.2f'))
    print('Surface [micrometer^2]:', format(total_surf,'.2f'))
    
# Compartmental Models of a T4 cell:
# low  temporal resolution (LTR, deltat=10  ms), with Mi1,4,9,Tm3,C3 and CT1 inputs as a function of the stimulus
    
def MultiCompT4_LTR(gMi1,gMi4,gMi9,gTm3,gCT1,gC3,gIhswitch=1):
    
    # 2.5 sec runtime for maxtime 1000
    
    maxtime=1000
    deltat=0.001     # 10 msec   
    
    # load T4 cell: soma=1735, axon=1998, dend=971   
    
    swcfile='T4a_swc_deposit/85.swc'   
    mycell=np.loadtxt(swcfile)
    mycell[:,2:6]=0.01*mycell[:,2:6]   
    interim=mycell.shape
    nofcomps=interim[0] 
    
    # passive membrane properties
    
    Rm=16000.0
    Ra=200.0
    Cm=0.6
       
    if gIhswitch==0: 
        IhComps=np.zeros(nofcomps) 
    if gIhswitch==1:
        IhComps=np.zeros(nofcomps)+1    # everything has Ih            
        IhComps[1535:1736]=0            # except soma fibre and soma    
    
    # calculates the matrix (0.15 s computing time)
    
    M,memcap=calc_Conductance_M(mycell, Rm=Rm, Ra=Ra, Cm=Cm, deltat=deltat)
    
    # defines Vm and gIh
        
    Vm = np.zeros((nofcomps,maxtime))   
    gIh= np.zeros((nofcomps,maxtime))
    
    # Receptor Channel Reversal Potentials
        
    EnAChR=+0.05
    EGABAR=-0.03
    EGluRa=-0.03
    
    # Ih Parameters -----------
 
    EIh=+0.050      # = +50 mV  
    gIhmax=100.0*10**(-12)      # pico Siemens    
    umidv=-28.0
    uslope=-0.25
    utau=0.800
    myx=np.linspace(0,200,201)-100
    uss=1.0/(1.0+np.exp((umidv-myx)*uslope))  
    u=np.zeros(nofcomps)
    
    gMi1=gMi1*(10**(-12)) # pS
    gMi4=gMi4*(10**(-12)) # pS
    gMi9=gMi9*(10**(-12)) # pS
    gTm3=gTm3*(10**(-12)) # pS
    gCT1=gCT1*(10**(-12)) # pS
    gC3 = gC3*(10**(-12)) # pS
    
    M_actual=1.0*M
        
    for t in range(1,maxtime):
        
        print('.', end=' ')
        
        # calculate Ih
        
        Vindex=rect(1000.0*Vm[:,t-1],-99)
        Vindex=ceil(Vindex,+99)
        Vindex=Vindex.astype(int)+100
        u[:]=deltat/utau*(uss[Vindex]-u[:])+u[:]
        gIh[:,t]=gIhmax*u*IhComps
        
        M_actual.setdiag(M.diagonal()+gMi1[:,t]+gMi4[:,t]+gMi9[:,t]+gTm3[:,t]+gCT1[:,t]+gC3[:,t]+gIh[:,t]+memcap/deltat)
        rightsideofeq=Vm[:,t-1]*memcap/deltat+EnAChR*(gMi1[:,t]+gTm3[:,t])+EGABAR*(gMi4[:,t]+gCT1[:,t]+gC3[:,t])+EGluRa*gMi9[:,t]+EIh*gIh[:,t]
        Vm[:,t] = spsolve(M_actual,rightsideofeq)
    
    Vm[:,:] = 1000.0*Vm # mV
    
    print()
    
    return Vm
    
# Two Compartmental Models of a T4: CClamp and VClamp
    
def MultiCompT4_CClamp(injcompswitch=1,HHon=1,Ihon=1,curramp=10*(10**(-12))):
    
    # 25 sec runtime for maxtime 1000
    
    maxtime=1000
    deltat=0.0002  #  0.2 msec   
    
    # load T4 cell: soma=1735,dend=971  
    
    swcfile='T4a_swc_deposit/85.swc'
    mycell=np.loadtxt(swcfile)
    mycell[:,2:6]=0.01*mycell[:,2:6]   
    interim=mycell.shape
    nofcomps=interim[0] 
    
    # Areas:    mycell[:,1]=200
    #           mycell[1535:1736,1]=100 # soma fibre and soma
    #           mycell[1750:2012,1]=150 # axon
    
    # Comp Numbers
    
    soma=1735
    dend=1107
    axterm=2000
        
    if injcompswitch==1: injcomp=soma
    if injcompswitch==2: injcomp=dend
    if injcompswitch==3: injcomp=axterm
    
    # passive membrane properties
    
    Rm=26000.0
    Ra=400.0
    Cm=0.6
    
    # calculates the matrix
    
    M,memcap,gleak=calc_Conductance_M(mycell, Rm=Rm, Ra=Ra, Cm=Cm, deltat=deltat)
    
    # define Vm and currinj
        
    Vm = np.zeros((nofcomps,maxtime)) 
    currinj= np.zeros((nofcomps,maxtime)) 
    tstart=200
    tstop=800 
    currinj[injcomp,tstart:tstop]=curramp
    
    # define active compartments

    HHComps=np.zeros(nofcomps)
    IhComps=np.zeros(nofcomps)
     
    # only axon has HH
    
    HHComps[1860:2012]=1.0
    
    # only dendrite has Ih
    
    IhComps[0:1535]=1.0
    
    # set HH and Ih according to switches
    
    if HHon==0:
        HHComps=0*HHComps
    if Ihon==0:
        IhComps=0*IhComps
    
    # Active Stuff -----------
    
    ENa=+0.090      # = +90 mV
    EK =-0.030      # = -30 mV
    EIh=+0.050      # = +50 mV
         
    gNa=np.zeros((nofcomps,maxtime))
    gK =np.zeros((nofcomps,maxtime))
    gIh=np.zeros((nofcomps,maxtime))
    
    gNamax=200.0*10**(-12)      # pico Siemens
    gKmax = 50.0*10**(-12)      # pico Siemens
    gIhmax=100.0*10**(-12)      # pico Siemens
    
    mmidv=20.0
    mslope=0.35
    mtau=0.001

    hmidv=10.0
    hslope=-0.15
    htau=0.003

    nmidv=20.0
    nslope=0.15
    ntau=0.004
        
    umidv=-28.0
    uslope=-0.25
    utau=0.800

    myx=np.linspace(0,200,201)-100
    
    mss=1.0/(1.0+np.exp((mmidv-myx)*mslope))
    hss=1.0/(1.0+np.exp((hmidv-myx)*hslope))
    nss=1.0/(1.0+np.exp((nmidv-myx)*nslope))
    uss=1.0/(1.0+np.exp((umidv-myx)*uslope))
           
    m=np.zeros(nofcomps)
    h=np.zeros(nofcomps)
    n=np.zeros(nofcomps)    
    u=np.zeros(nofcomps)
    
    # end of initializing
    
    def update_mnhu(Vm):
        
        Vindex=rect(1000.0*Vm,-99)
        Vindex=ceil(Vindex,+99)
        Vindex=Vindex.astype(int)+100
        
        m[:]=deltat/mtau*(mss[Vindex]-m[:])+m[:]
        n[:]=deltat/ntau*(nss[Vindex]-n[:])+n[:]
        h[:]=deltat/htau*(hss[Vindex]-h[:])+h[:]
        u[:]=deltat/utau*(uss[Vindex]-u[:])+u[:]
        
        return m,n,h,u
    
    M_actual=1.0*M
    
    for t in range(1, maxtime):
        
        print('.', end=' ')
        
        m,n,h,u=update_mnhu(Vm[:,t-1])
        
        gNa[:,t]=gNamax*(m**3)*h*HHComps
        gK[:,t] =gKmax *(n**4)*HHComps
        gIh[:,t]=gIhmax*u*IhComps
        
        M_actual.setdiag(M.diagonal()+gNa[:,t]+gK[:,t]+gIh[:,t]+memcap/deltat)
        rightsideofeq=Vm[:,t-1]*memcap/deltat+currinj[:,t]+ENa*gNa[:,t]+EK*gK[:,t]+EIh*gIh[:,t]
        
        Vm[:,t] = spsolve(M_actual,rightsideofeq)
    
    Vm[:,:] = 1000.0*Vm # mV
    
    return Vm
    
def MultiCompT4T5_VClamp(T4T5switch,injcompswitch=1,passive=0,plotswitch=1,ClampPot=-0.05):
    
    # 25 sec runtime for maxtime 1000
    
    maxtime=2000
    deltat=0.0002  #  0.2 msec     
    
    # load T4 cell: soma=1735,dend=971  
    # load T5 cell: soma=264, dend=284 
    
    if T4T5switch==1:
        swcfile='T4a_swc_deposit/85.swc' 
        soma=1735
        dend=971
        
    if T4T5switch==2: 
        swcfile='T5a_swc_deposit/365withsoma.swc' 
        soma=264
        dend=284
        
    if injcompswitch==1: 
        injcomp=soma
        
    if injcompswitch==2:
        injcomp=dend
          
    mycell=np.loadtxt(swcfile)
    mycell[:,2:6]=0.01*mycell[:,2:6]   
    interim=mycell.shape
    nofcomps=interim[0] 
    
    # passive membrane properties
    
    Rm=16000.0
    Ra=200.0
    Cm=0.6
    
    # calculates the matrix
    
    M,memcap=calc_Conductance_M(mycell, Rm=Rm, Ra=Ra, Cm=Cm, deltat=deltat)
    
    # define Vm and currinj
        
    Vm = np.zeros((nofcomps,maxtime)) 
    currinj= np.zeros(nofcomps) 
    VmClamp=np.zeros(maxtime)
    curramp=np.zeros(maxtime)
    
    tstart=500
    tstop=1500 
    
    currinj[injcomp]=1.0
    VmClamp[tstart:tstop]=ClampPot
    
    # define active compartments

    HHComps=np.zeros(nofcomps)+1    # everything has HH
    IhComps=np.zeros(nofcomps)+1    # everything has Ih 
    
    if T4T5switch==1:
        HHComps[1535:1736]=0            # except soma fibre and soma
        IhComps[1535:1736]=0            # except soma fibre and soma  
        
    if T4T5switch==2:          
        HHComps[85:264]=0               # except soma fibre and soma
        IhComps[85:264]=0               # except soma fibre and soma  
        
    if passive==1:
        HHComps=0*HHComps
        IhComps=0*IhComps
    
    # Active Stuff -----------
    
    ENa=+0.090      # = +90 mV
    EK =-0.060      # = -30 mV
    EIh=+0.050      # = +50 mV
         
    gNa=np.zeros((nofcomps,maxtime))
    gK =np.zeros((nofcomps,maxtime))
    gIh=np.zeros((nofcomps,maxtime))
    
    gNamax=200.0*10**(-12)      # pico Siemens
    gKmax = 50.0*10**(-12)      # pico Siemens  
    gIhmax=100.0*10**(-12)      # pico Siemens
    
    mmidv=20.0
    mslope=0.35
    mtau=0.002

    hmidv=10.0
    hslope=-0.15
    htau=0.003

    nmidv=20.0
    nslope=0.15
    ntau=0.004
        
    umidv=-28.0
    uslope=-0.25
    utau=0.800

    myx=np.linspace(0,200,201)-100
    
    mss=1.0/(1.0+np.exp((mmidv-myx)*mslope))
    hss=1.0/(1.0+np.exp((hmidv-myx)*hslope))
    nss=1.0/(1.0+np.exp((nmidv-myx)*nslope))
    uss=1.0/(1.0+np.exp((umidv-myx)*uslope))
           
    m=np.zeros(nofcomps)
    h=np.zeros(nofcomps)
    n=np.zeros(nofcomps)    
    u=np.zeros(nofcomps)
    
    # end of initializing
    
    def update_mnhu(Vm):
        
        Vindex=rect(1000.0*Vm,-99)
        Vindex=ceil(Vindex,+99)
        Vindex=Vindex.astype(int)+100
        
        m[:]=deltat/mtau*(mss[Vindex]-m[:])+m[:]
        n[:]=deltat/ntau*(nss[Vindex]-n[:])+n[:]
        h[:]=deltat/htau*(hss[Vindex]-h[:])+h[:]
        u[:]=deltat/utau*(uss[Vindex]-u[:])+u[:]
        
        return m,n,h,u
    
    gain=500.0
    
    M_actual=1.0*M
    
    print('busy ...')
    
    for t in range(1, maxtime):
        
        m,n,h,u=update_mnhu(Vm[:,t-1])
        
        gNa[:,t]=gNamax*(m**3)*h*HHComps
        gK[:,t] =gKmax *(n**4)*HHComps
        gIh[:,t]=gIhmax*u*IhComps
        
        curramp[t]=gain*(VmClamp[t]-Vm[injcomp,t-1])*10**(-12)+curramp[t-1]
        
        M_actual.setdiag(M.diagonal()+gNa[:,t]+gK[:,t]+gIh[:,t]+memcap/deltat)
        rightsideofeq=Vm[:,t-1]*memcap/deltat+currinj[:]*curramp[t]+ENa*gNa[:,t]+EK*gK[:,t]+EIh*gIh[:,t]
        
        Vm[:,t] = spsolve(M_actual,rightsideofeq)
    
    t=maxtime/2
    
    Rin=Vm[injcomp,t]/curramp[t]/(10**9) # GOhm
    
    Vm[:,:] = 1000.0*Vm # mV
    
    if plotswitch==1:
    
        plt.figure(figsize=(10,6))
        myt=np.arange(maxtime)*deltat
        plt.plot(myt,Vm[soma],linewidth=2,color='blue',label='Vm Soma [mV]')
        plt.plot(myt,Vm[dend],linewidth=2,color='green',label='Vm Dendrite [mV]')
        plt.plot(myt,VmClamp*1000.0,linewidth=2,color='black',label='Vm Clamp [mV]')
        plt.plot(myt,curramp*10**(12),linewidth=2,color='red',label='I inj [pA]')
        plt.xlabel('time [sec]')
        plt.legend(loc=2,frameon=False, fontsize=10)
    
    print('R input = ', Rin, ' [GOhm]')
    
    return Vm
    
# -----------------------------------------------------------------------------------------------------
    
def add_colorbar(Vm,Vm_min,Vm_max):
    
    myrange=(Vm_max-Vm_min) 
    setmyaxes(0.9,0.1,0.1,0.8)
    mybar=np.arange(500)
    mybar=mybar.reshape(100,5)
    plt.imshow(mybar,origin='lower')
    plt.xticks(np.arange(5),['','','','',''])
    plt.yticks(np.arange(6)*20.0,np.arange(6)*0.2*myrange+Vm_min)
    plt.text(-13,50,'Membrane Potential [mV]',verticalalignment='center',rotation=90)
    
def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
                   
def rot_tree(tree,rot_axis,angle):
    
    angle=angle/360.0*2.*np.pi
    
    if rot_axis==0:
        M=Rx(angle)
    if rot_axis==1:
        M=Ry(angle)
    if rot_axis==2:
        M=Rz(angle)
    
    nofcomps=tree.shape[0]
    
    new_tree=1.0*tree
    
    for i in range(nofcomps):
        
        new_tree[i,2:5]=M.dot(tree[i,2:5])
        
    return new_tree
    
def draw_Vm_tree(Vm,Vm_min,Vm_max,fname='T4a_swc_deposit/85.swc',dim1=2,dim2=3,sfac=10):

    plt.figure(figsize=(15,7)) 
    
    setmyaxes(0.1,0.1,0.75,0.8)
    
    tree=np.loadtxt(fname)
    tree[:,2:6]=0.01*tree[:,2:6]   
    n=tree.shape
    nofcomps=n[0]
    
    tree=rot_tree(tree,2,180)
    tree=rot_tree(tree,1,80)
    
    myrange=(Vm_max-Vm_min)   
    MyVm=limit(Vm,Vm_min,Vm_max)
    MyVm=(Vm-Vm_min)/myrange*255
    
    print(MyVm)
    
    for i in range(1,nofcomps,1):
        intensity=int(MyVm[i])
        if tree[i,6]!=-1:
            a=tree[i,0]-1
            b=tree[i,6]-1
            x1=tree[a,dim1]
            x2=tree[b,dim1]
            y1=tree[a,dim2]
            y2=tree[b,dim2]
            diam=tree[i,5]+tree[i-1,5]
            mylw=int(sfac*diam)
            plt.plot([x1,x2],[y1,y2],linewidth=mylw,color=plt.cm.rainbow(intensity))
            
    add_colorbar(Vm,Vm_min,Vm_max)
            
def find_comp_number(mycell,dim1=2,dim2=3):
    
    plt.scatter(mycell[:,dim1],mycell[:,dim2],c=mycell[:,0], cmap='viridis')
    plt.colorbar()
    
def create_colorswcfile(Vm,inputfname,outputfname,Vm_min,Vm_max):
    
    mycell=np.loadtxt(inputfname)
    
    # need to keep original dimensions (i.e.100 times as large)
    
    myrange=(Vm_max-Vm_min)   
    MyVm=rect(Vm,Vm_min)
    MyVm=ceil(MyVm,Vm_max)
    MyVm=(Vm-Vm_min)/myrange*255+20
    MyVm=MyVm.astype(int)
    
    mycell[:,1]=MyVm
    
    np.savetxt(outputfname,mycell.astype(int),fmt='%-5d')
    
def test_compmod(swcfile,injcomp=0,injcurr=10**(-11)):
       
    mycell=np.loadtxt(swcfile)
    mycell[:,2:6]=0.01*mycell[:,2:6]   
    interim=mycell.shape
    nofcomps=interim[0] 
    
    # passive membrane properties
    
    Rm=8000.0
    Ra=400.0
    Cm=0.6
    deltat=0.001
    
    # calculates the matrix
    
    M,memcap=calc_Conductance_M(mycell, Rm=Rm, Ra=Ra, Cm=Cm, deltat=deltat)
    
    currvec=np.zeros(nofcomps)
    currvec[injcomp]=injcurr
    
    Vm=spsolve(M,currvec)
    
    Vm=1000.0*Vm
    
    return Vm
    
#-------------------------
# IMPORTANT STUFF
#-------------------------
# row first: Y-Axis!
# savefig('fname')
        
# ASCII
# np.savetxt('fname', arrayname)
# arrayname=np.loadtxt('fname')  
        
# Binary File
# np.save('fname', arrayname)
# arrayname=np.load('fname')  
# a=scipy.misc.imread(filename)
        
# ax = plt.Subplot(fig, 111)      
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
        
# Cmaps: 
# Diverging: bwr, seismic, coolwarm
# linear:    hot, rainbow

    
    
          
    
        
        