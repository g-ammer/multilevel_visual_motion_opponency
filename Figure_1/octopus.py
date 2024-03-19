import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import os
import scipy.io
from scipy import stats




# Define Function for Cartesian/Polar Vector Transformation

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi = phi + 2*np.pi
    return(rho, phi)



def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y




# Define Function for calculating Motion Opponency index

def MOI(rot_rad, radius):

# Find indices of Depolarizing/Hyperpolarizing Responses
    D_idx = np.asarray(np.where(radius>0))[0,:]
    H_idx = np.asarray(np.where(radius<0))[0,:]

#Calculate Vectors and Magnitudes for Depolarization/Hyperpolarization
    D_cart = np.array(pol2cart(radius[D_idx],rot_rad[D_idx]))
    D_vect = np.sum(D_cart,1)
    D_vect_pol = cart2pol(D_vect[0], D_vect[1])
    print ('D_magn =' + str(D_vect_pol[0]) + '   D_deg = ' + str(np.degrees(D_vect_pol[1])))

    H_cart = np.array(pol2cart(radius[H_idx],rot_rad[H_idx]))
    H_vect = np.sum(H_cart,1)
    H_vect_pol = cart2pol(H_vect[0], H_vect[1])
    print ('H_magn =' + str(H_vect_pol[0]) + '   H_deg = ' + str(np.degrees(H_vect_pol[1])))

# Calculate MOI (Motion Opponency index)
    if D_vect_pol < H_vect_pol:
        MOI = (np.cos(D_vect_pol[1]-H_vect_pol[1]))*(D_vect_pol[0]/H_vect_pol[0])
    else:
        MOI = (np.cos(D_vect_pol[1]-H_vect_pol[1]))*(H_vect_pol[0]/D_vect_pol[0])

    print ('MOI = ' + str(MOI))
    
    return MOI




# Define Function for calculating the Direction Selectivity Index LDir
    
def LDir(rot_rad, radius):
    R_cart = np.array(pol2cart(radius,rot_rad))
    R_vect = np.sum(R_cart, 1)
    R_vect_pol = cart2pol(R_vect[0], R_vect[1])
    R_indiv_magn = np.sum(np.sqrt(R_cart[0]**2 + R_cart[1]**2))   
    
    R_deg = (np.degrees(R_vect_pol[1]))
    LDir = R_vect_pol[0] / R_indiv_magn
    
    print ('R_magn =' + str(R_vect_pol[0]) + '   R_deg = ' + str(np.degrees(R_vect_pol[1])))
    print ('R_indiv_magn = ' + str(R_indiv_magn))
    print ('LDir = ' + str(LDir))
    
    return LDir

def R_deg(rot_rad, radius):
    R_cart = np.array(pol2cart(radius,rot_rad))
    R_vect = np.sum(R_cart, 1)
    R_vect_pol = cart2pol(R_vect[0], R_vect[1])
    R_indiv_magn = np.sum(np.sqrt(R_cart[0]**2 + R_cart[1]**2))   
    
    R_deg = (np.degrees(R_vect_pol[1]))
    LDir = R_vect_pol[0] / R_indiv_magn
    
    #print 'R_magn =' + str(R_vect_pol[0]) + '   R_deg = ' + str(np.degrees(R_vect_pol[1]))
    #print 'R_indiv_magn = ' + str(R_indiv_magn)
    #print 'LDir = ' + str(LDir)
    
    return R_deg

def R_magn(rot_rad, radius):
    R_cart = np.array(pol2cart(radius,rot_rad))
    R_vect = np.sum(R_cart, 1)
    R_vect_pol = cart2pol(R_vect[0], R_vect[1])
    R_indiv_magn = np.sum(np.sqrt(R_cart[0]**2 + R_cart[1]**2))   
    
    R_deg = (np.degrees(R_vect_pol[1]))
    LDir = R_vect_pol[0] / R_indiv_magn
    
    #print 'R_magn =' + str(R_vect_pol[0]) + '   R_deg = ' + str(np.degrees(R_vect_pol[1]))
    #print 'R_indiv_magn = ' + str(R_indiv_magn)
    #print 'LDir = ' + str(LDir)
    
    return R_vect_pol[0]
