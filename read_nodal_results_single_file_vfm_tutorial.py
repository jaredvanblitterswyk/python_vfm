# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:22:50 2020

@author: Jared
"""
#%% Load in packages and define data locations
import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp2d, griddata
from numpy.linalg import inv

# read in and plot results from ansys simulation
base_dir = 'C:/Users/Jared/Documents/NIST-PREP/'
data_dir = 'simulation/unnotched_iosipescu_cfrp/'
path_data = os.path.join(base_dir, data_dir)

filename1 = 'vfm_tutorial_exx.csv'
filename2 = 'vfm_tutorial_eyy.csv'
filename3 = 'vfm_tutorial_gxy.csv'
filename4 = 'vfm_tutorial_X.csv'
filename5 = 'vfm_tutorial_Y.csv'
file1 = path_data+filename1
file2 = path_data+filename2
file3 = path_data+filename3
file4 = path_data+filename4
file5 = path_data+filename5

#%% Reference stiffness parameters
Q11 = 41
Q12 = 3.1
Q22 = 10.3
Q66 = 4

print('Calculated stiffness paramters:')
print('Q_{11}: '+str(round(Q11,2))+' (GPa)')
print('Q_{22}: '+str(round(Q22,2))+' (GPa)')
print('Q_{12}: '+str(round(Q12,2))+' (GPa)')
print('Q_{66}: '+str(Q66)+' (GPa)')

#%% Load in data 
# ----------------------------------------------------------------
e_xx = pd.read_csv(file1, delimiter=',', skiprows = 0, header=None)
e_yy = pd.read_csv(file2, delimiter=',', skiprows = 0, header=None)
g_xy = pd.read_csv(file3, delimiter=',', skiprows = 0, header=None)
X = pd.read_csv(file4, delimiter=',', skiprows = 0, header=None)
Y = pd.read_csv(file5, delimiter=',', skiprows = 0, header=None)

# convert to meters
e_xx = e_xx
e_yy = e_yy
g_xy = g_xy
X = X/1000
Y = Y/1000

ny,nx = X.shape
'''
# --------------------------------------------------------------
print('Average shear strain: '+ str(np.mean(np.mean(g_xy))) + ' (m/m)')
print('Average x strain: '+ str(np.mean(np.mean(e_xx))) + ' (m/m)')
print('Average y strain: '+ str(np.mean(np.mean(e_yy))) + ' (m/m)')
'''
#%% Identify stiffness parameters using the virtual fields method
N = nx*ny
F = 702 # from MATLAB data set provided with VFM textbook
t = 0.0023
L = 0.03
w = 0.02

X = X*1000
Y = Y*1000
L = L*1000

u1s_1 = X*(L-X)*Y
u2s_1 = -0.5*L*X*X + 1/3*X*X*X

u1s_2 = np.zeros(X.shape)
u2s_2 = X*(L-X)*Y

u1s_3 = L/(2*m.pi)*np.sin(2*m.pi*X/L)
u2s_3 = np.zeros(X.shape)

u1s_4 = np.zeros(X.shape)
u2s_4 = -X

eps1s_1 = (L-2*X)*Y
eps2s_1 = np.zeros(X.shape)
eps6s_1 = np.zeros(X.shape)

eps1s_2 = np.zeros(X.shape)
eps2s_2 = X*(L-X)
eps6s_2 = Y*(L-2*X)

eps1s_3 = np.cos(2*m.pi*X/L)
eps2s_3 = np.zeros(X.shape)
eps6s_3 = np.zeros(X.shape)

eps1s_4 = np.zeros(X.shape)
eps2s_4 = np.zeros(X.shape)
eps6s_4 = -1*np.ones(X.shape)

# define entries in matrices to solve
# eq'n 1
A11 = np.sum(np.sum(eps1s_1*e_xx))/N
A12 = np.sum(np.sum(eps1s_1*e_yy))/N
A13 = 0
A14 = 0
B1 = F*L*L/6/w/t # virtual work of external forces

# eq'n 2
A21 = 0
A22 = np.sum(np.sum(eps2s_2*e_xx))/N
A23 = np.sum(np.sum(eps2s_2*e_yy))/N
A24 = np.sum(np.sum(eps6s_2*g_xy))/N
B2 = 0 # no virtual work of external forces

# eq'n 3
A31 = np.sum(np.sum(eps1s_3*e_xx))/N
A32 = np.sum(np.sum(eps1s_3*e_yy))/N
A33 = 0
A34 = 0
B3 = 0 # no virtual work of external forces
# eq'n 4
A41 = 0
A42 = 0
A43 = 0
A44 = np.sum(np.sum(eps6s_4*g_xy))/N
B4 = F/(w*t) # virtual work of external forces

G_ident = F/(w*t)/np.sum(np.sum(eps6s_4*g_xy))*N/1e09
print('Check G:'+str(round(G_ident,3))+' (GPa)')
A = np.matrix([[A11, A12, A13, A14],[A21, A22, A23, A24],[A31, A32, A33, A34],[A41, A42, A43, A44]])
B = np.matrix([B1,B2,B3,B4])

Q = inv(A)*np.transpose(B) 
Q = np.asarray(Q)
print('-----------------------------------------')
print('Identified stiffness paramters:')
print('Q_{11}: '+str(round(Q[0][0]/1e09,2))+' (GPa)')
print('Q_{22}: '+str(round(Q[2][0]/1e09,2))+' (GPa)')
print('Q_{12}: '+str(round(Q[1][0]/1e09,2))+' (GPa)')
print('Q_{66}: '+str(round(Q[3][0]/1e09,2))+' (GPa)')

eQ11 = (round(Q[0][0]/1e09,2)-Q11)/Q11*100
eQ12 = (round(Q[1][0]/1e09,2)-Q12)/Q12*100
eQ22 = (round(Q[2][0]/1e09,2)-Q22)/Q22*100
eQ66 = (round(Q[3][0]/1e09,2)-Q66)/Q66*100

print('-----------------------------------------')
print('Error:')
print('Q_{11}: '+str(round(eQ11,1))+' (%)')
print('Q_{22}: '+str(round(eQ22,1))+' (%)')
print('Q_{12}: '+str(round(eQ12,1))+' (%)')
print('Q_{66}: '+str(round(eQ66,1))+' (%)')
#%% ----------------------------------------------------------------------------

# plot virtual fields

def plot_single_var_contour(X,Y,C,title_string,path,file):
    plt.figure(figsize=(5,3))
    f = plt.contourf(X,Y,C, 40, alpha=0.7)
    plt.colorbar()
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(title_string)
    plt.axis('scaled')
    plt.contour(X,Y,C, 40, alpha = 1.0, linewidths = 0.4)
    plt.clim(0.5*np.min(np.min(C)),0.5*np.max(np.max(C)))
    plt.tight_layout()
    plt.savefig(path+file, dpi=None, facecolor='w', edgecolor='w')
    
def plotVF_disp(X,Y,VF1,VF2,num,path,file):
    fig = plt.figure(figsize=(10,3))
    ax1 = plt.subplot(1, 2, 1)
    ax1 = plt.contourf(X,Y,VF1,40, cmap = 'viridis', alpha = 0.7)
    plt.colorbar()
    ax1s = plt.contour(X,Y,VF1, 40, alpha = 1.0, linewidths = 0.4)
    plt.title('$u_1^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    plt.clim(0.5*np.min(np.min(VF1)),0.5*np.max(np.max(VF1)))
    
    ax2 = plt.subplot(1, 2, 2)
    ax2 = plt.contourf(X,Y,VF2,40, cmap = 'viridis', alpha = 0.7)
    plt.colorbar()
    ax2s = plt.contour(X,Y,VF2, 40, alpha = 1.0, linewidths = 0.4)
    plt.title('$u_2^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    plt.clim(0.5*np.min(np.min(VF2)),0.5*np.max(np.max(VF2)))
    plt.savefig(path+file, dpi=None, facecolor='w', edgecolor='w')
      
def plotVF_strain(X,Y,VF1,VF2,VF3,num,path,file):
    fig = plt.figure(figsize=(15,3))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1 = plt.contourf(X,Y,VF1,40, cmap = 'viridis', alpha = 0.7)
    plt.colorbar()
    ax1s = plt.contour(X,Y,VF1,40, alpha = 1.0, linewidths = 0.4)
    plt.title('$\epsilon_1^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    
    ax2 = plt.subplot(1, 3, 2)
    ax2 = plt.contourf(X,Y,VF2,40, cmap = 'viridis', alpha = 0.7)
    plt.colorbar()
    ax1s = plt.contour(X,Y,VF2,40, alpha = 1.0, linewidths = 0.4)
    plt.title('$\epsilon_2^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    
    ax3 = plt.subplot(1, 3, 3)
    ax3 = plt.contourf(X,Y,VF3,40, cmap = 'viridis', alpha = 0.7)
    plt.colorbar()
    ax1s = plt.contour(X,Y,VF3, 40, alpha = 1.0, linewidths = 0.4)
    plt.title('$\epsilon_6^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    plt.savefig(path+file, dpi=None, facecolor='w', edgecolor='w')
    
    
results_dir = 'results_manual_vfm/'
path_results = os.path.join(path_data, results_dir)

fname = 'vfm_tutorial_manual_vfs_ustar_a'
plotVF_disp(X,Y,u1s_1,u2s_1,1,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_epsstar_a'
plotVF_strain(X,Y,eps1s_1,eps2s_1,eps6s_1,1,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_ustar_b'
plotVF_disp(X,Y,u1s_2,u2s_2,2,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_epsstar_b'
plotVF_strain(X,Y,eps1s_2,eps2s_2,eps6s_2,2,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_ustar_c'
plotVF_disp(X,Y,u1s_3,u2s_3,3,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_epsstar_c'
plotVF_strain(X,Y,eps1s_3,eps2s_3,eps6s_3,3,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_ustar_d'
plotVF_disp(X,Y,u1s_4,u2s_4,4,path_results,fname)
fname = 'vfm_tutorial_manual_vfs_epsstar_d'
plotVF_strain(X,Y,eps1s_4,eps2s_4,eps6s_4,4,path_results,fname)    

'''    
# plot strain maps

plot_single_var_contour(X,Y,e_xx,'$\epsilon_{xx}$ (m/m)')
plot_single_var_contour(X,Y,e_yy,'$\epsilon_{yy}$ (m/m)')
plot_single_var_contour(X,Y,g_xy,'$\gamma_{xy}$ (m/m)')
'''
