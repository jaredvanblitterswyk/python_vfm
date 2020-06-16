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
results_dir = 'simulation/unnotched_iosipescu_cfrp/'
path_results = os.path.join(base_dir, results_dir)

filename1 = 'vfm_tutorial_exx.csv'
filename2 = 'vfm_tutorial_eyy.csv'
filename3 = 'vfm_tutorial_gxy.csv'
filename4 = 'vfm_tutorial_X.csv'
filename5 = 'vfm_tutorial_Y.csv'
file1 = path_results+filename1
file2 = path_results+filename2
file3 = path_results+filename3
file4 = path_results+filename4
file5 = path_results+filename5

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
# --------------------------------------------------------------
print('Average shear strain: '+ str(np.mean(np.mean(g_xy))) + ' (m/m)')
print('Average x strain: '+ str(np.mean(np.mean(e_xx))) + ' (m/m)')
print('Average y strain: '+ str(np.mean(np.mean(e_yy))) + ' (m/m)')
#%% Identify stiffness parameters using the virtual fields method
N = nx*ny
F = 702 # from MATLAB data set provided with VFM textbook
t = 0.0023
L = 0.03
w = 0.02

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
print('Q_{11}: '+str(Q[0][0]/1e09)+' (GPa)')
print('Q_{22}: '+str(Q[2][0]/1e09)+' (GPa)')
print('Q_{12}: '+str(Q[1][0]/1e09)+' (GPa)')
print('Q_{66}: '+str(Q[3][0]/1e09)+' (GPa)')

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

# VF1
def plot_pair_fields(X,Y,VF1,VF2,title_string1,title_string2):
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1 = plt.pcolor(X,Y,VF1)
    plt.title(title_string1)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    cbaxes = fig.add_axes([0.75, 0.55, 0.03, 0.4])  # [left bottom width height]
    cb = plt.colorbar(ax1, cax = cbaxes) 
    
    ax2 = plt.subplot(2, 1, 2)
    ax2 = plt.pcolor(X,Y,VF2)
    plt.title(title_string2)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    plt.tight_layout()
    cbaxes = fig.add_axes([0.75, 0.05, 0.03, 0.4])  # [left bottom width height]
    cb = plt.colorbar(ax2, cax = cbaxes) 
       
def plotVF_strain(X,Y,VF1,VF2,VF3,num):
    fig = plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    ax1 = plt.pcolor(X,Y,VF1)
    plt.title('$\epsilon_1^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    cbaxes = fig.add_axes([0.75, 0.7, 0.03, 0.2])  # [left bottom width height]
    cb = plt.colorbar(ax1, cax = cbaxes) 
    
    ax2 = plt.subplot(3, 1, 2)
    ax2 = plt.pcolor(X,Y,VF2)
    plt.title('$\epsilon_2^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    plt.tight_layout()
    cbaxes = fig.add_axes([0.75, 0.35, 0.03, 0.2])  # [left bottom width height]
    cb = plt.colorbar(ax2, cax = cbaxes)    
    
    ax3 = plt.subplot(3, 1, 3)
    ax3 = plt.pcolor(X,Y,VF3)
    plt.title('$\epsilon_6^*$'+'$^{('+str(num)+')}$')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('scaled')
    plt.tight_layout()
    cbaxes = fig.add_axes([0.75, 0.05, 0.03, 0.2])  # [left bottom width height]
    cb = plt.colorbar(ax3, cax = cbaxes)     

def plot_single_var_contour(X,Y,C,title_string):
    plt.figure()
    f = plt.contourf(X,Y,C, 40, alpha=0.7)
    plt.colorbar()
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(title_string)
    plt.axis('scaled')
    plt.contour(X,Y,C, 40, alpha = 1.0, linewidths = 0.4)
    plt.clim(np.min(np.min(C)),np.max(np.max(C)))
    plt.tight_layout()

'''    
plot_pair_fields(X,Y,u1s_1,u2s_1,'$u_1^*$ (1)','$u_2^*$ (1)')
plot_pair_fields(X,Y,u1s_2,u2s_2,'$u_1^*$ (2)','$u_2^*$ (2)')
plot_pair_fields(X,Y,u1s_3,u2s_3,'$u_1^*$ (3)','$u_2^*$ (3)')
plot_pair_fields(X,Y,u1s_4,u2s_4,'$u_1^*$ (4)','$u_2^*$ (4)')
'''
plot_pair_fields(X,Y,eps1s_1*e_xx,eps1s_1*e_yy,'$\epsilon_1^*(1)\epsilon_{xx}$','$\epsilon_1^*(1)\epsilon_{yy}$')
plot_pair_fields(X,Y,eps1s_3*e_xx,eps1s_3*e_yy,'$\epsilon_1^*(3)\epsilon_{xx}$','$\epsilon_1^*(3)\epsilon_{yy}$')
plot_single_var_contour(X,Y,eps6s_4*g_xy,'$\epsilon_6^*\gamma_{xy}$')

plotVF_strain(X,Y,eps1s_1,eps2s_1,eps6s_1,1)
plotVF_strain(X,Y,eps1s_2,eps2s_2,eps6s_2,2)
plotVF_strain(X,Y,eps1s_3,eps2s_3,eps6s_3,3)
plotVF_strain(X,Y,eps1s_4,eps2s_4,eps6s_4,4)

# plot strain maps

plot_single_var_contour(X,Y,e_xx,'$\epsilon_{xx}$ (m/m)')
plot_single_var_contour(X,Y,e_yy,'$\epsilon_{yy}$ (m/m)')
plot_single_var_contour(X,Y,g_xy,'$\gamma_{xy}$ (m/m)')

