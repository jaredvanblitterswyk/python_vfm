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

loadstrain = True

filename1 = 'ux_gauge_nodes_vfm_tutorial_2Dmodel.txt'
filename2 = 'uy_gauge_nodes_vfm_tutorial_2Dmodel.txt'

file1 = path_results+filename1
file2 = path_results+filename2

if loadstrain:
    filename3 = 'exx_gauge_nodes_vfm_tutorial_2Dmodel.txt'
    filename4 = 'eyy_gauge_nodes_vfm_tutorial_2Dmodel.txt'
    filename5 = 'gxy_gauge_nodes_vfm_tutorial_2Dmodel.txt'
    file3 = path_results+filename3
    file4 = path_results+filename4
    file5 = path_results+filename5

#%% Compute stiffness parameters from ANSYS input values
Exx = 40.07e09
Eyy = 10.07e09
Ezz = 10.07e09
PRxy = 0.30
PRyz = 0.25875
PRxz = 0.30
Gxy = 4.0e09
Gyz = Eyy/(2*(1+PRyz))
Gxz = 4.0e09

# calculate minor Poisson's ratios
PRyx = PRxy*Eyy/Exx
PRzy = PRyz*Ezz/Eyy
PRzx = PRxz*Ezz/Exx

print('PRyx: '+str(PRyx))
print('PRzy: '+str(PRzy))
print('PRzx: '+str(PRzx))

#S = np.matrix([[1/Exx, -PRxy/Exx, -PRzx/Ezz, 0, 0, 0],[-PRxy/Exx, 1/Eyy, -PRyz/Eyy, 0, 0, 0],[-PRzx/Ezz, -PRyz/Eyy, 1/Ezz, 0, 0, 0],[0, 0, 0, 1/Gyz, 0, 0], [0, 0, 0, 0, 1/Gxz,0],[0, 0, 0, 0, 0, 1/Gxy]])
S = np.matrix([[1/Exx, -PRyx/Eyy, -PRzx/Ezz, 0, 0, 0],
               [-PRxy/Exx, 1/Eyy, -PRzy/Ezz, 0, 0, 0],
               [-PRxz/Exx, -PRyz/Eyy, 1/Ezz, 0, 0, 0],
               [0, 0, 0, 1/Gyz, 0, 0], 
               [0, 0, 0, 0, 1/Gxz,0],
               [0, 0, 0, 0, 0, 1/Gxy]])
Q = inv(S)
'''
Q11 = Q[0,0]/1e09
Q12 = Q[0,1]/1e09
Q22 = Q[1,1]/1e09
Q33 = Q[2,2]/1e09
Q66 = Q[5,5]/1e09
'''
S2 = np.matrix([[1/Exx, -PRxy/Exx, 0],
               [-PRxy/Exx, 1/Eyy,  0],
               [0, 0, 1/Gxy]])

Q = inv(S2)

Q11 = Q[0,0]/1e09
Q12 = Q[0,1]/1e09
Q22 = Q[1,1]/1e09
Q66 = Q[2,2]/1e09

#print(Q2/1e09)

print('Calculated stiffness paramters:')
print('Q_{11}: '+str(round(Q11,2))+' (GPa)')
print('Q_{22}: '+str(round(Q22,2))+' (GPa)')
#print('Q_{33}: '+str(round(Q33,2))+' (GPa)')
print('Q_{12}: '+str(round(Q12,2))+' (GPa)')
print('Q_{66}: '+str(round(Q66,2))+' (GPa)')

#%% Load in data and interpolate to regular, sorted grid for processing
# ----------------------------------------------------------------
# load in data in dataframe
df1 = pd.read_csv(file1, delim_whitespace=True, skiprows = 1, header=None, nrows = 10000)
df1.columns = ["node", "X", "Y", "Z", "V"]

df2 = pd.read_csv(file2, delim_whitespace=True, skiprows = 1, header=None, nrows = 10000)
df2.columns = ["node", "X", "Y", "Z", "V"]

if loadstrain:
    df3 = pd.read_csv(file3, delim_whitespace=True, skiprows = 1, header=None, nrows = 10000)
    df3.columns = ["node", "X", "Y", "Z", "V"]
    
    df4 = pd.read_csv(file4, delim_whitespace=True, skiprows = 1, header=None, nrows = 10000)
    df4.columns = ["node", "X", "Y", "Z", "V"]
    
    df5 = pd.read_csv(file5, delim_whitespace=True, skiprows = 1, header=None, nrows = 10000)
    df5.columns = ["node", "X", "Y", "Z", "V"]

# -------------------------------------------------------------
# plot raw data to check upload
'''
plt.figure()
plt.scatter(df1.X,df1.Y,s = 5, c = df1.V)
plt.colorbar()
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('$u_x$ (mm)')
plt.tight_layout()

plt.figure()
plt.scatter(df2.X,df2.Y,s = 5, c = df2.V)
plt.colorbar()
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('$u_y$ (mm)')
plt.tight_layout()
'''
# --------------------------------------------------------------

# save data frame arrays to variables
x_raw = np.asarray(df1.X)
y_raw = np.asarray(df1.Y)
ux_raw = np.asarray(df1.V)
uy_raw = np.asarray(df2.V)

if loadstrain:
    exx_raw = np.asarray(df3.V)
    eyy_raw = np.asarray(df4.V)
    gxy_raw = np.asarray(df5.V)

# add functionality to read strain fields directly

# ----------------------------------------------------------------
# create structured coordinate arrays and interpolate variable onto structured coordinates
# ----------------------------------------------------------------

# find mesh spacing in x and y
dx = round(np.max(np.diff(sorted(x_raw))),5)
dy = round(np.max(np.diff(sorted(y_raw))),5)

# find dimensions based on exported coordinates
xmin = np.min(x_raw); xmax = np.max(x_raw)
ymin = np.min(y_raw); ymax = np.max(y_raw)

L = xmax-xmin
w = ymax-ymin

nx = int(L/dx) # number of points in x direction - set to current mesh density
ny = int(w/dx) # number of points in y direction

# create structured coordinates according to range of coordinates in data file
x_coords = np.linspace(xmin+5*dx/2, xmax-5*dx/2, num = nx-4)
y_coords = np.linspace(ymin+dy/2, ymax-dy/2, num = ny)

#x_coords = np.linspace(np.amin(x_raw), np.amax(x_raw), num = (nx-1))
#y_coords = np.linspace(np.amin(y_raw), np.amax(y_raw), num = (ny-1))

# create regular grid of coordinates for plotting

X,Y = np.meshgrid(x_coords,y_coords)

# interpolate raw values onto regular grid
ux_interp = griddata((x_raw,y_raw), ux_raw, (X, Y), method ='linear')
uy_interp = griddata((x_raw,y_raw), uy_raw, (X, Y), method ='linear')

if loadstrain:
    e_xx = griddata((x_raw,y_raw), exx_raw, (X, Y), method ='linear')
    e_yy = griddata((x_raw,y_raw), eyy_raw, (X, Y), method ='linear')
    g_xy = griddata((x_raw,y_raw), gxy_raw, (X, Y), method ='linear')

# ---------------------------------------------------------------------------
if not loadstrain:
    dx = dx
    dy = dy
    print(dx)
    # displacement gradients    
    dux_dy, dux_dx = np.gradient(ux_interp,dx,edge_order=2)
    duy_dy, duy_dx = np.gradient(uy_interp,dy,edge_order=2)
    
    # calculate strains (small strain theory)
    e_xx = dux_dx # x strain
    e_yy = duy_dy # y strain
    g_xy = dux_dy + duy_dx # shear strain

# shift origin of coordinates
X = X - (xmin + 2*dx)    
Y = Y - 0.5*w  
print('Average shear strain: '+ str(np.mean(np.mean(g_xy))) + ' (m/m)')
print('Average x strain: '+ str(np.mean(np.mean(e_xx))) + ' (m/m)')
print('Average y strain: '+ str(np.mean(np.mean(e_yy))) + ' (m/m)')
#%% Identify stiffness parameters using the virtual fields method
ny,nx = X.shape
N = nx*ny
F = 2128 # from ANSYS - reaction force
L = 0.026 # considering area encompassed by elements - strains expressed at element centroid
w = 0.02
t = 0.0023

# ----- VF 1 ------
u1s_1 = X*(L-X)*Y
u2s_1 = -0.5*L*X*X + 1/3*X*X*X

eps1s_1 = (L-2*X)*Y
eps2s_1 = np.zeros(X.shape)
eps6s_1 = np.zeros(X.shape)

# ----- VF 2 -----
u1s_2 = np.zeros(X.shape)
u2s_2 = X*(L-X)*Y

eps1s_2 = np.zeros(X.shape)
eps2s_2 = X*(L-X)
eps6s_2 = Y*(L-2*X)

# ----- VF 3 -----
u1s_3 = L/(2*m.pi)*np.sin(2*m.pi*X/L)
u2s_3 = np.zeros(X.shape)

eps1s_3 = np.cos(2*m.pi*X/L)
eps2s_3 = np.zeros(X.shape)
eps6s_3 = np.zeros(X.shape)

# ----- VF 4 -----
u1s_4 = np.zeros(X.shape)
u2s_4 = -X

eps1s_4 = np.zeros(X.shape)
eps2s_4 = np.zeros(X.shape)
eps6s_4 = -1*np.ones(X.shape)
#eps6s_4 = -3*X*X
# ----------------

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
    plt.clim(0.5*np.min(np.min(C)),0.5*np.max(np.max(C)))
    plt.tight_layout()

'''  
plot_pair_fields(X,Y,u1s_1,u2s_1,'$u_1^*$ (1)','$u_2^*$ (1)')
plot_pair_fields(X,Y,u1s_2,u2s_2,'$u_1^*$ (2)','$u_2^*$ (2)')
plot_pair_fields(X,Y,u1s_3,u2s_3,'$u_1^*$ (3)','$u_2^*$ (3)')
plot_pair_fields(X,Y,u1s_4,u2s_4,'$u_1^*$ (4)','$u_2^*$ (4)')

plot_pair_fields(X,Y,eps1s_1*e_xx,eps1s_1*e_yy,'$\epsilon_1^*(1)\epsilon_{xx}$','$\epsilon_1^*(1)\epsilon_{yy}$')
plot_pair_fields(X,Y,eps1s_3*e_xx,eps1s_3*e_yy,'$\epsilon_1^*(3)\epsilon_{xx}$','$\epsilon_1^*(3)\epsilon_{yy}$')
plot_single_var_contour(X,Y,eps6s_4*g_xy,'$\epsilon_6^*\gamma_{xy}$')

plotVF_strain(X,Y,eps1s_1,eps2s_1,eps6s_1,1)
plotVF_strain(X,Y,eps1s_2,eps2s_2,eps6s_2,2)
plotVF_strain(X,Y,eps1s_3,eps2s_3,eps6s_3,3)
plotVF_strain(X,Y,eps1s_4,eps2s_4,eps6s_4,4)
'''
# plot strain maps

plot_single_var_contour(X,Y,e_xx,'$\epsilon_{xx}$ (m/m)')
plot_single_var_contour(X,Y,e_yy,'$\epsilon_{yy}$ (m/m)')
plot_single_var_contour(X,Y,g_xy,'$\gamma_{xy}$ (m/m)')

