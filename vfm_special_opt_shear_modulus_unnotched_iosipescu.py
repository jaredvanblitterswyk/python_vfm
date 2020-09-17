# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Piecewise special optimised virtual fields for unnotched iosipescu test

This script loads in the fields provided in the VFM textbook by Pierron & Grediac
(see Sec. 13.4) and implements piecewise virtual fields to identify the 
in-plane shear modulus (Q66) for an orthotropic composite.

The virtual fields are built up using an FE type mesh of linear quadratic elements.
A similar approach could be implemented using polynomial virtual fields (Sec. 13.3), 
however, the low-order of the shape functions means that the mesh density 
can be increased with lower risk of stability issues.

Note that the virtual fields are 'optimum' in the sense of minimal sensitivity
to strain noise.

The eta parameters for each stiffness parameter are also calculated. Eta represents
the sensitivity of each parameter to measurement noise.  

National Institute of Standards and Technology
Securities Technology Group (643.10)

Author: Jared Van Blitterswyk (Intl. Assoc.)
Date last updated: 17 September 2020

"""
#%% Load in packages and define data locations
import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp2d, griddata
from numpy.linalg import inv
from f_piecewise_optimized_vfm_off_axis_tension import pw_opt_vfs_offaxis

# print header
print('======================================================================')
print('Piecewise special optimised virtual fields identification')
print('======================================================================')

# read in and plot results from ansys simulation
base_dir = 'C:/Users/Jared/Documents/NIST-PREP/'
#results_dir = 'simulation/off_axis_45_tension_gfrp/'
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

S = np.matrix([[1/Exx, -PRxy/Exx, 0],
               [-PRxy/Exx, 1/Eyy,  0],
               [0, 0, 1/Gxy]])

Q = inv(S)

Q11 = Q[0,0]/1e09
Q12 = Q[0,1]/1e09
Q22 = Q[1,1]/1e09
Q66 = Q[2,2]/1e09

print('----------------------------------------------------------------------')
print('Calculated stiffness paramters:')
print('Q11: '+str(round(Q11,2))+' GPa')
print('Q22: '+str(round(Q22,2))+' GPa')
print('Q12: '+str(round(Q12,2))+' GPa')
print('Q66: '+str(round(Q66,2))+' GPa')
print('----------------------------------------------------------------------')
#%% Load in data
e_xx = pd.read_csv(file1, delimiter=',', skiprows = 0, header=None)
e_yy = pd.read_csv(file2, delimiter=',', skiprows = 0, header=None)
g_xy = pd.read_csv(file3, delimiter=',', skiprows = 0, header=None)
X = pd.read_csv(file4, delimiter=',', skiprows = 0, header=None)
Y = pd.read_csv(file5, delimiter=',', skiprows = 0, header=None)

e_xx = np.asmatrix(e_xx)
e_yy = np.asmatrix(e_yy)
g_xy = np.asmatrix(g_xy)
X = np.asmatrix(X)
Y = np.asmatrix(Y)

# convert to meters
e_xx = e_xx
e_yy = e_yy
g_xy = g_xy
X = X/1000
Y = Y/1000

#%% Identify stiffness parameters using the virtual fields method
ny,nx = X.shape
N = nx*ny
F = -702 # from MATLAB data set provided with VFM textbook
t = 0.0023*1000
L = 0.03*1000
H = 0.02*1000

#Q = pw_opt_vfs_offaxis(X,Y,e_11,e_22,g_12,L,H,F,t)

#%%
# optimized virtual fields for off-axis shear test
m = 2 # number of elements in y direction
n = 3 # number of elements in x direction

num_ele = m*n # number of elements
num_nodes = (m+1)*(n+1) # number of nodes

L_ele = L/n # length of element along x direction
H_ele = H/m # height of element along y direction

num_points = N

# move coordinates to lower left of region of interest
Y = Y - np.min(np.min(Y)) + H/1000/ny/2

# reshape strain fields into vector of length 'num_points'
X = np.reshape(X,(num_points,1),-1)*1000
Y = np.reshape(Y,(num_points,1),-1)*1000
e11 = np.reshape(e_xx,(num_points,1),-1)
e22 = np.reshape(e_yy,(num_points,1),-1)
g12 = np.reshape(g_xy,(num_points,1),-1)

# =============================================================================
# Build virtual fields
# =============================================================================
# find node index associated with each element in the virtual mesh

ele_i = np.floor(X*n/L)+1; # element index along x direction affiliated with coordinates
ele_j = np.floor(Y*m/H)+1; # element index along y direction affiliated with coordinates

# define parametric coordinates for each element (-1 to 1) - used to define shape functions
xsi1 = 2*X/L_ele - ele_i*2+1
xsi2 = 2*Y/H_ele - ele_j*2+1

# define virtual displacements and strains based on quadrilateral elements - linear shape functions
# virtual strains
v0 = np.zeros((num_points,1))

Eps6_elem = 0.5/L_ele*np.column_stack((-(1-xsi2), (1-xsi2), (1+xsi2), -(1+xsi2)))

# virtual displacements
u0 = np.zeros((num_points,1))

u2_elem = 0.25*np.column_stack((np.multiply((1-xsi1),(1-xsi2)), np.multiply((1+xsi1),(1-xsi2)), np.multiply((1+xsi1),(1+xsi2)), np.multiply((1-xsi1),(1+xsi2))));

# define nodal connectivity for each element
# number of the first node of the related element for each data point
n1 = (ele_i-1)*(m+1)+ele_j; 
# number of the second node of the related element for each data point
n2 = ele_i*(m+1)+ele_j;
# number of the third node of the related element for each data point
n3 = ele_i*(m+1)+ele_j+1;
# number of the fourth node of the related element for each data point
n4 = (ele_i-1)*(m+1)+ele_j+1;

# -----------------------------------------------------------------------------
# construct speciality conditions and components of the Hessian matrix
# speciality conditions enforce direct identification of each stiffness parameters from VFs

# Bij will be used for the speciality conditions
# Hij will be used to compute the Hessian matrix

# Initialize variables
B66 = np.zeros((1,num_nodes))
#
H66 = np.zeros((num_nodes,num_nodes))
    
# Matrix of dofs affected by each data point
# [node1_dof1, node1_dof2, node2_dof1, node2_dof2, node3_dof1, node3_dof2, node4_dof1, node4_dof2]
assemble = np.column_stack((n1, n2, n3, n4)) # 2 dofs for each node

r,c = assemble.shape
#print('Size of assemble: ('+str(r)+', '+str(c)+')')

assemble = assemble.astype(int)
# loop through all points and populate matrices
     
assemble_py_ind = assemble - 1 # convert to python index
temp, n_dof_ele = assemble_py_ind[0,:].shape
for i in range(0,num_nodes):
    dof = i
    point_list = np.where(assemble_py_ind == dof)[0]
    # pre-allocate memory
    b66 = np.zeros((len(point_list),1))
    
    h66 = np.zeros((num_nodes,num_nodes,len(point_list)))
    
    count = 0
    for p in point_list:
        # find index of degree of freedom for that point
        ind_dof = np.where(assemble_py_ind[p][:] == dof)[1][0]
        b66[count,0] = Eps6_elem[p,ind_dof]*g12[p]*L*H/num_points
        
        for j in range(0,n_dof_ele):
            ind_dof2 = assemble_py_ind[p,j]              
            h66[dof,ind_dof2,count] = Eps6_elem[p,ind_dof]*Eps6_elem[p,j]
            
        count += 1
    # sum all contributions in time
    h66 = np.sum(h66,axis = 2)
    
    # assign to main vector
    B66[0,i] = np.sum(b66)
    
    # assign to main matrix
    H66[i,:] = h66[i,:]  
    
# -----------------------------------------------------------------------------
# Define Virtual Boundary Conditions
num_cond = m*n+m+1
Aconst = np.zeros((num_cond, num_nodes)) # there are 4*(n+1) boundary conditions - 2 dofs per node

for j in range(0,(m+1)): # set u1 = 0 at y = 0: (n+1) conditions
    Aconst[j, j] = 1
    
for i in range(1,(n+1)):
    for j in range(0,m): 
        Aconst[i*(n-1)+(j+1), i*(m+1)+j] = 1
        Aconst[i*(n-1)+(j+1), i*(m+1)+(j+1)] = -1 
    
# debugging - print table of dofs and constraints applied
#dof_label = np.linspace(1,2*num_nodes,2*num_nodes)
#dof_label.astype(int)
#print('Aconst matrix:')
#for item in dof_label:
#    print(item,end=" ")
#print('----------------------------------------------------------------------------------')
#for line in Aconst:
#    print(*line)  
    
# -----------------------------------------------------------------------------    
# Construct Z vector of zeros except for conditions of speciality
# Za is the vector used to find the virtual field to identify Q11
Z = np.zeros((1,num_nodes+num_cond));

spec_cond = np.matrix([1]) # speciality condition for Q66

# append to original Z vector of zeros
Z = np.hstack((Z,spec_cond))

# -----------------------------------------------------------------------------
# Build up the A matrix containing all constraints:
# boundary conditions and speciality and the B matrix containing zeros

A = np.row_stack((Aconst,B66)) # obtained from the constraint equation

rows,cols = A.shape
B = np.zeros([rows,rows])

# -----------------------------------------------------------------------------
# solve the system - optimization
# -----------------------------------------------------------------------------
Q = 1 # first guess for Q matrix

n_iter = 20; # maximum number of iterations
delta_lim = 0.001; # tollerance
delta = 10;

i = 1; # set counter
Qold = 1; # store value of Q for previous iteration

while i < n_iter and delta > delta_lim:

    # Calculate Hessian matrix 
    Hess = (L*H/num_points)**2*(Q**2*H66)
    
    #NOTE: to avoid the numerical "Warning: Matrix is close to singular or
    #      badly scaled" matrix Opt can be scaled with the parameter corr.
    #      It does not change the results of the optimization.
    #      To not use put corr=1;
    
    corr = np.max(np.max(A))/np.max(np.max(Hess)) # error with this line - need to fix
    #corr = 1
    
    # build optimization matrix      
    upper_lr = np.column_stack((Hess/2*corr,np.transpose(A)*corr))
    lower_lr = np.column_stack((A,B))
    
    
    OptM = np.row_stack((upper_lr,lower_lr))
    if i == 1:
        tempOptM = OptM
    
    rows,cols = OptM.shape

    #print('Size of OptM (rows, cols): '+str(rows)+', '+str(cols))
    #print('OptM:')
    
    #for line in OptM:
    #    print(*line)
    
    # Vector containing the polynomial coefficients and the Lagrange multipliers
    Yd = inv(OptM)*np.transpose(Z) # for Q66
    
    # Remove the Lagrange multipliers from the Y vectors because they are of no interest
    Yd = Yd[0:num_nodes]
    
    # Calculating Q66 from the fourth optimized virtual field
    Q = float(Yd[num_nodes-1]*F/t)
    
    #print('Q66:'+str(Q66))
    
    delta=(Qold-Q)**2/Q**2
    #print(delta)
    i=i+1
    Qold = Q
    
# -----------------------------------------------------------------------------
# Final result
# -----------------------------------------------------------------------------
Hess = (L*H/num_points)**2*(Q**2*H66)    

# calculate eta parameters (sensitivity to noise) 
eta66 = np.sqrt(float(np.dot(np.transpose(Yd)*Hess,Yd)))   

# print results
print('----------------------------------------------------------------------')    
print('Identified stiffness paramters:')    
print('Q66: '+str(round(Q/1e03,2))+ ' GPa')
print('----------------------------------------------------------------------')
print('Normalized sensitivity to strain noise (eta_ij/Q_ij):')    
print('eta66/Q66: '+str(round(eta66/Q,2)))
print('----------------------------------------------------------------------')
print('Nodes: '+str(num_nodes)+', Elements: '+ str(num_ele))

#%% ----------------------------------------------------------------------------
# Reconstruct virtual fields for illustration
# u1v: virtual displacement fields in x direction (a, ... d) for (Q11, ... Q66)
# u2v: virtual displacement fields in y direction (a, ... d) for (Q11, ... Q66)
# Eps1: virtual x strain fields (a, ... d) for (Q11, ... Q66)
# Eps2: virtual y strain fields (a, ... d) for (Q11, ... Q66)
# Eps6: virtual xy strain fields (a, ... d) for (Q11, ... Q66)

# pre-allocate memory
u1vd = np.zeros((num_points,1))
u2vd = np.zeros((num_points,1))
eps6vd = np.zeros((num_points,1))

for k in range(0,num_points):
    # degrees of freedom associated with point k
    assemble_k = assemble_py_ind[k,:]
    # define zero vector where contribution at relevant dofs will be stored for each point
    var_dofs_k = np.zeros((1,num_nodes)) 
    
    # Virtual displacement fields, 2 component
    var_dofs_k[0,assemble_k] = u2_elem[k,:]
    
    u2vd[k] = np.dot(var_dofs_k,Yd) # For Q66
    
    var_dofs_k = np.zeros((1,num_nodes)) # reset before next variable
        
    # Virtual strain fields, 6 component
    var_dofs_k[0,assemble_k] = Eps6_elem[k,:]
    
    eps6vd[k] = np.dot(var_dofs_k,Yd) # For Q66

# Reshape data into matrix form
u1vd = np.reshape(u1vd,(ny,nx),-1)
u2vd = np.reshape(u2vd,(ny,nx),-1)
eps6vd = np.reshape(eps6vd,(ny,nx),-1)

#%% Plot fields (virtual and mechanical)
X = np.reshape(np.asarray(X),(ny,nx),-1)
Y = np.reshape(np.asarray(Y),(ny,nx),-1)

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
   

results_dir = 'results_pw_optim_vfm/'
path_results = os.path.join(path_data, results_dir)

fname = 'vfm_tutorial_pw_optim_vfs_ustar_d'
plotVF_disp(X,Y,u1vd,u2vd,4,path_results,fname)
fname = 'vfm_tutorial_pw_optim_vfs_epsstar_d'
plot_single_var_contour(X,Y,eps6vd,'$\epsilon_{6}^{*}$ (m/m)',path_results,fname)

fname = 'vfm_tutorial_exx'
plot_single_var_contour(X,Y,e_xx,'$\epsilon_{xx}$ (m/m)',path_results,fname)
fname = 'vfm_tutorial_eyy'
plot_single_var_contour(X,Y,e_yy,'$\epsilon_{yy}$ (m/m)',path_results,fname)
fname = 'vfm_tutorial_gxy'
plot_single_var_contour(X,Y,g_xy,'$\gamma_{xy}$ (m/m)',path_results,fname)


                     