# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Piecewise special optimised virtual fields for unnotched iosipescu test

This script loads in the fields provided in the VFM textbook by Pierron & Grediac
(see Sec. 13.4) and implements piecewise virtual fields to identify the 
4 in-plane stiffness parameters (Q11, Q12, Q22, Q66) for an orthotropic composite.

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
Date last updated: 16 June 2020

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

# read in and plot results from ansys simulation
base_dir = 'C:/Users/Jared/Documents/NIST-PREP/'
#results_dir = 'simulation/off_axis_45_tension_gfrp/'
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


'''
loadstrain = True
filename1 = 'ux_gauge_nodes.txt'
filename2 = 'uy_gauge_nodes.txt'

file1 = path_results+filename1
file2 = path_results+filename2

if loadstrain:
    filename3 = 'e11_gauge_nodes.txt'
    filename4 = 'e22_gauge_nodes.txt'
    filename5 = 'g12_gauge_nodes.txt'
    file3 = path_results+filename3
    file4 = path_results+filename4
    file5 = path_results+filename5
'''
# print header
print('======================================================================')
print('Piecewise special optimised virtual fields identification')
print('======================================================================')
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

#print('PRyx: '+str(PRyx))
#print('PRzy: '+str(PRzy))
#print('PRzx: '+str(PRzx))

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
print('----------------------------------------------------------------------')
print('Calculated stiffness paramters:')
print('Q11: '+str(round(Q11,2))+' GPa')
print('Q22: '+str(round(Q22,2))+' GPa')
#print('Q_{33}: '+str(round(Q33,2))+' (GPa)')
print('Q12: '+str(round(Q12,2))+' GPa')
print('Q66: '+str(round(Q66,2))+' GPa')
print('----------------------------------------------------------------------')
#%% Load in data and interpolate to regular, sorted grid for processing
# ----------------------------------------------------------------
'''
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

# save data frame arrays to variables
x_raw = np.asarray(df1.X)
y_raw = np.asarray(df1.Y)
ux_raw = np.asarray(df1.V)
uy_raw = np.asarray(df2.V)

if loadstrain:
    e11_raw = np.asarray(df3.V)
    e22_raw = np.asarray(df4.V)
    g12_raw = np.asarray(df5.V)

# ----------------------------------------------------------------
# create structured coordinate arrays and interpolate variable onto structured coordinates
# ----------------------------------------------------------------

# find mesh spacing in x and y
dx = round(np.max(np.diff(sorted(x_raw))),4)
dy = round(np.max(np.diff(sorted(y_raw))),4)

# find dimensions based on exported coordinates
xmin = np.min(x_raw); xmax = np.max(x_raw)
ymin = round(np.min(y_raw),3); ymax = round(np.max(y_raw),3)

L = xmax-xmin
H = ymax-ymin

nx = int(L/dx) # number of points in x direction - set to current mesh density
ny = int(H/dx) # number of points in y direction

# create structured coordinates according to range of coordinates in data file
x_coords = np.linspace(xmin+dx/2, xmax-dx/2, num = nx)
y_coords = np.linspace(ymin+dy/2, ymax-dy/2, num = ny)

# create regular grid of coordinates for plotting

X,Y = np.meshgrid(x_coords,y_coords)

# interpolate raw values onto regular grid
ux_interp = griddata((x_raw,y_raw), ux_raw, (X, Y), method ='linear')
uy_interp = griddata((x_raw,y_raw), uy_raw, (X, Y), method ='linear')

if loadstrain:
    e_11 = griddata((x_raw,y_raw), e11_raw, (X, Y), method ='linear')
    e_22 = griddata((x_raw,y_raw), e22_raw, (X, Y), method ='linear')
    g_12 = griddata((x_raw,y_raw), g12_raw, (X, Y), method ='linear')

Y = Y- ymin
# ---------------------------------------------------------------------------
if not loadstrain: # need to update and include rotation to material coords
    dx = dx
    dy = dy
    print(dx)
    # displacement gradients    
    dux_dy, dux_dx = np.gradient(ux_interp,dx,edge_order=1)
    duy_dy, duy_dx = np.gradient(uy_interp,dy,edge_order=1)
    
    # calculate strains (small strain theory)
    e_xx = dux_dx # x strain
    e_yy = duy_dy # y strain
    g_xy = dux_dy + duy_dx # shear strain
'''

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

#print('Average shear strain: '+ str(np.mean(np.mean(g_12))) + ' (m/m)')
#print('Average x strain: '+ str(np.mean(np.mean(e_11))) + ' (m/m)')
#print('Average y strain: '+ str(np.mean(np.mean(e_22))) + ' (m/m)')
#%% Identify stiffness parameters using the virtual fields method
ny,nx = X.shape
N = nx*ny
'''
F = 1900 # from ANSYS - reaction force
H = 0.075*1000 # considering area encompassed by elements - strains expressed at element centroid
L = 0.025*1000 # width of specimen
t = 0.00079*1000 # thickness of specimen
'''
F = -702 # from MATLAB data set provided with VFM textbook
t = 0.0023*1000
L = 0.03*1000
H = 0.02*1000

#Q = pw_opt_vfs_offaxis(X,Y,e_11,e_22,g_12,L,H,F,t)
'''
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

print('-----------------------------------------')
#print('Matrix condition number (close to zero is well-defined):' + str(round(np.linalg.cond(A),2)))
'''
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

Eps1_elem = 0.5/L_ele*np.column_stack((-(1-xsi2), v0, (1-xsi2), v0, (1+xsi2), v0, -(1+xsi2), v0));

Eps2_elem = 0.5/H_ele*np.column_stack((v0, -(1-xsi1), v0, -(1+xsi1), v0, (1+xsi1), v0, (1-xsi1)));

Eps6_elem = 0.5*np.column_stack((-(1-xsi1)/H_ele, -(1-xsi2)/L_ele, -(1+xsi1)/H_ele, (1-xsi2)/L_ele, (1+xsi1)/H_ele, (1+xsi2)/L_ele, (1-xsi1)/H_ele, -(1+xsi2)/L_ele))

# virtual displacements
u0 = np.zeros((num_points,1));

u1_elem = 0.25*np.column_stack((np.multiply((1-xsi1),(1-xsi2)), u0, np.multiply((1+xsi1),(1-xsi2)), u0, np.multiply((1+xsi1),(1+xsi2)), u0, np.multiply((1-xsi1),(1+xsi2)), u0))

u2_elem = 0.25*np.column_stack((u0, np.multiply((1-xsi1),(1-xsi2)), u0, np.multiply((1+xsi1),(1-xsi2)), u0, np.multiply((1+xsi1),(1+xsi2)), u0, np.multiply((1-xsi1),(1+xsi2))));

r,c = u1_elem.shape
#print('Size of u1_elem: ('+str(r)+', '+str(c)+')')
r,c = u2_elem.shape
#print('Size of u2_elem: ('+str(r)+', '+str(c)+')')

r,c = Eps1_elem.shape
#print('Size of Eps1_elem: ('+str(r)+', '+str(c)+')')
r,c = Eps2_elem.shape
#print('Size of Eps2_elem: ('+str(r)+', '+str(c)+')')
r,c = Eps6_elem.shape
#print('Size of Eps6_elem: ('+str(r)+', '+str(c)+')')

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
B11 = np.zeros((1,2*num_nodes))
B22 = np.zeros((1,2*num_nodes))
B12 = np.zeros((1,2*num_nodes))
B66 = np.zeros((1,2*num_nodes))

#
H11 = np.zeros((2*num_nodes,2*num_nodes))
H22 = np.zeros((2*num_nodes,2*num_nodes))
H12 = np.zeros((2*num_nodes,2*num_nodes))
H66 = np.zeros((2*num_nodes,2*num_nodes))


r,c = B11.shape
#print('Size of B11,B12,B22,B66: ('+str(r)+', '+str(c)+')')
r,c = H11.shape
#print('Size of H11,H12,H22,H66: ('+str(r)+', '+str(c)+')')    
# Matrix of dofs affected by each data point
# [node1_dof1, node1_dof2, node2_dof1, node2_dof2, node3_dof1, node3_dof2, node4_dof1, node4_dof2]
assemble = np.column_stack((n1*2-1, n1*2, n2*2-1, n2*2, n3*2-1, n3*2, n4*2-1, n4*2)) # 2 dofs for each node

r,c = assemble.shape
#print('Size of assemble: ('+str(r)+', '+str(c)+')')

assemble = assemble.astype(int)
# loop through all points and populate matrices
     
assemble_py_ind = assemble - 1 # convert to python index
temp, n_dof_ele = assemble_py_ind[0,:].shape
for i in range(0,2*num_nodes):
    dof = i
    point_list = np.where(assemble_py_ind == dof)[0]
    # pre-allocate memory
    b11 = np.zeros((len(point_list),1))
    b22 = np.zeros((len(point_list),1))
    b12 = np.zeros((len(point_list),1))
    b66 = np.zeros((len(point_list),1))
    
    h11 = np.zeros((2*num_nodes,2*num_nodes,len(point_list)))
    h22 = np.zeros((2*num_nodes,2*num_nodes,len(point_list)))
    h12 = np.zeros((2*num_nodes,2*num_nodes,len(point_list)))
    h66 = np.zeros((2*num_nodes,2*num_nodes,len(point_list)))
    
    count = 0
    for p in point_list:
        # find index of degree of freedom for that point
        ind_dof = np.where(assemble_py_ind[p][:] == dof)[1][0]
        b11[count,0] = Eps1_elem[p,ind_dof]*e11[p]*L*H/num_points
        b22[count,0] = Eps2_elem[p,ind_dof]*e22[p]*L*H/num_points
        b12[count,0] = (Eps1_elem[p,ind_dof]*e22[p]+Eps2_elem[p,ind_dof]*e11[p])*L*H/num_points
        b66[count,0] = Eps6_elem[p,ind_dof]*g12[p]*L*H/num_points
        
        for j in range(0,n_dof_ele):
            ind_dof2 = assemble_py_ind[p,j]              
            h11[dof,ind_dof2,count] = Eps1_elem[p,ind_dof]*Eps1_elem[p,j]
            h22[dof,ind_dof2,count] = Eps2_elem[p,ind_dof]*Eps2_elem[p,j]
            h12[dof,ind_dof2,count] = Eps1_elem[p,ind_dof]*Eps2_elem[p,j]
            h66[dof,ind_dof2,count] = Eps6_elem[p,ind_dof]*Eps6_elem[p,j]
            
        count += 1
    # sum all contributions in time
    h11 = np.sum(h11,axis = 2)
    h22 = np.sum(h22,axis = 2)
    h12 = np.sum(h12,axis = 2)
    h66 = np.sum(h66,axis = 2)
    
    # assign to main vector
    B11[0,i] = np.sum(b11)
    B22[0,i] = np.sum(b22)
    B12[0,i] = np.sum(b12)
    B66[0,i] = np.sum(b66)
    
    # assign to main matrix
    H11[i,:] = h11[i,:]
    H22[i,:] = h22[i,:] 
    H12[i,:] = h12[i,:] 
    H66[i,:] = h66[i,:]  
    
# -----------------------------------------------------------------------------
# Define Virtual Boundary Conditions
'''
num_cond = 3*(n+1)+n
Aconst = np.zeros((num_cond, 2*num_nodes)) # there are 4*(n+1) boundary conditions - 2 dofs per node

for j in range(0,(n+1)): # set u1 = 0 at y = 0: (n+1) conditions
    Aconst[j, 2*j*(m+1)] = 1
    
for j in range(0,(n+1)): # set u2 = 0 at y=0: n+1 conditions
    Aconst[j +(n+1), 2*j*(m+1)+1] = 1
   
for j in range(0,(n+1)): # set u1 = 0 at y=H: n+1 conditions
    Aconst[j + 2*(n+1), 2*(j+1)*(m+1)-2] = 1
    
for j in range(0,n): # u2(y=H) = const. for n = 2, m = 3: u2(8) - u2(4) = 0: n conditions
    Aconst[j + 3*(n+1), 2*(j+1)*(m+1)-1] = 1
    Aconst[j + 3*(n+1), 2*(j+1)*(m+1)+2*(m+1)-1] = -1     
'''
num_cond = 4*m+3
Aconst = np.zeros((num_cond, 2*num_nodes)) # there are 4*(n+1) boundary conditions - 2 dofs per node

for j in range(0,2*(m+1)): # set u1 = 0 at y = 0: (n+1) conditions
    Aconst[j, j] = 1
     
for j in range(0,(m+1)): # set u1 = 0 at y=H: n+1 conditions
    Aconst[j + 2*(m+1), 2*(num_nodes-n+j)] = 1
    
for j in range(0,m): # u2(y=H) = const. for n = 2, m = 3: u2(8) - u2(4) = 0: n conditions
    Aconst[j + 3*(m+1), 2*(num_nodes-m+j)-1] = 1
    Aconst[j + 3*(m+1), 2*(num_nodes-m+j)+1] = -1  
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
Za = np.zeros((1,2*num_nodes+num_cond));
Zb=Za; Zc=Za; Zd=Za;

spec_cond1 = np.matrix([1,0,0,0]) # speciality condition for Q11
spec_cond2 = np.matrix([0,1,0,0]) # speciality condition for Q22
spec_cond3 = np.matrix([0,0,1,0]) # speciality condition for Q12
spec_cond4 = np.matrix([0,0,0,1]) # speciality condition for Q66

# append to original Z vector of zeros
Za = np.hstack((Za,spec_cond1))
Zb = np.hstack((Zb,spec_cond2))
Zc = np.hstack((Zc,spec_cond3))
Zd = np.hstack((Zd,spec_cond4))

r,c = Za.shape
#print('Size of Za,Zb,Zc,Zd: ('+str(r)+', '+str(c)+')')

# -----------------------------------------------------------------------------
# Build up the A matrix containing all constraints:
# boundary conditions and speciality and the B matrix containing zeros

A = np.row_stack((Aconst,B11,B22,B12,B66)) # obtained from the constraint equation

rows,cols = A.shape

#print('Size of A (rows, cols): '+str(rows)+', '+str(cols))

B = np.zeros([rows,rows])

# -----------------------------------------------------------------------------
# solve the system - optimization
# -----------------------------------------------------------------------------
Q = np.matrix([1, 1, 1, 1]) # first guess for Q matrix

n_iter = 20; # maximum number of iterations
delta_lim = 0.001; # tollerance
delta = 10;

i = 1; # set counter
Qold = np.matrix([1,1,1,1]); # store value of Q for previous iteration

while i < n_iter and delta > delta_lim:

    # Calculate Hessian matrix 
    Hess =(L*H/num_points)**2*((Q[0,0]**2+Q[0,2]**2)*H11 + (Q[0,1]**2+Q[0,2]**2)*H22 + Q[0,3]**2*H66 + 2*(Q[0,0] + Q[0,1])*Q[0,2]*H12)
    

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
    Ya = inv(OptM)*np.transpose(Za) # for Q11
    Yb = inv(OptM)*np.transpose(Zb) # for Q22
    Yc = inv(OptM)*np.transpose(Zc) # for Q12
    Yd = inv(OptM)*np.transpose(Zd) # for Q66
    
    # Remove the Lagrange multipliers from the Y vectors because they are of no interest
    Ya = Ya[0:2*num_nodes]
    Yb = Yb[0:2*num_nodes]
    Yc = Yc[0:2*num_nodes]
    Yd = Yd[0:2*num_nodes]
    
    # Calculating Q11 from the first optimized virtual field
    Q11 = (Ya[2*num_nodes-1]*F/t)
    # Calculating Q22 from the second optimized virtual field
    Q22 = (Yb[2*num_nodes-1]*F/t)
    # Calculating Q12 from the third optimized virtual field
    Q12 = (Yc[2*num_nodes-1]*F/t)
    # Calculating Q66 from the fourth optimized virtual field
    Q66 = (Yd[2*num_nodes-1]*F/t)
    
    #print('Q11:'+str(Q11))
    #print('Q22:'+str(Q22))
    #print('Q12:'+str(Q12))
    #print('Q66:'+str(Q66))
    
    Q[0,0] = Q11
    Q[0,1] = Q22
    Q[0,2] = Q12
    Q[0,3] = Q66
    
    delta=np.sum(np.multiply((Qold-Q),(Qold-Q))/(np.multiply(Q,Q)))
    #print(delta)
    i=i+1
    Qold[0,0] = Q11
    Qold[0,1] = Q22
    Qold[0,2] = Q12
    Qold[0,3] = Q66
    
# -----------------------------------------------------------------------------
# Final result
# -----------------------------------------------------------------------------
Hess = (L*H/num_points)**2*((Q[0,0]**2+Q[0,2]**2)*H11 + (Q[0,1]**2+Q[0,2]**2)*H22 + Q[0,3]**2*H66 + 2*(Q[0,0] + Q[0,1])*Q[0,2]*H12)    

# calculate eta parameters (sensitivity to noise)    

# print results
print('----------------------------------------------------------------------')    
print('Identified stiffness paramters:')    
print('Q11: '+str(round(Q[0,0]/1e03,2))+ ' GPa')
print('Q22: '+str(round(Q[0,1]/1e03,2))+ ' GPa')
print('Q12: '+str(round(Q[0,2]/1e03,2))+ ' GPa')
print('Q66: '+str(round(Q[0,3]/1e03,2))+ ' GPa')
print('----------------------------------------------------------------------')

#%% ----------------------------------------------------------------------------



# plot virtual fields
'''
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

plot_pair_fields(X,Y,u1s_1,u2s_1,'$u_1^*$ (1)','$u_2^*$ (1)')
plot_pair_fields(X,Y,u1s_2,u2s_2,'$u_1^*$ (2)','$u_2^*$ (2)')
plot_pair_fields(X,Y,u1s_3,u2s_3,'$u_1^*$ (3)','$u_2^*$ (3)')
plot_pair_fields(X,Y,u1s_4,u2s_4,'$u_1^*$ (4)','$u_2^*$ (4)')

plot_pair_fields(X,Y,eps1s_2*e_11,eps1s_2*e_22,'$\epsilon_1^*(2)\epsilon_{11}$','$\epsilon_1^*(2)\epsilon_{22}$')
plot_pair_fields(X,Y,eps2s_2*e_11,eps6s_2*g_12,'$\epsilon_2^*(2)\epsilon_{11}$','$\epsilon_6^*(2)\gamma_{12}$')


plot_pair_fields(X,Y,eps2s_3*e_11,eps2s_3*e_22,'$\epsilon_2^*(3)\epsilon_{11}$','$\epsilon_2^*(3)\epsilon_{22}$')
plot_pair_fields(X,Y,eps2s_4*e_11,eps2s_4*e_22,'$\epsilon_2^*(4)\epsilon_{11}$','$\epsilon_2^*(4)\epsilon_{22}$')
plot_pair_fields(X,Y,eps6s_4*g_12,eps2s_4*e_22,'$\epsilon_6^*(4)\gamma_{12}$','$\epsilon_2^*(4)\epsilon_{22}$')

plot_single_var_contour(X,Y,eps6s_4*g_12,'$\epsilon_6^*\gamma_{xy}$')

plotVF_strain(X,Y,eps1s_1,eps2s_1,eps6s_1,1)
plotVF_strain(X,Y,eps1s_2,eps2s_2,eps6s_2,2)
plotVF_strain(X,Y,eps1s_3,eps2s_3,eps6s_3,3)
plotVF_strain(X,Y,eps1s_4,eps2s_4,eps6s_4,4)

# plot strain maps

plot_single_var_contour(X,Y,e_11,'$\epsilon_{11}$ (m/m)')
plot_single_var_contour(X,Y,e_22,'$\epsilon_{22}$ (m/m)')
plot_single_var_contour(X,Y,g_12,'$\gamma_{12}$ (m/m)')
'''
