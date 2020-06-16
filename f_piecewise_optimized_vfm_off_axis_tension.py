# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:26:08 2020

@author: Jared
"""
import numpy as np
from numpy.linalg import inv

def pw_opt_vfs_offaxis(X,Y,e11,e22,g12,L,H,F,t):
    
    # optimized virtual fields for off-axis shear test
    m = 3 # number of elements in y direction
    n = 2 # number of elements in x direction
    
    num_ele = m*n # number of elements
    num_nodes = (m+1)*(n+1) # number of nodes
    
    L_ele = L/n # length of element along x direction
    H_ele = H/m # height of element along y direction
    
    rows,cols = X.shape
    
    num_points = rows*cols
    
    # reshape strain fields into vector of length 'num_points'
    X = np.reshape(X,(num_points,1))
    Y = np.reshape(Y,(num_points,1))
    e11 = np.reshape(e11,(num_points,1))
    e22 = np.reshape(e22,(num_points,1))
    g12 = np.reshape(g12,(num_points,1))
    
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
    
    u1_elem = 0.25*np.column_stack(((1-xsi1)*(1-xsi2), u0, (1+xsi1)*(1-xsi2), u0, (1+xsi1)*(1+xsi2), u0, (1-xsi1)*(1+xsi2), u0))
    
    u2_elem = 0.25*np.column_stack((u0, (1-xsi1)*(1-xsi2), u0, (1+xsi1)*(1-xsi2), u0, (1+xsi1)*(1+xsi2), u0, (1-xsi1)*(1+xsi2)));
    
    r,c = u1_elem.shape
    print('Size of u1_elem: ('+str(r)+', '+str(c)+')')
    r,c = u2_elem.shape
    print('Size of u2_elem: ('+str(r)+', '+str(c)+')')
    
    r,c = Eps1_elem.shape
    print('Size of Eps1_elem: ('+str(r)+', '+str(c)+')')
    r,c = Eps2_elem.shape
    print('Size of Eps2_elem: ('+str(r)+', '+str(c)+')')
    r,c = Eps6_elem.shape
    print('Size of Eps6_elem: ('+str(r)+', '+str(c)+')')
    
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
    B22 = B11; B12 = B11; B66 = B11
    #
    H11 = np.zeros((2*num_nodes,2*num_nodes))
    H22 = H11; H12 = H11; H66 = H11
    
    r,c = B11.shape
    print('Size of B11,B12,B22,B66: ('+str(r)+', '+str(c)+')')
    r,c = H11.shape
    print('Size of H11,H12,H22,H66: ('+str(r)+', '+str(c)+')')    
    # Matrix of dofs affected by each data point
    # [node1_dof1, node1_dof2, node2_dof1, node2_dof2, node3_dof1, node3_dof2, node4_dof1, node4_dof2]
    assemble = np.column_stack((n1*2-1, n1*2, n2*2-1, n2*2, n3*2-1, n3*2, n4*2-1, n4*2)) # 2 dofs for each node
    
    r,c = assemble.shape
    print('Size of assemble: ('+str(r)+', '+str(c)+')')
    
    assemble = assemble.astype(int)
    # loop through all points and populate matrices
    for k in range(0,num_points):  
        assemble1 = assemble[k,:]-1
        
        print(Eps1_elem)
        print(assemble1)
        # temporary assembly variable with dofs tied to current point
        #for j in range(0,len(assemble1)):
        #    ind = assemble1[j]
        #    print(str(ind)+', '+str(Eps1_elem[k,j]))
        #print(B11[0,ind])
        # these expressions come from the expanded format of eq'm (constant reps. approx. of integrals with discrete sums)
        # iterate to add up the contribution in each element
        B11[0,assemble1] = B11[0,assemble1] + float(e11[k])*Eps1_elem[k,:]*L*H/num_points
        
        B22[0,assemble1] = B22[0,assemble1] + float(e22[k])*Eps2_elem[k,:]*L*H/num_points
        
        B12[0,assemble1] = B12[0,assemble1] + (float(e11[k])*Eps2_elem[k,:] + float(e22[k])*Eps1_elem[k,:])*L*H/num_points
        
        B66[0,assemble1] = B66[0,assemble1] + float(g12[k])*Eps6_elem[k,:]*L*H/num_points        
        print(assemble1)
        H11[assemble1,assemble1] = H11[assemble1,assemble1] + np.transpose(Eps1_elem[k,:])*Eps1_elem[k,:]
            
        H22[assemble1,assemble1] = H22[assemble1,assemble1] + np.transpose(Eps2_elem[k,:])*Eps2_elem[k,:]
            
        H12[assemble1,assemble1] = H12[assemble1,assemble1] + np.transpose(Eps1_elem[k,:])*Eps2_elem[k,:]
            
        H66[assemble1,assemble1] = H66[assemble1,assemble1] + np.transpose(Eps6_elem[k,:])*Eps6_elem[k,:]
            
    return B11
    # -----------------------------------------------------------------------------
    # Define Virtual Boundary Conditions
    num_cond = 3*(n+1)+n
    Aconst = np.zeros((num_cond, 2*num_nodes)) # there are 4*(n+1) boundary conditions - 2 dofs per node
    
    i = 0 
    for j in range(0,2*(n+1),2): # set u1 = 0 at y = 0: (n+1) conditions
        Aconst[j, 2*i*(m+1)] = 1
        i +=1
    i = 0    
    for j in range(1,2*(n+1)+1,2): # set u2 = 0 at y=0: n+1 conditions
        Aconst[j, 2*i*(m+1)+1] = 1
        i += 1
        
    for j in range(0,(n+1)): # set u1 = 0 at y=H: n+1 conditions
        Aconst[j + 2*(n+1), 2*(j+1)*(m+1)-2] = 1
        
    for j in range(0,n): # u2(y=H) = const. for n = 2, m = 3: u2(8) - u2(4) = 0: n conditions
        Aconst[j + 3*(n+1), 2*(j+1)*(m+1)-1] = 1
        Aconst[j + 3*(n+1), 2*(j+1)*(m+1)+2*(m+1)-1] = -1     
    
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
    print('Size of Za,Zb,Zc,Zd: ('+str(r)+', '+str(c)+')')
    
    # -----------------------------------------------------------------------------
    # Build up the A matrix containing all constraints:
    # boundary conditions and speciality and the B matrix containing zeros
    
    A = np.row_stack((Aconst,B11,B22,B12,B66)) # obtained from the constraint equation
    
    rows,cols = A.shape
    
    print('Size of A (rows, cols): '+str(rows)+', '+str(cols))

    B = np.zeros([rows,rows])
    
    # -----------------------------------------------------------------------------
    # solve the system - optimization
    # -----------------------------------------------------------------------------
    Q = np.matrix([1, 1, 1, 1]) # first guess for Q matrix
    
    n_iter = 20; # maximum number of iterations
    delta_lim = 0.001; # tollerance
    delta = 10;
    
    i = 1; # set counter
    Qold = Q; # store value of Q for previous iteration
    
    while i < n_iter and delta > delta_lim:
    
        # Calculate Hessian matrix 
        Hess =(L*H/num_points)**2*((Q[0,0]**2+Q[0,2]**2)*H11 + (Q[0,1]**2+Q[0,2]**2)*H22 + Q[0,3]**2*H66 + 2*(Q[0,0] + Q[0,1])*Q[0,2]*H12)
        
    
        #NOTE: to avoid the numerical "Warning: Matrix is close to singular or
        #      badly scaled" matrix Opt can be scaled with the parameter corr.
        #      It does not change the results of the optimization.
        #      To not use put corr=1;
        
        #corr = np.max(np.max(A))/np.max(np.max(H)) # error with this line - need to fix
        corr = 1
        
        # build optimization matrix      
        upper_lr = np.column_stack((Hess/2*corr,np.transpose(A)*corr))
        lower_lr = np.column_stack((A,B))
        
        
        OptM = np.row_stack((upper_lr,lower_lr))
        return OptM
        rows,cols = OptM.shape
    
        print('Size of OptM (rows, cols): '+str(rows)+', '+str(cols))
        print('OptM:')
        np.set_printoptions(precision=3)
        for line in OptM:
            print(*line)
        
        # Vector containing the polynomial coefficients and the Lagrange multipliers
        Ya = inv(OptM)*np.transpose(Za) # for Q11
        Yb = inv(OptM)*np.transpose(Zb) # for Q22
        Yc = inv(OptM)*np.transpose(Zc) # for Q12
        Yd = inv(OptM)*np.transpose(Zd) # for Q66
        
        # Remove the Lagrange multipliers from the Y vectors because they are of no interest
        Ya[2*num_nodes+1:len(Ya)] = []
        Yb[2*num_nodes+1:len(Yb)] = []
        Yc[2*num_nodes+1:len(Yc)] = []
        Yd[2*num_nodes+1:len(Yd)] = []
        
        # Calculating Q11 from the first optimized virtual field
        Q[0,0] = (Ya[2*num_nodes]*F/t)
        # Calculating Q22 from the second optimized virtual field
        Q[0.1] = (Yb[2*num_nodes]*F/t)
        # Calculating Q12 from the third optimized virtual field
        Q[0,2] = (Yc[2*num_nodes]*F/t)
        # Calculating Q66 from the fourth optimized virtual field
        Q[0,3] = (Yd[2*num_nodes]*F/t)
    
        delta=np.sum((Qold-Q)**2/(Q**2))
        i=i+1
        Qold=Q
        
    return Q