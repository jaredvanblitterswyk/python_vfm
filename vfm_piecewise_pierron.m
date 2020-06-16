% The Virtual Fields Methods: Extracting constitutive mechanical parameters
% from full-field deformation measurements, based on experimental data.
% F. Pierron, M. Grédiac
% Springer
% April 2010
% Based on the paper "Sensitivity to the virtual fields method to noisy
% data", Avril, Grediac, Pierron, - Comp. Mech. (2004)

%%%%%%%%%%% Chapter C.II - CASE STUDY II %%%%%%%%%%%%%%%%%
%%%%%%%%%%% UNNOTCHED IOSIPESCU TEST %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% ORTHOTROPIC MATERIAL %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% SIMULATED DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% OPTIMIZED SPECIAL PIECEWISE VIRTUAL FIELDS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DESCRIPTION OF THE INPUT FILE: Iosipescu_isotropic.mat
% Geometrical parameters
% e1 is the horizontal axis, e2 the vertical one (shear force is along e2)
% X1 is the matrix containing the abscissa of the data points centroids
% X2 is the vector containing the ordinates of the data points centroids
% L is the length of the active area (S2) in mm
% w is the width of the specimen in mm
% t is the thickness of the specimen in mm

% Load
% F is the shear load in N (negative because pointing downwards)

% Deformations
% Eps1 is the matrix containing the epsilon 1 strain values
% Eps2 is the matrix containing the epsilon 2 strain values
% Eps6 is the matrix containing the epsilon 6 strain values (engineering
% shear strain)

% Material
% Q11 is the material's in-plane stiffness in the e1 direction, which is the fibre direction (to be identified)
% Q22 is the material's in-plane stiffness in the e2 direction, which is the transverse direction (to be identified)
% Q12 is the material's in-plane stiffness associated to Poisson's effect (to be identified)
% Q66 is the material's in-plane shear stiffness (to be identified)

% Virtual fields
% m,n are the number of elements respecively along e1 and e2

% Units are in N and mm: stiffnesses will be obtained in MPa

% DESCRIPTION OF THE OUTPUTS
% Q is the vector formed by (Q11,Q22,Q12,Q66)
% eta contains the sensitivity to noise parameters for the Q components
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter definition and data formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Choice of the number of elements
m=3; % Number of elements along e1
n=2; % Number of elements along e2

% Parameter definition
n_nodes=(m+1)*(n+1); % number of nodes
n_elem=m*n; % number of elements
n_row=size(Eps1,1); % size of the array of data points in vertical direction
n_column=size(Eps1,2); % size of the array of data points in horizontal direction
n_points=size(Eps1,1)*size(Eps1,2); % number of data points

L_el=L/m; % Element length along e1
w_el=w/n; % Element width along e2

% Data formatting
% Sets the origin at the bottom-left angle
X2=X2-min(X2(:))+w/size(X2,1)/2;
% Transforms the strain and coordinate matrices into vectors of length
% 'n_points'
X1=reshape(X1,n_points,1);
X2=reshape(X2,n_points,1);
Eps1=reshape(Eps1,n_points,1);
Eps2=reshape(Eps2,n_points,1);
Eps6=reshape(Eps6,n_points,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Building up the virtual fields
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Intermediary quantities
% tag of element belonging along X1
iii=floor(X1*m/L)+1;
% tag of element belonging along X2
jjj=floor(X2*n/w)+1; 


% Definition of the parametric coordinates
xsi1=2*X1/L_el-iii*2+1; % Parametric coordinate for each element 
xsi2=2*X2/w_el-jjj*2+1; % Parametric coordinate for each element 

% Calculation of the virtual strains
v0=zeros(n_points,1);
Eps1elem=[-(1-xsi2) v0 (1-xsi2) v0 (1+xsi2) v0 -(1+xsi2) v0]*1/2/L_el;
Eps2elem=[v0 -(1-xsi1) v0 -(1+xsi1) v0 (1+xsi1) v0 (1-xsi1)]*1/2/w_el;
Eps6elem=[-(1-xsi1)/w_el -(1-xsi2)/L_el -(1+xsi1)/w_el (1-xsi2)/L_el ...
    (1+xsi1)/w_el (1+xsi2)/L_el (1-xsi1)/w_el -(1+xsi2)/L_el]*1/2;

% Calculation of the virtual displacements
u0=zeros(n_points,1);
u1elem=0.25*[(1-xsi1).*(1-xsi2) u0 (1+xsi1).*(1-xsi2) u0...
    (1+xsi1).*(1+xsi2) u0 (1-xsi1).*(1+xsi2) u0];
u2elem=0.25*[u0 (1-xsi1).*(1-xsi2) u0 (1+xsi1).*(1-xsi2)...
    u0 (1+xsi1).*(1+xsi2) u0 (1-xsi1).*(1+xsi2)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construction of the quantities based on the 
% virtual fields in the final optimization matrix 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bij will be used for the speciality conditions
% Hij will be used to compute the Hessian matrix

% Initialization of the variables
B11=zeros(1,2*n_nodes);
B22=zeros(1,2*n_nodes);
B12=zeros(1,2*n_nodes);
B66=zeros(1,2*n_nodes);
H11=zeros(2*n_nodes,2*n_nodes);
H22=zeros(2*n_nodes,2*n_nodes);
H12=zeros(2*n_nodes,2*n_nodes);
H66=zeros(2*n_nodes,2*n_nodes);

% Definition of the nodes
% number of the first node of the related element for each data point
n1 = (iii-1)*(n+1)+jjj; 
% number of the second node of the related element for each data point
n2 = iii*(n+1)+jjj;
% number of the third node of the related element for each data point
n3 = iii*(n+1)+jjj+1;
% number of the fourth node of the related element for each data point
n4 = (iii-1)*(n+1)+jjj+1;

% Matrix containing the degrees of freedom affected by each data point
assemble=[n1*2-1 n1*2 n2*2-1 n2*2 n3*2-1 n3*2 n4*2-1 n4*2];
count = 1;
dof = 1;
point_list = find(assemble == dof);
res = zeros(length(point_list),1);
strain = zeros(length(point_list),1);
eps1s = zeros(length(point_list),1);

for k=1:n_points
    assemble1=assemble(k,:); 
    
    if ismember(k,point_list)
        ind_dof = find(assemble1 == 1);
        res(count) = Eps1(k)*Eps1elem(k,ind_dof)*L*w/n_points;
        strain(count) = Eps1(k);
        eps1s(count) = Eps1elem(k,ind_dof);
        count = count +1;
    end
    
      
    B11(assemble1)=B11(assemble1)+Eps1(k)*Eps1elem(k,:)*L*w/n_points;
    B22(assemble1)=B22(assemble1)+Eps2(k)*Eps2elem(k,:)*L*w/n_points;
    B12(assemble1)=B12(assemble1)+(Eps1(k)*Eps2elem(k,:)...
        +Eps2(k)*Eps1elem(k,:))*L*w/n_points;
    B66(assemble1)=B66(assemble1)+Eps6(k)*Eps6elem(k,:)*L*w/n_points;

    H11(assemble1,assemble1)=H11(assemble1,assemble1)+...
        Eps1elem(k,:)'*Eps1elem(k,:);
    H22(assemble1,assemble1)=H22(assemble1,assemble1)+...
        Eps2elem(k,:)'*Eps2elem(k,:);
    H12(assemble1,assemble1)=H12(assemble1,assemble1)+...
        Eps1elem(k,:)'*Eps2elem(k,:);
    H66(assemble1,assemble1)=H66(assemble1,assemble1)+...
        Eps6elem(k,:)'*Eps6elem(k,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Virtual boundary conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Aconst=zeros(4*n+3,2*n_nodes); % there are 4n+3 conditions here

for i=1:2*(n+1)
% setting u1*=u2*=0 on the left boundary, 2(n+1) conditions
    Aconst(i,i)=1; 
end
% setting u1*=0 on the right boundary, n+1 conditions
for i=1:(n+1)    
    Aconst(i+2*(n+1),2*n_nodes-2*(n+1)+2*i-1)=1;
end

for i=1:n
% setting u2*=constant on the right boundary, n conditions
    Aconst(i+3*(n+1),2*n_nodes-2*(n+1)+2*i)=1;
    Aconst(i+3*(n+1),2*n_nodes-2*(n+1)+2*(i+1))=-1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constitution of the Z vector containing zeros 
% except for the conditions of speciality
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Za=zeros(1,2*n_nodes+size(Aconst,1));
Zb=Za;Zc=Za;Zd=Za;

Za(2*n_nodes+size(Aconst,1)+1:2*n_nodes+size(Aconst,1)+4)=[1 0 0 0];
Zb(2*n_nodes+size(Aconst,1)+1:2*n_nodes+size(Aconst,1)+4)=[0 1 0 0];
Zc(2*n_nodes+size(Aconst,1)+1:2*n_nodes+size(Aconst,1)+4)=[0 0 1 0];
Zd(2*n_nodes+size(Aconst,1)+1:2*n_nodes+size(Aconst,1)+4)=[0 0 0 1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Building up the A matrix containing all constraints:
% boundary conditions and speciality and the 
% B matrix containing zeros
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=[Aconst;B11;B22;B12;B66]; % obtained from the constraint equation
B=zeros(size(A,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solving the system (optimization)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q=[1 1 1 1]; % first guess for Q

n_iter=20; % maximum number of iteration
delta_lim=0.001; % tollerance
delta=10;

i=1;
Qold=Q;

while i<n_iter && delta>delta_lim

    % Hessian matrix 
    H =(L*w/n_points)^2*((Q(1)^2+Q(3)^2)*H11+(Q(2)^2+Q(3)^2)*H22...
        +Q(4)^2*H66+2*(Q(1)+Q(2))*Q(3)*H12);
    

    %NOTE: to avoid the numerical "Warning: Matrix is close to singular or
    %      badly scaled" matrix Opt can be scaled with the parameter corr.
    %      It does not change the results of the optimization.
    %      To not use put corr=1;
    
    corr=max(max(A))/max(max(H));
    OptM=[H/2*corr,A'*corr;A,B];
    if i == 1
        tempOptM = OptM;
    end
    % Vector containing the polynomial coefficients 
    % and the Lagrange multipliers
    Ya=OptM\Za'; % for Q11
    Yb=OptM\Zb'; % for Q22
    Yc=OptM\Zc'; % for Q12
    Yd=OptM\Zd'; % for Q66
    
    % Removing the Lagrange multipliers from the Y vectors
    % because they are of no interest
    Ya(2*n_nodes+1:size(Ya))=[];
    Yb(2*n_nodes+1:size(Yb))=[];
    Yc(2*n_nodes+1:size(Yc))=[];
    Yd(2*n_nodes+1:size(Yd))=[];
    
    % Calculating Q11 from the first optimized virtual field
    Q(1)=(Ya(2*n_nodes)*F/t);
    % Calculating Q22 from the second optimized virtual field
    Q(2)=(Yb(2*n_nodes)*F/t);
    % Calculating Q12 from the third optimized virtual field
    Q(3)=(Yc(2*n_nodes)*F/t);
    % Calculating Q66 from the fourth optimized virtual field
    Q(4)=(Yd(2*n_nodes)*F/t);

    delta=sum((Qold-Q).^2./Q.^2);
    i=i+1;
    Qold=Q;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Final Hessian matrix for computing the eta parameters
H =(L*w/n_points)^2*((Q(1)^2+Q(3)^2)*H11+(Q(2)^2+Q(3)^2)*H22+...
    Q(4)^2*H66+2*(Q(1)+Q(2))*Q(3)*H12);

eta2(1) = sqrt(Ya'*H*Ya);
eta2(2) = sqrt(Yb'*H*Yb);
eta2(3) = sqrt(Yc'*H*Yc);
eta2(4) = sqrt(Yd'*H*Yd);

% Reconstruction of the virtual fields
% for visualization  purposes
for k=1:n_points,   
    
    % Virtual displacement fields, e1 component
    u1vv=zeros(1,2*n_nodes);
    assemble1=assemble(k,:);
    u1vv(assemble1)=u1elem(k,:);
    u1va(k)=u1vv*Ya; % For Q11
    u1vb(k)=u1vv*Yb; % For Q22
    u1vc(k)=u1vv*Yc; % For Q12
    u1vd(k)=u1vv*Yd; % For Q66
    
    % Virtual displacement fields, e2 component
    u2vv=zeros(1,2*n_nodes);
    assemble1=assemble(k,:);
    u2vv(assemble1)=u2elem(k,:);
    u2va(k)=u2vv*Ya;
    u2vb(k)=u2vv*Yb;
    u2vc(k)=u2vv*Yc;
    u2vd(k)=u2vv*Yd;
    
    % Virtual strain fields, 1 components
    Eps1vv=zeros(1,2*n_nodes);
    Eps1vv(assemble1)=Eps1elem(k,:);
    Eps1va(k)=Eps1vv*Ya;
    Eps1vb(k)=Eps1vv*Yb;
    Eps1vc(k)=Eps1vv*Yc;
    Eps1vd(k)=Eps1vv*Yd;
    
    % Virtual strain fields, 2 components
    Eps2vv=zeros(1,2*n_nodes);
    Eps2vv(assemble1)=Eps2elem(k,:);
    Eps2va(k)=Eps2vv*Ya;
    Eps2vb(k)=Eps2vv*Yb;
    Eps2vc(k)=Eps2vv*Yc;
    Eps2vd(k)=Eps2vv*Yd;
    
    % Virtual strain fields, 6 components
    Eps6vv=zeros(1,2*n_nodes);
    Eps6vv(assemble1)=Eps6elem(k,:);
    Eps6va(k)=Eps6vv*Ya;
    Eps6vb(k)=Eps6vv*Yb;
    Eps6vc(k)=Eps6vv*Yc;
    Eps6vd(k)=Eps6vv*Yd;    
end
% Reshaping the data in matrix form
u1va=reshape(u1va',n_row,n_column);
u1vb=reshape(u1vb',n_row,n_column);
u1vc=reshape(u1vc',n_row,n_column);
u1vd=reshape(u1vd',n_row,n_column);

u2va=reshape(u2va',n_row,n_column);
u2vb=reshape(u2vb',n_row,n_column);
u2vc=reshape(u2vc',n_row,n_column);
u2vd=reshape(u2vd',n_row,n_column);

Eps1va=reshape(Eps1va',n_row,n_column);
Eps2va=reshape(Eps2va',n_row,n_column);
Eps6va=reshape(Eps6va',n_row,n_column);
Eps1vb=reshape(Eps1vb',n_row,n_column);
Eps2vb=reshape(Eps2vb',n_row,n_column);
Eps6vb=reshape(Eps6vb',n_row,n_column);
Eps1vc=reshape(Eps1vc',n_row,n_column);
Eps2vc=reshape(Eps2vc',n_row,n_column);
Eps6vc=reshape(Eps6vc',n_row,n_column);
Eps1vd=reshape(Eps1vd',n_row,n_column);
Eps2vd=reshape(Eps2vd',n_row,n_column);
Eps6vd=reshape(Eps6vd',n_row,n_column);

% Displays outputs 
    fprintf(['Q11 = ','%7.2f','\t','eta11/Q11 = ','%3.4f','\t','\n']...
        ,Q(1),eta2(1)/Q(1));
    fprintf(['Q22 = ','%7.2f','\t','eta22/Q22 = ','%3.4f','\t','\n']...
        ,Q(2),eta2(2)/Q(2));
    fprintf(['Q12 = ','%7.2f','\t','eta12/Q12 = ','%3.4f','\t','\n']...
        ,Q(3),eta2(3)/Q(3));
    fprintf(['Q66 = ','%7.2f','\t','eta66/Q66 = ','%3.4f','\t','\n']...
        ,Q(4),eta2(4)/Q(4));
    fprintf(['nodes = ' '%6.0f;' '\t' 'elements =' '%6.0f;' '\t' '\n']...
        ,n_nodes,n_elem);

