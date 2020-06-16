function [identStiffVFOpt,VFOptDiag] = func_VFDynPWSpecOptOrthoLinElas(VFOpts,...
    pos,specimen,material,accel,strain)
% Author: Lloyd Fletcher
% PhotoDyn Group, University of Southampton
% Date: 19/9/2017
%
% Uses piece-wise bi-linear FE special optimised virtual fields to find 
% the stiffness components Q11,Q22,Q12,Q66 for a dynamically impacted 
% linear elastic orthotropic material. Specimen is rectangular with 
% dimensions L*w and it is assumed that the specimen is impacted on the 
% edge x1 = L.

% Turn off warnings for ill conditioned matrices
warning('off', 'MATLAB:singularMatrix'); 
warning('off', 'MATLAB:nearlySingularMatrix');

%--------------------------------------------------------------------------
% Parameter Initialisation
% Number of elements in the virtual mesh
m = VFOpts.nElemX;  % number of elements along x1
n = VFOpts.nElemY;  % number of elements along x2

% Geometry and material properties
rho = material.rho;
L = specimen.length;
w = specimen.height;

% Number of speciality constraints, 4 for orthotropic
numSpecConst = 4;

% Mesh geometry parameters
nNodes = (m+1)*(n+1);
nElems = m*n;
[nRow,nCol,nFrames] = size(strain.x); 
nPoints = nCol*nRow;
lElem = L/m;
wElem = w/n;

% Vectorise the fields to speed up the calculation
% NOTE: in the FE formulation co-ord system is critical
x1 = reshape(pos.xGrid,nPoints,1);
x2 = reshape(pos.yGrid,nPoints,1);

%--------------------------------------------------------------------------
% Construct the Virtual Fields
% Piecewise functions - Bi-linear 4-node elements
% Element col number for element along x1
elemColNum = floor(x1*m/L)+1;
% Element row number for element along x2
elemRowNum = floor(x2*n/w)+1;

% Parametric co-ords in element co-ord system
x1Para = 2*x1/lElem-elemColNum*2+1;
x2Para = 2*x2/wElem-elemRowNum*2+1;

% Virtual Displacement Calculation
u0 = zeros(nPoints,1);
u1StarElem = 0.25*[(1-x1Para).*(1-x2Para) , u0,...
                   (1+x1Para).*(1-x2Para) , u0,...
                   (1+x1Para).*(1+x2Para) , u0,...
                   (1-x1Para).*(1+x2Para) , u0];
u2StarElem = 0.25*[u0, (1-x1Para).*(1-x2Para),...
                   u0, (1+x1Para).*(1-x2Para),...
                   u0, (1+x1Para).*(1+x2Para),...
                   u0, (1-x1Para).*(1+x2Para)];

% Virtual Strain Calculation
v0 = zeros(nPoints,1);
eps1StarElem = 1/2/lElem*[-(1-x2Para), v0,...
                           (1-x2Para), v0,...
                           (1+x2Para), v0,...
                          -(1+x2Para), v0];
eps2StarElem = 1/2/wElem*[v0, -(1-x1Para),...
                          v0, -(1+x1Para),...
                          v0,  (1+x1Para),...
                          v0,  (1-x1Para)];
eps6StarElem = 1/2*[-(1-x1Para)/wElem, -(1-x2Para)/lElem,...
                    -(1+x1Para)/wElem,  (1-x2Para)/lElem,...
                     (1+x1Para)/wElem,  (1+x2Para)/lElem,...
                     (1-x1Para)/wElem, -(1+x2Para)/lElem];

%--------------------------------------------------------------------------
% Construct Speciality Conditions and Components of the Hessian Matrix
% SCij are the speciality conditions
% Hij and the components of the Hessian

% Intitalise the SCij and Hij vars to zero
SC11 = zeros(1,2*nNodes); 
SC22 = SC11; SC12 = SC11; SC66 = SC11;
H11 = zeros(2*nNodes,2*nNodes);
H22 = H11; H12 = H11; H66 = H11;

% Definition of Node Connectivity
node1 = (elemColNum-1)*(n+1)+elemRowNum;     % First node of each elem (bottom LH)
node2 = elemColNum*(n+1)+elemRowNum;         % Second node of each elem (bottom RH)
node3 = elemColNum*(n+1)+elemRowNum+1;       % Third node of each elem (top RH)
node4 = (elemColNum-1)*(n+1)+elemRowNum+1;   % Fourth node of each elem (top LH)
% Matrix giving dofs affected by each data point
assemMat = [2*node1-1, 2*node1, 2*node2-1, 2*node2,...
            2*node3-1, 2*node3, 2*node4-1, 2*node4];

% Build the nodal components of the Hessian matrix, Hij 
for pp = 1:nPoints
    assemPt = assemMat(pp,:);   
    H11(assemPt,assemPt) = H11(assemPt,assemPt) + ...
        eps1StarElem(pp,:)'* eps1StarElem(pp,:);
    H22(assemPt,assemPt) = H22(assemPt,assemPt) + ...
        eps2StarElem(pp,:)'* eps2StarElem(pp,:);
    H12(assemPt,assemPt) = H12(assemPt,assemPt) + ...
        eps1StarElem(pp,:)'* eps2StarElem(pp,:);
    H66(assemPt,assemPt) = H66(assemPt,assemPt) + ...
        eps6StarElem(pp,:)'* eps6StarElem(pp,:); 
end

%--------------------------------------------------------------------------
% Define the Virtual Boundary Conditions and Constraint Matrix
Aconst = zeros((n+1),2*nNodes);
% u1*=0 on the right boundary (impacted edge of the sample)
for bb = 1:(n+1)
    Aconst(bb,2*nNodes-2*(n+1)+2*bb-1)=1;
end
% Block of zeros to fill out M matrix
B = zeros(size(Aconst,1)+numSpecConst); 

%--------------------------------------------------------------------------
% Construct the Z vectors for solving the system
Z11 = zeros(1,2*nNodes+size(Aconst,1)+numSpecConst); 
Z22=Z11; Z12=Z11; Z66=Z11; 
Z11(end-numSpecConst+1:end) = [1,0,0,0];    % Speciality condition for Q11
Z22(end-numSpecConst+1:end) = [0,1,0,0];    % Speciality condition for Q22
Z12(end-numSpecConst+1:end) = [0,0,1,0];    % Speciality condition for Q12
Z66(end-numSpecConst+1:end) = [0,0,0,1];    % Speciality condition for Q66

%--------------------------------------------------------------------------
% Initialise the loop storage vars to zero
Q11t = zeros(1,nFrames);
Q22t = zeros(1,nFrames);
Q12t = zeros(1,nFrames);
Q66t = zeros(1,nFrames);
nIterLog = zeros(1,nFrames); 
mMatrixCond = zeros(1,nFrames); 
eta11t = zeros(1,nFrames);
eta22t = zeros(1,nFrames);
eta12t = zeros(1,nFrames);
eta66t = zeros(1,nFrames);

%--------------------------------------------------------------------------
% Loop over each frame and identify the stiffness
startFrame = VFOpts.startFrame;
for ff = startFrame:nFrames
    %----------------------------------------------------------------------
    % Reshape the strain and accel fields for this frame
    eps1 = reshape(strain.x(:,:,ff),nPoints,1);
    eps2 = reshape(strain.y(:,:,ff),nPoints,1);
    eps6 = reshape(strain.s(:,:,ff),nPoints,1);
    accel1 = reshape(accel.x(:,:,ff),nPoints,1);
    accel2 = reshape(accel.y(:,:,ff),nPoints,1);
    
    %----------------------------------------------------------------------
    % Assemble the speciality conditions and constraint matrix 
    
    % Set initial SCij vector to zero
    SC11 = zeros(1,2*nNodes); 
    SC22 = SC11; SC12 = SC11; SC66 = SC11;
    % Calculate SCij summing over all points
    for pp = 1:nPoints
        assemPt = assemMat(pp,:);
        SC11(assemPt) = SC11(assemPt)+(L*w/nPoints)*...
            (eps1(pp)*eps1StarElem(pp,:));
        SC22(assemPt) = SC22(assemPt)+(L*w/nPoints)*...
            (eps2(pp)*eps2StarElem(pp,:));
        SC12(assemPt) = SC12(assemPt)+(L*w/nPoints)*...
            (eps2(pp)*eps1StarElem(pp,:)+eps1(pp)*eps2StarElem(pp,:));
        SC66(assemPt) = SC66(assemPt)+(L*w/nPoints)*...
            (eps6(pp)*eps6StarElem(pp,:));
    end
    
    % Combine all constraints into the A matrix, BCs and speciality
    A = [Aconst; SC11; SC22; SC12; SC66];

    %----------------------------------------------------------------------
    % Solve the System Iteratively
    Q = [10,1,0.5,0.5]*10^9;% Initialise Q to initialise the first H matrix
    maxIter = 10;           % Max number of iters before breaking loop
    deltaLim = 0.001;       % Tolerance for convergence
    delta = 10;             % Set initial difference value to higher than tol
    Qprev = Q;              % Intialise prev Q matrix for conv check
    
    % Start Convergence loop
    ii = 1;
    while ii<maxIter && delta>deltaLim
        % Assemble the Hessian matrix
        H =(L*w/nPoints)^2*...
           ((Q(1)^2+Q(3)^2)*H11+...
           (Q(2)^2+Q(3)^2)*H22+...
           2*(Q(1)+Q(2))*Q(3)*H12+...
           Q(4)^2*H66);
        
        % Coefficient for normalising the matrices prior to inversion
        normCoeff = max(max(A))/max(max(H));

        % Assemble the M matrix to find the coeffs for the virtual fields
        OptM = [H/2*normCoeff,A';A,B];

        % Find Virtual Nodal Displacement for each Special Field
        Y11 = OptM\Z11';
        Y22 = OptM\Z22';
        Y12 = OptM\Z12';
        Y66 = OptM\Z66';

        % Remove the Lagrange multipliers
        Y11(2*nNodes+1:end) = [];
        Y22(2*nNodes+1:end) = [];
        Y12(2*nNodes+1:end) = [];
        Y66(2*nNodes+1:end) = [];
        
        % Pre-alloc for speed and clear the vars to zero
        uStar1Eval11 = zeros(1,nPoints);
        uStar2Eval11 = zeros(1,nPoints);
        uStar1Eval22 = zeros(1,nPoints);
        uStar2Eval22 = zeros(1,nPoints);
        uStar1Eval12 = zeros(1,nPoints);
        uStar2Eval12 = zeros(1,nPoints);
        uStar1Eval66 = zeros(1,nPoints);
        uStar2Eval66 = zeros(1,nPoints);
        % Evaluate the optimised virtual displacement fields at all
        % measurement points
        for pp = 1:nPoints
            u1vv = zeros(1,2*nNodes);
            u2vv = zeros(1,2*nNodes);
            assemPt = assemMat(pp,:);
            u1vv(assemPt) = u1StarElem(pp,:);
            u2vv(assemPt) = u2StarElem(pp,:);
            
            % Virtual disp fields for Q11
            uStar1Eval11(pp) = u1vv*Y11;
            uStar2Eval11(pp) = u2vv*Y11;
            % Virtual disp fields for Q22
            uStar1Eval22(pp) = u1vv*Y22;
            uStar2Eval22(pp) = u2vv*Y22;
            % Virtual disp fields for Q12
            uStar1Eval12(pp) = u1vv*Y12;
            uStar2Eval12(pp) = u2vv*Y12;
            % Virtual disp fields for Q66
            uStar1Eval66(pp) = u1vv*Y66;
            uStar2Eval66(pp) = u2vv*Y66;
        end
        
        % Calculate the Qij values using the virtual disp fields and the
        % measured acceleration fields
        Q(1) = -rho*L*w*(mean(uStar1Eval11'.*accel1) + mean(uStar2Eval11'.*accel2));
        Q(2) = -rho*L*w*(mean(uStar1Eval22'.*accel1) + mean(uStar2Eval22'.*accel2));
        Q(3) = -rho*L*w*(mean(uStar1Eval12'.*accel1) + mean(uStar2Eval12'.*accel2));
        Q(4) = -rho*L*w*(mean(uStar1Eval66'.*accel1) + mean(uStar2Eval66'.*accel2));

        % Calculate the difference in the stiffnesses for convergence checking
        delta = sum((Qprev-Q).^2./Q.^2);
        
        % Move to the next iteration
        Qprev = Q;
        ii = ii + 1;
    end
    %----------------------------------------------------------------------
    % Store diagnostics for the identification
    nIterLog(ff) = ii;
    mMatrixCond(ff) = cond(OptM);
    % Hessian matrix for the final stiffnesses
    H =(L*w/nPoints)^2*...
       ((Q(1)^2+Q(3)^2)*H11+...
       (Q(2)^2+Q(3)^2)*H22+...
       2*(Q(1)+Q(2))*Q(3)*H12+...
       Q(4)^2*H66);
    % Calculate the noise sensitivity
    eta11t(ff) = sqrt(Y11'*H*Y11);
    eta22t(ff) = sqrt(Y22'*H*Y22);
    eta12t(ff) = sqrt(Y12'*H*Y12);
    eta66t(ff) = sqrt(Y66'*H*Y66);
    
    %----------------------------------------------------------------------
    % Store the stiffnesses found for this frame
    Q11t(ff) = Q(1);
    Q22t(ff) = Q(2);
    Q12t(ff) = Q(3);
    Q66t(ff) = Q(4);
end

% Push the diagnostics into a data struct to return
VFOptDiag.nIters = nIterLog;
VFOptDiag.mMatrixCond = mMatrixCond; 
VFOptDiag.eta11VsT = eta11t;
VFOptDiag.eta22VsT = eta22t;
VFOptDiag.eta12VsT = eta12t;
VFOptDiag.eta66VsT = eta66t;

% Push the identified stiffnesses into a struct to return
identStiffVFOpt.Q11VsT = Q11t;
identStiffVFOpt.Q22VsT = Q22t;
identStiffVFOpt.Q12VsT = Q12t;
identStiffVFOpt.Q66VsT = Q66t;

% Calculate medians over the identification time to reject outliers
% Note: this is a first guess at the identified value
identStiffVFOpt.Q11AvgOverT = nanmedian(identStiffVFOpt.Q11VsT);
identStiffVFOpt.Q22AvgOverT = nanmedian(identStiffVFOpt.Q22VsT);
identStiffVFOpt.Q12AvgOverT = nanmedian(identStiffVFOpt.Q12VsT);
identStiffVFOpt.Q66AvgOverT = nanmedian(identStiffVFOpt.Q66VsT);

% Turn back on warnings for ill conditioned matrices
warning('on', 'MATLAB:singularMatrix'); 
warning('on', 'MATLAB:nearlySingularMatrix');

end

