function [identQVFOpt,uxStar,eps6Star] = f_VFSpecialOptimised_Shear(virtualField, pos, strain, accel, material)
% choose the number of elements in the x and y direction
m = virtualField.nElemX; % number of nodes in the X direction
n = virtualField.nElemY; % number of nodes in the Y direction
L = max(pos.x) + min(pos.x);
w = max(pos.y) + min(pos.y);
% L = pos.x(end-49);-pos.x(50);
[rows, cols, frames] = size(strain.sRE);

n_nodes = (m+1)*(n+1); % number of nodes
n_points = rows*cols; % number of measurement points

L_el = L/m; % length of elements
w_el = w/n; % width of elements

%reshape coordinate matrices into vectors
X1 = reshape(pos.xGrid, n_points,1);
X2 = reshape(pos.yGrid, n_points,1);

%% Build Up Virtual Fields
% assign an element to each measurement point
iii = floor(X1*m/L)+1; % elements in the x1 direction
jjj = floor(X2*n/w)+1; % elements in the x2 direction

%define parametric coordinates
xsi1 = 2*X1/L_el - 2*iii +1; % along x1 direction
xsi2 = 2*X2/w_el - 2*jjj +1; % along x2 direction

% calculate virtual displacements - 1 dof per node
u1elem = 0.25*[(1-xsi1).*(1-xsi2) (1+xsi1).*(1-xsi2) (1+xsi1).*(1+xsi2) (1-xsi1).*(1+xsi2)]; % first row of N matrix (condensed version from pg. 51 textbook)
    
% calculate virtual strains
% Eps6elem = [-(1-xsi2) (1-xsi2) (1+xsi2) -(1+xsi2)]*1/2/L_el; % see pg. 67
% of first notebook for derivation - axial shape functions
Eps6elem = [-(1-xsi1) -(1+xsi1) (1+xsi1) (1-xsi1)]*1/2/w_el;

%% Consruct the matrices in the form of the final optimization matrix 
% Bij will be used for the speciality condition
% Hij will be used to compute the Hessian matrix
B66 = zeros(1,n_nodes);
H66 = zeros(n_nodes, n_nodes);

% Define the nodes
n1 = (iii-1)*(n+1) + jjj; % first node
n2 = iii*(n+1) + jjj; % second node
n3 = iii*(n+1) + jjj +1; % third node
n4 = (iii-1)*(n+1) + jjj +1; % fourth node

% matrix containing the degrees of freedom affected by each data point
assemble = [n1 n2 n3 n4];

%% Define Virtual Boundary Conditions
% Aconst = zeros(m+1, n_nodes); %there are (m+1) boundary conditions
Aconst = zeros(m*n+m+1, n_nodes); %there are (m+1) boundary conditions and 1 degrees of freedom per node
for j = 1:(m+1) % u1(y = H) = 0 (m+1) conditions
    Aconst(j, j*(n+1)) = 1;
end

% constrain horizontal virtual displacements to be the same within each
% horizontal plane
for j = 1:n % logic written up on pg. 134 of logbook
    for i = 1:m
        Aconst(j*m+1+i, j+(i-1)*(n+1)) = 1;
        Aconst(j*m+1+i, j+i*(n+1)) = -1;
    end
end

%% Construct Z vector of zeros except for conditions of speciality
% Za is the vector used to find the virtual field to identify Q11
Zd = zeros(1, n_nodes + size(Aconst,1)+1); % 32 columns plus 4 for describing the boundary conditions
Zd(n_nodes + size(Aconst,1)+1 : n_nodes + size(Aconst,1)+1) = 1; % special VF condition for Q66

%% Compute stiffness for each frame using VFs with minimal noise sensitivity
% allocate memory for vector holding stiffness values as a function of time
% processFrames = 85;
identQVFOpt = zeros(1,frames);
u1vd = zeros(1,n_points);

% uxStar = zeros(size(strain.s));
% eps6Star = uxStar;

warning('off','all')
for j = 1:frames
    % reshape matrices for each frame9:end-9,1:end-9,:
    Eps6 = reshape(squeeze(strain.sRE(:,:,j)), n_points,1);
    A1 = reshape(squeeze(accel.x(:,:,j)), n_points,1);

    for k = 1:n_points
        assemble1 = assemble(k,:);
        B66(assemble1) = B66(assemble1) + (Eps6(k)*Eps6elem(k,:))*(L*w/n_points); % multiply actual strains by virtual strains

        % assemble Hessian matrix (minimization of sensitivity to noise)
        H66(assemble1,assemble1) = H66(assemble1,assemble1) + Eps6elem(k,:)'*Eps6elem(k,:);
    end
    
    % build up A matrix with all constraints
    A = [Aconst; B66];
    % speciality conditions
    B = zeros(size(A,1));

    %% Solving the optimization problem - same as for polynomials
    Q = 1; % initial guesses for stiffness values - required since H relies on Qij

    n_iter = 20; % maximum iterations
    delta_lim = 0.001; % tolerance on optimization

    delta = 10; % starting with a tolerance larger than delta_lim
    i = 1;
    Qold = Q; % stiffnesses from previous iteration required to compute error

    while i<n_iter && delta> delta_lim
        % Hessian matrix
        H = (L*w/n_points)^2*(Q^2)*H66;

        % NOTE: to avoid numerical "Warning: Matrix is close to singular or
        % badly scaled" matrix Opt can be scaled with the parameter corr.
        % It does not change the results of the optimization.
        % To avoid using, put corr = 1

%         corr = max(max(A))/max(max(H)); % normalization coefficient
        corr = 1;
        OptM = [H*corr, A'*corr; A,B]; % matrix for virtual fields optimization

        %vector containing the polynomial coefficients for Q11 and the
        %Lagrange multipliers
        Yd = OptM\Zd';

        %remove the lagrangian multipliers from the Y vectors because they are
        %of no interest
        Yd(n_nodes + 1: size(Yd)) = [];

        for k = 1:n_points
            % virtual displacement field
            u1vv = zeros(1,n_nodes);
            assemble1 = assemble(k,:);
            u1vv(assemble1) = u1elem(k,:);
            u1vd(k) = u1vv*Yd; % for Q66
        end
        % dynamics code
        %calculating Q11 from the first optimized virtual field
        Q = -material.rho*(L*w/n_points)*(sum(u1vd'.*A1));

        % compute difference between the current and previous identified values
        delta = sum((Qold-Q).^2./Q.^2);

        Qold = Q; % store current parameters as old before the next iteration
        i = i+1; % increment the step
    end

    % Final results
    identQVFOpt(j) = Q;

    for k = 1:n_points
        % virtual strain fields
        Eps6vv = zeros(1,n_nodes);
        Eps6vv(assemble1) = Eps6elem(k,:);
        Eps6vd(k) = Eps6vv*Yd;
    end

    %reshape vectors in matrices
    uxStar(:,:,j) = reshape(u1vd, rows, cols);
    eps6Star(:,:,j) = reshape(Eps6vd, rows, cols);

    %reset for next frame
    B66 = zeros(1,n_nodes);
    u1vd = zeros(1,n_points);

end

end

