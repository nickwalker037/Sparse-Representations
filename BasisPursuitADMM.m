function v = bp_admm(CA, b, lambda)
% BP_ADMM Solve Basis Pursuit problem via ADMM
%
% Solves the following problem:
%   min_x 1/2*||b - CAx||_2^2 + lambda*|| x ||_1
%
% The solution is returned in the vector v.
 
% Set the accuracy tolerance of ADMM, run for at most max_admm_iters
tol_admm = 1e-4;
max_admm_iters = 10; %was 100
 
CAtb = [];
for col_no = 1:size(CA,2)
  inner_prod = abs(CA(:,col_no)'*b); % maximize the inner product
  CAtb = [CAtb inner_prod];
end


M = CA'*CA + eye(size(CA,2));
L = chol(M, 'lower');
%U = L';
%M2 = L*L';




% Force Matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
 
% Initialize v
v = zeros(size(CA,2),1);
 
% Initialize u
u = zeros(size(CA,2),1);
 
% Initialize the previous estimate of v, used for convergence test
v_prev = zeros(size(CA,2),1);
 
% main loop
for i = 1:max_admm_iters
 
    % x-update via Cholesky factorization. Solve the linear system
    % (CA'*CA + I)x = (CAtb + v - u)   
    x = inv(CA'*CA + eye(size(CA,2)))*(CA'*b+v_prev-u);

    % v-update via soft thresholding
    v = thresh(x+u,lambda,'soft');

    %  u-update according to the ADMM formula
    u = u + x - v;
    
    % Check if converged   
    if norm(v) && (norm((v - v_prev)) / norm(v)) < tol_admm
         break;
    end
    
    % Save the previous estimate in v_prev
    v_prev = v;
 
end
 
end

