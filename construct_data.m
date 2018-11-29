function [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, k)
% CONSTRUCT_DATA Generate Mondrian-like synthetic image
%
% Input:
%  A     - Dictionary of size (n^2 x m)
%  p     - Percentage of known data in the range (0 1]
%  sigma - Noise std
%  k     - Cardinality of the representation of the synthetic image in the
%          range [1 max(m,n)]
%
% Output:
%  x0 - A sparse vector creating the Mondrian-like image b0
%  b0 - The original image of size n^2
%  noise_std  - The standard deviation of the noise added to b0
%  b0_noisy   - A noisy version of b0 of size n^2
%  C  - Sampling matrix of size (p*n^2 x n^2), 0 < p <= 1
%  b  - The corrupted image (noisy and subsampled version of b0) of size p*n^2
 
 
% Get the size of the image and number of atoms
[n_squared, m] = size(A);
n = sqrt(n_squared);
 
%% generate a Mondrian image
%  by drawing at random a sparse vector x0 of length m with cardinality k
 
% TODO: Draw at random the locations of the non-zeros
% Write your code here... nnz_locs = ????;
nnz_locs = sparse(randperm(m, k)); % m represents the column chosen, k represents how many columns to choose

 
% TODO: Draw at random the values of the coefficients
% Write your code here... nnz_vals = ????;
nnz_vals = randn(1, k);
 
% TODO: Create a k-sparse vector x0 of length m given the nnz_locs and nnz_vals
% Write your code here... x0 = ????;
x0 = sparse(zeros(m,1));
x0(nnz_locs) = nnz_vals;

 
% TODO: Given A and x0, compute the signal b0
% Write your code here... b0 = ????;
b0 = A*x0;
 
%% Create the measured data vector b of size n^2
 
% TODO: Compute the dynamic range
% Write your code here... dynamic_range = ????;
dynamic_range = max(b0) - min(b0);
 
% Create a noise vector
noise_std = sigma*dynamic_range;
noise = noise_std*randn(n^2,1);
 
% TODO: Add noise to the original image
% Write your code here... b0_noisy = ????;
b0_noisy = A*x0 + randn(1)*(sigma*dynamic_range);
 
 
%% Create the sampling matrix C of size (p*n^2 x n^2), 0 < p <= 1
 
% TODO: Create an identity matrix of size (n^2 x n^2)
% Write your code here... I = ????;

 
% TODO: Draw at random the indices of rows to be kept
% Write your code here... keep_inds = ????;

 
% TODO: Create the sampling matrix C of size (p*n^2 x n^2) by keeping rows
% from I that correspond to keep_inds
% Write your code here... C = ????;

 
% TODO: Create a subsampled version of the noisy image
% Write your code here... b = ????;

 
end
 
