% In this project we will solve a variant of the P0^\epsilon for filling-in 
% missing pixels (also known as "inpainting") in a synthetic image.
 
close all;
%clear; clc;
 
%% Parameters
 
% TODO: Set the size of the desired image is (n x n)
% Write your code here... n = ????;
n = 40^2;

% TODO: Set the number of atoms
% Write your code here... m = ????;
m = 2*n;

% TODO: Set the percentage of known data
% Write your code here... p = ????;
p = .05;

% TODO: Set the noise std
% Write your code here... sigma = ????;
sigma = 0.5;

% TODO: Set the cardinality of the representation
% Write your code here... true_k = ????;
true_k = 15;

% Base seed - A non-negative integer used to reproduce the results
% TODO: Set an arbitrary value for base_seed
% Write your code here... base_seed = ????;
base_seed = 120;


% Run the different algorithms for num_experiments and average the results
num_experiments = 10;
 
 
%% Create a dictionary A of size (n^2 x m) for Mondrian-like images
 
% TODO: initialize A with zeros
% Write your code here... A = ????;
A = sparse(zeros(n, m));
 
% In this part we construct A by creating its atoms one by one, where
% each atom is a rectangle of random size (in the range 5-20 pixels),
% position (uniformly spread in the area of the image), and sign. 
% Lastly we will normalize each atom to a unit norm.
for i=1:size(A,2) % for each column

    % Choose a specific random seed to reproduce the results
    randn('seed',i + base_seed); % my version of Octave doesn't have "rng" functionality
    
    empty_atom_flag = 1;    
    while empty_atom_flag
        
        % TODO: Create a rectangle of random size and position
        % Write your code here... atom = ????;
        x_rand = round((20-5).*rand(1)+5);
        y_rand = round((20-5).*rand(1)+5);
        position_x_rand = round(((40-x_rand)-1).*rand(1)+1);
        position_y_rand = round(((40-y_rand)-1).*rand(1)+1); % by including the y_rand we can guarantee the entire atom rectangle is within the image
        
        sign_rand = ((rand(1) > 0.5)*2 - 1);
        
        atom = ones(x_rand, y_rand)*sign_rand;
        
        A(position_x_rand:position_x_rand+x_rand,position_y_rand:position_y_rand+y_rand) = atom;
        
        
        % Verify that the atom is not empty or nearly so
        if norm(atom(:)) > 1e-5
            empty_atom_flag = 0;
            
            % TODO: Normalize the atom
            % Write your code here... atom = ????;
        
            
            
            % Assign the generated atom to the matrix A
            A(:,i) = atom(:);
        end
        
    end
    
end
 
%% Oracle Inpainting
 
% Allocate a vector to store the PSNR results
PSNR_oracle = zeros(num_experiments,1);
 
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Choose a specific random seed to reproduce the results
    rng(experiment + base_seed);
    
    % Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k);
    
    % TODO: Compute the subsampled dictionary
    % Write your code here... A_eff = ????;
    
    
    % TODO: Compute the oracle estimation
    % Write your code here... x_oracle = ????;
        
    
    % Compute the estimated image    
    b_oracle = A*x_oracle;
    
    % Compute the PSNR
    PSNR_oracle(experiment) = compute_psnr(b0, b_oracle);
    
end
 
% Display the average PSNR of the oracle
fprintf('Oracle: Average PSNR = %.3f\n', mean(PSNR_oracle));
 
%% Greedy: OMP Inpainting
 
% We will sweep over k = 1 up-to k = max_k and pick the best result
max_k = min(2*true_k, m);
 
% Allocate a vector to store the PSNR estimations per each k
PSNR_omp = zeros(num_experiments,max_k);
 
% Loop over the different realizations
for experiment = 1:num_experiments
    
    % Choose a specific random seed to reproduce the results
    rng(experiment + base_seed);
    
    % Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k);
    
    % Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
   
    % Run the OMP for various values of k and pick the best results
    for k_ind = 1:max_k
        
        % Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, k_ind);
        
        % Un-normalize the coefficients
        x_omp = x_omp./atoms_norm';
        
        % Compute the estimated image        
        b_omp = A*x_omp;
        
        % Compute the current PSNR
        PSNR_omp(experiment, k_ind) = compute_psnr(b0, b_omp);
        
        % Save the best result of this realization, we will present it later
        if PSNR_omp(experiment, k_ind) == max(PSNR_omp(experiment, :))
            best_b_omp = b_omp;
        end
        
    end
    
end
 
% Compute the best PSNR, computed for different values of k
PSNR_omp_best_k = max(PSNR_omp,[],2);
 
% Display the average PSNR of the OMP (obtained by the best k per image)
fprintf('OMP: Average PSNR = %.3f\n', mean(PSNR_omp_best_k));
 
% Plot the average PSNR vs. k
psnr_omp_k = mean(PSNR_omp,1);
figure(1); plot(1:max_k, psnr_omp_k, '-*r', 'LineWidth', 2);
ylabel('PSNR [dB]'); xlabel('k'); grid on;
title(['OMP: PSNR vs. k, True Cardinality = ' num2str(true_k)]);
 
 
%% Convex relaxation: Basis Pursuit Inpainting via ADMM
 
% We will sweep over various values of lambda
num_lambda_values = 10;
 
% Allocate a vector to store the PSNR results obtained for the best lambda
PSNR_admm_best_lambda = zeros(num_experiments,1);
 
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Choose a specific random seed to reproduce the results
    rng(experiment + base_seed);
    
    % Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k);
    
    % Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
    
    % Run the BP for various values of lambda and pick the best result
    lambda_max = norm( A_eff_normalized'*b, 'inf' );
    lambda_vec = logspace(-5,0,num_lambda_values)*lambda_max;    
    psnr_admm_lambda = zeros(1,num_lambda_values);
    
    % Loop over various values of lambda
    for lambda_ind = 1:num_lambda_values
        
        % Compute the BP estimation
        x_admm = bp_admm(A_eff_normalized, b, lambda_vec(lambda_ind));
        
        % Un-normalize the coefficients
        x_admm = x_admm./atoms_norm';
        
        % Compute the estimated image        
        b_admm = A*x_admm;
        
        % Compute the current PSNR
        psnr_admm_lambda(lambda_ind) = compute_psnr(b0, b_admm);
        
        % Save the best result of this realization, we will present it later
        if psnr_admm_lambda(lambda_ind) == max(psnr_admm_lambda)
            best_b_admm = b_admm;
        end
        
    end
    
    % Save the best PSNR
    PSNR_admm_best_lambda(experiment) = max(psnr_admm_lambda);
    
end
 
% Display the average PSNR of the BP
fprintf('BP via ADMM: Average PSNR = %.3f\n', mean(PSNR_admm_best_lambda));
 
% Plot the PSNR vs. lambda of the last realization
figure(2); semilogx(lambda_vec, psnr_admm_lambda, '-*r', 'LineWidth', 2);
ylabel('PSNR [dB]'); xlabel('\lambda'); grid on;
title('BP via ADMM: PSNR vs. \lambda');
 
%% show the results
 
% Show the images obtained in the last realization, along with their PSNR
figure(3); 
subplot(2,3,1); imagesc(reshape(full(b0),n,n)); 
colormap(gray); axis equal;
title(['Original Image, k = ' num2str(true_k)]);
 
subplot(2,3,2); imagesc(reshape(full(b0_noisy),n,n)); 
colormap(gray); axis equal;
title(['Noisy Image, PSNR = ' num2str(compute_psnr(b0, b0_noisy))]);
 
subplot(2,3,3); imagesc(reshape(full(C'*b),n,n)); 
colormap(gray); axis equal;
title(['Corrupted Image, PSNR = ' num2str(compute_psnr(b0, C'*b))]);
 
subplot(2,3,4); imagesc(reshape(full(b_oracle),n,n)); 
colormap(gray); axis equal;
title(['Oracle, PSNR = ' num2str(compute_psnr(b0, b_oracle))]);
 
subplot(2,3,5); imagesc(reshape(full(best_b_omp),n,n)); 
colormap(gray); axis equal;
title(['OMP, PSNR = ' num2str(compute_psnr(b0, best_b_omp))]);
 
subplot(2,3,6); imagesc(reshape(full(best_b_admm),n,n));
colormap(gray); axis equal;
title(['BP-ADMM, PSNR = ' num2str(compute_psnr(b0, best_b_admm))]);
 
%% Compare the results

% Show a bar plot of the average PSNR value obtained per each algorithm
figure(4);
mean_psnr = [mean(PSNR_oracle) mean(PSNR_omp_best_k) mean(PSNR_admm_best_lambda)];
bar(mean_psnr);
set(gca,'XTickLabel',{'Oracle','OMP','BP-ADMM'});
ylabel('PSNR [dB]'); xlabel('Algorithm');
 
%% Run OMP with fixed cardinality and increased percentage of known data
 
% TODO: Set the noise std
% Write your code here... sigma = ????;



% TODO: Set the cardinality of the representation
% Write your code here... true_k = ????;



% TODO: Create a vector of increasing values of p in the range [0.4 1]. The
% length of this vector equal to num_values_of_p = 7.
% Write your code here... num_values_of_p = ????; p_vec = ????;




% We will repeat the experiment for num_experiments realizations
num_experiments = 100;
 
% Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_p = zeros(num_values_of_p,1);
 
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Loop over various values of p
    for p_ind = 1:num_values_of_p
        
        % Choose a specific random seed to reproduce the results
        rng(experiment + base_seed);
        
        % Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p_vec(p_ind), sigma, true_k);
                
        % Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
        
        % Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k);
        
        % Un-normalize the coefficients
        x_omp = x_omp./atoms_norm';
        
        % Compute the estimated image        
        b_omp = A*x_omp;
                
        % TODO: Compute the MSE of the estimate
        % Write your code here... cur_mse = ????;

        
                
        % Compute the current normalized MSE and aggregate
        mse_omp_p(p_ind) = mse_omp_p(p_ind) + cur_mse / noise_std^2;
 
    end
    
end
 
% Compute the average PSNR over the different realizations
mse_omp_p = mse_omp_p / num_experiments;
 
% Plot the average normalized MSE vs. p
figure(5); plot(p_vec, mse_omp_p, '-*r', 'LineWidth', 2);
ylabel('Normalized-MSE'); xlabel('p'); grid on;
title(['OMP with k = ' num2str(true_k) ', Normalized-MSE vs. p'])
 
 
%% Run OMP with fixed cardinality and increased noise level
 
% TODO: Set the cardinality of the representation
% Write your code here... true_k = ????;



% TODO: Set the percentage of known data
% Write your code here... p = ????;



% TODO: Create a vector of increasing values of sigma in the range [0.15 0.5].
% The length of this vector equal to num_values_of_sigma = 10.
% Write your code here... num_values_of_sigma = ????; sigma_vec = ????;



% Repeat the experiment for num_experiments realizations
num_experiments = 100;
 
% Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_sigma = zeros(num_values_of_sigma,1);
 
% Loop over num_experiments
for experiment = 1:num_experiments
    
    % Loop over increasing noise level
    for sigma_ind = 1:num_values_of_sigma
        
        % Choose a specific random seed to reproduce the results
        rng(experiment + base_seed);
        
        % Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma_vec(sigma_ind), true_k);
        
        % Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A);
        
        % Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k);
        
        % Un-normalize the coefficients
        x_omp = x_omp./atoms_norm';
        
        % Compute the estimated image        
        b_omp = A*x_omp;
                
        % TODO: Compute the MSE of the estimate
        % Write your code here... cur_mse = ????;

        
        
        % Compute the current normalized MSE and aggregate
        mse_omp_sigma(sigma_ind) = mse_omp_sigma(sigma_ind) + cur_mse / noise_std^2;
 
    end
    
end
 
% Compute the average PSNR over the different realizations
mse_omp_sigma = mse_omp_sigma / num_experiments;
    
% Plot the average normalized MSE vs. sigma
figure(6); plot(sigma_vec, mse_omp_sigma, '-*r', 'LineWidth', 2);
ylim([0.5*min(mse_omp_sigma) 5*max(mse_omp_sigma)]);
ylabel('Normalized-MSE'); xlabel('sigma'); grid on;
title(['OMP with k = ' num2str(true_k) ', Normalized-MSE vs. sigma']);


