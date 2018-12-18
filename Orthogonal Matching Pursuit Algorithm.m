
function y = omp(CA, b, k)
% OMP Solve the sparse coding problem via OMP
%
% Solves the following problem:
%   min_x ||b - CAx||_2^2 s.t. ||x||_0 <= k
%
% The solution is returned in the vector x

% Initialize the vector x
x = zeros(size(CA,2),1);


res = b;            % initialize the residual to equal b
col_set = [];       % set of columns that we keep
col_ind = [];       % indexes of the columns we keep 
iter_no = 1;        % initializing the iteration variable


while(iter_no < k+1)
  max_inner_prod = 0;
  
  
  % Find the most contributing column:
  for col_no = 1:size(CA,2)
    inner_prod = abs(CA(:,col_no)'*res); % maximize the absolute inner product
    if inner_prod > max_inner_prod
      max_inner_prod = inner_prod;
      curr_col = col_no;
    end
  end
  
  col_ind = [col_ind curr_col];         % keep indices of columns
  col_set = [col_set CA(:,curr_col)];   % keep atoms
  
  
  % Updating the x coefficient:
  min_res = Inf;
  for x_val = -1.0:0.01:-0.01
    x(col_ind(end)) = x_val;
    res_temp = norm((CA*x)-b,2); 
    if res_temp < min_res
      min_res = res_temp;
      curr_val = x_val;
    end
  end
  for x_val = 0.01:0.01:1.0
    x(col_ind(end)) = x_val;
    res_temp = norm((CA*x)-b,2); 
    if res_temp < min_res
      min_res = res_temp;
      curr_val = x_val;
    end
  end
  
  
  
  x(col_ind(end)) = curr_val;
  
  % Updating the residual: 
  res = b - col_set*x(col_ind);

  % Next iteration:
  iter_no = iter_no+1;


end

y = x;




