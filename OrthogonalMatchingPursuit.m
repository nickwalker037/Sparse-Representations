% Orthogonal Matching Pursuit Algorithm

A = [ 0.1817   0.5394 -0.1197  0.6404;
      0.6198   0.1994  0.0946 -0.3121;
     -0.7634  -0.8181  0.9883  0.7018];

b = [1.1862; -0.1158; -0.1093];

xval = [];

% -------------------------------------------------------------
% ----------------- Version 1 ---------------------------------
% -------------------------------------------------------------

% This creates arrays of the residual value for different values of our support vector x
% We search for the best solution with a support of only 1 atom
      % This is done by finding the column for A that leads to the best match between Ax and b
for j = 1:4
  x = [0; 0; 0; 0];
  xval = [];
  residual = [];
  for i = -5.0:0.001:5.0
    x(j) = i; % iterate the test values through each row in the x vector
    res = norm((A*x)-b,2); % calculate the residual based on the new x value
    xval = [xval, i];
    residual = [residual, res];
    
  end;
  fprintf('The minimum residual value for row number %.0f where x = %.3f is: %.4f \n', j, xval(find(residual == min(residual))),min(residual))
end


%x = [0;0.915;0;0.719];







% -------------------------------------------------------------
% ----------------- Version 2 ---------------------------------
% -------------------------------------------------------------



resid = norm((A*x)-b,2);
while resid > k
  col_res_values = [];
  col_x_values = [];
  for j = 1:size(A,2)
    if x(j) == 0
      res = [];
      xvals = [];
      for i = -3.0:0.01:3.0
        x(j) = i;
        res_temp = norm((A*x)-b,2); 
        res = [res, res_temp]; 
        xvals = [xvals, i]; 
      end
      min_res = min(res);
      min_x = xvals(find(res == min_res));
      x(j)=0;
    else
      min_res = 100;
      min_x = 100;
    end
  col_res_values = [col_res_values, min_res]; 
  col_x_values = [col_x_values, min_x];
  end
col_index = find(col_res_values == min(abs(col_res_values)));
x(col_index) = col_x_values(col_index);
resid = abs(norm((A*x)-b,2));
s=nnz(sparse(x));
end
y = x;

















