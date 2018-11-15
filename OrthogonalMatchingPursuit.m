% Orthogonal Matching Pursuit Algorithm

A = [ 0.1817   0.5394 -0.1197  0.6404;
      0.6198   0.1994  0.0946 -0.3121;
     -0.7634  -0.8181  0.9883  0.7018];

b = [1.1862; -0.1158; -0.1093];

xval = [];


% This creates arrays of the residual value for different values of our support vector x
% We search for the best solution with a support of only 1 atom
      % This is done by finding the column for A that leads to the best match between Ax and b
for j = 1:4
  x = [0; 0; 0; 0];
  xval = [];
  residual = [];
  for i = -5.0:0.01:5.0
    x(j) = i; % iterate the test values through each row in the x vector
    res = norm((A*x)-b,2); % calculate the residual based on the new x value
    xval = [xval, i];
    residual = [residual, res];
  end;
  fprintf('The minimum residual value for row number %.0f is: %.4f \n', j, min(residual))
end


%x = [0;0.915;0;0.719];




















