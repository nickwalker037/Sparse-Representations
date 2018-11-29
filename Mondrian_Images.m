
% Generating a random Mondian-like rectangle within a matrix A

A=zeros(40,40)
x_rand = round((20-5).*rand(1)+5)
y_rand = round((20-5).*rand(1)+5)
position_x_rand = round(((40-x_rand)-1).*rand(1)+1)
position_y_rand = round(((40-y_rand)-1).*rand(1)+1)
rect = ones(x_rand,y_rand)

A(position_x_rand:x_rand+position_x_rand,position_y_rand:y_rand+position_y_rand)
A(position_x_rand:x_rand+position_x_rand-1,position_y_rand:y_rand+position_y_rand-1)=rect

% AND THEN THIS RECTANGLE IS REPRESENTED USING A COLUMN OF THE DICTIONARY..........
...... SO THE e.g. 150 VALUES OF THE RECTANGLE WILL BE REPRESENTED BY 15 SPARSE VALUES IN ATOM COLUMN X

i.e. EACH COLUMN OF THE DICTIONARY A (ATOM) REPRESENTS A RECTANGLE OF RANDOM HEIGHT AND WIDTH
