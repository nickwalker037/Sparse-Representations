
% in this script, we'll use 3 methods to calculate the Spark of a matrix (the minimum # of linearly dependent columns)




% Creating the matrix A
rand('seed',120);
n=50; m=150; %50 rows by 150 columns
A=randn(n,m); %initializing the matrix entries with an iid Gaussian distribution
  % the above step implies the Spark = 51 since every subset of 50 columns from it must be linearly independent
Spark=13;
A(:,m)=mean(A(:,1:Spark-1),2); % replace the last column in A by the mean of the first 12 columns
  % thus, the spark drops to 13 since there is a set of 13 columns that are linearly dependent

% Normalizing the columns: -- this has no effect on the Spark but makes calculating the coherence easier later on
for k=1:1:m
  A(:,k) = A(:,k)/norm(A(:,k));
end



% =====================================
% Method 1 - Mutual Coherence 
G = A'*A;
G=abs(G);
for k=1:1:m
  G(k,k)=0; % setting the diagonals = to 0
end

mu=max(G(:)); % getting the max normalized value in G
SparkEst1 = ceil(1+1/mu); % calculate the lower bound using mu
fprintf('A lower bound on the spark via mutual coherence is: %d\n',SparkEst1);




% =====================================
% Method 2 - Babel Function
G = A'*A;
G=abs(G);
for k=1:1:m
  G(k,:)=sort(G(k,:),'descend');
end
G = G(:,2:end); % getting rid of the first columns (all 1's taken from the diagonal entries)
G = cumsum(G,2); % get the cumulative sum of each row
mu1 = zeros(20,1); % 20 rows of zero's
for k=1:1:20
  mu1(k)=max(G(:,k)); % get the max value of each of the first 20 columns. 20 is arbitrary, we just need to include as many columns as necessary to find the first mu1>1
end
SparkEst2 = find(mu1>1,1)+1; 
fprintf('A lower bound on the Spark via the Babel function is: %d\n',SparkEst2);



% =====================================
% Method 3 - Evaluating the spark by the Upper-bound

options = optimoptions('linprog','Algorithm','dual-simplex','Display','none','OptimalityTolerance',1.0000e-07);

Z = zeros(m,m);
Zcount=zeros(m,1);
h=waitbar(0,'Sweeping through the LP problems');
set(h,'Position',[500 100 270 56]);
for k=1:1:m
  waitbar(k/m);
  % we convert the problem min ||z||_1 s.t. Az=0 and z_k=1 to Linear Programming by splitting z into the positive and negative entries z=u-v, u, v>=0
  c = ones(2*m,1);
  Aeq = [A,-A];
  indicator = zeros(1,2*m);
  indicator(k) = 1;
  



























%{
final notes:
  - the Spark estimate retrieved via the Babel function is usually > that retrieved via mutual coherence
      - this makes sense as the Babel function is slightly more informative than the mutual coherence


}%




