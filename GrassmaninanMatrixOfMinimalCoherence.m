% Code to synthesize a Grassmaninan matrix of minimal coherence using Tropp's algorithm:


% Create a matrix of size n*m such that its coherence is minimal
n=50; m=100;
rand('seed',10);
A0 = randn(n,m);
A0 = A0*diag(1./sqrt(diag(A0'*A0))); 


% iteratively update a frame such that
%   1. minimal M (mutual coherence)
%   2. Rank(G) = n
%   3. columns are normalized

Iter=1e3; dd1=0.95; dd2=0.95; % 0.95 here is used to identify the entries in G that are the top 5 percent in their absolute values and then shrink them by 0.95
A=A0;
G=A'*A; % compute the Gram matrix
mu=sqrt((m-n)/(n*(m-1)));

Res = zeros(Iter,3);
for k=1:1:Iter
  % shrink the highest inner products
  gg=sort(abs(G(:)));
  pos = find(abs(G(:))>gg(round(dd1*(m^2-m))) & abs(G(:)-1)>1e-6);
  G(pos)=G(pos)*dd2;
  
  % reduce the rank back to n via singular value decomposition
  [U,S,V]=svd(G);
  S(n+1:end,1+n:end)=0;
  G=U*S*V';
  
  % normalize the columns
  % do this by multiplying G from the left and right by a matrix that causes the main diagonal entries to =1
  G=diag(1./sqrt(diag(G)))*G*diag(1./sqrt(diag(G))); 
  
  % show status
  gg=sort(abs(G(:)));
  pos=find(abs(G(:))>gg(round(dd1*(m^2-m))) & abs(G(:)-1)>1e-6);
  %print the iteration #, the ideal mu we'd like to get, the avg abs(off-diagonal elements in G) -- the avg. coherence, and the most extreme value in the off-diagonal elements in G -- the current mutual coherence
  fprintf(1,'%6i %12.8f %12.8f %12.8f \n',[k,mu,mean(abs(G(pos))),max(abs(G(pos)))]);
  Res(k,:)=[mu,mean(abs(G(pos))),max(abs(G(pos)))];
end

% Use SVD in order to extract the resulting matrix A from its Gram matrix -- parallels a sqrt computation
[U,S,V]=svd(G);
Aout=S(1:n,1:n).^0.5*U(:,1:n)';


% Show the results

% Figure 1 displays a tendency to get closer and closer to the welch bound
h=figure(1); clf
set(h,'Position',[415 100 400 200]);
h=semilogx(Res(:,3),'b'); hold on; 
set(h,'LineWidth',2); 
h=semilogx(Res(:,2),'r'); hold on; 
set(h,'LineWidth',2); 
h=semilogx(Res(:,1),'g');
set(h,'LineWidth',2); 
axis([0 10000 0 0.6]); 
legend({'Obtained \mu','mean coherence','Optimal \mu'}); 
set(gca,'FontSize',12);

% in Figure 2, we plot the entries of the abs(G) after omitting the main diagonal
% This shows the final gram is getting very close to the ideal form (having all of its entries equal)
h=figure(2); clf; 
set(h,'Position',[820 100 400 200]);
gg=abs(A'*A); gg=sort((gg(:))); gg=gg(1:end-m); h=plot(gg); 
set(h,'LineWidth',2); 
hold on; 
gg=abs(Aout'*Aout); gg=sort((gg(:))); gg=gg(1:end-m); h=plot(gg,'r'); 
set(h,'LineWidth',2); 
optmu=sqrt((m-n)/n/(m-1)); h=plot([1,m*(m-1)],[optmu,optmu],'g'); 
set(h,'LineWidth',2); 
legend({'Initial Gram','Final Gram','Optimal \mu'}); 
grid on; 
axis([0 m^2-m 0 0.6]);
set(gca,'FontSize',12);
























