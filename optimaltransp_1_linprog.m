%% Optimal Transport with Linear Programming
perform_toolbox_installation('signal', 'general');

%% Optimal Transport of Discrete Distribution
%%

n0 = 3;
n1 = 4;

gauss = @(q,a,c)a*randn(2,q)+repmat(c(:), [1 q]);
X0 = randn(2,n0)*.3;
X1 = [gauss(n1/2,.5, [0 1.6]) gauss(n1/4,.3, [-1 -1]) gauss(n1/4,.3, [1 -1])];

normalize = @(a)a/sum(a(:));
p0 = normalize(rand(n0,1));
p1 = normalize(rand(n1,1));

%%

myplot = @(x,y,ms,col)plot(x,y, 'o', 'MarkerSize', ms, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth', 2);

clf; hold on;
for i=1:length(p0)
    myplot(X0(1,i), X0(2,i), p0(i)*length(p0)*10, 'b');
end
for i=1:length(p1)
    myplot(X1(1,i), X1(2,i), p1(i)*length(p1)*10, 'r');
end
axis([min(X1(1,:)) max(X1(1,:)) min(X1(2,:)) max(X1(2,:))]); axis off;

%%
C = repmat( sum(X0.^2)', [1 n1] ) + ...
    repmat( sum(X1.^2), [n0 1] ) - 2*X0'*X1;

%Define a function handle that transform a x matrix in a column vector
flat = @(x)x(:);
%Define two function handles that generate an efficient sparse matrix
Cols = @(n0,n1)sparse( flat(repmat(1:n1, [n0 1])), ...
             flat(reshape(1:n0*n1,n0,n1) ), ...
             ones(n0*n1,1) );
Rows = @(n0,n1)sparse( flat(repmat(1:n0, [n1 1])), ...
             flat(reshape(1:n0*n1,n0,n1)' ), ...
             ones(n0*n1,1) );
 % Define a function handle that expresses the coupling constraint matrix
Sigma = @(n0,n1)[Rows(n0,n1);Cols(n0,n1)];

%%
%Maximum number of iterations
maxit = 1e4; 
%Tolerance
tol = 1e-9;

otransp = @(C,p0,p1)reshape( perform_linprog( ...
        Sigma(length(p0),length(p1)), ...
        [p0(:);p1(:)], C(:), 0, maxit, tol), [length(p0) length(p1)] );
    
gamma = otransp(C,p0,p1);

%%
fprintf('Number of non-zero: %d (n0+n1-1=%d)\n', full(sum(gamma(:)~=0)), n0+n1-1);

%%
fprintf('Constraints deviation (should be 0): %.2e, %.2e.\n', norm(sum(gamma,2)-p0(:)),  norm(sum(gamma,1)'-p1(:)));

%% Displacement Interpolation
%%
[I,J,gammaij] = find(gamma);

clf;
tlist = linspace(0,1,6);
for i=1:length(tlist)
    t=tlist(i);
    Xt = (1-t)*X0(:,I) + t*X1(:,J);
    subplot(2,3,i);
    hold on;
    for i=1:length(gammaij)
        myplot(Xt(1,i), Xt(2,i), gammaij(i)*length(gammaij)*6, [t 0 1-t]);
    end
    title(['t=' num2str(t,2)]);
    axis([min(X1(1,:)) max(X1(1,:)) min(X1(2,:)) max(X1(2,:))]); axis off;
end


%% Optimal Assignement
%%
n0 = 4;
n1 = n0;

X0 = randn(2,n0)*.3;
X1 = [gauss(n1/2,.5, [0 1.6]) gauss(n1/4,.3, [-1 -1]) gauss(n1/4,.3, [1 -1])];

p0 = ones(n0,1)/n0;
p1 = ones(n1,1)/n1;
%%
clf; hold on;
myplot(X0(1,:), X0(2,:), 10, 'b');
myplot(X1(1,:), X1(2,:), 10, 'r');
axis equal; axis off;

%%
C = repmat( sum(X0.^2)', [1 n1] ) + ...
    repmat( sum(X1.^2), [n0 1] ) - 2*X0'*X1;
%%
gamma = otransp(C,p0,p1);

%%
clf;
imageplot(gamma);

%%
clf; hold on;
[I,J,~] = find(gamma);
for k=1:length(I)
    h = plot( [X0(1,I(k)) X1(1,J(k))], [X0(2,I(k)) X1(2,J(k))], 'k' );
    set(h, 'LineWidth', 2);
end
myplot(X0(1,:), X0(2,:), 10, 'b');
myplot(X1(1,:), X1(2,:), 10, 'r');
axis equal; axis off;

    