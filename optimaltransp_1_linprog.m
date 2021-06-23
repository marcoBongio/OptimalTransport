%% Optimal Transport with Linear Programming
perform_toolbox_installation('signal', 'general');

%% Optimal Transport of Discrete Distribution
%%
% Dimensions \(n_0, n_1\) of the clouds.
n0 = 3;
n1 = 4;

% Compute a first point cloud \(X_0\) that is Gaussian.
% and a second point cloud \(X_1\) that is Gaussian mixture.
gauss = @(q,a,c)a*randn(2,q)+repmat(c(:), [1 q]);
X0 = randn(2,n0)*.3;
X1 = [gauss(n1/2,.5, [0 1.6]) gauss(n1/4,.3, [-1 -1]) gauss(n1/4,.3, [1 -1])];

% Density weights \(p_0, p_1\).
normalize = @(a)a/sum(a(:));
p0 = normalize(rand(n0,1));
p1 = normalize(rand(n1,1));

%%
% Test values for x and y used for the report.

X0 = [0.05856478057150 0.02076654631178 -0.4996929353644; 
      0.2666586461648 0.7460463960892 -0.1247783834556 ];

X1 = [ -0.04211230323545 0.7280736002377 -1.034463175576 1.225448457091;
       1.644625973754 1.709726759319 -0.9794192591551 -1.206828117377 ];
  
%%
% Shortcut for display.
myplot = @(x,y,ms,col)plot(x,y, 'o', 'MarkerSize', ms, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth', 2);

% Display the point clouds.
% The size of each dot is proportional to its probability density weight.

clf; hold on;
for i=1:length(p0)
    myplot(X0(1,i), X0(2,i), p0(i)*length(p0)*10, 'b');
end
for i=1:length(p1)
    myplot(X1(1,i), X1(2,i), p1(i)*length(p1)*10, 'r');
end
axis([min(X1(1,:)) max(X1(1,:)) min(X1(2,:)) max(X1(2,:))]); axis off;

%%
% Compute the weight matrix
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

% Compute the optimal transport plan.
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
% Find the \(i,j\) with non-zero \(\ga_{i,j}^\star\).
[I,J,gammaij] = find(gamma);

% Display the evolution of \(\mu_t\) for a varying value of \(t \in [0,1]\).
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
% Same number of points
n0 = 4;
n1 = n0;

% Compute points clouds.
X0 = randn(2,n0)*.3;
X1 = [gauss(n1/2,.5, [0 1.6]) gauss(n1/4,.3, [-1 -1]) gauss(n1/4,.3, [1 -1])];

% Constant distributions.
p0 = ones(n0,1)/n0;
p1 = ones(n1,1)/n1;

%%
% Test values for x and y used for the report.

X0 = [-0.1422404202120 -0.6219079297018 -0.09457187428494 -0.2639644186302;
       0.2962145979051 0.3732078359950 0.4197881996802 -0.2767417366056 ];

X1 = [1.157201449266 -0.1548838173838 -0.6689947798329 0.3208020332562;
        1.894006602047 1.083517926325 -0.8973673207685 -1.270985745005];

%%
% Display the coulds.
clf; hold on;
myplot(X0(1,:), X0(2,:), 10, 'b');
myplot(X1(1,:), X1(2,:), 10, 'r');
axis equal; axis off;

%%
% Display the coulds.
C = repmat( sum(X0.^2)', [1 n1] ) + ...
    repmat( sum(X1.^2), [n0 1] ) - 2*X0'*X1;
%%
% Solve the optimal transport.
gamma = otransp(C,p0,p1);

%%
% Show that \(\ga\) is a binary permutation matrix.
clf;
imageplot(gamma);

%%
% Display the optimal assignement.
clf; hold on;
[I,J,~] = find(gamma);
for k=1:length(I)
    h = plot( [X0(1,I(k)) X1(1,J(k))], [X0(2,I(k)) X1(2,J(k))], 'k' );
    set(h, 'LineWidth', 2);
end
myplot(X0(1,:), X0(2,:), 10, 'b');
myplot(X1(1,:), X1(2,:), 10, 'r');
axis equal; axis off;

    