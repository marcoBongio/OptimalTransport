%% Entropic Regularization of Optimal Transport
perform_toolbox_installation('signal', 'general');

%% Transport Between Point Clouds
%%
% Number of points in each cloud.

N = [5,4];

%%
% Dimension of the clouds.

d = 2;

%%
% Point cloud x, of N_1 points inside a square.

x = rand(2,N(1))-.5;

%%
% Point cloud y, of N_2 points inside a ring.

theta = 2*pi*rand(1,N(2));
r = .8 + .2*rand(1,N(2));
y = [cos(theta).*r; sin(theta).*r];

%%
% Test values for x and y used for the report.
x = [0.2922 0.1557 0.3491 0.1787 0.2431;
     0.4594 -0.4642 0.4339 0.2577 -0.1077];
y = [-0.4786 0.3845 -0.2234 0.9454;
     -0.7089 0.7120 -0.7883 0.1926];
%%
% Shortcut for displaying point clouds.

plotp = @(x,col)plot(x(1,:)', x(2,:)', 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', col, 'LineWidth', 2);

%%
% Display of the two clouds.

clf; hold on;
plotp(x, 'b');
plotp(y, 'r');
axis('off'); axis('equal');

%%
% Target histograms, here uniform histograms.

p = ones(N(1),1)/N(1);
q = ones(N(2),1)/N(2);

%%
% Regularization strength.

gamma = .01;

%%
% Cost matrix.

x2 = sum(x.^2,1); y2 = sum(y.^2,1);
C = repmat(y2,N(1),1)+repmat(x2.',1,N(2))-2*x.'*y;

%%
% Gibbs Kernel.

xi = exp(-C/gamma);

%% Sinkhorn algorithm
% Display the evolution of the constraints satisfaction errors.

b = ones(N(2),1);
niter = 300;
Err_p = []; Err_q = [];
for i=1:niter
    a = p ./ (xi*b);
    Err_q(end+1) = norm( b .* (xi'*a) - q )/norm(q);
    b = q ./ (xi'*a);
    Err_p(end+1) = norm( a .* (xi*b) - p )/norm(p);
end

% Compute the final matrix.

Pi = diag(a)*xi*diag(b);

%%
% Display it.

clf;
imageplot(Pi);

%%
% Display the violation of constraint error in log-plot.

clf;
subplot(2,1,1);
plot(log10(Err_p)); axis tight; title('log|\pi 1 - p|');
subplot(2,1,2);
plot(log10(Err_q)); axis tight; title('log|\pi^T 1 - q|');

%%
% Keep only the highest entries of the coupling matrix, and use them to
% draw a map between the two clouds.

clf;
hold on;
A = sparse( Pi .* (Pi> min(1./N)*.7) ); [i,j,~] = find(A);
h = plot([x(1,i);y(1,j)], [x(2,i);y(2,j)], 'k');
set(h, 'LineWidth', 2); % weaker connections.
A = sparse( Pi .* (Pi> min(1./N)*.3) ); [i,j,~] = find(A);
h = plot([x(1,i);y(1,j)], [x(2,i);y(2,j)], 'k:');
set(h, 'LineWidth', 1);
plotp(x, 'b'); % plot the two point clouds.
plotp(y, 'r');
axis('off'); axis('equal');

%% Display the regularized transport solution for various values of \(\gamma\).
%% For a too small value of \(\gamma\), what do you observe ?
glist = [.1 .01 .005 .001 ];
niter = 300;
clf;
for k=1:length(glist)
    gamma = glist(k);
    xi = exp(-C/gamma);
    b = ones(N(2),1);
    for i=1:niter
        a = p ./ (xi*b);
        b = q ./ (xi'*a);
    end
    Pi = diag(a)*xi*diag(b);
    imageplot(Pi, ['\gamma=' num2str(gamma)], 2,2,k);
end

%%
% Compute the obtained optimal \(\pi\).

Pi = diag(a)*xi*diag(b);

%%
gamma = 0.002;
xi = exp(-C/gamma);
b = ones(N(2),1);

a = p ./ (xi*b);
b = q ./ (xi'*a);

%%
% Keep only the highest entries of the coupling matrix, and use them to
% draw a map between the two clouds.

clf;
hold on;
A = sparse( Pi .* (Pi> min(1./N)*.7) ); [i,j,~] = find(A);
h = plot([x(1,i);y(1,j)], [x(2,i);y(2,j)], 'k');
set(h, 'LineWidth', 2); % weaker connections.
A = sparse( Pi .* (Pi> min(1./N)*.3) ); [i,j,~] = find(A);
h = plot([x(1,i);y(1,j)], [x(2,i);y(2,j)], 'k:');
set(h, 'LineWidth', 1);
plotp(x, 'b'); % plot the two point clouds.
plotp(y, 'r');
axis('off'); axis('equal');

%% Log-Domain Sinkhorn's Algorithm
%%

p = ones(N(1),1)/N(1); 
q = ones(1,N(2))/N(2);

%%
% Compute the regularized minumum operators.

minp = @(H,gamma)-gamma*log( sum(p .* exp(-H/gamma),1) );
minq = @(H,gamma)-gamma*log( sum(q .* exp(-H/gamma),2) );

%%
% Stabilize the regularized minumum operator defined this way is non-stable, but it can

minpp = @(H,gamma)minp(H-min(H,[],1),gamma) + min(H,[],1);
minqq = @(H,gamma)minq(H-min(H,[],2),gamma) + min(H,[],2);

%% Implement Sinkhorn in log domain.

gamma = .01;
e = 1000;
f = zeros(N(1),1);
Err = [];
for it=1:e
    g = minpp(C-f,gamma);
    f = minqq(C-g,gamma);
    % generate the coupling
    Pi = p .* exp((f+g-C)/gamma) .* q;
    % check conservation of mass
    Err(it) = norm(sum(Pi,1)-q,1);    
end

%%
% Display error evolution.
clf;
plot(log10(Err), 'LineWidth', 2);

%%
% Display coupling matrix.
clf;
imageplot(Pi);

%%
% Compute different coupling matrices for different values of gamma.
glist = [.1 .01 .005 .001 ];
e = 300;
clf;
for k=1:length(glist)
    gamma = glist(k);
    f = zeros(N(1),1);
    for i=1:e
        g = minpp(C-f,gamma);
        f = minqq(C-g,gamma);
    end
    Pi = p .* exp((f+g-C)/gamma) .* q;
    imageplot( Pi , ['\gamma=' num2str(gamma)], 2,2,k);
end

%% Transport Between Histograms
%%
% Size N of the histograms.

N = 100;
t = (0:N-1)'/N;

Gaussian = @(t0,sigma)exp( -(t-t0).^2/(2*sigma^2) );
sigma = .06;
p = Gaussian(.25,sigma);
q = Gaussian(.8,sigma);

%%
% Add some minimal mass and normalize.

normalize = @(p)p/sum(p(:));
vmin = .02;
p = normalize( p+max(p)*vmin);
q = normalize( q+max(q)*vmin);

%%
% Display the histograms.

clf;
subplot(2,1,1);
bar(t, p, 'k'); axis tight;
subplot(2,1,2);
bar(t, q, 'k'); axis tight;

%%
% Regularization strength gamma.

gamma = (.03)^2;

% The Gibbs kernel is a Gaussian convolution.

[Y,X] = meshgrid(t,t);
xi = exp( -(X-Y).^2 / gamma); %matrix 100x100

%% Sinkhorn's Algorithm.
% Display the evolution of the constraints satisfaction errors.

b = ones(N,1);
niter = 2000;
Err_p = []; Err_q = [];
for i=1:niter
    a = p ./ (xi*b);
    Err_q(end+1) = norm( b .* (xi*a) - q )/norm(q);
    b = q ./ (xi'*a);
    Err_p(end+1) = norm( a .* (xi'*b) - p )/norm(p);
end
Pi = diag(a)*xi*diag(b);

%%
% Display the violation of constraint error in log-plot.
clf;
subplot(2,1,1);
plot(log10(Err_p)); axis tight; title('log|\pi 1 - p|');
subplot(2,1,2);
plot(log10(Err_q)); axis tight; title('log|\pi^T 1 - q|');

%%
% Display the coupling.

clf;
imageplot(Pi);

%%
% Display the coupling. Use a log domain plot to better vizualize it.

clf;
imageplot(log(Pi+1e-5));

%%
% One can compute an approximation of the transport plan
% between the two measure by computing the so-called barycentric projection
% map.
% This computation can thus be done using only multiplication with the
% kernel xi.

s = (xi*(b.*t)) .* a ./ p;

%%
% Display the transport map over the coupling.

clf; hold on;
imagesc(t,t,log(Pi+1e-5)); colormap gray(256);
plot(s,t, 'r', 'LineWidth', 3);
axis image; axis off; axis ij;
