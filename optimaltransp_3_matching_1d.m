%% Optimal Transport in 1-D
perform_toolbox_installation('signal', 'general');

%% Optimal Transport and Assignement
%%
% Load an image.

n = 256;
f = rescale( load_image('lena', n) );

%%
% Display it.

clf;
imageplot(f);

%%
% Load another image.

g = rescale( mean(load_image('fingerprint', n),3) );

%%
% Display it.

clf;
imageplot(g);

%%
% Compute the histogram of the first image.

Q = 50;
[h,t] = hist(f(:), Q);

%%
% Display the histogram.

clf;
bar(t,h*Q/n^2); axis('tight');

%% Compare the two histograms.

Q = 50;
[h0,t] = hist(f(:), Q);
[h1,t] = hist(g(:), Q);
clf;
subplot(2,1,1);
bar(t,h0*Q/n^2); axis([0 1 0 6]);
subplot(2,1,2);
bar(t,h1*Q/n^2); axis([0 1 0 6]);

%% 1-D Optimal Assignement
%%
% Sort f and g.

[~,sigmaf] = sort(f(:)); %column vector with 65536 elements
[~,sigmag] = sort(g(:)); %column vector with 65536 elements

%%
% Compute the inverse permutation.

sigmafi = []; %column vector with 65536 elements
sigmafi(sigmaf) = 1:n^2;

%%
% Compute the optimal permutation.

sigma = sigmag(sigmafi); %column vector with 65536 elements

%%
% Compute the projection.
clf;
f1 = reshape(g(sigma), [n n]);
imageplot(f1);
%%
% Compare before/after equalization.

clf;
imageplot(f, 'f', 1,2,1);
imageplot(f1, '\pi_g(f)',  1,2,2);

%%
% Create an image with flat histogram.

for i=1:n
    for j=1:n
        g(i,j)=i/(n);
    end
end
clf;
imageplot(g);

%%
% Check the histogram.
Q = 64;
[h,t] = hist(g(:), Q);
clf;
bar(t,h*Q/n^2); axis("tight"); axis([0 1 0 2.5]);

%%
% Histogtam equalization with flat histogram.

[~,sigmaf] = sort(f(:)); 
[~,sigmag] = sort(g(:)); 
sigmafi = []; 
sigmafi(sigmaf) = 1:n^2;
sigma = sigmag(sigmafi); 
f1 = reshape(g(sigma), [n n]);
imageplot(f1);
clf;
imageplot(f, 'f', 1,2,1);
imageplot(f1, '\pi_g(f)',  1,2,2);

%%
% Histogram equalization using histeq().
J = histeq(f);
clf;
subplot(3,1,1);
imageplot(f, "Original:");
subplot(3,1,2);
imageplot(f1, "Optimal Tranport Equalization:");
subplot(3,1,3);
imageplot(J, "Image Processing Toolbox:");

%%
%To display the histeq() histogram.

[h,t] = hist(J(:), Q);
clf;
bar(t,h*Q/n^2); axis([0 1 0 2.5]);

%% Histogram Interpolation
%%
% Define the interpolation operator.

ft = @(t)reshape( t*f1 + (1-t)*f, [n n]); 

%%
% The midway equalization is obtained for \(t=1/2\).

clf;
imageplot(ft(1/2));

%% Display the progression of the interpolation of the histograms.

p = 64;
tlist = linspace(0,1,5);
for i=1:length(tlist)
    a = ft(tlist(i));
    subplot(length(tlist), 1, i);
    [h,t] = hist(a(:), p);
    bar(t,h*p/n^2);
    axis([0 1 0 2.5]);
end