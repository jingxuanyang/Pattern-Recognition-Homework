%% errorRateEstimation.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.21
% Project: Pattern Recognition
% Purpose: Estimate error rate of Bayesian classifier
% Note:  

clc;
clear;
close all;

%% parameters
% parameter of p(x|\omega_1)
mu1 = [-1, 0];
Sigma1 = eye(2);

% parameter of p(x|\omega_2)
mu2 = [1, 0];
Sigma2 = eye(2);

% draw 1000 samples from p(x|\omega_1), p(x|\omega_2)
N = 1000;
X1 = mvnrnd(mu1,Sigma1,N);
X2 = mvnrnd(mu2,Sigma2,N);

% generate data for EM method
data1 = mvnrnd(mu1,Sigma1,2*N);
data2 = mvnrnd(mu2,Sigma2,2*N);
select = rand(2*N,1) < 0.5;
data = select .* data1 + (1 - select) .* data2;

% range
x = linspace(-5,5);
y = linspace(-5,5);

% testing times
times = 10;

%% calculations
% Gaussian window
peGaussian1 = zeros(1,times);
peGaussian2 = zeros(1,times);
mseGaussian1 = zeros(1,times);
mseGaussian2 = zeros(1,times);
pnGaussian1 = zeros(length(x),length(y));
pnGaussian2 = zeros(length(x),length(y));
for k = 1:times
    a = 0.3;
    X1 = mvnrnd(mu1,Sigma1,N);
    X2 = mvnrnd(mu2,Sigma2,N);
    for i = 1:length(x)
        for j = 1:length(y)
            pnGaussian1(i,j) = sum(mvnpdf(X1,[x(i) y(j)],a^2*Sigma1)) / N;
            pnGaussian2(i,j) = sum(mvnpdf(X2,[x(i) y(j)],a^2*Sigma1)) / N;
            pdf1 = mvnpdf([x(i) y(j)],mu1,Sigma1);
            pdf2 = mvnpdf([x(i) y(j)],mu2,Sigma2);
            if pnGaussian1(i,j) >= pnGaussian2(i,j)
                peGaussian2(k) = peGaussian2(k) + pdf2 / 100;
            else
                peGaussian1(k) = peGaussian1(k) + pdf1 / 100;
            end
            mseGaussian1(k) = mseGaussian1(k) + (pdf1 - pnGaussian1(i,j))^2 / 100;
            mseGaussian2(k) = mseGaussian2(k) + (pdf2 - pnGaussian2(i,j))^2 / 100;
        end
    end
end

% total error
peGaussian = 0.5 * peGaussian1 + 0.5 * peGaussian2;

% cubic window
peCube1 = zeros(1,times);
peCube2 = zeros(1,times);
mseCube1 = zeros(1,times);
mseCube2 = zeros(1,times);
pnCube1 = zeros(length(x),length(y));
pnCube2 = zeros(length(x),length(y));
for k = 1:times
    a = 1;
    X1 = mvnrnd(mu1,Sigma1,N);
    X2 = mvnrnd(mu2,Sigma2,N);
    for i = 1:length(x)
        for j = 1:length(y)
            pnCube1(i,j) = sum(cubicwin((X1-[x(i) y(j)]) / a)) / (N * a^2);
            pnCube2(i,j) = sum(cubicwin((X2-[x(i) y(j)]) / a)) / (N * a^2);
            pdf1 = mvnpdf([x(i) y(j)],mu1,Sigma1);
            pdf2 = mvnpdf([x(i) y(j)],mu2,Sigma2);
            if pnCube1(i,j) >= pnCube2(i,j)
                peCube2(k) = peCube2(k) + pdf2 / 100;
            else
                peCube1(k) = peCube1(k) + pdf1 / 100;
            end
            mseCube1(k) = mseCube1(k) + (pdf1 - pnCube1(i,j))^2 / 100;
            mseCube2(k) = mseCube2(k) + (pdf2 - pnCube2(i,j))^2 / 100;
        end
    end
end

% total error
peCube = 0.5 * peCube1 + 0.5 * peCube2;

% mean of error and mean squared error
peMean = mean([peGaussian; peCube],2);
mseMean = mean([mseGaussian1;mseGaussian2;mseCube1;mseCube2],2);

% EM method
times = 10;
param = cell(1,times);
logLikelihoodMax = zeros(1,times);
peEM1 = zeros(1,times);
peEM2 = zeros(1,times);
mseEM1 = zeros(1,times);
mseEM2 = zeros(1,times);
pnEM1 = zeros(length(x),length(y));
pnEM2 = zeros(length(x),length(y));
for t = 1:times
    [param{t},~,ll] = em_mix(data,2,1);
    logLikelihoodMax(t) = max(ll);
    for i = 1:length(x)
        for j = 1:length(y)
            pnEM1(i,j) = mvnpdf([x(i) y(j)],param{1,t}(1).mean,param{1,t}(1).cov);
            pnEM2(i,j) = mvnpdf([x(i) y(j)],param{1,t}(2).mean,param{1,t}(2).cov);
            pdf1 = mvnpdf([x(i) y(j)],mu1,Sigma1);
            pdf2 = mvnpdf([x(i) y(j)],mu2,Sigma2);
            if pnEM1(i,j) >= pnEM2(i,j)
                peEM2(t) = peEM2(t) + pdf2 / 100;
            else
                peEM1(t) = peEM1(t) + pdf1 / 100;
            end
            mseEM1(t) = mseEM1(t) + (pdf1 - pnEM1(i,j))^2 / 100;
            mseEM2(t) = mseEM2(t) + (pdf2 - pnEM2(i,j))^2 / 100;
        end
    end
end

% deal with abnormal cases
peEM1(peEM1 >= 0.5) = 1 - peEM1(peEM1 >= 0.5);
peEM2(peEM2 >= 0.5) = 1 - peEM2(peEM2 >= 0.5);

% total error, mean and variance
peEM = 0.5 * peEM1 + 0.5 * peEM2;
peEMMean = mean(peEM);
peEMVar = var(peEM);

%% functions
% cubic window
function y = cubicwin(x)
    y = max(abs(x),[],2) <= 1/2;
end

% estimate Gaussian mixture model via EM algorithm
function [param,history,ll] = em_mix(data,m,equalcov,eps)
    % INPUT:
    %   data: input data, N * dims double
    %   m: assumed number of components, int
    %   equalcov: flag of equal covariance matrix, binary
    %   eps(optional): stopping criterion, float
    %
    % OUTPUT:
    %   param: params of different gaussians, list
    %   history(optional): params of different gaussians during iteration, dict
    %   ll(optional): log-likelihood of the data during iteration, list

    % set stopping criterion
    if nargin < 3
        equalcov = 0;
        eps = min(1e-3, 1 / (size(data,1) * 100));
    elseif nargin < 4
        eps = min(1e-3, 1 / (size(data,1) * 100));  
    end

    % initialize GMM
    param = initialize_mixture(data,m);

    history = {}; 
    ll = [];

    cont = 1; 
    it = 1; 
    log_likel = 0; 

    while cont

        % one step EM
        if equalcov
            [param,new_log_likel] = one_EM_iteration_Equal_Cov(data,param);
        else
            [param,new_log_likel] = one_EM_iteration(data,param);
        end

        history{length(history)+1} = param; %# ok
        ll(length(ll)+1) = new_log_likel; %# ok

        % when to stop
        cont = new_log_likel - log_likel > eps * abs(log_likel);
        cont = cont | it < 10; 
        it = it + 1;

        log_likel = new_log_likel; 

    end
    
end

% plot the visualization results
function plot_all(data,param,dim1,dim2) %# ok
    % INPUT:
    %   data: input data, N * dims double
    %   param: params of different gaussians, list
    %   dim1(optional): the first plot dim of data, int
    %   dim2(optional): the second plot dim of data, int

    if nargin < 3 
        dim1 = 1; 
        dim2 = 2; 
    end

    % plot data
    [n,~] = size(data);
    log_prob = zeros(n,length(param)); 
    
    for i = 1:length(param)
        log_prob(:,i) = gaussian_log_prob(data,param(i)) + log(param(i).p);
    end
    
    [~, index] = max(log_prob, [], 2);
    scatter(data(:,dim1),data(:,dim2), [], index); 
    hold on;
    colormap jet;
    myaxis = axis;

    % plot Gaussian
    for i = 1:length(param)
        plot_gaussian(param(i),dim1,dim2,'k');
        axis(myaxis);
    end

    hold off;
    
end

% plot the Gaussian distribution
function plot_gaussian(param,dim1,dim2,st)
    % INPUT:
    %   param: params of different gaussians, list
    %   dim1: the first plot dim of data, int
    %   dim2: the second plot dim of data, int
    %   st: the plot color, str

    [V,E] = eig(param.cov);
    V = V';
    
    % standard deviations
    s = diag(sqrt(E)); 

    t=(0:0.05:2*pi)'; 
    X = s(dim1) * cos(t) * V(dim1,:) + s(dim2) * sin(t) * V(dim2,:); 
    X = X + repmat(param.mean,length(t),1);

    plot(X(:,1),X(:,2),st);
    
end

% calculate one iteration of EM algorithm
function [param,log_likel] = one_EM_iteration(data,param)
    % INPUT:
    %   data: input data, N * dims double
    %   param: params of different gaussians, list
    %
    % OUTPUT:
    %   param: params of different gaussians, list
    %   log_likel: log-likelihood of the data, double

    [n,d] = size(data);

    % E-step
    log_prob = zeros(n,length(param)); 
    for i = 1:length(param)
        log_prob(:,i) = gaussian_log_prob(data,param(i)) + log(param(i).p);
    end

    log_likel = sum(log(sum(exp(log_prob),2)));

    post_prob = exp(log_prob);
    post_prob = post_prob ./ repmat(sum(post_prob,2),1,length(param));

    % M-step
    for i = 1:length(param)
        post_n = sum(post_prob(:,i));

        param(i).p = post_n / n; 
        param(i).mean = post_prob(:,i)' * data / post_n; 

        Z = data - repmat(param(i).mean,n,1);
        weighted_cov = (repmat(post_prob(:,i),1,d) .* Z)' * Z;
        param(i).cov = weighted_cov / post_n;
    end
    
end

% calculate one iteration of EM algorithm with equal covariance matrices
function [param,log_likel] = one_EM_iteration_Equal_Cov(data,param)
    % INPUT:
    %   data: input data, N * dims double
    %   param: params of different gaussians, list
    %
    % OUTPUT:
    %   param: params of different gaussians, list
    %   log_likel: log-likelihood of the data, double

    [n,d] = size(data);

    % E-step
    log_prob = zeros(n,length(param)); 
    for i = 1:length(param)
        log_prob(:,i) = gaussian_log_prob(data,param(i))+log(param(i).p);
    end

    log_likel = sum(log(sum(exp(log_prob),2)));

    post_prob = exp(log_prob);
    post_prob = post_prob ./ repmat(sum(post_prob,2),1,length(param));

    % M-step, modified for equal covariance matrices
    total_n = 0;
    total_cov = 0;
    for i = 1:length(param)
        post_n = sum(post_prob(:,i));

        param(i).p = post_n / n; 
        param(i).mean = post_prob(:,i)' * data / post_n; 

        Z = data - repmat(param(i).mean,n,1);
        weighted_cov = (repmat(post_prob(:,i),1,d) .* Z)' * Z;
        
        total_n = total_n + post_n;
        total_cov = total_cov + weighted_cov;
    end
    
    for i = 1:length(param)
        param(i).cov = total_cov / total_n;
    end
    
end

% calculate log probability of Gaussian distribution
function log_prob = gaussian_log_prob(data,param)
    % INPUT:
    %   data: input data, N * dims double
    %   param: params of different gaussians, list
    %
    % OUTPUT:
    %   log_prob: log probability of gaussian distribution, double

    [n,d] = size(data);
    Ci = inv(param.cov);

    Z = data-repmat(param.mean,n,1); 
    log_prob = (-sum((Z/param.cov) .* Z, 2) + log(det(Ci)) - d*log(2*pi)) / 2;
    
end

% initialize parameters of different Gaussians
function param = initialize_mixture(data,m)
    % INPUT:
    %   data: input data, N * dims double
    %   m: assumed number of components, int
    %
    % OUTPUT:
    %   param: params of different gaussians, list

    [n,d] = size(data);
    
    % initial spherical covariance matrix
    Covar = median(eig(cov(data))) * eye(d); 
    
    % random ordering of the examples
    [~,I] = sort(rand(n,1)); 

    param = [];

    for i = 1:m
        
        % random point as the initial mean
        prm.mean = data(I(i),:);
        
        % spherical covariance as the initial covariance matrix
        prm.cov = Covar;
        
        % uniform frequency as the init mixing
        prm.p = 1/m; 
        
        param = [param; prm]; %# ok
        
    end

end

