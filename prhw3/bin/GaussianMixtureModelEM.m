%% GaussianMixtureModelEM.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.14
% Project: Pattern Recognition
% Purpose: Solve Gaussian mixture model with EM algorithm
% Note:  

clc;
clear;
close all;

%% parameters
% load data file
load('emdata.mat');
[n,d] = size(data);

% number of components
m = [2,3,4,5];
choices = length(m);

% number of rerun times
times = 200;

% initialize parameters
paramTotal = cell(times,choices);
paramTotalEqcov = cell(times,choices);
logLikelihoodMaxTotal = zeros(times,choices);
logLikelihoodMaxTotalEqcov = zeros(times,choices);

%% calculations
for i = 1:choices
    for t = 1:times
        % unequal variance matrices
        [paramTotal{t,i},~,ll] = em_mix(data,m(1,i));
        logLikelihoodMaxTotal(t,i) = max(ll);
        
        % equal variance matrices
        [paramTotalEqcov{t,i},~,llEqcov] = em_mix(data,m(1,i),1);
        logLikelihoodMaxTotalEqcov(t,i) = max(llEqcov);
    end
end

% find maximum log likelihood
[logLikelihood,indexTimes] = max(logLikelihoodMaxTotal);
[logLikelihoodEqcov,indexTimesEqcov] = max(logLikelihoodMaxTotalEqcov);

% calculate BIC, Bayesian Information Criterion
BIC = -2 * logLikelihood + log(n) * (1 + d + d^2) * m;
BICEqcov = -2 * logLikelihoodEqcov + log(n) * (1 + d + d^2) * m;

% find minimum BIC and its index
[minBIC,indexBIC] = min(BIC);
[minBICEqcov,indexBICEqcov] = min(BICEqcov);

% select the parameters of the minimum BIC
param = paramTotal{indexTimes(indexBIC),indexBIC};
paramEqcov = paramTotalEqcov{indexTimesEqcov(indexBICEqcov),indexBICEqcov};

%% figures
% GMM with unequal covariance matrices
figure(1)
for i = 1:choices
    subplot(2,2,i);
    plot_all(data, paramTotal{indexTimes(i),i});
    title(strcat('$m=',num2str(i+1),'$'), 'Interpreter', 'latex');
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    set(gca, 'fontsize', 14, 'fontname', 'Euclid');
end

% exportgraphics(gcf,'gmmem-1.pdf','ContentType','vector');

% GMM with equal covariance matrices
figure(2)
for i = 1:choices
    subplot(2,2,i);
    plot_all(data, paramTotalEqcov{indexTimesEqcov(i),i});
    title(strcat('$m=',num2str(i+1),'$'), 'Interpreter', 'latex');
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    set(gca, 'fontsize', 14, 'fontname', 'Euclid');
end

% exportgraphics(gcf,'gmmemEqcov.pdf','ContentType','vector');

%% functions
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
function plot_all(data,param,dim1,dim2)
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

