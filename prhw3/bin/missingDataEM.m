%% missingDataEM.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.14
% Project: Pattern Recognition
% Purpose: Deal with missing data via EM algorithm
% Note:  

clc;
clear;
close all;

%% parameters
% data points
x = [0.42   -0.087 0.58;
     -0.2   -3.3   -3.4;
     1.3    -0.32  1.7;
     0.39   0.71   0.23;
     -1.6   -5.3   -0.15;
     -0.029 0.89   -4.7;
     -0.23  1.9    2.2;
     0.27   -0.3   -0.87;
     -1.9   0.76   -2.1;
     0.87   -1     -2.6];

% size of data points
[xNum,xDim] = size(x);

% stopping criteria
eps = 1e-5;

% initialize parameters
mu = zeros(xDim,1);
Sigma = eye(xDim);
diff = ones(4,1);

%% case: missing even-numbered x_3 data
% please see paper for detailed explanation of E, D, W and N
while sum(diff > eps)
    
    % calculate expectation and variance of conditional prob(x3|x1,x2)
    E = zeros(xNum,1);
    D = zeros(xNum,1);
    for i = 2:2:xNum
        E(i) = mu(3) + Sigma(3,1:2) / Sigma(1:2,1:2) * [x(i,1) - mu(1); x(i,2) - mu(2)];
        D(i) = Sigma(3,3) - Sigma(3,1:2) / Sigma(1:2,1:2) * Sigma(1:2,3);
    end
    
    % obtain new mean
    muNew = zeros(xDim,1);
    muNew(1) = sum(x(:,1)) / xNum;
    muNew(2) = sum(x(:,2)) / xNum;
    muNew(3) = sum(x(1:2:end,3)) / xNum + sum(E(2:2:end,1)) / xNum;
    
    % difference
    diff(1:3) = abs(muNew - mu);
    mu = muNew;
    
    % calculate W_i
    W = zeros(xDim,xDim,xNum);
    for i = 2:2:xNum
        W(1,1,i) = (x(i,1) - mu(1))^2;
        W(2,1,i) = (x(i,2) - mu(2)) * (x(i,1) - mu(1));
        W(2,2,i) = (x(i,2) - mu(2))^2;
        W(3,1,i) = (E(i) - mu(3)) * (x(i,1) - mu(1));
        W(3,2,i) = (E(i) - mu(3)) * (x(i,2) - mu(2));
        W(3,3,i) = D(i) + E(i)^2 - 2 * mu(3) * E(i) + mu(3)^2;
        W(1,2,i) = W(2,1,i);
        W(1,3,i) = W(3,1,i);
        W(2,3,i) = W(3,2,i);
    end
    
    % calculate N_i
    N = zeros(xDim,xDim,xNum);
    for i = 1:2:xNum - 1
        N(:,:,i) = (x(i,:)' - mu) * (x(i,:)' - mu)';
    end
    
    % obtain new covariance matrix
    SigmaNew = (sum(N,3) + sum(W,3)) / xNum;
    
    % difference
    diff(4) = max(max(abs(SigmaNew - Sigma)));
    Sigma = SigmaNew;
    
end

%% case: missing all even-numbered data
% mean
muHalf = mean(x(1:2:end,:))';
% covariance matrix
SigmaHalf = cov(x(1:2:end,:)) * (xNum - 2) / xNum;

%% case: complete data
% mean
muComplete = mean(x)';
% covariance matrix
SigmaComplete = cov(x) * (xNum - 1) / xNum;

