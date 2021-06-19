%% ParzenWindowEstimation.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.08
% Project: Pattern Recognition
% Purpose: Parzen window density estimation with different window
%          functions, such as square, Gaussian, triangular, cosine and 
%          expnential functions
% Note:  

clc;
clear;
close all;

%% parameters
% generate samples from $p(x) \sim 0.2N(-1,1) + 0.8N(1,1)$
alpha = 0.2;
N = 5000;
U = rand(N,1);
I = U < alpha;
xi = I .* (randn(N,1) - 1) + (1-I) .* (randn(N,1) + 1);

% window width
aArray = [0.25 0.5 1 2 4 8];
widthNum = length(aArray);

% number of samples
nArray = [5,10,50,100,1000,5000];
sampleNum = length(nArray);

% ground truth distribution
x = linspace(-5,5);
xNum = length(x);
p = 0.2 * normpdf(x,-1,1) + 0.8 * normpdf(x,1,1);

% simulation times
times = 20;

%% figures
% square window function
figure(1);

% initialize mean and variance of MSE
squareMeanMSE = zeros(sampleNum,widthNum);
squareVarMSE = zeros(sampleNum,widthNum);

% iterate for different sample numbers and window width
for j = 1:sampleNum
    n = nArray(j);
    for k = 1:widthNum
        a = aArray(k);
        MSE = zeros(times,1);
        for t = 1:times
            px = zeros(xNum,1);
            pn = zeros(xNum,1);
            
            % generate random index to choose from samples at each time
            index = randperm(N,n);
            for i = 1:length(x)
                px(i) = sum(squareWindow(x(i)-xi(index),a))/n;
                pn(i) = sum(squareWindow(x(i)-xi(1:n),a))/n;
            end
            
            % calculate MSE
            temp = px - p';
            MSE(t) = temp' * temp / xNum;
        end
        
        % set values for mean and variance of MSE
        squareMeanMSE(j,k) = mean(MSE);
        squareVarMSE(j,k) = var(MSE);
        
        % draw subplot figures
        subplot(sampleNum, widthNum, widthNum * (j - 1) + k);
        drawSubplot(x,p,pn,a,n,j,k);
    end
end

% exportgraphics(gcf,'squareWindow.pdf','ContentType','vector');

% Gaussian window function
figure(2);

% initialize mean and variance of MSE
gaussianMeanMSE = zeros(sampleNum,widthNum);
gaussianVarMSE = zeros(sampleNum,widthNum);

% iterate for different sample numbers and window width
for j = 1:sampleNum
    n = nArray(j);
    for k = 1:widthNum
        a = aArray(k);
        sigma = a/sqrt(n);
        MSE = zeros(times,1);
        for t = 1:times
            px = zeros(xNum,1);
            pn = zeros(xNum,1);
            
            % generate random index to choose from samples at each time
            index = randperm(N,n);
            for i = 1:length(x)
                px(i) = sum(gaussianWindow(x(i)-xi(index),sigma),1)/n;
                pn(i) = sum(gaussianWindow(x(i)-xi(1:n),sigma),1)/n;
            end
            
            % calculate MSE
            temp = px - p';
            MSE(t) = temp' * temp / xNum;
        end
        
        % set values for mean and variance of MSE
        gaussianMeanMSE(j,k) = mean(MSE);
        gaussianVarMSE(j,k) = var(MSE);
        
        % draw subplot figures
        subplot(sampleNum, widthNum, widthNum * (j - 1) + k);
        drawSubplot(x,p,pn,a,n,j,k);
    end
end

% exportgraphics(gcf,'gaussianWindow.pdf','ContentType','vector');

% triangular window function
figure(3);

% initialize mean and variance of MSE
triangularMeanMSE = zeros(sampleNum,widthNum);
triangularVarMSE = zeros(sampleNum,widthNum);

% iterate for different sample numbers and window width
for j = 1:sampleNum
    n = nArray(j);
    for k = 1:widthNum
        a = aArray(k);
        MSE = zeros(times,1);
        for t = 1:times
            px = zeros(xNum,1);
            pn = zeros(xNum,1);
            
            % generate random index to choose from samples at each time
            index = randperm(N,n);
            for i = 1:length(x)
                px(i) = sum(triangularWindow((x(i)-xi(index))/a),1)/(n*a);
                pn(i) = sum(triangularWindow((x(i)-xi(1:n))/a),1)/(n*a);
            end
            
            % calculate MSE
            temp = px - p';
            MSE(t) = temp' * temp / xNum;
        end
        
        % set values for mean and variance of MSE
        triangularMeanMSE(j,k) = mean(MSE);
        triangularVarMSE(j,k) = var(MSE);
        
        % draw subplot figures
        subplot(sampleNum, widthNum, widthNum * (j - 1) + k);
        drawSubplot(x,p,pn,a,n,j,k);
    end
end

% exportgraphics(gcf,'triangularWindow.pdf','ContentType','vector');

% cosine window function
figure(4);

% initialize mean and variance of MSE
cosineMeanMSE = zeros(sampleNum,widthNum);
cosineVarMSE = zeros(sampleNum,widthNum);

% iterate for different sample numbers and window width
for j = 1:sampleNum
    n = nArray(j);
    for k = 1:widthNum
        a = aArray(k);
        MSE = zeros(times,1);
        for t = 1:times
            px = zeros(xNum,1);
            pn = zeros(xNum,1);
            
            % generate random index to choose from samples at each time
            index = randperm(N,n);
            for i = 1:length(x)
                px(i) = sum(cosineWindow((x(i)-xi(index))/a),1)/(n*a);
                pn(i) = sum(cosineWindow((x(i)-xi(1:n))/a),1)/(n*a);
            end
            
            % calculate MSE
            temp = px - p';
            MSE(t) = temp' * temp / xNum;
        end
        
        % set values for mean and variance of MSE
        cosineMeanMSE(j,k) = mean(MSE);
        cosineVarMSE(j,k) = var(MSE);
        
        % draw subplot figures
        subplot(sampleNum, widthNum, widthNum * (j - 1) + k);
        drawSubplot(x,p,pn,a,n,j,k);
    end
end

% exportgraphics(gcf,'cosineWindow.pdf','ContentType','vector');

% exponential window function
figure(5);

% initialize mean and variance of MSE
exponentialMeanMSE = zeros(sampleNum,widthNum);
exponentialVarMSE = zeros(sampleNum,widthNum);

% iterate for different sample numbers and window width
for j = 1:sampleNum
    n = nArray(j);
    for k = 1:widthNum
        a = aArray(k);
        MSE = zeros(times,1);
        for t = 1:times
            px = zeros(xNum,1);
            pn = zeros(xNum,1);
            
            % generate random index to choose from samples at each time
            index = randperm(N,n);
            for i = 1:length(x)
                px(i) = sum(exponentialWindow((x(i)-xi(index))/a),1)/(n*a);
                pn(i) = sum(exponentialWindow((x(i)-xi(1:n))/a),1)/(n*a);
            end
            
            % calculate MSE
            temp = px - p';
            MSE(t) = temp' * temp / xNum;
        end
        
        % set values for mean and variance of MSE
        exponentialMeanMSE(j,k) = mean(MSE);
        exponentialVarMSE(j,k) = var(MSE);
        
        % draw subplot figures
        subplot(sampleNum, widthNum, widthNum * (j - 1) + k);
        drawSubplot(x,p,pn,a,n,j,k);
    end
end

% exportgraphics(gcf,'exponentialWindow.pdf','ContentType','vector');

%% functions
% square window function
function y = squareWindow(x,a)
    y = zeros(length(x),1);
    index = x >= -a/2 & x <= a/2;
    y(index) = 1/a;
end

% Gaussian window function
function y = gaussianWindow(x,sigma)
    y = (1 / (sqrt(2*pi) * sigma)) * exp(-1/2 * (x/sigma) .^ 2);
end

% triangular window function
function y = triangularWindow(x)
    y = zeros(length(x),1);
    index = find(x >= -1 & x <= 1);
    y(index) = 1 - abs(x(index));
end

% cosine window function
function y = cosineWindow(x)
    y = zeros(length(x),1);
    index = find( x >= -1 & x <= 1);
    y(index) = pi/4 * cos(pi/2 * x(index));
end

% exponential kernal function
function y = exponentialWindow(x)
    y = 1/2 * exp(-abs(x));
end

% draw each subplot figure
function drawSubplot(x,p,pn,a,n,j,k)
    plot(x,pn);
    hold on;
    plot(x,p,'r');
    set(gca, 'fontsize', 12, 'fontname', 'Euclid');
    if k == 1
        yline = ['$n=',num2str(n),'$'];
        ylabel({yline}, 'Interpreter', 'latex');
    end
    if j == 1
        line = ['$a=',num2str(a),'$'];
        title({line}, 'Interpreter', 'latex');
    end
    grid on;
end

