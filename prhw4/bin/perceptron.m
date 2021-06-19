%% perceptron.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.20
% Project: Pattern Recognition
% Purpose: perceptron, with margin or not
% Note:  

clc;
clear;
close all;

%% generate sample data points
% if not generated
if ~exist('percepData.mat','file')
    
    % set parameters
    w = [-1;1];
    b = 0;
    N = 100;
    Sigma = eye(2);
    mu1 = [3,7];
    mu2 = [7,3];
    x = zeros(2*N,2);
    
    % randomly select data points of first class
    i = 1;
    while i <= N
        x(i,:) = mvnrnd(mu1,Sigma);
        if x(i,:)*w + b > 0
            i = i + 1;
        end
    end
    
    % select data points of second class
    while i <= 2*N
        x(i,:) = mvnrnd(mu2,Sigma);
        if x(i,:)*w + b < 0
            i = i + 1;
        end
    end
    
    % set label to data points
    y = [ones(N,1); -ones(N,1)];
    
    % save variables to file
    save('percepData.mat','x','y','N');
    
else
    load('percepData.mat');
end

% draw data points
figure(1)
drawData(x);
legend('$\rm{label}=1$','$\rm{label}=-1$', 'Interpreter', 'latex');

% exportgraphics(gcf,'dataPoints.pdf','ContentType','vector');

%% use classical perceptron algorithm
% add ones to the first column of x
xAug = [ones(2*N,1),x];

% find perceptron
a = classicalPerceptron(xAug,y);

% draw figures
figure(2)
drawData(x);
x0 = linspace(0,10);
if a(2) ~= 0
    plot(x0,-(x0*a(2)+a(1))/a(3));
else
    plot(-ones(1,length(x0)),x0);
end
legend('$\rm{label}=1$','$\rm{label}=-1$', 'Interpreter', 'latex');

% exportgraphics(gcf,'classicalPrecep.pdf','ContentType','vector');

%% margin perceptron algorithm
% set margin
gamma = 10;

% find perceptron
[a,~] = marginPerceptron(xAug,y,gamma);

% draw figures
figure(3)
drawData(x);
if a(2) ~= 0
    plot(x0,-(x0*a(2)+a(1))/a(3));
else
    plot(-ones(1,length(x0)),x0);
end
legend('$\rm{label}=1$','$\rm{label}=-1$', ...
       strcat('$\gamma=',num2str(gamma),'$'), 'Interpreter', 'latex');

% exportgraphics(gcf,'marginPrecep.pdf','ContentType','vector');

%% margin perceptron algorithm with different margins
% set margin array
gamma = 0:100;
times = zeros(length(gamma),1);
a = zeros(length(gamma),3);

% find perceptron
for i = 1:length(gamma)
    [a(i,:),times(i)] = marginPerceptron(xAug,y,gamma(i));
end

% draw figures
figure(4)
plot(gamma,times,'-');
xlabel('$\gamma$', 'Interpreter', 'latex');
ylabel('Number of Iterations');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');

% exportgraphics(gcf,'marginEffect.pdf','ContentType','vector');

% draw classification boundary with different margins
gammaMargin = 0:10:80;
aMargin = zeros(length(gammaMargin),3);
figure(5)
for i = 1:length(gammaMargin)
    [aMargin(i,:),~] = marginPerceptron(xAug,y,gammaMargin(i));
    subplot(3,3,i);
    drawData(x);
    plot(x0,-(x0*aMargin(i,2)+aMargin(i,1))/aMargin(i,3));
    title(strcat('$\gamma=',num2str(gammaMargin(i)),'$'), 'Interpreter', 'latex');
end

% exportgraphics(gcf,'marginEffect2.pdf','ContentType','vector');

%% functions
% draw data points
function drawData(x)
    % INPUT: 
    %  x: the training data
    
    N = length(x) / 2;
    plot(x(1:N,1),x(1:N,2),'o');
    hold on;
    plot(x(N+1:2*N,1),x(N+1:2*N,2),'*');
    axis equal;
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    set(gca, 'fontsize', 14, 'fontname', 'Euclid');
    
end

% classical perception algorithm
function a = classicalPerceptron(x,y)
    % INPUT: 
    %  x: the training data
    %  y: the label of x
    % 
    % OUTPUT:
    %  a: the parameters of hyperplane
    
    a = zeros(3,1);
    n = size(x,1);
    k = 0;
    i = 0;
    
    while i < n
        k = mod(k,n) + 1;
        if y(k) * x(k,:) * a <= 0
            a = a + y(k) * x(k,:)';
            i = 0;
        else
            i = i + 1;
        end
    end
    
end

% margin perception algorithm
function [a,times] = marginPerceptron(x,y,theta)
    % INPUT:
    %  x: the training data
    %  y: the label of x
    %  theta: the eps
    %
    % OUTPUT:
    %  a: the parameters of hyperplane
    %  times: iteration times
    
    a = zeros(3,1);
    n = size(x,1);
    k = 0;
    i = 0;
    times = 0;
    
    while i < n
        k = mod(k,n) + 1;
        if y(k) * x(k,:) * a <= theta
            a = a + y(k) * x(k,:)';
            i = 0;
        else
            i = i + 1;
        end
        times = times + 1;
    end
    
end

