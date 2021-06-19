%% svmClassifier.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.26
% Project: Pattern Recognition
% Purpose: Train an SVM classifier with built-in function fitcsvm
% Note:  

clc;
clear;
close all;
warning off;

%% parameters
N = 10;
x = [-3.0 -2.9;
      0.5  8.7;
      2.9  2.1;
     -0.1  5.2;
     -4.0  2.2;
     -1.3  3.7;
     -3.4  6.2;
     -4.1  3.4; 
     -5.1  1.6;
      1.9  5.1;
     -2.0 -8.4;
     -8.9  0.2;
     -4.2 -7.7;
     -8.5 -3.2;
     -6.7 -4.0;
     -0.5 -9.2;
     -5.3 -6.7;
     -8.7 -6.4;
     -7.1 -9.7;
     -8.0 -6.3];

% data preprocess
xt = hignDimMap(x);
yt = [ones(N,1); -ones(N,1)];

% initialize parameters
svmModel = cell(N,1);
margin = zeros(N,1);
bias = zeros(N,1);
beta = zeros(N,6);

% iterate of points
for i = 1:10
    
    % train dataset
    xtrain = [xt(1:i,:); xt(N+1:N+i,:)];
    ytrain = [yt(1:i,:); yt(N+1:N+i,:)];
    
    % fit train dataset via SVM
    svmModel{i} = fitcsvm(xtrain,ytrain);
    
    % g(x) = x' * beta + bias, margin = 2/||beta||
    beta(i,:) = svmModel{i}.Beta;
    margin(i) = 2/norm(beta(i,:));
    bias(i) = svmModel{i}.Bias;
    
    % draw figures
    if i == 1
        figure(1)
    else
        figure(2)
        subplot(3,3,i-1)
    end
    
    plot(x(1:i,1),x(1:i,2),'ro');
    hold on;
    plot(x(N+1:N+i,1),x(N+1:N+i,2),'r*');
    plot(x(i+1:N,1),x(i+1:N,2),'ko');
    plot(x(N+i+1:2*N,1),x(N+i+1:2*N,2),'k*');
    xlabel('$x_1$', 'Interpreter', 'latex');
    ylabel('$x_2$', 'Interpreter', 'latex');
    set(gca, 'fontsize', 14, 'fontname', 'Euclid');
    
    % discriminant function
    fd = @(x1,x2) hignDimMap([x1,x2]) * svmModel{i}.Beta + svmModel.Bias;
    
    % draw decision boundary
    fimplicit(fd,[-15,15,-20,15],'b');
    
    if i == 1
        legend('Selected $\omega_1$', 'Selected $\omega_2$', ...
               '$\omega_1$', '$\omega_2$', 'decision hyperplane', ...
               'Interpreter', 'latex');
        % exportgraphics(gcf,'svmPoint1.pdf','ContentType','vector');
    else
        title(strcat('$\rm{Train}~\rm{Points} =', num2str(i), '$'), ...
              'Interpreter', 'latex');
    end
    
end

% exportgraphics(gcf,'svmPoint2-10.pdf','ContentType','vector');

% construct total parameter table
paramTbl = [beta,bias,margin];

%% functions
% map initial data to high dimention (6-D)
function y = hignDimMap(x)
    y = [ones(size(x,1),1), x, x(:,1) .* x(:,1), x(:,1) .* x(:,2), x(:,2) .* x(:,2)];
end

