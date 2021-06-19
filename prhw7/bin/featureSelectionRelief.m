%% featureSelectionRelief.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.04.10
% Project: Pattern Recognition
% Purpose: Feature selection with Relief-based method
% Note:  

clc;
clear;
close all;
warning off;

%% parameters
% load csv file without first line
% since it can not serve as variable names
wm = readtable('watermelon_3.csv','ReadVariableNames',false,'HeaderLines',1);

% preprocess data
% group those cells in Chinese and change into index, then form a matrix
wmPredictor = [grp2idx(wm.Var2), grp2idx(wm.Var3), grp2idx(wm.Var4), ...
               grp2idx(wm.Var5), grp2idx(wm.Var6), grp2idx(wm.Var7), ...
               wm.Var8, wm.Var9];

% select class of each data sample           
wmClass = wm.Var10;

%% calculations
% select features with ReliefF method
[idx,weights] = relieff(wmPredictor,wmClass,3);

% fit train dataset via SVM
wmTrain = [wmPredictor(:,idx(1)), wmPredictor(:,idx(2))];
svmModel = fitcsvm(wmTrain,wmClass);

%% figures
figure(1)

% sample data of class 1
plot(wmTrain(1:8,1),wmTrain(1:8,2),'o');
hold on;

% sample data of class 2
plot(wmTrain(9:end,1),wmTrain(9:end,2),'*');

% discriminant function
fd = @(x1,x2) [x1,x2] * svmModel.Beta + svmModel.Bias;

% draw decision boundary
fimplicit(fd,[0,4,0,0.5],'b');

% misc settings
xlabel('$x_4$', 'Interpreter', 'latex');
ylabel('$x_8$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
legend('$\omega_1$', '$\omega_2$', 'decision hyperplane','Interpreter', 'latex');

% exportgraphics(gcf,'featureSVM.pdf','ContentType','vector');

