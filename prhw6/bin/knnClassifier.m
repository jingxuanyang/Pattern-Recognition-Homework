%% knnClassifier.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.04.02
% Project: Pattern Recognition
% Purpose: Train a k-NN classifier
% Note:  

clc;
clear;
close all;

%% parameters
% load mnist dataset
load('mnist.mat');

% size of data
N = size(train_X,1);
r = randperm(N);
[m,n] = size(test_X);

% k-NN parameters
sampleArray = [100,300,1000,3000,10000];
kArray = [1:2:11,15:5:100];
pArray = [1,2,4,Inf];

% initialize tuples
time = zeros(length(sampleArray),length(kArray),length(pArray));
accuracy = time;

%% calculations
% select s = 3000, k = 1, p = 2 as training parameters
for i = 1:length(sampleArray)
    s = sampleArray(i);
    for j = 1:length(kArray)
        k = kArray(j);
        for l = 1:length(pArray)
            if s == 3000 || k == 1 || p == 2
                p = pArray(l);
                tic;
                predict = kNN(train_X(r(1:s),:),train_Y(r(1:s)),test_X,k,p);
                time(i,j,l) = toc;
                accuracy(i,j,l) = mean(predict == test_Y);
                fprintf('s=%d, k=%d, p=%d, time=%f\n',s,k,p,time(i,j,l));
            end
        end
    end
end

%% figures
figure(1)
plot(sampleArray,squeeze(time(:,1,2)),'-*');
xlabel('Training Set Size $n$', 'Interpreter', 'latex');
ylabel('Running Time', 'Interpreter', 'latex');
legend('$k=1,~p=2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
grid on;
% exportgraphics(gcf,'sampleTime.pdf','ContentType','vector');

figure(2)
plot(sampleArray,squeeze(accuracy(:,1,2)),'-o');
xlabel('Training Set Size $n$', 'Interpreter', 'latex');
ylabel('Accuracy', 'Interpreter', 'latex');
ylim([0.7 1]);
legend('$k=1,~p=2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
grid on;
% exportgraphics(gcf,'sampleAccuracy.pdf','ContentType','vector');

figure(3)
plot(kArray,squeeze(time(4,:,2)));
xlabel('$k$-NN Parameter $k$', 'Interpreter', 'latex');
ylabel('Running time', 'Interpreter', 'latex');
legend('$n=3000,~p=2$', 'Interpreter', 'latex');
ylim([2.5 3]);
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
grid on;
% exportgraphics(gcf,'kTime.pdf','ContentType','vector');

figure(4)
plot(kArray,squeeze(accuracy(4,:,2)));
xlabel('$k$-NN Parameter $k$', 'Interpreter', 'latex');
ylabel('Accuracy', 'Interpreter', 'latex');
legend('$n=3000,~p=2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
grid on;
% exportgraphics(gcf,'kAccuracy.pdf','ContentType','vector');

figure(5)
plot(1:4,squeeze(time(4,1,:)),'-*');
xticklabels({'1','','2','','4','','\infty'});
xlabel('Minkowski Metric $p$', 'Interpreter', 'latex');
ylabel('Running time', 'Interpreter', 'latex');
legend('$k=1,~n=3000$', 'Interpreter', 'latex', 'Location','northwest');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
grid on;
% exportgraphics(gcf,'pTime.pdf','ContentType','vector');

figure(6)
plot(1:4,squeeze(accuracy(4,1,:)),'-o');
xticklabels({'1','','2','','4','','\infty'});
xlabel('Minkowski Metric $p$', 'Interpreter', 'latex');
ylabel('Accuracy', 'Interpreter', 'latex');
ylim([0.6 1]);
legend('$k=1,~n=3000$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
grid on;
% exportgraphics(gcf,'pAccuracy.pdf','ContentType','vector');

%% functions
% k-NN method for classification
function c = kNN(data,label,x,k,p)
    % Input: 
    %  data: m*n matrix, each row represents a sample
    %  label: 1*m vector, label(i) is the class of data(i,:)
    %  x: xm*n matrix, test dataset
    %  k: the k parameter of KNN
    %  p: select distance measure
    %
    % Ouput: 
    %  c: classification result
    
    [m,~] = size(x);
    dis = pdist2(data,x,'minkowski',p);
    c = zeros(1,m);
    [~,index] = sort(dis);
    
    % select the mode of label in k nearest neighbors
    [~,~,C] = mode(label(index(1:k,:)),1);
    
    for i = 1:m
        
        Cl = C{i};
        
        % random pick one label if there are not only one mode
        cindex = randperm(length(Cl));
        c(i) = Cl(cindex(1));
        
    end

end

