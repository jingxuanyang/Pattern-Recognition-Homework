%% drawExponentialPDF.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.05
% Project: Pattern Recognition
% Purpose: draw a exponential probability density function
% Note:  

clear;
clc;

%% parameters
theta1 = 1;
x1 = 0:0.1:10;

theta2 = 0:0.01:5;
x2 = 2;

%% calculations
p1 = theta1 * exp(-theta1 * x1);
p2 = theta2 .* exp(-theta2 * x2);

%% figures
figure
plot(x1,p1);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$p(x|\theta)$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
exportgraphics(gcf,'epdf-1.pdf','ContentType','vector');

figure
plot(theta2,p2);
xlabel('$\theta$', 'Interpreter', 'latex');
ylabel('$p(x|\theta)$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
exportgraphics(gcf,'epdf-2.pdf','ContentType','vector');

