%% drawCauchyDistribution.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.21
% Project: Pattern Recognition
% Purpose: draw Cauchy distribution
% Note:  

clc;
clear;

%% parameters
a1 = 3;
a2 = 5;
b = 1;

%% calculations
x = -20:0.1:20;
temp1 = 1 + ((x - a1) / b) .^ 2;
temp2 = 1 + ((x - a2) / b) .^ 2;
p1 = (1/(pi*b)) ./ temp1;
p2 = (1/(pi*b)) ./ temp2;

%% draw figures
figure(1)
plot(x,p1);
hold on;
plot(x,p2);
legend('$P(x|\omega_1)$','$P(x|\omega_2)$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('Cauchy Probability Density Function');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
% exportgraphics(gcf,'postpdf.pdf','ContentType','vector');

