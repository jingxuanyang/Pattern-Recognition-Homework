%% drawPosteriorProb.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.02.25
% Project: Pattern Recognition
% Purpose: draw a posterior probability density function
% Note:  

clear;
clc;

%% parameters
a1 = 3;
a2 = 5;
b = 1;

%% calculations
x = -100:0.1:100;
temp1 = 1 + ((x - a1) / b) .^ 2;
temp2 = 1 + ((x - a2) / b) .^ 2;
p1 = temp2 ./ (temp1 + temp2);
p2 = 1 - p1;

%% draw figures
plot(x,p1);
hold on;
plot(x,p2);
legend('$P(\omega_1|x)$','$P(\omega_2|x)$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('Posterior Probability Density Function');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
exportgraphics(gcf,'postpdf.pdf','ContentType','vector');


