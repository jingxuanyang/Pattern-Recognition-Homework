%% drawDecisionSurface2.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.02.25
% Project: Pattern Recognition
% Purpose: draw decision surface
% Note:  

clear;
clc;

%% parameters
mu1 = [0,0];% 均值向量
Sigma1 = [1 0;0 1];% 协方差矩阵
[X1,Y1] = meshgrid(-3:0.1:3,-3:0.1:3);%在XOY面上，产生网格数据
p1 = mvnpdf([X1(:) Y1(:)],mu1,Sigma1);%求取联合概率密度，相当于Z轴
p1 = reshape(p1,size(X1));%将Z值对应到相应的坐标上

mu2 = [1,1];% 均值向量
Sigma2 = [1 0;0 1];% 协方差矩阵
[X2,Y2] = meshgrid(-3:0.1:3,-3:0.1:3);%在XOY面上，产生网格数据
p2 = mvnpdf([X2(:) Y2(:)],mu2,Sigma2);%求取联合概率密度，相当于Z轴
p2 = reshape(p2,size(X2));%将Z值对应到相应的坐标上

%% draw figures
figure
contour(X1,Y1,p1)
hold on;
contour(X2,Y2,p2,'--')
line([0,1], [0,1], 'Marker', '*')
line([3,-2-log(4)], [-2-log(4),3], 'Color','red')
axis equal
legend('$p(x|\omega_1)$','$p(x|\omega_2)$','mean line','decision boundary','Interpreter', 'latex');
xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
% exportgraphics(gcf,'mvnpdf-2.pdf','ContentType','vector');


