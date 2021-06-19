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
q1 = 3/4;
q2 = 1/4;

%%
fun3 = @(x,y) 1 - 3*x/4 - 3*y/4;
ymax3 = @(x) 2/3 - x;
q3 = 1/2 * integral2(fun3,0,1/3,1/3,ymax3);

%%
fun4 = @(x,y) 3/4*(x + y) - 1/2;
ymin4 = @(x) 4/3 - x;
q4 = 1/2 * integral2(fun4,1/3,2/3,ymin4,1);

%%
e = 1/8 * q1 + 5/8 * q2 + 9/4 * q3 + 9/4 * q4;




