%% naiveBayesCalculation.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.02.26
% Project: Pattern Recognition
% Purpose: draw a posterior probability density function
% Note:  

clear;
clc;

%% parameters


%% calculations
P1 = 0.9*0.3 / (0.9*0.3 + 0.4*0.8);
P2 = 0.9*0.7 / (0.9*0.7 + 0.4*0.2);
P3 = 0.1*0.3 / (0.1*0.3 + 0.6*0.8);
P4 = 0.1*0.7 / (0.1*0.7 + 0.6*0.2);

E = 0.5*(1 - 0.9*0.7) + 0.5*0.4*0.2;

Pj = 0.5*0.9*0.3;
