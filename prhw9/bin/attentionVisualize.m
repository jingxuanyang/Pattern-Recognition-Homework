%% attentionVisualize.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.04.25
% Project: Pattern Recognition
% Purpose: Visualize the single-headed and multi-headed attention
% Note:  

clc;
clear;
close all;

%% parameters
va = [0; 1];
mu1 = va;

vb = [1; 0];
mu2 = vb;

alpha = 1e-10;
M = 10;
N = 10;

Sigma_i = alpha * eye(2);
Sigma_a = alpha * eye(2) + 0.5 * (mu1 * mu1');

%% calculations
% c-i
q_ci = M * (mu1 + mu2);
ka_ci = mvnrnd(mu1,Sigma_i,N);
kb_ci = mvnrnd(mu2,Sigma_i,N);

alphaa_ci = exp(ka_ci * q_ci) ./ (exp(ka_ci * q_ci) + exp(kb_ci * q_ci));
alphab_ci = exp(kb_ci * q_ci) ./ (exp(ka_ci * q_ci) + exp(kb_ci * q_ci));

c_ci = [alphaa_ci, alphab_ci] * [va, vb];

% c-ii
q_cii = M * (mu1 + mu2);
ka_cii = mvnrnd(mu1,Sigma_a,N);
kb_cii = mvnrnd(mu2,Sigma_i,N);

alphaa_cii = exp(ka_cii * q_cii) ./ (exp(ka_cii * q_cii) + exp(kb_cii * q_cii));
alphab_cii = exp(kb_cii * q_cii) ./ (exp(ka_cii * q_cii) + exp(kb_cii * q_cii));

c_cii = [alphaa_cii, alphab_cii] * [va, vb];

% d-i
q1_di = M * mu1;
q2_di = M * mu2;
ka_di = mvnrnd(mu1,Sigma_i,N);
kb_di = mvnrnd(mu2,Sigma_i,N);

alphaa_di = exp(ka_di * q1_di) ./ (exp(ka_di * q1_di) + exp(kb_di * q1_di));
alphab_di = exp(kb_di * q2_di) ./ (exp(ka_di * q2_di) + exp(kb_di * q2_di));

c_di = 0.5 * [alphaa_di, (1 - alphaa_di)] * [va, vb] + ...
       0.5 * [(1 - alphab_di), alphab_di] * [va, vb];

% d-ii
M = 30;
q1_dii = M * mu1;
q2_dii = M * mu2;
ka_dii = mvnrnd(mu1,Sigma_a,N);
kb_dii = mvnrnd(mu2,Sigma_i,N);

alphaa_dii = exp(ka_dii * q1_dii) ./ (exp(ka_dii * q1_dii) + exp(kb_dii * q1_dii));
alphab_dii = exp(kb_dii * q2_dii) ./ (exp(ka_dii * q2_dii) + exp(kb_dii * q2_dii));

c_dii = 0.5 * [alphaa_dii, (1 - alphaa_dii)] * [va, vb] + ...
        0.5 * [(1 - alphab_dii), alphab_dii] * [va, vb];

%% figures
figure(1)
scatter(c_ci(:,1),c_ci(:,2));
xlabel('$c_1$', 'Interpreter', 'latex');
ylabel('$c_2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
% exportgraphics(gcf,'ci.pdf','ContentType','vector');

figure(2)
scatter(c_cii(:,1),c_cii(:,2));
xlabel('$c_1$', 'Interpreter', 'latex');
ylabel('$c_2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
% exportgraphics(gcf,'cii.pdf','ContentType','vector');

figure(3)
scatter(c_di(:,1),c_di(:,2));
xlabel('$c_1$', 'Interpreter', 'latex');
ylabel('$c_2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
% exportgraphics(gcf,'di.pdf','ContentType','vector');

figure(4)
scatter(c_dii(:,1),c_dii(:,2));
xlabel('$c_1$', 'Interpreter', 'latex');
ylabel('$c_2$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
% exportgraphics(gcf,'dii.pdf','ContentType','vector');

