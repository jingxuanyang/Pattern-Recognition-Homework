%% optimalWindowWidthCube.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.21
% Project: Pattern Recognition
% Purpose: Find optimal window width of hypercubic window method
% Note:  

clc;
clear;
close all;

%% parameters
mu1 = [-1, 0];
Sigma1 = eye(2);
mu2 = [1, 0];
Sigma2 = eye(2);

N = 1000;
X1 = mvnrnd(mu1,Sigma1,N);
X2 = mvnrnd(mu2,Sigma2,N);

x = linspace(-5,5);
y = linspace(-5,5);
mvn1 = mvnpdf([x' y'],mu1,Sigma1);

% window width
aArray = [0.01:0.01:0.1,0.15:0.05:3,3:0.5:10];
widthNum = length(aArray);

%% calculations
% cubic window
peCube1 = zeros(1,widthNum);
peCube2 = zeros(1,widthNum);
mseCube1 = zeros(1,widthNum);
mseCube2 = zeros(1,widthNum);
pnCube1 = zeros(length(x),length(y));
pnCube2 = zeros(length(x),length(y));
for k = 1:widthNum
    a = aArray(k);
    for i = 1:length(x)
        for j = 1:length(y)
            pnCube1(i,j) = sum(cubicwin((X1-[x(i) y(j)]) / a)) / (N * a^2);
            pnCube2(i,j) = sum(cubicwin((X2-[x(i) y(j)]) / a)) / (N * a^2);
            pdf1 = mvnpdf([x(i) y(j)],mu1,Sigma1);
            pdf2 = mvnpdf([x(i) y(j)],mu2,Sigma2);
            if pnCube1(i,j) >= pnCube2(i,j)
                peCube2(k) = peCube2(k) + pdf2 / 100;
            else
                peCube1(k) = peCube1(k) + pdf1 / 100;
            end
            mseCube1(k) = mseCube1(k) + (pdf1 - pnCube1(i,j))^2 / 100;
            mseCube2(k) = mseCube2(k) + (pdf2 - pnCube2(i,j))^2 / 100;
        end
    end
end

% total error
peCube = 0.5 * peCube1 + 0.5 * peCube2;

% minimum of mean squared error
[min1,index1]=min(mseCube1);
[min2,index2]=min(mseCube2);

%% figures
figure(1)
semilogy(aArray,mseCube1);
hold on
semilogy(aArray,mseCube2);
xlabel('$a$', 'Interpreter', 'latex');
ylabel('Log of Mean Squared Error', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
legend('$\log[\epsilon(p_n(x|\omega_1))]$', ...
       '$\log[\epsilon(p_n(x|\omega_2))]$', 'Interpreter', 'latex');

% exportgraphics(gcf,'mseCube.pdf','ContentType','vector');

%% functions
% cubic window
function y = cubicwin(x)
    y = max(abs(x),[],2) <= 1/2;
end

