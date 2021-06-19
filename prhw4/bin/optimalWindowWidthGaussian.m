%% optimalWindowWidthGaussian.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.03.21
% Project: Pattern Recognition
% Purpose: Find optimal window width of Gaussian window method
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

% window width
aArray = [0.01:0.01:0.1,0.15:0.05:3,3:0.5:10];
widthNum = length(aArray);

%% calculations
% Gaussian window
peGaussian1 = zeros(1,widthNum);
peGaussian2 = zeros(1,widthNum);
mseGaussian1 = zeros(1,widthNum);
mseGaussian2 = zeros(1,widthNum);
pnGaussian1 = zeros(length(x),length(y));
pnGaussian2 = zeros(length(x),length(y));
for k = 1:widthNum
    a = aArray(k);
    for i = 1:length(x)
        for j = 1:length(y)
            pnGaussian1(i,j) = sum(mvnpdf(X1,[x(i) y(j)],a^2*Sigma1)) / N;
            pnGaussian2(i,j) = sum(mvnpdf(X2,[x(i) y(j)],a^2*Sigma1)) / N;
            pdf1 = mvnpdf([x(i) y(j)],mu1,Sigma1);
            pdf2 = mvnpdf([x(i) y(j)],mu2,Sigma2);
            if pnGaussian1(i,j) >= pnGaussian2(i,j)
                peGaussian2(k) = peGaussian2(k) + pdf2 / 100;
            else
                peGaussian1(k) = peGaussian1(k) + pdf1 / 100;
            end
            mseGaussian1(k) = mseGaussian1(k) + (pdf1 - pnGaussian1(i,j))^2 / 100;
            mseGaussian2(k) = mseGaussian2(k) + (pdf2 - pnGaussian2(i,j))^2 / 100;
        end
    end
end

% total error
peGaussian = 0.5 * peGaussian1 + 0.5 * peGaussian2;

% minimum of mean squared error
[min1,index1]=min(mseGaussian1);
[min2,index2]=min(mseGaussian2);

%% figures
figure(1)
semilogy(aArray,mseGaussian1);
hold on
semilogy(aArray,mseGaussian2);
xlabel('$a$', 'Interpreter', 'latex');
ylabel('Log of Mean Squared Error', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');
legend('$\log[\epsilon(p_n(x|\omega_1))]$', ...
       '$\log[\epsilon(p_n(x|\omega_2))]$', 'Interpreter', 'latex');

% exportgraphics(gcf,'mseGaussian.pdf','ContentType','vector');

