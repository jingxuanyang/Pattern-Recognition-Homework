%% dimenReductionISOMAP.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.05.02
% Project: Pattern Recognition
% Purpose: Create manifold of shape Z and reduce dimention via ISOMAP
% Note:  

clc;
clear;
close all;

%% parameters
% create manifold of shape Z
% number of points per surface
N = 500;

% top surface of Z: z = 1, x,y = [-1,1]
x1 = 2 * rand(1,N) - 1;
y1 = 2 * rand(1,N) - 1;
z1 = ones(1,N);

% middle surface of Z: z = y, x,y = [-1,1]
x2 = 2 * rand(1,N) - 1;
y2 = 2 * rand(1,N) - 1;
z2 = y2;

% bottom surface of Z: z = 0, x,y = [-1,1]
x3 = rand(1,N)*2 - 1;
y3 = rand(1,N)*2 - 1;
z3 = zeros(1,N) -1;

% combine three surfaces together to form shape Z
X = [x1,x2,x3]';
Y = [y1,y2,y3]';
Z = [z1,z2,z3]';

%% figures
% show manifold of shape Z
figure(1);
m = length(X);
S = zeros(m,1) + 15;
C = linspace(0,255,m)';
scatter3(X,Y,Z,S,C,'filled');
box on;
grid on;
axis([-1.5,1.5,-1.5,1.5,-1.5,1.5]);
view(100,25);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');

% exportgraphics(gcf,'manifoldZ.pdf','ContentType','vector');

% reduce dimension via ISOMAP method
figure(2);
k = 20;
dim = 2;
data = [X,Y,Z];
reducedData = isomap(data,k,dim);
scatter(reducedData(:,1),reducedData(:,2),S,C,'filled');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');

% exportgraphics(gcf,'reducedZ.pdf','ContentType','vector');

%% functions
% implement ISOMAP algorithm
function dataReduced = isomap(data,k,dim)
    % Input:
    %  data: data as N*D matrix (D = dimensionality, N = #points)
    %  k: number of neighbors
    %  dim: max embedding dimensionality
    % Ouput:
    %  dataReduced: embedding as N*dim matrix
    
    % compute pairwise distances & find neighbors 
    m = length(data);
    dis = pdist2(data,data,'euclidean');
    [~,index] = sort(dis,2);
    
    % initialize geodesic distance as infinity
    geoDis = zeros(m) + inf;
    for i = 1:m
        geoDis(i,index(i,1:k+1)) = dis(i,index(i,1:k+1));
    end
    
    % calculate the geodesic distance
    for k = 1:m
        for i = 1:m
            for j = 1:m
                geoDis(i,j) = min(geoDis(i,j), geoDis(i,k) + geoDis(k,j)); 
            end
        end
    end
    
    % multidimensional scaling
    geoDis = geoDis.^2;
    J = eye(m) - ones(m) / m;
    B = - J * geoDis * J / 2;
    [V,E] = eig(B);
    [e,I]= sort(diag(E),'descend');
    dataReduced = V(:,I(1:dim)) * diag(sqrt(e(1:dim)));

end

