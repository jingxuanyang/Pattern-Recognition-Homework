%% dimenReductionLLE.m
% Author:  Jingxuan Yang
% E-mail:  yangjx20@mails.tsinghua.edu.cn
% Page:    www.jingxuanyang.com
% Date:    2021.05.02
% Project: Pattern Recognition
% Purpose: Create manifold of shape W and reduce dimention via LLE
% Note:  

clc;
clear;
close all;

%% parameters
% create manifold of shape W
% number of points per surface
N = 500;

% slope of W's surface
s = sqrt(3);

% left surface of W: x,z = [-1,1], z = -sqrt(3) * y - 3
x1 = 2 * rand(1,N) - 1;
z1 = 2 * rand(1,N) - 1;
y1 = - (3 + z1) / s;

% middle left surface of W: x,z = [-1,1], z = sqrt(3) * y + 1
x2 = 2 * rand(1,N) - 1;
z2 = 2 * rand(1,N) - 1;
y2 = (z2 - 1) / s;

% middle right surface of W: x,z = [-1,1], z = -sqrt(3) * y + 1
x3 = 2 * rand(1,N) - 1;
z3 = 2 * rand(1,N) - 1;
y3 = (1 - z3) / s;

% right surface of W: x,z = [-1,1], z = sqrt(3) * y - 3
x4 = 2 * rand(1,N) - 1;
z4 = 2 * rand(1,N) - 1;
y4 = (z4 + 3) / s;

% combine three surfaces together to form shape W
X = [x1,x2,x3,x4]';
Y = [y1,y2,y3,y4]';
Z = [z1,z2,z3,z4]';

%% figures
% show manifold of shape W
figure(1);
m = length(X);
S = zeros(m,1) + 15;
C = linspace(0,255,m)';
scatter3(X,Y,Z,S,C,'filled');
box on;
grid on;
axis([-1.5,1.5,-3,3,-1.5,1.5]);
view(100,10);
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');

% exportgraphics(gcf,'manifoldW.pdf','ContentType','vector');

% reduce dimension via LLE method
figure(2);
k = 40;
dim = 2;
data = [X,Y,Z];
reducedData = lle(data,k,dim);
scatter(reducedData(:,1),reducedData(:,2),S,C,'filled');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'fontsize', 14, 'fontname', 'Euclid');

% exportgraphics(gcf,'reducedW.pdf','ContentType','vector');

%% functions
% implement LLE algorithm
% ref: https://cs.nyu.edu/~roweis/lle/code/lle.m
function dataReduced = lle(data,k,dim)
    % Input:
    %  data: data as N*D matrix (D = dimensionality, N = #points)
    %  k: number of neighbors
    %  dim: max embedding dimensionality
    % ouput:
    %  dataReduced: embedding as N*dim matrix
    
    % compute pairwise distances & find neighbors 
    [m,D] = size(data);
    dis = pdist2(data,data,'euclidean');
    [~,index] = sort(dis,2);
    
    % regularlizer in case constrained fits are ill conditioned
    if k > D
        tol = 1e-3;  
    else
        tol = 0;
    end
    
    % solve for reconstruction weights
    W = zeros(m,k);
    for i = 1:m
        
        % shift ith pt to origin
        tp = repmat(data(i,:),k,1) - data(index(i,2:k+1),:);
        
        % local covariance
        C = tp * tp';
        
        % regularlization if k > D
        C = C + eye(k) * tol * trace(C);
        
        % solve CW = 1
        W(i,:) = (C \ ones(k,1))';
        
        % enforce sum(W) = 1
        W(i,:) = W(i,:) / sum(W(i,:));
        
    end
    
    % compute embedding from eigenvects of cost matrix M = (I-W)'(I-W)
    M = sparse(m,m);
    for i = 1:m
        M(i,index(i,2:k+1)) = W(i,:); %# ok
    end
    
    % calculation of embedding
    M = (eye(m) - M') * (eye(m) - M);
    [V,E] = eig(M);
    [~,I]= sort(diag(E),'ascend');
    dataReduced = V(:,I(2:dim+1));
    
end

