clear all

table = readtable('train.csv','Delimiter',',');
train = table{:,2:94}; %table2array
table = readtable('test.csv','Delimiter',',');
test = table{:,2:94}; %table2array

%% Statistics

%Class Ranges (needed because of how data was imported)
C1 = 1:1929;
C2 = 1930:18051;
C3 = 18052:26055;
C4 = 26056:28746;
C5 = 28747:31485;
C6 = 31486:45620;
C7 = 45621:48459;
C8 = 48460:56923;
C9 = 56924:61878;

%Label Vector
label = zeros(61878,1);
label(C1,1) = 1; %Class 1
label(C2,1) = 2; %Class 2
label(C3,1) = 3; %Class 3
label(C4,1) = 4; %Class 4
label(C5,1) = 5; %Class 5
label(C6,1) = 6; %Class 6
label(C7,1) = 7; %Class 7
label(C8,1) = 8; %Class 8
label(C9,1) = 9; %Class 9

%Class Distributions
figure(1)
hist(label,[1 2 3 4 5 6 7 8 9]);
xlabel('Classes')
ylabel('Data Points')
title('Class-Data Points Distribution')
grid

%Prior Probabilities
prob = hist(label,[1 2 3 4 5 6 7 8 9])./N;

%Mean across N observations
mean = mean(train(:,:));

%Variance across N observations
var = var(train(:,:));

%Median across N observations
median = median(train(:,:));



	
%% Downloaded Toolbox
%the data
data.X = train;
[N,n]=size(data.X);

%data normalization
data = clust_normalize(data,'range');
figure;
plot(data.X(:,1),data.X(:,2),'.')
hold on

%parameters given
param.c=9;
param.vis=1;
param.val=1;

%result=Kmeans(data,param);
result=Kmeans(data,param);
hold on
plot(result.cluster.v(:,1),result.cluster.v(:,2),'ro')

result = validity(result,data,param);
result.validity

%% MATLAB Toolbox

X=train;

figure;
plot(X(:,1),X(:,2),'.');

[idx,C] = kmeans(X,9,'Distance','cityblock','Display','final','Replicates',5);

figure;
colors={'r.' 'b.' 'g.' 'y.' 'm.' 'c.' 'r.' 'b*' 'g*' };
for i=1:9
    hold all
    plot(X(idx==i,1),X(idx==i,2),colors{i},'MarkerSize',12)
end
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3)
title 'Cluster Assignments and Centroids'
hold off

%figure;
%[silh3,h] = silhouette(X,idx,'cityblock');
%h = gca;
%h.Children.EdgeColor = [.8 .8 1];
%xlabel 'Silhouette Value';
%ylabel 'Cluster';

figure;
histogram(idx)
xlabel('Class')
ylabel('Frequency')

%% Spectral Clustering

X=train;
W = getAdjacencyMatrix(X);

