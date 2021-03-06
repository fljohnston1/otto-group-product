clear all

table = readtable('train.csv','Delimiter',',');
train = table{:,2:94}; %table2array
table = readtable('test.csv','Delimiter',',');
test = table{:,2:94}; %table2array

%% Statistics

[n,d]=size(test);


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
prob = hist(label,[1 2 3 4 5 6 7 8 9])./n;

%Mean across N observations
mean = mean(train(:,:));

%Variance across N observations
var = var(train(:,:));

%Median across N observations
median = median(train(:,:));

%Correlation
[r,p] = corrcoef(train); % Compute sample correlation and p-values.
[i,j] = find(p>0.05);
R = [i,j];

for i=1:93
    x(i) = length(find(R(:,2)==i));
end
max(x)


%% Pre-processing

var_idx = find(var>23);
train_red = train(:,var_idx);

%% MATLAB Toolbox

X=train_red;

figure;
plot(X(:,1),X(:,2),'.');

[idx,C] = kmeans(X,9,'Display','final','Replicates',5);

figure;
colors={'r.' 'b.' 'g.' 'y.' 'm.' 'c.' 'r*' 'b*' 'g*' };
for i=1:9
    hold all
    plot(X(idx==i,1),X(idx==i,2),colors{i},'MarkerSize',10)
end
plot(C(:,1),C(:,2),'kx','MarkerSize',12,'LineWidth',3)
title 'Cluster Assignments and Centroids'
hold off

figure;
histogram(idx)
xlabel('Class')
ylabel('Frequency')

%% Spectral Clustering

X=train;
W = getAdjacencyMatrix(X);

%[C, L, U] = SpectralClustering(W, 9, 2);


