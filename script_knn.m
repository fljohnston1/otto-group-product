table_train = readtable('train.csv','Delimiter',',');
train = table_train{:,2:94}; %table2array
label = double(categorical(table_train.target));

%%

table_test = readtable('test.csv','Delimiter',',');
test = table_test{:,2:94}; %table2array

%%
rng(1)
priors = countcats(categorical(table_train.target));
ns = round(logspace(3, log10(15000), 10));
times = zeros(size(ns));
losses = zeros(size(ns));
pca_times = zeros(size(ns));
pca_losses = zeros(size(ns));
num_neighbours = 3;
for i = 1:length(ns)
    disp(ns(i));
    [small, idx] = datasample(train, ns(i), 'Replace', false);
    
    tic;
    mdl = fitcknn(small, label(idx),...
        'NumNeighbors', num_neighbours, 'KFold', 10);
    losses(i) = kfoldLoss(mdl, 'Lossfun', @(C,S,W,COST) logloss(C,S));
    times(i) = toc;
    
    tic;
    small = pca(small, 10);
    
    mdl = fitcknn(small, label(idx),...
        'NumNeighbors', num_neighbours, 'KFold', 10);
    pca_losses(i) = kfoldLoss(mdl, 'Lossfun', @(C,S,W,COST) logloss(C,S));
    pca_times(i) = toc;
end

%% Try to find the best k to use
rng(1)  % For reproducibility

[small, idx] = datasample(train, 2000, 'Replace', false);
% data = pca(small, 7);
data = small;
params = unique(round(logspace(log10(15), log10(151), 10)));
n_params = length(params);
min_loss = zeros(1, n_params);
mean_loss = zeros(1, n_params);
max_loss = zeros(1, n_params);
for i = 1:n_params
    inner_losses = zeros(10, 1);
    tic
    for k = 1:10
        mdl = fitcknn(data, label(idx), 'NumNeighbors', params(i), 'KFold', 10);
        inner_losses(k) = kfoldLoss(mdl, 'Lossfun', @(C,S,W,COST) logloss(C,S));
    end
    min_loss(i) = min(inner_losses);
    mean_loss(i) = mean(inner_losses);
    max_loss(i) = max(inner_losses);
    t = toc;
    fprintf('k = %d, loss = %f, t = %f\n', params(i), mean_loss(i), t);
end

figure
plot(params, mean_loss, params, max_loss, '.-', params, min_loss, '.-');

%% Try to find the relationship between k and n

rng(1)  % For reproducibility

params = unique(round(logspace(log10(1000), log10(length(train)), 15)));
n_params = length(params);
losses = zeros(1, n_params);
ks = zeros(1, n_params);
k = 15;
inicount = 3;
omega = nthroot(2, 5);
for i = 1:n_params
    tic
    [small, idx] = datasample(train, params(i), 'Replace', false);
    small = pca(small, 7);
    best_loss = inf;
    countdown = inicount;
    best_k = k;
    while countdown > 0 && k <= length(train)
        mdl = fitcknn(small, label(idx), 'NumNeighbors', k, 'KFold', 10);
        loss = kfoldLoss(mdl, 'Lossfun', @(C,S,W,COST) logloss(C,S));
        if loss <= best_loss
            countdown = inicount;
            best_k = k;
        else
            countdown = countdown - 1;
        end
        k = ceil(k * omega);
        best_loss = min(best_loss, loss);
    end
    losses(i) = best_loss;
    ks(i) = best_k;
    t = toc;
    fprintf('n = %d, k = %d, loss = %f, t = %f\n',...
        params(i), best_k, best_loss, t);
    k = round(best_k / 2);
end

figure
plot(params, losses);


