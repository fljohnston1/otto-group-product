%% Load datasets
table_train = readtable('train.csv','Delimiter',',');
train = table_train{:,2:94}; %table2array
label = double(categorical(table_train.target));

table_test = readtable('test.csv','Delimiter',',');
test = table_test{:,2:94}; %table2array

%% Logistic Regression

[small, idx] = datasample(train, 200, 'Replace', false);
mdl = fitcnb(small, label(idx),...
    'Distribution', 'kernel', 'KFold', 10);
disp(kfoldLoss(mdl, 'Lossfun', @(C,S,W,COST) logloss(C,S)));

%%
