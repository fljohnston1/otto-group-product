table_train = readtable('train.csv','Delimiter',',');
train = table_train{:,2:94};
label = double(categorical(table_train.target));

trainl = log1p(train);
[trainlp, mappinglp] = pca(trainl, 7);
[trainp, mappingp] = pca(train, 7);

table_test = readtable('test.csv','Delimiter',',');
test = table_test{:,2:94}; %table2array
testl = log1p(test);
testlp = bsxfun(@minus, testl, mappinglp.mean) * mappinglp.M;
testp = bsxfun(@minus, test, mappingp.mean) * mappingp.M;

%% Default MN Logistic Regression
rng('default') % for reproducibility

holdout = cvpartition(label, 'HoldOut', 0.98);
X = trainp(holdout.training, :);
y = label(holdout.training);

X_test = trainp(holdout.test, :);
y_test = label(holdout.test);

tic
B = mnrfit(X, y);
toc

test_p = mnrval(B, X_test);

disp(logloss(double(y_test), test_p));


%% 1 vs. all Logistic Regression (converges better)
rng('default') % for reproducibility

in = trainlp;

holdout = cvpartition(label, 'HoldOut', 0.9);
X = in(holdout.training, :);
y = label(holdout.training);

X_test = in(holdout.test, :);
y_test = label(holdout.test);

[B, B_all] = logit1all(X, y);

[test_p, test_p2] = logit1allval(B, X_test, B_all);

disp(logloss(double(y_test), test_p));
disp(logloss(double(y_test), test_p2));

%% Make predictions for submission
rng('default') % for reproducibility

train_proportion = 0.1;
best_loss = inf;
best_B = 0;
plain_losses = zeros(10, 1);
for k = 1:10
    fprintf('Plain MVL on %.0f%% of data...', train_proportion * 100);
    h = cvpartition(label, 'HoldOut', 1 - train_proportion);
    X_plain = trainlp(h.training, :);
    y_plain = label(h.training);
    
    X_val = trainlp(h.test, :);
    y_val = label(h.test);
    
    tic
    B = mnrfit(X_plain, y_plain);
    t = toc;
    fprintf(' done (%fs)\n', t);
    
    train_p = mnrval(B, X_plain);
    val_p = mnrval(B, X_val);
    plain_losses(k) = logloss(y_val, val_p);
    fprintf('Training loss: %f\n', logloss(y_plain, train_p));
    fprintf('Validation loss: %f\n', plain_losses(k));
    
    if plain_losses(k) < best_loss
        best_loss = plain_losses(k);
        best_B = B;
    end
end

boxplot(plain_losses);

fprintf('Preparing test submission\n');
test_p = mnrval(best_B, testlp);
save_submission(test_p, 'plain_logit.csv');  % 0.94506 on Kaggle

%% 1 vs rest

rng('default') % for reproducibility

fprintf('1 vs rest\n');

one_losses = crossval(@(Xtrain, Ytrain, Xtest, Ytest) ...
    logloss(Ytest, logit1allval(logit1all(Xtrain, Ytrain), Xtest)), ...
    trainlp, label, 'KFold', 10, 'stratify', label);

boxplot(one_losses);

best_B = logit1all(trainlp, label);
fprintf('Preparing test submission\n');
test_p = logit1allval(best_B, testlp);
save_submission(test_p, '1all_logit.csv');  % 1.01918 on Kaggle

%% Two level  Terrible, don't run :)

rng('default') % for reproducibility

train_proportion = 0.10;
fprintf('Two level on %.0f%% of data\n', train_proportion * 100);
best_loss = inf;
best_B = 0;
best_B_all = 0;
two_losses = zeros(10, 1);
for k = 1:10
    h = cvpartition(label, 'HoldOut', 1 - train_proportion);
    X_plain = trainlp(h.training, :);
    y_plain = label(h.training);
    
    X_val = trainlp(h.test, :);
    y_val = label(h.test);
    
    [B, B_all] = logit1all(X_plain, y_plain);
    
    train_p = logit1allval(B, X_plain, B_all);
    val_p = logit1allval(B, X_val, B_all);
    two_losses(k) = logloss(y_val, val_p);
    fprintf('Training loss: %f\n', logloss(y_plain, train_p));
    fprintf('Validation loss: %f\n', two_losses(k));
    
    if two_losses(k) < best_loss
        best_loss = two_losses(k);
        best_B = B;
        best_B_all = B_all;
    end
end

boxplot(two_losses);

fprintf('Preparing test submission\n');
test_p = logit1allval(best_B, testlp, best_B_all);
save_submission(test_p, 'two_level.csv');  % 0.94506 on Kaggle


