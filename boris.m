table = readtable('train.csv','Delimiter',',');
train = table{:,2:94}; %table2array
label = double(categorical(table.target));

%%

estdim = intrinsic_dim(train, 'EigValue');

%%
figure
[mapped, mapping] = compute_mapping(train, 'PCA', 93);
plot(cumsum(mapping.lambda / sum(mapping.lambda)));
grid;
title('Cumulative Variance of Principal Components')
xlabel('Number of Principal components')
ylabel('Proportional of Variance')
hold on;
plot([estdim estdim], [0.1 1]);
hold off;
saveTightFigure('cumVarPCA.pdf')

%%

scatter(mapped(:, 1), mapped(:, 2), 5, label);
title('First Two Principal Components')
saveTightFigure('2PCs.pdf')

%%

perm = randperm(length(train));
small_perm = perm(1:1000);
small = train(small_perm, :);

[mapped, ~] = compute_mapping(...
    small, 'KernelPCA', 93, 'poly', 1, 2);
scatter(mapped(:, 1), mapped(:, 2), 15, label(small_perm));
percentiles = prctile(mapped, [2, 98], 1);
title('Kernel-PCA with Polynomial Kernel')
axis(percentiles(1:4) * 1.2);
saveTightFigure('pca-poly.pdf');

%%

perm = randperm(length(train));
small_perm = perm(1:1000);
small = train(small_perm, :);

[mapped, ~] = compute_mapping(...
    small, 'KernelPCA', 93, 'gauss', 5);
scatter(mapped(:, 1), mapped(:, 2), 15, label(small_perm));
title('Kernel-PCA with Gaussian Kernel')
saveTightFigure('pca-gauss.pdf');

%%

%%
nfigs = 0;
for i = 1:93
    if mod(i, 32) == 1
        nfigs = nfigs + 1;
        figure(nfigs);
        clf;
        base = i - 1;
    end
    subplot(4, 8, i - base);
    xs = unique(train(:, i));
    cs = histc(train(:, i), xs);
    plot(log1p(xs), log1p(cs));
    ax = gca;
    ax.XTickLabel = {};
    ax.YTickLabel = {};
end

for f = 1:nfigs
    fig = figure(f);
    saveTightFigure(fig, sprintf('log-log-features-%d.pdf', f));
end

%%

