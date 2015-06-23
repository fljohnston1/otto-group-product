function [B, B_all] = logit1all(X, Y)

Y = double(Y);
B = cell(9, 1);
for class = unique(Y)';
    tic
    fprintf('Training for class %d (n = %d)...', class, sum(Y == class));
    B{class} = mnrfit(X, (Y ~= class) + 1);
    t = toc;
    fprintf('done (%fs)\n', t);
end

if nargout == 2
    tic
    p = logit1allval(B, X);
    toc
    tic
    B_all = mnrfit(p, Y);
    toc
end

end

