function loss = logloss(Y, P)
% LOGLOSS Calculate log loss as defined in the Otto Group Product Classification Challenge
%  LOSS = LOGLOSS(Y, P)
%
% Calculates the log loss of the prediced probabilities P w.r.t. to the true
% class labels Y.
%
% P - an n-by-l matrix of the predicted conditional class probabilities.
% A - an n-by-one or n-by-l matrix of true class labels. If A is n-by-one then
%     A must contain the index of the correct class where 1 <= index <= l. If
%     A is n-by-l then A(i,j) == 1 iff example i has class j.
%
% For more information see:
%  https://www.kaggle.com/c/otto-group-product-classification-challenge/details/evaluation

[N, L] = size(P);

assert(size(Y, 1) == N);

% Transform Y into an indicator matrix if needed
if size(Y, 2) == 1
  Y_ind = sparse(1:N, Y, 1, N, L);
else
  Y_ind = Y;
end

assert(size(Y_ind, 2) == L);

% Normalise the probabilities
P_norm = bsxfun(@times, P, 1 ./ sum(P, 2));
P_norm = max(min(P_norm, 1 - 1e-15), 1e-15);

loss = full(-1 / N * sum(sum(Y_ind .* log(P_norm))));

end % function
