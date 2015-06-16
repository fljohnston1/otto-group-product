function [W] = getAdjacencyMatrix(X)

[m, n] = size(X);

X_size = m*n;

% 1-off diagonal elements
V = repmat([ones(m-1,1); 0],n, 1);
V = V(1:end-1); % remove last zero

% n-off diagonal elements
U = ones(m*(n-1), 1);

% get the upper triangular part of the matrix
W = sparse(1:(X_size-1),    2:X_size, V, X_size, X_size)...
  + sparse(1:(X_size-m),(m+1):X_size, U, X_size, X_size);

% finally make W symmetric
W = W + W';

end

