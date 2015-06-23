function [p, p2] = logit1allval(B, X, B_all)
num_classes = 9;
p = zeros(size(X, 1), num_classes);

for class = 1:num_classes
    pcvall = mnrval(B{class}, X);
    p(:, class) = pcvall(:, 1);
end

p2 = 0;
if nargin == 3
    p2 = mnrval(B_all, p);
end

end

