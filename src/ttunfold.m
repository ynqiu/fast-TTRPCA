function Xk = ttunfold(X,k)
% unfold high-order tensor using TT-formate
dim = size(X);
N = numel(dim);
Xk = reshape(X, [prod(dim(1:k)), prod(dim(k+1:N))]);
end

