function g = TTg_rand(I,R)
N = numel(I);
g = cell(N,1);
for i=2:N-1
    g{i} =  1/sqrt(I(i)) * randn(R(i),I(i),R(i+1));
end
g{1} = 1/sqrt(I(1)) * randn(1,I(1),R(1));
g{N} = 1/sqrt(I(N)) * randn(R(N),I(N),1);
end
