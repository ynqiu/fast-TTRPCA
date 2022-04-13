function T = shrinkage_tt( X, alpha )
% soft-thresholding on tensor X
% T=sign(x)*max{0,x-alpha};

Xv = double( X );
Tv = shrinkage_v( Xv, alpha );
T  = Tv;
end