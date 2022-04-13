function [tnsr,g]=TT_rand(I,R)
g=TTg_rand(I,R);

tnsr = coreten2tt(g);
end