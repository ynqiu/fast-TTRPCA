%==========================================================================
% This script compares Color Image Denoising methods
% listed as follows:
%   1. BRTC
%   2. HoRPCA
%   3. tensor-SVD-based RPCA method (t-SVD)
%   4. TT-RPCA
%   5. Fast TT-RPCA
%
% Four quality assessment (QA) indices -- Fmeasure
% -- are calculated for each methods after denoising.
%
% You can:
%       Type 'Demo_RPCA_Color_Image' to to run various methods and see the pre-computed results.
%
% more detail can be found in [1]
% [1] Y. Qiu, G. Zhou, Z. Huang, Q. Zhao, and S. Xie, 
%“Efficient Tensor Robust PCA under Hybrid Model of Tucker and Tensor Train,” 
%  IEEE Signal Process. Lett., vol. 14, no. 8, pp. 1–1, 2022, doi: 10.1109/LSP.2022.3143721.
%
% by Yuning Qiu
% 11/5/2022
%==========================================================================
clear;close all;
addpath(genpath('lib'));
addpath(genpath('mylib'));
addpath(genpath('compete_code_TRPCA'));
addpath(genpath('src'));
addpath(genpath('evaluation'));
rng(2022,'twister')
% Set enable bits
EN_BRTC            = 0; 
EN_HoRPCA          = 0;
EN_t_SVD           = 0;
EN_TT_RPCA         = 0;
EN_fast_TT_RPCA    = 1;

% initial Data
methodname  = {'BRTC','SNN','t-SVD','TT-RPCA','fast TT-RPCA'};
Xim = double(imread('data/images/giant512.png'));


% sparse component
NR     = 0.2;
D      = Xim;
idx    = randsample(numel(Xim),round(NR*numel(Xim)));
D(idx) = randi(256,1,length(idx))-1;

% initialization of the parameters
alpha   = ones(1, 3);
alpha   = alpha / sum(alpha);
maxIter = 1000;
mu      = 1e-2; 
epsilon = 1e-5;

sizeD     = size(D);
ndim      = length(sizeD);

% normalization
Nway   = [4,4,8,4, 4,4,4,8, 3];
Ndim   = [16,16,32,32,3];
order  = [1,5,2,6,3,7,4,8,9];
inNway = [4,4,4,4, 8,4,4,8, 3];

enList = [];
i      = 0;

fprintf( '=== The variance of noise is %.2f ===\n',NR);
%% BRTC
i  = i + 1;
if EN_BRTC
    disp(['performing ',methodname{i}, ' ... ']);
    tic;
    model    = BayesRCP(D, 'init', 'ml', 'maxRank', 100, 'maxiters', 100,  'verbose', 0);
    Xre{i}   = double(ktensor(model.Z));
    % save the results
    Time(i)  = toc;
    PSNR(i) = PSNR_RGB(Xhat, Xim);
    RSE(i)  = perfscore(Xhat, Xim);
    SSIM(i) = ssim_index( rgb2gray(uint8(Xhat)), rgb2gray(uint8(Xim)));
    enList = [enList,i];
end
%% Use SNN
i  = i + 1;
if EN_HoRPCA
    disp(['performing ',methodname{i}, ' ... ']);
    
    data.T      = D;
    data.X      = D;

    ParH = struct('lambda',.3/sqrt(max(sizeD)),'mu1',5*std(D(:)), 'mu2',5*std(D(:)),...
        'max_iter',100,'verbose',false,'T',D,'X',D,'X0',D,'E0',zeros(size(D)),...
        'opt_tol',1e-8);
    for vi = 1:3
        ParH.V0{vi}  = zeros(size(D));
    end
    tic;
    resultsHOrpca    = tensor_rpca_adal(data, ParH);
    Xre{i}           = double(resultsHOrpca.X);
    Sre{i}           = double(resultsHOrpca.E);
    % save the results
    Time(i) = toc; 
    PSNR(i) = PSNR_RGB(Xhat, Xim);
    RSE(i)  = perfscore(Xhat, Xim);
    SSIM(i) = ssim_index( rgb2gray(uint8(Xhat)), rgb2gray(uint8(Xim)));
    enList = [enList,i];
end
%% Use t-SVD
i  = i + 1;
if EN_t_SVD
    disp(['performing ',methodname{i}, ' ... ']);
    tic;
    Dmax    = max(D(:));
    Dnor    = D ./ Dmax;
    [Xre{i}, Sre{i}] =   tensor_rpca( D , 0.03/sqrt(size(D,1)));
    Xre{i}  = Xre{i} * Dmax;
    Time(i) = toc;
 
    PSNR(i) = PSNR_RGB(Xre{i}, Xim);
    RSE(i)  = perfscore(Xre{i}, Xim);
    SSIM(i) = ssim_index( rgb2gray(uint8(Xre{i})), rgb2gray(uint8(Xim)));
    enList = [enList,i];
end

%% Use TT-RPCA
i  = i + 1;
if EN_TT_RPCA
    disp(['performing ',methodname{i}, ' ... ']);

    
    Dh     = reshape(permute(reshape(D,Nway),order),Ndim);
    
    ParTT = struct('mu1',1e-3,'mu2',1e-3,'maxit',1000,'verbose',false,'X0',Dh,...
        'E0',zeros(size(Dh)),'tol',1e-8,'ro',1.1);
    
    tic;
    resultsTTrpca    = tensor_rpca_TT(Dh, ParTT);
    Xre{i}           = double(resultsTTrpca.X);
    Sre{i}           = double(resultsTTrpca.S);
    % save the results
    Time(i) = toc; 
    
    Xre{i}           = reshape(ipermute(reshape(Xre{i},inNway),order),size(D));
    
    
    PSNR(i) = PSNR_RGB(Xre{i}, Xim);
    RSE(i)  = perfscore(Xre{i}, Xim);
    SSIM(i) = ssim_index( rgb2gray(uint8(Xre{i})), rgb2gray(uint8(Xim)));
    enList = [enList,i];
end

%% Use fast TT-RPCA
i  = i + 1;
if EN_fast_TT_RPCA
    disp(['performing ',methodname{i}, ' ... ']);
   
    Dh     = reshape(permute(reshape(D,Nway),order),Ndim);
        
    ParTTFast = struct('mu1',1e-4,'mu2',1e-4,'mu3',1e-4,'maxit',1000,...
        'verbose',false,'X0',Dh,'E0',zeros(size(Dh)),'tol',1e-9,'ro',1.2,...
        'muMax',1e+10);
    
    R0               = [8,8,16,16,3];    
    Ttemp            = tucker_als(tensor(Dh), R0,'printitn',0);
    U                = Ttemp.U;
    Xhat             = double(Ttemp.core);
    ParTTFast.Xhat   = Xhat;
    ParTTFast.U      = U;
    
    tic;
    resultFastTT     = tensor_rpca_fast_TT(Dh, ParTTFast);
    Xre{i}           = double(resultFastTT.X);
    Sre{i}           = double(resultFastTT.S);
    
    Xre{i}           = reshape(ipermute(reshape(Xre{i},inNway),order),size(D));
    
    Time(i) = toc;
    PSNR(i) = PSNR_RGB(Xre{i}, Xim);
    RSE(i)  = perfscore(Xre{i}, Xim);
    SSIM(i) = ssim_index( rgb2gray(uint8(Xre{i})), rgb2gray(uint8(Xim)));
    
    enList = [enList,i];
end

%% Show result
fprintf('\n');
fprintf('================== Result ============================\n');
fprintf(' %9.9s    %8.8s   %8.8s  %8.8s %8.8s\n','method','PSNR', 'RSE', 'SSIM', 'Time');
for i = 1:length(enList)
    fprintf(' %9.9s        %4.4f    %4.4f    %4.4f    %4.4f\n',...
        methodname{enList(i)},PSNR(enList(i)),...
        RSE(enList(i)),SSIM(enList(i)),Time(enList(i)));
end
fprintf('================== Result ============================\n');
