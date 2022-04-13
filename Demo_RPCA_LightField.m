%==========================================================================
% This script compares Light Field Image Denoising methods
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
% addpath(genpath('lib'));
% addpath(genpath('mylib'));
% addpath(genpath('compete_code_TRPCA'));
% addpath(genpath('src'));
% addpath(genpath('evaluation'));
% addpath(genpath('data'));
addpath(genpath(pwd));
rng(2022,'twister')
% Set enable bits
EN_BRTC            = 0; 
EN_HoRPCA          = 0;
EN_t_SVD           = 0;
EN_TT_RPCA         = 0;
EN_fast_TT_RPCA    = 1;

% initial Data
methodname  = {'Original','BRTC','HoRPCA','t-SVD','TT-RPCA','Fast TT-RPCA'};
dataName = {'greek', 'medieval2','pillows','vinyl'};

load('vinyl.mat');

% sparse component (case 1)
NR     = 0.2;
D      = vinyl;
Xim    = D;
idx    = randsample(numel(Xim),round(NR*numel(Xim)));
D(idx) = randi(256,1,length(idx))-1;

% initialization of the parameters
alpha   = ones(1, 3);
alpha   = alpha / sum(alpha);


sizeD     = size(D);
ndim      = length(sizeD);

enList = [];
i      = 1;

%% originial data
[PSNR,RSE,SSIM] = ImageQualityMeasure(Xim, D);
PSNRArr(i)=PSNR;   RSEArr(i)=RSE;   SSIMArr(i)=SSIM;
enList = [enList,i];

fprintf( '=== The variance of noise is %.2f ===\n',NR);
%% BRTC
i  = i + 1;
if EN_BRTC
    disp(['performing ',methodname{i}, ' ... ']);
    tic;
    model    = BayesRCP(D, 'init', 'ml', 'maxRank', 100, 'maxiters', 200,  'verbose', 0);
    Xre{i}   = double(ktensor(model.Z));
    % save the results
    Time(i)  = toc;
    [PSNR,RSE,SSIM] = ImageQualityMeasure(Xim, Xre{i});
    PSNRArr(i)=PSNR;   RSEArr(i)=RSE;   SSIMArr(i)=SSIM;
    enList = [enList,i];
end
%% Use SNN
i  = i + 1;
if EN_HoRPCA
    disp(['performing ',methodname{i}, ' ... ']);
    
    data.T      = D;
    data.X      = D;

    ParH = struct('lambda',.3/sqrt(max(sizeD)),'mu1',5*std(D(:)), 'mu2',5*std(D(:)),...
        'max_iter',1000,'verbose',false,'T',D,'X',D,'X0',D,'E0',zeros(size(D)),...
        'opt_tol',1e-8);
    for vi = 1:length(size(D))
        ParH.V0{vi}  = zeros(size(D));
    end
    tic;
    resultsHOrpca    = tensor_rpca_adal(data, ParH);
    Xre{i}           = double(resultsHOrpca.X);
    Sre{i}           = double(resultsHOrpca.E);
    % save the results
    Time(i) = toc; 
    [PSNR,RSE,SSIM] = ImageQualityMeasure(Xim, Xre{i});
    PSNRArr(i)=PSNR;   RSEArr(i)=RSE;   SSIMArr(i)=SSIM;
    enList = [enList,i];
end
%% Use TNN
i  = i + 1;
if EN_t_SVD
    disp(['performing ',methodname{i}, ' ... ']);
    
    opts.mu = 1e-4;
    opts.tol = 1e-8;
    opts.rho = 1.05;
    opts.max_iter = 1000;
    opts.DEBUG = 0;
    
    tic;
    Dlow    = reshape(D, [128*128, 3, 81]);
    [n1,n2,n3] = size(Dlow);
    lambda = 1/sqrt(max(n1,n2)*n3);
    [XreTNN,SreTNN,err,iter] = trpca_tnn(Dlow,lambda,opts);
    Xre{i}  = reshape(XreTNN, [128, 128, 3, 81]);
    Sre{i}  = reshape(SreTNN, [128, 128, 3, 81]);
    Time(i) = toc;
 
    [PSNR,RSE,SSIM] = ImageQualityMeasure(Xim, Xre{i});
    PSNRArr(i)=PSNR;   RSEArr(i)=RSE;   SSIMArr(i)=SSIM;
    enList = [enList,i];
end

%% Use TT-RPCA
i  = i + 1;
if EN_TT_RPCA
    disp(['performing ',methodname{i}, ' ... ']);

    
    Dh     = D;%reshape(permute(reshape(D,Nway),order),Ndim);
    
    ParTT = struct('mu1',1e-4,'mu2',1e-4,'maxit',1000,'verbose',false,'X0',Dh,...
        'E0',zeros(size(Dh)),'tol',1e-8,'ro',1.1);
    ParTT.alpha = [0.1,0.8,0.1];
    tic;
    resultsTTrpca    = tensor_rpca_TT(Dh, ParTT);
    Xre{i}           = double(resultsTTrpca.X);
    Sre{i}           = double(resultsTTrpca.S);
    % save the results
    Time(i) = toc; 
    
%     Xre{i}           = reshape(ipermute(reshape(Xre{i},inNway),order),size(D));
    
    
    [PSNR,RSE,SSIM] = ImageQualityMeasure(Xim, Xre{i});
    PSNRArr(i)=PSNR;   RSEArr(i)=RSE;   SSIMArr(i)=SSIM;
    enList = [enList,i];
end

%% Use fast TT-RPCA
i  = i + 1;
if EN_fast_TT_RPCA
    disp(['performing ',methodname{i}, ' ... ']);
   
    Dh     = D; %reshape(permute(reshape(D,Nway),order),Ndim);
        
    ParTTFast = struct('mu1',1e-4,'mu2',1e-4,'mu3',1e-4,'maxit',1000,...
        'verbose',false,'X0',Dh,'E0',zeros(size(Dh)),'tol',1e-8,'ro',1.1,...
        'muMax',1e+10);
    
    ParTTFast.alpha = [0.1,0.8,0.1];
    
    R0               = [80,80,3,10];    
    Ttemp            = tucker_als(tensor(Dh), R0,'printitn',0);
    U                = Ttemp.U;
    Xhat             = double(Ttemp.core);
    ParTTFast.Xhat   = Xhat;
    ParTTFast.U      = U;
    
    tic;
    resultFastTT     = tensor_rpca_fast_TT(Dh, ParTTFast);
    Xre{i}           = double(resultFastTT.X);
    Sre{i}           = double(resultFastTT.S);
    
%     Xre{i}           = reshape(ipermute(reshape(Xre{i},inNway),order),size(D));
    
    Time(i) = toc;
    [PSNR,RSE,SSIM] = ImageQualityMeasure(Xim, Xre{i});
    PSNRArr(i)=PSNR;   RSEArr(i)=RSE;   SSIMArr(i)=SSIM;
    enList = [enList,i];
end

%% Show result
fprintf('\n');
fprintf('================== Result ============================\n');
fprintf(' %9.9s    %8.8s    %8.8s   %8.8s    %8.8s\n','method','PSNR', 'RSE', 'SSIM', 'Time');
for i = 1:length(enList)
    fprintf(' %9.9s        %4.4f    %4.4f    %4.4f    %4.4f\n',...
        methodname{enList(i)},PSNRArr(enList(i)),...
        RSEArr(enList(i)),SSIMArr(enList(i)),Time(enList(i)));
end
fprintf('================== Result ============================\n');
