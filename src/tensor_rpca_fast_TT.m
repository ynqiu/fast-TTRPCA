function results = tensor_rpca_fast_TT( Y, opts )

K = length( size(opts.X0) );
xSize = size(Y);

X0 = Y;
S0 = zeros(xSize);

alphaTemp = zeros(1,K-1);
tau       = 0;
for i=1:K-1
    Xtemp         = ttunfold(Y,i);
    alphaTemp(i)  = min(size(Xtemp));
    tau           = tau+1/sqrt(max(size(Xtemp)));
end
alphaTemp = alphaTemp/sum(alphaTemp);

tau = tau/(K-1);

if isfield(opts,'alpha');      alpha   = opts.alpha;      else; alpha   = alphaTemp;       end
if isfield(opts,'tau');        tau     = opts.tau;        else; tau     = tau;             end
if isfield(opts,'mu1');        mu1     = opts.mu1;        else; mu1     = 1;               end
if isfield(opts,'mu2');        mu2     = opts.mu2;        else; mu2     = 1;               end
if isfield(opts,'mu3');        mu3     = opts.mu3;        else; mu3     = 1;               end
if isfield(opts,'ro');         ro      = opts.ro;         else; ro      = 1.1;             end
if isfield(opts,'muMax');      muMax   = opts.muMax;      else; muMax   = 1e10;            end
if isfield(opts,'maxit');      maxit   = opts.maxit;      else; maxit   = 500;             end
if isfield(opts,'tol');        tol     = opts.tol;        else; tol     = 1e-5;            end
if isfield(opts,'verbose');    verbose = opts.verbose;    else; verbose = 1;               end
if isfield(opts,'R0');         R0      = opts.R0;         else; R0      = 1/2*xSize;       end

if isfield(opts,'X0');         X       = opts.X0;         else;  X      = zeros(xSize);    end
if isfield(opts,'E0');         S       = opts.E0;         else;  S      = zeros(xSize);    end
if isfield(opts,'Xhat');       Xhat    = opts.Xhat;       else;  Xhat   = zeros(R0);       end
if isfield(opts,'U');          U       = opts.U;          else;  U      = cell(1,K-1);     end

% auxilarly variables and dual variables initialization
M = cell( 1, K-1 );
Q = cell( 1, K );

xSizeHat = size(Xhat);
%% I am here.

for k = 1:K-1
    M{k} = Xhat;
    Q{k} = zeros(xSizeHat);
end

E    = zeros(xSize); % dual variables for Y=X+S
P    = zeros(xSize); % dual variables for X=Xhat*U1*...*UK

activeMode=1:K;
for iter = 1:maxit
    % solve M_i's 
    for k = 1:K-1
        XQi   = ttunfold(Xhat-Q{k}/mu2, k);
        Mi    = Pro2TraceNorm(XQi, alpha(k)/mu2);
        M{k}  = ttfold(Mi, xSizeHat);
    end
    
    % solve Uk    
    for k = 1:K
        % computing Bk
        nindices = activeMode(activeMode~=k);
        Vk = ttm(tensor(Xhat), U(nindices), nindices);
        VkD = double(tenmat(1/mu3*P+X,k))*double(tenmat(Vk,k))';
        [Ak,~,Bk] = svd(VkD, 'econ');
        U{k}=Ak*Bk';
    end
    
    % solve Xhat 
    Msum = zeros(xSizeHat);
    Qsum = zeros(xSizeHat);
    for k=1:K-1
        Msum = Msum + M{i};
        Qsum = Qsum + Q{i};
    end
    numXhat = mu2*Msum + Qsum + double(ttm(tensor(mu3*X+P),U,'t'));
    Xhat    = numXhat / ((K-1)*mu2 + mu3);
    
    % solve S  
    YXQ = Y - X + E/mu1;
    S = shrinkage_tt( YXQ, tau/mu1 );
    
    % solve X  
    Xprev = X;
    Xre  = double(ttm(tensor(Xhat), U));
    XP   = mu3*Xre - P;
    X = (XP + mu1*(Y-S) + E ) / (mu1+mu3);
    
    % compute optimality stats
 
    tdiffM = cell( 1, K-1 );
    for k = 1:K-1
        tdiffM{k} = M{k} - Xhat;
 
    end
    tdiffX = X - Xre;
    tdiffY = Y - X - S;
    
    % check convergence conditions
    reX = norm(X(:)-X0(:))/norm(X0(:));
    reS = norm(S(:)-S0(:))/norm(S0(:));
    % print
    if verbose
    fprintf('Iter: %d,   reX: %3.2e,   reS: %3.2e \n', iter, reX, reS );
    end

    if reX<tol && reS<tol
        break;
    end
    
    % update Lagrange multipliers
    for k = 1:K-1
        Q{k} = Q{k} + mu2*tdiffM{k};
    end
    P    = P    + mu3*tdiffX;
    E    = E    + mu1*tdiffY;    
    
    mu1 = min(ro*mu1, muMax);
    mu2 = min(ro*mu2, muMax);
    mu3 = min(ro*mu3, muMax);
    
    X0  = X;
    S0  = S;
%     if rem(iter,50) == 0
%         mu1 = max(mu1*0.8, params.mu_min);    mu2 = max(mu2*0.8, params.mu_min);
%     end
end


results.X = X;
results.S = S;


results.iter = iter;
results.mu = mu1;
results.lambda = tau;

end
