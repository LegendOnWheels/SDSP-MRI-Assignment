%% RPCA

clc; clear all; close all;

% Loading data

baseDir  = '/Users/pragun/Desktop/MRI/MRI_datasets';           
matFiles = dir(fullfile(baseDir, '**', '*.mat'));            
for k = 1:numel(matFiles)
    filePath = fullfile(matFiles(k).folder, matFiles(k).name);
    load(filePath);
end

x_bad1  = slice1_channel1_badData;  x_good1 = slice1_channel1_goodData;
x_bad2  = slice1_channel2_badData;  x_good2 = slice1_channel2_goodData;
x_bad3  = slice1_channel3_badData;  x_good3 = slice1_channel3_goodData;

%% Step 1: Parameters

% ALC parameters
kMAD   = 2.0;
radius = 1;     

% RPCA (+ PCP via ADMM ) parameters
lambdaRPCA   = [];     
muRPCA       = 1.0; 
rhoADMM      = 1.0;     
maxItersRPCA = 300;
tolRPCA      = 1e-4;    
verboseRPCA  = true;    
columnSparse = false;   
protectHW    = 6;      

%% Step 2: Outlier detection + restoration

[badCols1, ~] = alc_detect_cols(x_bad1, kMAD, radius);
[badCols2, ~] = alc_detect_cols(x_bad2, kMAD, radius);
[badCols3, ~] = alc_detect_cols(x_bad3, kMAD, radius);

fprintf('Detected bad columns (coil1): %s\n', mat2str(badCols1));
fprintf('Detected bad columns (coil2): %s\n', mat2str(badCols2));
fprintf('Detected bad columns (coil3): %s\n', mat2str(badCols3));

x_rpca1 = rpca_restore_cols(x_bad1, badCols1, lambdaRPCA, muRPCA, rhoADMM, maxItersRPCA, tolRPCA, protectHW, verboseRPCA, columnSparse);
x_rpca2 = rpca_restore_cols(x_bad2, badCols2, lambdaRPCA, muRPCA, rhoADMM, maxItersRPCA, tolRPCA, protectHW, verboseRPCA, columnSparse);
x_rpca3 = rpca_restore_cols(x_bad3, badCols3, lambdaRPCA, muRPCA, rhoADMM, maxItersRPCA, tolRPCA, protectHW, verboseRPCA, columnSparse);

%% Step 3: Convert to image domain

Data_img_bad(:,:,1) = ifftshift(ifft2(x_bad1),1);
Data_img_bad(:,:,2) = ifftshift(ifft2(x_bad2),1);
Data_img_bad(:,:,3) = ifftshift(ifft2(x_bad3),1);

Data_img_rpca(:,:,1) = ifftshift(ifft2(x_rpca1),1);
Data_img_rpca(:,:,2) = ifftshift(ifft2(x_rpca2),1);
Data_img_rpca(:,:,3) = ifftshift(ifft2(x_rpca3),1);

Data_img_good(:,:,1) = ifftshift(ifft2(x_good1),1);
Data_img_good(:,:,2) = ifftshift(ifft2(x_good2),1);
Data_img_good(:,:,3) = ifftshift(ifft2(x_good3),1);

%% Step 3: Combine channels + crop image domain

clear_comp   = linspace(10,0.1,size(Data_img_bad,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Data_img_bad,1) 1]);

eye_raw_bad   = sqrt(abs(Data_img_bad(:,:,1)).^2   + abs(Data_img_bad(:,:,2)).^2   + abs(Data_img_bad(:,:,3)).^2)   .* clear_matrix; 
eye_raw_rpca  = sqrt(abs(Data_img_rpca(:,:,1)).^2  + abs(Data_img_rpca(:,:,2)).^2  + abs(Data_img_rpca(:,:,3)).^2)  .* clear_matrix; 
eye_raw_good  = sqrt(abs(Data_img_good(:,:,1)).^2  + abs(Data_img_good(:,:,2)).^2  + abs(Data_img_good(:,:,3)).^2)  .* clear_matrix; 

crop_x = (128 + 60) : (348 - 33);
eye_visualize_bad  = reshape(eye_raw_bad(crop_x, :),  [128 128]); 
eye_visualize_rpca = reshape(eye_raw_rpca(crop_x, :), [128 128]); 
eye_visualize_good = reshape(eye_raw_good(crop_x, :), [128 128]); 

std_within = 0.995;
eye_visualize_bad  = hist_stretch_u16(eye_visualize_bad,  std_within);
eye_visualize_rpca = hist_stretch_u16(eye_visualize_rpca, std_within);
eye_visualize_good = hist_stretch_u16(eye_visualize_good, std_within);

%% Step 4: Visualize in k-space (coil 1) and image domain (SoS)

Kb = log1p(abs(x_bad1));      
Kc = log1p(abs(x_rpca1));  
Kg = log1p(abs(x_good1));     

figure(1);
subplot(1,3,1); imagesc(Kb); axis image off; colorbar; axis square; title('Bad Data (k-space, cropped)');
subplot(1,3,2); imagesc(Kc); axis image off; colorbar; axis square; title('Cleaned (k-space, cropped)');
subplot(1,3,3); imagesc(Kg); axis image off; colorbar; axis square; title('Good Data (k-space, cropped)');
sgtitle('ALC + RPCA in K-Space (coil 1)');

figure(2);
subplot(1,3,1); imshow(eye_visualize_bad,  [], 'InitialMagnification', 'fit'), title('Bad Data (Image Domain)');
subplot(1,3,2); imshow(eye_visualize_rpca, [], 'InitialMagnification', 'fit'), title('Cleaned by RPCA (Image Domain)');
subplot(1,3,3); imshow(eye_visualize_good, [], 'InitialMagnification', 'fit'), title('Good Data (Image Domain)');
sgtitle('Comparison in Image Domain (SoS)');

%% Step 5: Quantitative metrics

% K-space metrics: bad vs good, clean vs good

residual_bad_k   = x_bad1   - x_good1;
residual_clean_k = x_rpca1  - x_good1;

MSE_bad_k    = mean(abs(residual_bad_k(:)).^2);
MSE_clean_k  = mean(abs(residual_clean_k(:)).^2);
NMSE_bad_k   = MSE_bad_k   / mean(abs(x_good1(:)).^2);
NMSE_clean_k = MSE_clean_k / mean(abs(x_good1(:)).^2);
PSNR_bad_k   = 10*log10( max(abs(x_good1(:))).^2 / max(MSE_bad_k,   eps) );
PSNR_clean_k = 10*log10( max(abs(x_good1(:))).^2 / max(MSE_clean_k, eps) );
noiseVar_k   = var(real(residual_clean_k(:))) + var(imag(residual_clean_k(:)));

fprintf('\nK-space metrics (coil 1):\n');
fprintf('MSE  bad=%.3e | clean=%.3e\n',  MSE_bad_k,   MSE_clean_k);
fprintf('NMSE bad=%.3e | clean=%.3e\n',  NMSE_bad_k,  NMSE_clean_k);
fprintf('PSNR bad=%.2f dB | clean=%.2f dB\n', PSNR_bad_k, PSNR_clean_k);
fprintf('Noise variance (clean vs good): %.4e\n', noiseVar_k);

% Image-domain metrics: bad vs good, clean vs good

img_bad   = double(eye_visualize_bad);
img_clean = double(eye_visualize_rpca);
img_good  = double(eye_visualize_good);

residual_bad_i   = img_bad   - img_good;
residual_clean_i = img_clean - img_good;

MSE_bad_i    = mean(residual_bad_i(:).^2);
MSE_clean_i  = mean(residual_clean_i(:).^2);
NMSE_bad_i   = MSE_bad_i   / mean(img_good(:).^2);
NMSE_clean_i = MSE_clean_i / mean(img_good(:).^2);
PSNR_bad_i   = 10*log10( (max(img_good(:))^2) / max(MSE_bad_i,   eps) );
PSNR_clean_i = 10*log10( (max(img_good(:))^2) / max(MSE_clean_i, eps) );
if exist('ssim','file')
    SSIM_bad_i   = ssim(norm01(img_bad),  norm01(img_good));
    SSIM_clean_i = ssim(norm01(img_clean),norm01(img_good));
else
    SSIM_bad_i = NaN; SSIM_clean_i = NaN;
end

fprintf('\nImage-domain metrics (SoS):\n');
fprintf('MSE  bad=%.3e | clean=%.3e\n',  MSE_bad_i,   MSE_clean_i);
fprintf('NMSE bad=%.3e | clean=%.3e\n',  NMSE_bad_i,  NMSE_clean_i);
fprintf('PSNR bad=%.2f dB | clean=%.2f dB\n', PSNR_bad_i, PSNR_clean_i);
fprintf('SSIM bad=%.3f | clean=%.3f\n', SSIM_bad_i, SSIM_clean_i);

%% ====== HELPER FUNCTIONS =====

% Adjacent-line correlation (ALC) detection

function [badCols, scores] = alc_detect_cols(X, kMAD, radius)
    Nx = size(X,2); scores = zeros(1,Nx);
    for c = 1:Nx
        nn = [];
        for r = 1:radius
            if c-r>=1,  nn(end+1) = corrmag(X(:,c), X(:,c-r)); end
            if c+r<=Nx, nn(end+1) = corrmag(X(:,c), X(:,c+r)); end
        end
        scores(c) = 1 - median(nn);
    end
    thr = median(scores) + kMAD*(mad(scores,1)+eps);
    badCols = find(scores > thr);
end

% Magnitude of complex correlation

function r = corrmag(a,b)
    r = abs( sum(conj(a(:)).*b(:)) ) / max(eps, norm(a(:))*norm(b(:)));
end

% Histogram-based intensity cap

function y = hist_stretch_u16(x, frac)
    x = double(x);
    bins = linspace(0, max(x(:))+eps, 1000);
    h = histcounts(x(:), bins);
    c = cumsum(h)/sum(h);
    idx = find(c > frac, 1, 'first'); 
    if isempty(idx), thr = max(x(:)); else, thr = bins(idx); end
    y = uint16( min(x, thr) * (65535 / max(thr, eps)) );
end

% Normalize to [0, 1]

function y = norm01(x)
    x = double(x);
    x = x - min(x(:)); y = x / max(eps, max(x(:)));
end

% RPCA-based restoration using PCP

function K_clean = rpca_restore_cols(K_bad, badCols, lambda, mu, rho, maxIters, tol, protectHW, verbose, columnSparse)

    [Nx, Ny] = size(K_bad);
    if isempty(lambda)
        lambda = 1 / sqrt(max(Nx,Ny));  % theory-guided default
    end

    % Run RPCA/PCP on the full k-space (only use L on flagged cols)
    opts.lambda      = lambda;
    opts.mu          = mu;
    opts.rho         = rho;
    opts.maxIters    = maxIters;
    opts.tol         = tol;
    opts.verbose     = verbose;
    opts.columnSparse= columnSparse;

    [L, S, info] = rpca_pcp(K_bad, opts); 
    if verbose
        fprintf('[RPCA] iters=%d | final rel=%.2e | obj=%.3e\n', info.iters, info.rel, info.obj);
    end

    % DC protection
    mid = floor(Ny/2) + 1;
    protect = false(1, Ny);
    if protectHW > 0
        protect(max(1,mid-protectHW):min(Ny,mid+protectHW)) = true;
    end
    toReplace = badCols(~protect(badCols));
    skipped   = badCols(protect(badCols));
    if ~isempty(skipped)
        fprintf('[RPCA] Protected (not replaced) cols near DC: %s\n', mat2str(skipped));
    end

    % Replace only the flagged columns with low-rank L
    K_clean = K_bad;
    if ~isempty(toReplace)
        K_clean(:, toReplace) = L(:, toReplace);
    end
end

% RPCA via ADMM (complex valued SVD is used)

function [L, S, info] = rpca_pcp(K, opts)

    lambda = opts.lambda;
    mu     = opts.mu;
    rho    = opts.rho;
    maxIt  = opts.maxIters;
    tol    = opts.tol;
    verb   = opts.verbose;
    colsp  = opts.columnSparse;

    [Nx, Ny] = size(K);
    L = zeros(Nx, Ny, 'like', K);
    S = zeros(Nx, Ny, 'like', K);
    U = zeros(Nx, Ny, 'like', K);  % scaled dual

    prev_obj = Inf; rel = NaN; obj = NaN; iters = maxIt;

    for it = 1:maxIt
        % L-update: singular value soft-thresholding
        M  = K - S + U;
        L  = svt_shrink(M, mu/rho);

        % S-update: entrywise or column-wise soft-thresholding
        R  = K - L + U;
        if ~colsp
            S = soft_thresh(R, lambda/rho);
        else
            S = group_soft_cols(R, lambda/rho);
        end

        % Dual update
        E = K - L - S;         
        U = U + E;

        % Objective & stopping
        if ~colsp
            sparsity_term = lambda * sum(abs(S(:)));
        else
            sparsity_term = lambda * sum( sqrt(sum(abs(S).^2, 1)) );
        end

        svals = svd(L, 'econ'); nuc = mu * sum(svals);
        obj   = 0.5 * sum(abs(E(:)).^2) + nuc + sparsity_term;

        rel = norm(E,'fro') / max(norm(K,'fro'), eps);
        if verb && (mod(it,10)==0 || it==1)
            fprintf('[RPCA] it=%3d | rel=%.2e | obj=%.3e\n', it, rel, obj);
        end
        if rel < tol || abs(prev_obj-obj)/max(obj,1e-12) < tol
            iters = it; break;
        end
        prev_obj = obj; iters = it;
    end

    info.iters = iters;
    info.rel   = rel;
    info.obj   = obj;
end

% Singular Value Thresholding (complex SVD)

function L = svt_shrink(M, tau)
    [U,S,V] = svd(M, 'econ');
    s = diag(S);
    s = max(s - tau, 0);
    r = nnz(s);
    if r==0
        L = zeros(size(M), 'like', M);
    else
        L = U(:,1:r) * diag(s(1:r)) * V(:,1:r)';
    end
end

% Entrywise complex soft-thresholding

function Y = soft_thresh(X, tau)
    mag = abs(X);
    scale = max(0, 1 - tau ./ max(mag, eps));
    Y = scale .* X;
end

% Column-wise group soft thresholding

function Y = group_soft_cols(X, tau)
    mags = sqrt(sum(abs(X).^2, 1));      % 1 x Ny
    scale = max(0, 1 - (tau ./ max(mags, eps)));
    Y = X .* repmat(scale, size(X,1), 1);
end