%% Compressed Sensing

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

% CS parameters
lambdaCS   = 0.01;   
maxItersCS = 250;    
tolCS      = 1e-4;   
protectHW  = 6;      
verboseCS  = true;

%% Step 2: Outlier detection + restoration

[badCols1, scores1] = alc_detect_cols(x_bad1, kMAD, radius);
[badCols2, scores2] = alc_detect_cols(x_bad2, kMAD, radius);
[badCols3, scores3] = alc_detect_cols(x_bad3, kMAD, radius);

fprintf('Detected bad columns (coil1): %s\n', mat2str(badCols1));
fprintf('Detected bad columns (coil2): %s\n', mat2str(badCols2));
fprintf('Detected bad columns (coil3): %s\n', mat2str(badCols3));

x_cs1 = cs_restore_cols(x_bad1, badCols1, lambdaCS, maxItersCS, tolCS, protectHW, verboseCS);
x_cs2 = cs_restore_cols(x_bad2, badCols2, lambdaCS, maxItersCS, tolCS, protectHW, verboseCS);
x_cs3 = cs_restore_cols(x_bad3, badCols3, lambdaCS, maxItersCS, tolCS, protectHW, verboseCS);

%% Step 3: Convert to image domain

Data_img_bad(:,:,1) = ifftshift(ifft2(x_bad1),1);
Data_img_bad(:,:,2) = ifftshift(ifft2(x_bad2),1);
Data_img_bad(:,:,3) = ifftshift(ifft2(x_bad3),1);

Data_img_cs(:,:,1)  = ifftshift(ifft2(x_cs1),1);
Data_img_cs(:,:,2)  = ifftshift(ifft2(x_cs2),1);
Data_img_cs(:,:,3)  = ifftshift(ifft2(x_cs3),1);

Data_img_good(:,:,1) = ifftshift(ifft2(x_good1),1);
Data_img_good(:,:,2) = ifftshift(ifft2(x_good2),1);
Data_img_good(:,:,3) = ifftshift(ifft2(x_good3),1);

%% Step 3: Combine channels + crop image domain

clear_comp   = linspace(10,0.1,size(Data_img_bad,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Data_img_bad,1) 1]);

eye_raw_bad  = sqrt(abs(Data_img_bad(:,:,1)).^2  + abs(Data_img_bad(:,:,2)).^2  + abs(Data_img_bad(:,:,3)).^2)  .* clear_matrix; 
eye_raw_cs   = sqrt(abs(Data_img_cs(:,:,1)).^2   + abs(Data_img_cs(:,:,2)).^2   + abs(Data_img_cs(:,:,3)).^2)   .* clear_matrix; 
eye_raw_good = sqrt(abs(Data_img_good(:,:,1)).^2 + abs(Data_img_good(:,:,2)).^2 + abs(Data_img_good(:,:,3)).^2) .* clear_matrix; 

crop_x = (128 + 60) : (348 - 33);
eye_visualize_bad  = reshape(eye_raw_bad(crop_x, :), [128 128]); 
eye_visualize_cs   = reshape(eye_raw_cs(crop_x, :),  [128 128]); 
eye_visualize_good = reshape(eye_raw_good(crop_x, :),[128 128]); 

std_within = 0.995;
eye_visualize_bad  = hist_stretch_u16(eye_visualize_bad,  std_within);
eye_visualize_cs   = hist_stretch_u16(eye_visualize_cs,   std_within);
eye_visualize_good = hist_stretch_u16(eye_visualize_good, std_within);

%% Step 4: Visualize in k-space (coil 1) and image domain (SoS)

Kb = log1p(abs(x_bad1));      
Kc = log1p(abs(x_cs1));  
Kg = log1p(abs(x_good1));     

% Image domain (coil 1)

figure(1);
subplot(1,3,1); imagesc(Kb); axis image off; colorbar; axis square; title('Bad Data (k-space, cropped)');
subplot(1,3,2); imagesc(Kc); axis image off; colorbar; axis square; title('Cleaned (k-space, cropped)');
subplot(1,3,3); imagesc(Kg); axis image off; colorbar; axis square; title('Good Data (k-space, cropped)');
sgtitle('ALC + Compressed Sensing in K-Space (coil 1)');


% Image domain (SoS)

figure(2);
subplot(1,3,1); imshow(eye_visualize_bad, [], 'InitialMagnification', 'fit'), title('Bad Data (Image Domain)');
subplot(1,3,2); imshow(eye_visualize_cs,  [], 'InitialMagnification', 'fit'), title('Cleaned by CS (Image Domain)');
subplot(1,3,3); imshow(eye_visualize_good,[], 'InitialMagnification', 'fit'), title('Good Data (Image Domain)');
sgtitle('Comparison in Image Domain (SoS)');

%% Step 5: Quantitative metrics

% K-space metrics: bad vs good, clean vs good

residual_bad_k   = x_bad1 - x_good1;
residual_clean_k = x_cs1  - x_good1;

MSE_bad_k    = mean(abs(residual_bad_k(:)).^2);
MSE_clean_k  = mean(abs(residual_clean_k(:)).^2);
NMSE_bad_k   = MSE_bad_k   / mean(abs(x_good1(:)).^2);
NMSE_clean_k = MSE_clean_k / mean(abs(x_good1(:)).^2);
PSNR_bad_k   = 10*log10( max(abs(x_good1(:))).^2 / max(MSE_bad_k, eps) );
PSNR_clean_k = 10*log10( max(abs(x_good1(:))).^2 / max(MSE_clean_k, eps) );
noiseVar_k   = var(real(residual_clean_k(:))) + var(imag(residual_clean_k(:)));

fprintf('\nK-space metrics (coil 1):\n');
fprintf('MSE  bad=%.3e | clean=%.3e\n',  MSE_bad_k,   MSE_clean_k);
fprintf('NMSE bad=%.3e | clean=%.3e\n',  NMSE_bad_k,  NMSE_clean_k);
fprintf('PSNR bad=%.2f dB | clean=%.2f dB\n', PSNR_bad_k, PSNR_clean_k);
fprintf('Noise variance (clean vs good): %.4e\n', noiseVar_k);

% Image-domain metrics: bad vs good, clean vs good

img_bad   = double(eye_visualize_bad);
img_clean = double(eye_visualize_cs);
img_good  = double(eye_visualize_good);

residual_bad_i   = img_bad   - img_good;
residual_clean_i = img_clean - img_good;

MSE_bad_i    = mean(residual_bad_i(:).^2);
MSE_clean_i  = mean(residual_clean_i(:).^2);
NMSE_bad_i   = MSE_bad_i   / mean(img_good(:).^2);
NMSE_clean_i = MSE_clean_i / mean(img_good(:).^2);
PSNR_bad_i   = 10*log10( (max(img_good(:))^2) / max(MSE_bad_i, eps) );
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

% Reconstruction via CS
% Note the protected k-space

function K_clean = cs_restore_cols(K_bad, badCols, lambda, maxIters, tol, protectHW, verbose)

    if nargin < 7, verbose = false; end

    [Nx, Ny] = size(K_bad);
    % DC column index (assumes centered k-space)
    mid = floor(Ny/2) + 1;
    protect = false(1, Ny);
    if protectHW > 0
        protect(max(1,mid-protectHW):min(Ny,mid+protectHW)) = true;
    end

    toFill  = badCols(~protect(badCols));
    skipped = badCols(protect(badCols));
    if ~isempty(skipped)
        fprintf('[CS] Protected (not replaced) cols near DC: %s\n', mat2str(skipped));
    end
    % Build sampling mask: trusted = 1, missing flagged cols = 0
    M = true(Nx, Ny);
    M(:, toFill) = false;

    % Measured k-space on trusted set
    y = M .* K_bad;

    % Unit-normalized FFTs (||F||=1)
    fft2c  = @(x) fftshift(fft2(ifftshift(x))) / sqrt(numel(x));
    ifft2c = @(k) fftshift(ifft2(ifftshift(k))) * sqrt(numel(k));

    % Forward/adjoint and gradient
    A   = @(x) M .* fft2c(x);
    AH  = @(k) ifft2c(M .* k);
    grad= @(x) AH(A(x) - y);

    % DCT prox
    [dct2_, idct2_] = get_dct2_handles();
    soft = @(z,t) sign(z).*max(abs(z)-t,0);

    % Initialize with zero-filled inverse FFT
    x  = ifft2c(y);
    z  = x;
    tN = 1;
    L  = 1.0;                   

    prev_obj = Inf; iters = maxIters; rel = NaN; obj = NaN;
    for it = 1:maxIters
        g   = grad(z);
        v   = z - (1/L)*g;

        vT  = dct2_(v);
        x_new = idct2_( soft(vT, lambda/L) );

        t_new = (1 + sqrt(1 + 4*tN*tN))/2;
        z     = x_new + ((tN-1)/t_new)*(x_new - x);
        tN    = t_new;

        Ax  = A(x_new);
        obj = 0.5*norm(Ax(:)-y(:))^2 + lambda*sum(abs(dct2_(x_new)), 'all');
        rel = abs(prev_obj - obj)/max(obj,1e-12);
        if verbose && (mod(it,25)==0 || it==1)
            fprintf('[CS] FISTA it=%3d | obj=%.3e | rel=%.2e\n', it, obj, rel);
        end
        if rel < tol, iters = it; break; end
        x = x_new; prev_obj = obj; iters = it;
    end

    % Fill only the flagged columns from the CS estimate, keep trusted data
    K_hat  = fft2c(x_new);
    K_clean = K_bad;
    if ~isempty(toFill)
        K_clean(:, toFill) = K_hat(:, toFill);
    end
    
end

% Returns orthonormal 2D DCT/IDCT handles
function [dct2_, idct2_] = get_dct2_handles()
    hasBuiltin = (exist('dct2','file')==2) && (exist('idct2','file')==2);
    if hasBuiltin
        dct2_  = @(X) dct2(X);
        idct2_ = @(X) idct2(X);
    else
        dct2_  = @(X) dct2_orth(X);
        idct2_ = @(X) idct2_orth(X);
    end
end

% Orthonormal 2D DCT-II fallback (no toolboxes)
function Y = dct2_orth(X)
    persistent Dx Dy Nx_prev Ny_prev
    [Nx, Ny] = size(X);
    if isempty(Dx) || Nx ~= Nx_prev, Dx = dctmtx_orth(Nx); Nx_prev = Nx; end
    if isempty(Dy) || Ny ~= Ny_prev, Dy = dctmtx_orth(Ny); Ny_prev = Ny; end
    Y = Dx * X * Dy.';
end

% Inverse of orthonormal 2D DCT-II (transpose)
function X = idct2_orth(Y)
    persistent Dx Dy Nx_prev Ny_prev
    [Nx, Ny] = size(Y);
    if isempty(Dx) || Nx ~= Nx_prev, Dx = dctmtx_orth(Nx); Nx_prev = Nx; end
    if isempty(Dy) || Ny ~= Ny_prev, Dy = dctmtx_orth(Ny); Ny_prev = Ny; end
    X = Dx.' * Y * Dy;
end

% Orthonormal DCT-II matrix
function D = dctmtx_orth(N)
    n = 0:N-1;
    k = (0:N-1)';
    D = cos(pi*(n + 0.5).* (k/N));
    alpha = sqrt(2/N) * ones(N,1); alpha(1) = sqrt(1/N);
    D = diag(alpha) * D;
end