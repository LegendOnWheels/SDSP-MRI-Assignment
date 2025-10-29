%% Savitzky–Golay smoothing

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

% SG parameters
sg_win     = 7;   
sg_degree  = 2;   
protectHW  = 6; 

%% Step 2: Outlier detection + restoration

[badCols1, ~] = alc_detect_cols(x_bad1, kMAD, radius);
[badCols2, ~] = alc_detect_cols(x_bad2, kMAD, radius);
[badCols3, ~] = alc_detect_cols(x_bad3, kMAD, radius);

fprintf('Detected bad columns (coil1): %s\n', mat2str(badCols1));
fprintf('Detected bad columns (coil2): %s\n', mat2str(badCols2));
fprintf('Detected bad columns (coil3): %s\n', mat2str(badCols3));

x_smooth1 = sg_smooth_cols(x_bad1, badCols1, sg_win, sg_degree, protectHW);
x_smooth2 = sg_smooth_cols(x_bad2, badCols2, sg_win, sg_degree, protectHW);
x_smooth3 = sg_smooth_cols(x_bad3, badCols3, sg_win, sg_degree, protectHW);

%% Step 3: Convert to image domain

Data_img_bad(:,:,1)    = ifftshift(ifft2(x_bad1),1);
Data_img_bad(:,:,2)    = ifftshift(ifft2(x_bad2),1);
Data_img_bad(:,:,3)    = ifftshift(ifft2(x_bad3),1);

Data_img_smooth(:,:,1) = ifftshift(ifft2(x_smooth1),1);
Data_img_smooth(:,:,2) = ifftshift(ifft2(x_smooth2),1);
Data_img_smooth(:,:,3) = ifftshift(ifft2(x_smooth3),1);

Data_img_good(:,:,1)   = ifftshift(ifft2(x_good1),1);
Data_img_good(:,:,2)   = ifftshift(ifft2(x_good2),1);
Data_img_good(:,:,3)   = ifftshift(ifft2(x_good3),1);

%% Step 3: Combine channels + crop image domain

clear_comp   = linspace(10,0.1,size(Data_img_bad,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Data_img_bad,1) 1]);

eye_raw_bad    = sqrt(abs(Data_img_bad(:,:,1)).^2    + abs(Data_img_bad(:,:,2)).^2    + abs(Data_img_bad(:,:,3)).^2)    .* clear_matrix; 
eye_raw_smooth = sqrt(abs(Data_img_smooth(:,:,1)).^2 + abs(Data_img_smooth(:,:,2)).^2 + abs(Data_img_smooth(:,:,3)).^2) .* clear_matrix; 
eye_raw_good   = sqrt(abs(Data_img_good(:,:,1)).^2   + abs(Data_img_good(:,:,2)).^2   + abs(Data_img_good(:,:,3)).^2)   .* clear_matrix; 

crop_x = (128 + 60) : (348 - 33);
eye_visualize_bad    = reshape(eye_raw_bad(crop_x, :),   [128 128]); 
eye_visualize_smooth = reshape(eye_raw_smooth(crop_x, :),[128 128]); 
eye_visualize_good   = reshape(eye_raw_good(crop_x, :),  [128 128]); 

std_within = 0.995;
eye_visualize_bad    = hist_stretch_u16(eye_visualize_bad,    std_within);
eye_visualize_smooth = hist_stretch_u16(eye_visualize_smooth, std_within);
eye_visualize_good   = hist_stretch_u16(eye_visualize_good,   std_within);

%% Step 4: Visualize in k-space (coil 1) and image domain (SoS)

Kb = log1p(abs(x_bad1));      
Kc = log1p(abs(x_smooth1));  
Kg = log1p(abs(x_good1));     

figure(1);
subplot(1,3,1); imagesc(Kb); axis image off; colorbar; axis square; title('Bad Data (k-space, cropped)');
subplot(1,3,2); imagesc(Kc); axis image off; colorbar; axis square; title('Cleaned (k-space, cropped)');
subplot(1,3,3); imagesc(Kg); axis image off; colorbar; axis square; title('Good Data (k-space, cropped)');
sgtitle('ALC + Savitzky–Golay Smoothing in K-Space (coil 1)');

figure(2);
subplot(1,3,1); imshow(eye_visualize_bad,   []), title('Bad Data (Image Domain)');
subplot(1,3,2); imshow(eye_visualize_smooth,[]), title('Cleaned Data (Image Domain)');
subplot(1,3,3); imshow(eye_visualize_good,  []), title('Good Data (Image Domain)');
sgtitle('Comparison in Image Domain (SoS)');

%% Step 5: Quantitative metrics

% K-space metrics: bad vs good, clean vs good

residual_bad_k   = x_bad1    - x_good1;
residual_clean_k = x_smooth1 - x_good1;

MSE_bad_k    = mean(abs(residual_bad_k(:)).^2);
MSE_clean_k  = mean(abs(residual_clean_k(:)).^2);
NMSE_bad_k   = MSE_bad_k   / mean(abs(x_good1(:)).^2);
NMSE_clean_k = MSE_clean_k / mean(abs(x_good1(:)).^2);
PSNR_bad_k   = 10*log10( max(abs(x_good1(:))).^2 / MSE_bad_k );
PSNR_clean_k = 10*log10( max(abs(x_good1(:))).^2 / MSE_clean_k );
noiseVar_k   = var(real(residual_clean_k(:))) + var(imag(residual_clean_k(:)));

fprintf('\nK-space metrics (coil 1):\n');
fprintf('MSE  bad=%.3e | clean=%.3e\n',  MSE_bad_k,   MSE_clean_k);
fprintf('NMSE bad=%.3e | clean=%.3e\n',  NMSE_bad_k,  NMSE_clean_k);
fprintf('PSNR bad=%.2f dB | clean=%.2f dB\n', PSNR_bad_k, PSNR_clean_k);
fprintf('Noise variance (clean vs good): %.4e\n', noiseVar_k);

% Image-domain metrics: bad vs good, clean vs good

img_bad   = double(eye_visualize_bad);
img_clean = double(eye_visualize_smooth);
img_good  = double(eye_visualize_good);

residual_bad_i   = img_bad   - img_good;
residual_clean_i = img_clean - img_good;

MSE_bad_i    = mean(residual_bad_i(:).^2);
MSE_clean_i  = mean(residual_clean_i(:).^2);
NMSE_bad_i   = MSE_bad_i   / mean(img_good(:).^2);
NMSE_clean_i = MSE_clean_i / mean(img_good(:).^2);
PSNR_bad_i   = 10*log10( (max(img_good(:))^2) / MSE_bad_i );
PSNR_clean_i = 10*log10( (max(img_good(:))^2) / MSE_clean_i );
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

% 1D S–G smoothing restoration (along rows ONLY to specified columns)
% Note the protected k-space 
function Y = sg_smooth_cols(X, badCols, win, degree, protectHW)

    Y   = X;
    Nx  = size(X,2);
    mid = floor(Nx/2) + 1;                   

    % DC protection mask
    protect = false(1, Nx);
    if protectHW > 0
        protect(max(1,mid-protectHW):min(Nx,mid+protectHW)) = true;
    end
    toSmooth = badCols(~protect(badCols));
    skipped  = badCols(protect(badCols));
    if ~isempty(skipped)
        fprintf('[SG] Protected (not smoothed) cols near DC: %s\n', mat2str(skipped));
    end
    if isempty(toSmooth), return; end

    % Ensure valid SG parameters
    if mod(win,2)==0, win = win+1; end
    degree = min(degree, win-1);

    % Prepare SG operator
    has_sgolayfilt = exist('sgolayfilt','file')==2;
    if ~has_sgolayfilt
        if exist('sgolay','file')~=2
            error('Savitzky–Golay requires sgolayfilt or sgolay (Signal Processing Toolbox).');
        end
        [~, G] = sgolay(degree, win);
        w = G(:,1);
    end

    % Filter each selected column (cast to double for SG, then cast back)
    for c = toSmooth(:).'
        xr = double(real(X(:,c)));
        xi = double(imag(X(:,c)));

        if has_sgolayfilt
            yr = sgolayfilt(xr, degree, win, [], 1);
            yi = sgolayfilt(xi, degree, win, [], 1);
        else
            yr = conv(xr, w, 'same');
            yi = conv(xi, w, 'same');
        end

        yc = complex(yr, yi);
        if ~isa(X, 'double')
            Y(:,c) = cast(yc, 'like', X);
        else
            Y(:,c) = yc;
        end
    end
    
end