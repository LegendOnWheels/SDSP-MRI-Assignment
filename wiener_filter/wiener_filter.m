%% Smoothing methods
clc; clear all; close all;
baseDir = 'C:\Users\Nathan\OneDrive - Delft University of Technology\Documents\TU Delft\MASTER 1\Q1\SDSP\Assignment\MRI_datasets';  % <-- your base folder
matFiles = dir(fullfile(baseDir, '**', '*.mat'));  % find all .mat files recursively

for k = 1:numel(matFiles)
   
    filePath = fullfile(matFiles(k).folder, matFiles(k).name);
    load(filePath);  % loads the variable directly into workspace
end

%% Loading variables
x_bad1 = slice4_channel1_badData;               x_good1 = slice4_channel1_goodData;
x_bad2 = slice4_channel2_badData;               x_good2 = slice4_channel2_goodData;
x_bad3 = slice4_channel3_badData;               x_good3 = slice4_channel3_goodData;


%% Step 1: Visualize in image domain
% IFFT of k-space data
Data_img_bad(:,:,1) = ifftshift(ifft2(x_bad1),1);
% channel 2
Data_img_bad(:,:,2) = ifftshift(ifft2(x_bad2),1);
%channel 3
Data_img_bad(:,:,3) = ifftshift(ifft2(x_bad3),1);

%channel 1
Data_img_good(:,:,1) = ifftshift(ifft2(x_good1),1);
% channel 2
Data_img_good(:,:,2) = ifftshift(ifft2(x_good2),1);
%channel 3
Data_img_good(:,:,3) = ifftshift(ifft2(x_good3),1);

%% Step 2: Combine channels
% clear compensation, preparation, based on fourier transformed blinked 
% k-space data (Data_raw)
clear_comp = linspace(10,0.1,size(Data_img_bad,2)).^2; 
clear_matrix = repmat(clear_comp,[size(Data_img_bad,1) 1]);

% combine 3 channels sum of squares and add clear compensation
eye_raw_bad  = sqrt( abs(squeeze(Data_img_bad(:,:,1))).^2 + ...
           abs(squeeze(Data_img_bad(:,:,2))).^2 + ...
           abs(squeeze(Data_img_bad(:,:,3))).^2).* clear_matrix; 
eye_raw_good  = sqrt( abs(squeeze(Data_img_good(:,:,1))).^2 + ...
           abs(squeeze(Data_img_good(:,:,2))).^2 + ...
           abs(squeeze(Data_img_good(:,:,3))).^2).* clear_matrix; 

eye_raw_good1  = sqrt( abs(squeeze(Data_img_good(:,:,1))).^2).* clear_matrix;
eye_raw_good2  = sqrt( abs(squeeze(Data_img_good(:,:,2))).^2).* clear_matrix; 
eye_raw_good3  = sqrt( abs(squeeze(Data_img_good(:,:,3))).^2).* clear_matrix; 
% crop images because we are only interested in eye. Make it square 
% 128 x 128
crop_x = [128 + 60 : 348 - 33]; % crop coordinates
eye_raw_bad_all = eye_raw_bad(crop_x, :);
eye_raw_good_all = eye_raw_good(crop_x, :);
eye_raw_good1 = eye_raw_good1(crop_x, :);
eye_raw_good2 = eye_raw_good2(crop_x, :);
eye_raw_good3 = eye_raw_good3(crop_x, :);

% Visualize the images. 

% image
eye_visualize_bad = reshape(squeeze(eye_raw_bad_all(:,:)),[128 128]); 
eye_visualize_good = reshape(squeeze(eye_raw_good_all(:,:)),[128 128]);
eye_visualize_good1 = reshape(squeeze(eye_raw_good1(:,:)),[128 128]);
eye_visualize_good2 = reshape(squeeze(eye_raw_good2(:,:)),[128 128]);
eye_visualize_good3 = reshape(squeeze(eye_raw_good3(:,:)),[128 128]);

% For better visualization and contrast of the eye images, histogram based
% compensation will be done 

std_within = 0.995; 
% set maximum intensity to contain 99.5 % of intensity values per image
[aa, val] = hist(eye_visualize_bad(:),linspace(0,max(...
                                    eye_visualize_bad(:)),1000));
    thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));

% set threshold value to 65536
eye_visualize_bad = uint16(eye_visualize_bad * 65536 / thresh); 

[aa, val] = hist(eye_visualize_good(:),linspace(0,max(...
                                    eye_visualize_good(:)),1000));
    thresh = val(find(cumsum(aa)/sum(aa) > std_within,1,'first'));

% set threshold value to 65536
eye_visualize_good = uint16(eye_visualize_good * 65536 / thresh);
eye_visualize_good1 = uint16(eye_visualize_good1 * 65536 / thresh);
eye_visualize_good2 = uint16(eye_visualize_good2 * 65536 / thresh);
eye_visualize_good3 = uint16(eye_visualize_good3 * 65536 / thresh);

%% Step 3: Mask the image for applying the Wiener filter
I_raw = abs(eye_raw_good);  % magnitude image
I_raw_bad = abs(eye_raw_bad);

% Define how many pixels to extend the mask upward and downward
extend_px = 60;

% Compute safe extended mask limits
top_limit = max(1, min(crop_x) - extend_px);
bottom_limit = min(size(I_raw,1), max(crop_x) + extend_px+20);

% Create mask for entire image
mask = true(size(I_raw));
mask(top_limit:bottom_limit, :) = false;

% Extract noise pixels
noise_pixels = I_raw(mask);

% Optional: remove any outliers
noise_pixels = noise_pixels(noise_pixels < prctile(noise_pixels, 99));

% Compute noise statistics
noise_variance = var(noise_pixels(:));
noise_std = sqrt(noise_variance);

fprintf('Estimated noise variance (outside crop_x): %.6e\n', noise_variance);
fprintf('Estimated noise std: %.6e\n', noise_std);


I_wiener = wiener2(I_raw, [3 3], noise_variance);
I_wiener_bad = wiener2(I_raw_bad, [3 3], noise_variance);


% ---- Visualization ----
% Use same contrast scaling (99.5%) for fair comparison
I_raw_crop = I_raw(crop_x, :);
I_raw_bad_crop = I_raw_bad(crop_x, :);
I_wiener_bad_crop = I_wiener_bad(crop_x, :);
I_wiener_crop = I_wiener(crop_x, :);

% ---- Visualization ----
% Use same contrast scaling (99.5%) for fair comparison
std_within = 0.995;

% Compute contrast threshold from cropped raw image histogram
[aa, val] = hist(I_raw_crop(:), linspace(0, max(I_raw_crop(:)), 1000));
thresh = val(find(cumsum(aa)/sum(aa) > std_within, 1, 'first'));

% Scale both images to same visualization range (0–65536)
I_raw_vis = uint16(I_raw_crop * 65536 / thresh);
I_wiener_vis = uint16(I_wiener_crop * 65536 / thresh);
I_raw_bad_vis = uint16(I_raw_bad_crop * 65536 / thresh); 
I_wiener_bad_vis = uint16(I_wiener_bad_crop * 65536 / thresh);

% ---- Show before/after side by side ----
figure;
subplot(2,2,1);
imshow(I_raw_vis, []);
title('Before Wiener Filter (raw reconstruction)');

subplot(2,2,2);
imshow(I_wiener_vis, []);
title('After Wiener Filter (using estimated noise variance)');

subplot(2,2,3);
imshow(I_raw_bad_vis, []);
title('Bad data before Wiener Filter (raw reconstruction)');

subplot(2,2,4);
imshow(I_wiener_bad_vis, []);
title('Bad data after Wiener Filter (using good estimated noise variance)');

noise_pixels_after = I_wiener(mask);

% Optionally remove outliers
noise_pixels_after = noise_pixels_after(noise_pixels_after < prctile(noise_pixels_after, 99));

% Compute post-filter noise variance and std
noise_variance_after = var(noise_pixels_after(:));
noise_std_after = sqrt(noise_variance_after);

fprintf('\n--- Noise reduction results ---\n');
fprintf('Before Wiener filtering: variance = %.6e, std = %.6e\n', noise_variance, noise_std);
fprintf('After Wiener filtering:  variance = %.6e, std = %.6e\n', noise_variance_after, noise_std_after);
fprintf('Noise power reduction ratio: %.2f %%\n', ...
        100 * (1 - noise_variance_after / noise_variance));
