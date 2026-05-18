%% Integrated TwIST Reconstruction & Pseudo-Color Generation for HS CS
% Solves: min_x 0.5||y - Ax||_2^2 + tau * TV(x)
% Converts the reconstructed HS cube to sRGB using CIE 1931 2° CMFs.
clear; clc; close all;

%% -------------------- 1) User Settings --------------------
% TwIST settings
% regularizationTau = 0.005;  
regularizationTau = 0.01;  
maxMainIters   = 100;
minMainIters   = 50;
toleranceStop  = 1e-6;
lambdaMinEig   = 1e-4; 
tvInnerIters   = 50;

% Colorimetry settings
wavelength_nm = (400:10:700)';                      % 31 bands
rgb_gt_path   = '..\feathers_ms\feathers_RGB.bmp';  % RGB ground truth image
cmfMatFile    = 'cie1931_2deg_400_700_10nm.mat';    % CIE CMF file
rgbOutPng     = 'pseudoColor_sRGB.png';             % Output render

%% -------------------- 2) Load Data --------------------
fprintf('Loading CS measurement data...\n');
load('HS.mat', 'HS');                            % Ground truth cube: HxWxB
load('measurement_matrix.mat', 'A');             % Measurement mask/patterns: HxWxB
load('compressed_img.mat', 'compressed_img');    % Compressed image: HxW

if isfile('idxMap.mat')
    load('idxMap.mat', 'idxMap'); 
end

hypercube_gt   = single(HS);
compressed_img = single(compressed_img);
measureMatrix  = single(A);
[imgHeight, imgWidth, numBands] = size(hypercube_gt);

figure('Color', 'w'); 
imshow(compressed_img, []); 
title('Measurement (Compressed Image)');

%% -------------------- 3) Operator Normalization C --------------------
op_norm2_map = sum(measureMatrix.^2, 3);           % H x W map
operator_normC = sqrt(max(op_norm2_map(:)));       % Scalar max

if operator_normC == 0
    error('Operator norm C is zero; check measurement matrix.');
end
fprintf('Operator Norm C: %.4f\n', operator_normC);

% Define scaled operators
forwardOp  = @(cube_est) sum(measureMatrix .* cube_est, 3) ./ operator_normC;
adjointOp  = @(meas_in)  (measureMatrix .* meas_in) ./ operator_normC;
measurement_scaled = compressed_img ./ operator_normC;

%% -------------------- 4) TV Psi / Phi --------------------
psiTV = @(cube_in, tau_prox) tvdenoise_cube(cube_in, tau_prox, tvInnerIters);
phiTV = @(cube_in) TV_Phi_cube_tvdenoiseBC(cube_in); 

%% -------------------- 5) Run TwIST --------------------
% Initialization: x0 = A'*y
init_cube = adjointOp(measurement_scaled);
init_cube = psiTV(init_cube, 0.01 * regularizationTau); % Light TV pre-denoise

fprintf('\nStarting TwIST Reconstruction...\n');
[recon_cube, recon_debias, objectiveVals, cpuTimes, debiasStartIter, mseCurve] = TwIST( ...
    measurement_scaled, forwardOp, regularizationTau, ...
    'AT', adjointOp, ...
    'lambda', lambdaMinEig, ...
    'Psi', psiTV, ...
    'Phi', phiTV, ...
    'Initialization', init_cube, ...
    'Monotone', 1, ...
    'Sparse', 0, ...
    'StopCriterion', 1, ...
    'ToleranceA', toleranceStop, ...
    'MaxiterA', maxMainIters, ...
    'MiniterA', minMainIters, ...
    'Verbose', 1);

% Objective curve
figure;
plot(objectiveVals, '-');
xlabel('Iteration'); ylabel('Objective');
title('TwIST objective'); grid on;

% Bound the output to physical reflectance values [0, 1]
recon_cube = max(0, min(1, recon_cube));

% Show one band example side-by-side
bandShow = min(15, numBands); % Change this to view a different band
figure('Color', 'w', 'Position', [100 100 900 400]);
subplot(1,3,1); imshow(hypercube_gt(:,:,bandShow), []); title(sprintf('Ground Truth (Band %d)', bandShow));
subplot(1,3,2); imshow(recon_cube(:,:,bandShow), []);   title(sprintf('Reconstruction (Band %d)', bandShow));
subplot(1,3,3); imshow(abs(recon_cube(:,:,bandShow) - hypercube_gt(:,:,bandShow)), [0 0.2]); colormap(gca, 'hot'); colorbar; title('Absolute Error');

%% -------------------- 6) Quantitative Evaluation --------------------
band_mse = zeros(numBands, 1, 'single');
band_psnr = zeros(numBands, 1, 'single');

for bandIdx = 1:numBands
    diffBand = recon_cube(:,:,bandIdx) - hypercube_gt(:,:,bandIdx);
    mse_val = mean(diffBand(:).^2);
    band_mse(bandIdx) = mse_val;
    band_psnr(bandIdx) = 10 * log10(1 / mse_val);
end

fprintf('\nAverage MSE across all bands: %.6e\n', mean(band_mse));
fprintf('Average PSNR across all bands: %.2f dB\n', mean(band_psnr));

figure('Color', 'w', 'Position', [100 100 900 350]);
subplot(1,2,1); plot(1:numBands, band_mse, 'o-', 'LineWidth', 1.5);
xlabel('Band index'); ylabel('MSE'); title('Reconstruction MSE'); grid on;
subplot(1,2,2); plot(1:numBands, band_psnr, 'r^-', 'LineWidth', 1.5);
xlabel('Band index'); ylabel('PSNR (dB)'); title('Reconstruction PSNR'); grid on;

%% -------------------- 7) Convert HS -> XYZ -> sRGB --------------------
fprintf('\nGenerating Pseudo-Color sRGB Images...\n');

% Load CIE CMFs
cieProjection = load(cmfMatFile);
assert(numBands == numel(wavelength_nm), "HS cube bands (%d) must match wavelength list (%d).", numBands, numel(wavelength_nm));

% Convert Reconstructed Cube to RGB
rgb_recon = hs_to_srgb_CIE1931(recon_cube, cieProjection.wavelength_nm, cieProjection.xbar, cieProjection.ybar, cieProjection.zbar);

% Convert Ground Truth Cube to RGB (for fair comparison)
rgb_hs_gt = hs_to_srgb_CIE1931(hypercube_gt, cieProjection.wavelength_nm, cieProjection.xbar, cieProjection.ybar, cieProjection.zbar);

% Load actual Camera RGB Ground Truth
if isfile(rgb_gt_path)
    rgb_cam_gt = imread(rgb_gt_path);
else
    rgb_cam_gt = zeros(imgHeight, imgWidth, 3, 'uint8');
    warning('Camera RGB Ground Truth not found at %s', rgb_gt_path);
end

% Visualization
figure('Color', 'w', 'Position', [150 250 1200 450]);
subplot(1,3,1); imshow(rgb_cam_gt); 
title("Camera RGB (True GT)");
subplot(1,3,2); imshow(rgb_hs_gt); 
title("Projected GT Cube (CIE1931 2°)");
subplot(1,3,3); imshow(rgb_recon); 
title('Reconstructed Cube (CIE1931 2°)');

%% -------------------- 8) Save Results --------------------
imwrite(rgb_recon, rgbOutPng);
fprintf("Saved RGB image: %s\n", rgbOutPng);

save('HS_recon_TwIST_Integrated.mat', ...
    'recon_cube', 'objectiveVals', 'cpuTimes', 'band_mse', 'band_psnr', ...
    'regularizationTau', 'tvInnerIters', 'lambdaMinEig', 'operator_normC', ...
    'rgb_recon', '-v7.3');
fprintf('Saved reconstruction data: HS_recon_TwIST_Integrated.mat\n');


%% ============================================================
% Local Functions (TwIST & TV)
%% ============================================================

function cube_out = tvdenoise_cube(cube_in, tau_prox, iters)
    cube_in = single(cube_in);
    tau_prox = max(single(tau_prox), single(1e-12));   
    lambda_tv = 1 ./ tau_prox;
    cube_out = single(tvdenoise(double(cube_in), double(lambda_tv), iters));
end

function tvVal = TV_Phi_cube_tvdenoiseBC(cube_in)
    cube_in = single(cube_in);
    [H,W,B] = size(cube_in);
    tvVal = 0;
    for b = 1:B
        u = cube_in(:,:,b);
        u_right = u(:, [2:W W]);
        u_down  = u([2:H H], :);
        dx = u_right - u;
        dy = u_down  - u;
        tvVal = tvVal + sum( sqrt(dx(:).^2 + dy(:).^2) );
    end
end

%% ============================================================
% Local Functions (Colorimetry)
%% ============================================================

function rgb = hs_to_srgb_CIE1931(hypercube, wavelength_nm, xbar, ybar, zbar)
    hypercube = single(hypercube);
    [H,W,B] = size(hypercube);
    wavelength_nm = single(wavelength_nm(:));
    xbar = single(xbar(:));
    ybar = single(ybar(:));
    zbar = single(zbar(:));
    
    assert(numel(wavelength_nm)==B, "wavelength_nm size mismatch.");
    assert(numel(xbar)==B && numel(ybar)==B && numel(zbar)==B, "CMF size mismatch.");
    
    % Integration step (10 nm)
    dLambda = single(mean(diff(wavelength_nm)));
    
    % Vectorize
    spectralMat = reshape(hypercube, H*W, B);  % (HW)xB
    X = (spectralMat * xbar) * dLambda;
    Y = (spectralMat * ybar) * dLambda;
    Z = (spectralMat * zbar) * dLambda;
    xyz = [X Y Z];  % (HW)x3
    
    % Brightness normalization for display
    Yref = prctile(double(Y), 99);
    if Yref > 0
        xyz = xyz / single(Yref);
    end
    
    % XYZ -> linear sRGB (D65)
    M = single([ ...
        3.2406  -1.5372  -0.4986; ...
       -0.9689   1.8758   0.0415; ...
        0.0557  -0.2040   1.0570]);
    rgb_lin = xyz * M.';                 % (HW)x3
    rgb_lin = max(rgb_lin, 0);           % clip negative
    
    % Gamma encode
    rgb_enc = srgb_gamma_encode(rgb_lin);
    
    % Reshape
    rgb = reshape(rgb_enc, H, W, 3);
end

function rgb = srgb_gamma_encode(rgb_lin)
    rgb_lin = single(rgb_lin);
    a = single(0.055);
    thr = single(0.0031308);
    rgb = zeros(size(rgb_lin), 'single');
    
    idx = rgb_lin <= thr;
    rgb(idx)  = 12.92 * rgb_lin(idx);
    rgb(~idx) = (1+a) * (rgb_lin(~idx).^(1/2.4)) - a;
    rgb = min(max(rgb, 0), 1);
end