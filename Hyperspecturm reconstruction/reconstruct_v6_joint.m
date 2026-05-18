%% TwIST reconstruction for hyperspectral CS with joint spatial-Huber spectral regularization
% Objective:
%   min_X 0.5||Y - AX||_2^2 + tau * (TV_spatial(X) + spectralWeight * HuberSpectral(X))
% The main solver is still TwIST, while the proximal operator combines
% per-band spatial TV and per-pixel spectral Huber smoothing to better model
% smooth spectral continuity without the staircase effect of 1D TV.

clear; clc; close all;

%% -------------------- 1) User Settings --------------------
regularizationTau = 0.01;     % keep the same nominal value as the previous script
maxMainIters      = 100;
minMainIters      = 50;
toleranceStop     = 1e-6;
lambdaMinEig      = 1e-4;
spatialTvIters    = 50;
spectralTvIters   = 2;
jointProxSweeps   = 1;
% spectralTVWeight  = 0.002;
spectralTVWeight  = 1e-5;
spectralHuberDelta = 0.003;
outerAltIters      = 1;
innerTwISTIters    = 50;
useSpectralInTwIST = false;
ergasRatio         = 1;

% Colorimetry settings
wavelength_nm = (400:10:700)';                      % 31 bands
rgb_gt_path   = '..\feathers_ms\feathers_RGB.bmp'; % RGB ground truth image
cmfMatFile    = 'cie1931_2deg_400_700_10nm.mat';    % CIE CMF file
rgbOutPng     = 'pseudoColor_sRGB_v6_joint.png';    % Output render

%% -------------------- 2) Load Data --------------------
fprintf('Loading CS measurement data...\n');
load('HS.mat', 'HS');
load('measurement_matrix.mat', 'A');
load('compressed_img.mat', 'compressed_img');

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

%% -------------------- 3) Operator Normalization --------------------
op_norm2_map   = sum(measureMatrix.^2, 3);
operator_normC = sqrt(max(op_norm2_map(:)));

if operator_normC == 0
    error('Operator norm C is zero; check measurement matrix.');
end
fprintf('Operator Norm C: %.4f\n', operator_normC);

forwardOp = @(cube_est) sum(measureMatrix .* cube_est, 3) ./ operator_normC;
adjointOp = @(meas_in)  (measureMatrix .* meas_in) ./ operator_normC;

measurement_scaled = compressed_img ./ operator_normC;
tauTwIST = regularizationTau;

%% -------------------- 4) Joint Spatial-Spectral Regularizer --------------------
useSpectral = useSpectralInTwIST && spectralTVWeight > 0;
if useSpectral
    psiJoint = @(cube_in, tau_prox) joint_huber_prox( ...
        cube_in, tau_prox, spatialTvIters, spectralTvIters, jointProxSweeps, spectralTVWeight, spectralHuberDelta);
    phiJoint = @(cube_in) joint_huber_norm(cube_in, spectralTVWeight, spectralHuberDelta);
else
    psiJoint = @(cube_in, tau_prox) tvdenoise_cube(cube_in, tau_prox, spatialTvIters);
    phiJoint = @(cube_in) TV_Phi_cube_tvdenoiseBC(cube_in);
end

%% -------------------- 5) Run TwIST --------------------
init_cube = adjointOp(measurement_scaled);
init_cube = psiJoint(init_cube, 0.01 * regularizationTau);

fprintf('\nStarting TwIST Reconstruction with joint spatial-Huber spectral regularization...\n');
recon_cube = init_cube;
maxInnerIters = min(innerTwISTIters, maxMainIters);
minInnerIters = min(minMainIters, maxInnerIters);
objectiveVals = [];
cpuTimes = [];
debiasStartIter = [];
mseCurve = [];
for outerIdx = 1:outerAltIters
    fprintf('Outer iteration %d/%d...\n', outerIdx, outerAltIters);
    [recon_cube, recon_debias, objectiveVals, cpuTimes, debiasStartIter, mseCurve] = TwIST( ...
        measurement_scaled, forwardOp, tauTwIST, ...
        'AT', adjointOp, ...
        'lambda', lambdaMinEig, ...
        'Psi', psiJoint, ...
        'Phi', phiJoint, ...
        'Initialization', recon_cube, ...
        'Monotone', 1, ...
        'Sparse', 0, ...
        'StopCriterion', 1, ...
        'ToleranceA', toleranceStop, ...
        'MaxiterA', maxInnerIters, ...
        'MiniterA', minInnerIters, ...
        'Verbose', 1);

    recon_cube = max(0, min(1, recon_cube));
    if spectralTVWeight > 0
        fprintf('Applying spectral Huber denoise...\n');
        lambda_spectral = regularizationTau * spectralTVWeight;
        recon_cube = spectral_huber_denoise_cube(recon_cube, lambda_spectral, spectralHuberDelta, spectralTvIters);
        recon_cube = max(0, min(1, recon_cube));
    end
end

figure('Color', 'w');
plot(objectiveVals, '-', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective');
title('TwIST objective'); grid on;

bandShow = min(15, numBands);
figure('Color', 'w', 'Position', [100 100 1100 420]);
subplot(1,4,1); imshow(hypercube_gt(:,:,bandShow), []); title(sprintf('GT Band %d', bandShow));
subplot(1,4,2); imshow(recon_cube(:,:,bandShow), []); title(sprintf('Recon Band %d', bandShow));
subplot(1,4,3); imshow(abs(recon_cube(:,:,bandShow) - hypercube_gt(:,:,bandShow)), [0 0.2]); colormap(gca, 'hot'); colorbar; title('Abs Error');
subplot(1,4,4); plotSpectrumExample(hypercube_gt, recon_cube, bandShow); title('Example spectrum');

%% -------------------- 6) Quantitative Evaluation --------------------
[band_mse, band_psnr] = compute_band_metrics(recon_cube, hypercube_gt);
[sam_map, sam_mean_deg] = compute_sam_map(recon_cube, hypercube_gt);
[ergasValue, band_rmse, band_ssim] = compute_ergas_ssim(recon_cube, hypercube_gt, ergasRatio);
[spectral_rmse_map, spectral_rmse_mean] = compute_spectral_curve_error(recon_cube, hypercube_gt);

fprintf('\nAverage MSE across all bands: %.6e\n', mean(band_mse));
fprintf('Average PSNR across all bands: %.2f dB\n', mean(band_psnr));
fprintf('Mean SAM: %.4f deg\n', sam_mean_deg);
fprintf('ERGAS: %.4f\n', ergasValue);
band_ssim_valid = band_ssim(~isnan(band_ssim));
if isempty(band_ssim_valid)
    meanBandSSIM = NaN;
else
    meanBandSSIM = mean(band_ssim_valid);
end
fprintf('Mean SSIM across bands: %.4f\n', meanBandSSIM);
fprintf('Mean per-pixel spectral RMSE: %.6e\n', spectral_rmse_mean);

figure('Color', 'w', 'Position', [100 100 1100 420]);
subplot(1,3,1); plot(1:numBands, band_mse, 'o-', 'LineWidth', 1.3); xlabel('Band'); ylabel('MSE'); title('Band MSE'); grid on;
subplot(1,3,2); plot(1:numBands, band_psnr, 'r^-', 'LineWidth', 1.3); xlabel('Band'); ylabel('PSNR (dB)'); title('Band PSNR'); grid on;
subplot(1,3,3); plot(1:numBands, band_ssim, 'ks-', 'LineWidth', 1.3); xlabel('Band'); ylabel('SSIM'); title('Band SSIM'); grid on;

figure('Color', 'w', 'Position', [120 120 1100 430]);
subplot(1,2,1); imagesc(sam_map); axis image off; colorbar; title(sprintf('SAM map (mean %.3f deg)', sam_mean_deg));
subplot(1,2,2); imagesc(spectral_rmse_map); axis image off; colorbar; title(sprintf('Per-pixel spectral RMSE (mean %.4e)', spectral_rmse_mean));

%% -------------------- 7) Convert HS -> XYZ -> sRGB --------------------
fprintf('\nGenerating pseudo-color sRGB images...\n');

cieProjection = load(cmfMatFile);
assert(numBands == numel(wavelength_nm), 'HS cube bands (%d) must match wavelength list (%d).', numBands, numel(wavelength_nm));

rgb_recon = hs_to_srgb_CIE1931(recon_cube, cieProjection.wavelength_nm, cieProjection.xbar, cieProjection.ybar, cieProjection.zbar);
rgb_hs_gt = hs_to_srgb_CIE1931(hypercube_gt, cieProjection.wavelength_nm, cieProjection.xbar, cieProjection.ybar, cieProjection.zbar);

if isfile(rgb_gt_path)
    rgb_cam_gt = imread(rgb_gt_path);
else
    rgb_cam_gt = zeros(imgHeight, imgWidth, 3, 'uint8');
    warning('Camera RGB ground truth not found at %s', rgb_gt_path);
end

figure('Color', 'w', 'Position', [150 250 1200 450]);
subplot(1,3,1); imshow(rgb_cam_gt); title('Camera RGB (True GT)');
subplot(1,3,2); imshow(rgb_hs_gt); title('Projected GT Cube (CIE1931 2°)');
subplot(1,3,3); imshow(rgb_recon); title('Reconstructed Cube (CIE1931 2°)');

%% -------------------- 8) Save Results --------------------
imwrite(rgb_recon, rgbOutPng);
fprintf('Saved RGB image: %s\n', rgbOutPng);

save('HS_recon_TwIST_JointHuber.mat', ...
    'recon_cube', 'objectiveVals', 'cpuTimes', 'band_mse', 'band_psnr', ...
    'sam_map', 'sam_mean_deg', 'ergasValue', 'band_rmse', 'band_ssim', ...
    'spectral_rmse_map', 'spectral_rmse_mean', 'regularizationTau', ...
    'spatialTvIters', 'spectralTvIters', 'jointProxSweeps', 'spectralTVWeight', ...
    'spectralHuberDelta', 'lambdaMinEig', 'operator_normC', 'tauTwIST', 'rgb_recon', '-v7.3');
fprintf('Saved reconstruction data: HS_recon_TwIST_JointHuber.mat\n');

%% ============================================================
% Local Functions: TwIST + Joint Regularization
%% ============================================================

function cube_out = joint_huber_prox(cube_in, tau_prox, spatialIters, spectralIters, jointSweeps, spectralWeight, huberDelta)
    cube_out = single(cube_in);
    tau_prox = max(single(tau_prox), single(1e-12));
    spatialLambda = tau_prox;
    spectralLambda = tau_prox * single(max(spectralWeight, 0));
    huberDelta = max(single(huberDelta), single(1e-6));

    for sweepIdx = 1:jointSweeps
        [~, ~, B] = size(cube_out);
        cube_out = tvdenoise_cube(cube_out, spatialLambda, spatialIters);
        if spectralLambda > 0 && spectralIters > 0
            cube_out = spectral_huber_denoise_cube(cube_out, spectralLambda, huberDelta, spectralIters);
        end
    end
end

function cube_out = spectral_huber_denoise_cube(cube_in, lambda_spectral, huberDelta, nIter)
    cube_in = double(cube_in);
    lambda_spectral = max(double(lambda_spectral), 0);
    huberDelta = max(double(huberDelta), 1e-12);

    if nIter <= 0 || lambda_spectral == 0
        cube_out = single(cube_in);
        return;
    end

    [H, W, B] = size(cube_in);
    numPixels = H * W;
    signalMat = reshape(cube_in, numPixels, B);

    if B < 2
        cube_out = single(cube_in);
        return;
    end

    rho = 1;
    e = ones(B, 1);
    Dmat = spdiags([-e e], [0 1], B - 1, B);
    systemMat = speye(B) + rho * (Dmat' * Dmat);
    dualVar = zeros(numPixels, B - 1);
    bregVar = zeros(numPixels, B - 1);

    for iterIdx = 1:nIter
        rhs = signalMat + rho * ((dualVar - bregVar) * Dmat);
        signalMat = rhs / systemMat;

        diffMat = signalMat * Dmat';
        dualVar = huber_prox_scalar(diffMat + bregVar, lambda_spectral / rho, huberDelta);
        bregVar = bregVar + diffMat - dualVar;
    end

    cube_out = single(reshape(signalMat, H, W, B));
end

function z = huber_prox_scalar(v, alpha, delta)
    absV = abs(v);
    z = zeros(size(v));

    scaleMask = absV <= (delta + alpha);
    z(scaleMask) = (delta ./ (delta + alpha)) .* v(scaleMask);

    softMask = ~scaleMask;
    z(softMask) = sign(v(softMask)) .* max(absV(softMask) - alpha, 0);
end

function phiVal = joint_huber_norm(cube_in, spectralWeight, huberDelta)
    cube_in = single(cube_in);
    [~, ~, B] = size(cube_in);
    phiVal = 0;
    phiVal = TV_Phi_cube_tvdenoiseBC(cube_in);
    if B > 1
        spectralDiff = diff(cube_in, 1, 3);
        phiVal = phiVal + single(max(spectralWeight, 0)) * sum(huber_penalty(spectralDiff(:), single(huberDelta)));
    end
end

function y = huber_penalty(x, delta)
    ax = abs(x);
    y = zeros(size(x), 'like', x);
    quadMask = ax <= delta;
    y(quadMask) = 0.5 * (x(quadMask).^2) ./ delta;
    y(~quadMask) = ax(~quadMask) - 0.5 * delta;
end

function [band_mse, band_psnr] = compute_band_metrics(recon_cube, gt_cube)
    recon_cube = single(recon_cube);
    gt_cube = single(gt_cube);
    [~, ~, B] = size(gt_cube);

    band_mse = zeros(B, 1, 'single');
    band_psnr = zeros(B, 1, 'single');

    for bandIdx = 1:B
        diffBand = recon_cube(:,:,bandIdx) - gt_cube(:,:,bandIdx);
        mseVal = mean(diffBand(:).^2);
        band_mse(bandIdx) = mseVal;
        band_psnr(bandIdx) = safe_psnr(mseVal);
    end
end

function psnrVal = safe_psnr(mseVal)
    if mseVal <= 0
        psnrVal = inf;
    else
        psnrVal = 10 * log10(1 / mseVal);
    end
end

function [sam_map, sam_mean_deg] = compute_sam_map(recon_cube, gt_cube)
    reconMat = double(reshape(recon_cube, [], size(recon_cube, 3)));
    gtMat    = double(reshape(gt_cube,    [], size(gt_cube, 3)));

    dotProd = sum(reconMat .* gtMat, 2);
    reconNorm = sqrt(sum(reconMat.^2, 2));
    gtNorm    = sqrt(sum(gtMat.^2, 2));
    denom = reconNorm .* gtNorm + eps;

    cosTheta = dotProd ./ denom;
    cosTheta = max(-1, min(1, cosTheta));
    samRad = acos(cosTheta);
    samDeg = rad2deg(samRad);

    sam_map = reshape(single(samDeg), size(recon_cube, 1), size(recon_cube, 2));
    samValid = samDeg(~isnan(samDeg));
    if isempty(samValid)
        sam_mean_deg = NaN;
    else
        sam_mean_deg = mean(samValid);
    end
end

function [ergasValue, band_rmse, band_ssim] = compute_ergas_ssim(recon_cube, gt_cube, ratio)
    recon_cube = single(recon_cube);
    gt_cube = single(gt_cube);
    [~, ~, B] = size(gt_cube);

    band_rmse = zeros(B, 1, 'single');
    band_mean = zeros(B, 1, 'single');
    band_ssim = nan(B, 1, 'single');

    useSSIM = (exist('ssim', 'file') == 2);

    for bandIdx = 1:B
        gtBand = gt_cube(:,:,bandIdx);
        recBand = recon_cube(:,:,bandIdx);
        errBand = recBand - gtBand;

        band_rmse(bandIdx) = sqrt(mean(errBand(:).^2));
        band_mean(bandIdx) = mean(gtBand(:));

        if useSSIM
            try
                band_ssim(bandIdx) = single(ssim(recBand, gtBand, 'DynamicRange', 1));
            catch
                band_ssim(bandIdx) = nan;
            end
        end
    end

    normalizedRmse = band_rmse ./ (band_mean + eps('single'));
    ergasValue = 100 / ratio * sqrt(mean(normalizedRmse.^2));
end

function [spectral_rmse_map, spectral_rmse_mean] = compute_spectral_curve_error(recon_cube, gt_cube)
    reconMat = double(reshape(recon_cube, [], size(recon_cube, 3)));
    gtMat    = double(reshape(gt_cube,    [], size(gt_cube, 3)));

    spectral_rmse = sqrt(mean((reconMat - gtMat).^2, 2));
    spectral_rmse_map = reshape(single(spectral_rmse), size(recon_cube, 1), size(recon_cube, 2));
    spectral_rmse_mean = mean(spectral_rmse);
end

function plotSpectrumExample(gt_cube, recon_cube, bandShow)
    [H, W, B] = size(gt_cube);
    row = round(H / 2);
    col = round(W / 2);
    gtSpec = squeeze(gt_cube(row, col, :));
    recSpec = squeeze(recon_cube(row, col, :));
    plot(1:B, gtSpec, 'k-o', 'LineWidth', 1.1, 'MarkerSize', 4); hold on;
    plot(1:B, recSpec, 'r-s', 'LineWidth', 1.1, 'MarkerSize', 4);
    grid on;
    xlabel('Band'); ylabel('Intensity');
    legend('GT', 'Recon', 'Location', 'best');
    title(sprintf('Central pixel spectrum (band %d)', bandShow));
end

%% ============================================================
% Local Functions: Colorimetry
%% ============================================================

function rgb = hs_to_srgb_CIE1931(hypercube, wavelength_nm, xbar, ybar, zbar)
    hypercube = single(hypercube);
    [H, W, B] = size(hypercube);
    wavelength_nm = single(wavelength_nm(:));
    xbar = single(xbar(:));
    ybar = single(ybar(:));
    zbar = single(zbar(:));

    assert(numel(wavelength_nm) == B, 'wavelength_nm size mismatch.');
    assert(numel(xbar) == B && numel(ybar) == B && numel(zbar) == B, 'CMF size mismatch.');

    dLambda = single(mean(diff(wavelength_nm)));
    spectralMat = reshape(hypercube, H * W, B);
    X = (spectralMat * xbar) * dLambda;
    Y = (spectralMat * ybar) * dLambda;
    Z = (spectralMat * zbar) * dLambda;
    xyz = [X Y Z];

    Yref = prctile(double(Y), 99);
    if Yref > 0
        xyz = xyz / single(Yref);
    end

    M = single([ ...
        3.2406  -1.5372  -0.4986; ...
       -0.9689   1.8758   0.0415; ...
        0.0557  -0.2040   1.0570]);
    rgb_lin = xyz * M.';
    rgb_lin = max(rgb_lin, 0);
    rgb_enc = srgb_gamma_encode(rgb_lin);
    rgb = reshape(rgb_enc, H, W, 3);
end

function rgb = srgb_gamma_encode(rgb_lin)
    rgb_lin = single(rgb_lin);
    a = single(0.055);
    thr = single(0.0031308);
    rgb = zeros(size(rgb_lin), 'single');

    idx = rgb_lin <= thr;
    rgb(idx)  = 12.92 * rgb_lin(idx);
    rgb(~idx) = (1 + a) * (rgb_lin(~idx).^(1/2.4)) - a;
    rgb = min(max(rgb, 0), 1);
end

function cube_out = tvdenoise_cube(cube_in, tau_prox, iters)
    cube_in = single(cube_in);
    tau_prox = max(single(tau_prox), single(1e-12));
    lambda_tv = 1 ./ tau_prox;
    cube_out = single(tvdenoise(double(cube_in), double(lambda_tv), iters));
end

function tvVal = TV_Phi_cube_tvdenoiseBC(cube_in)
    cube_in = single(cube_in);
    [H, W, B] = size(cube_in);
    tvVal = 0;
    for b = 1:B
        u = cube_in(:,:,b);
        u_right = u(:, [2:W W]);
        u_down  = u([2:H H], :);
        dx = u_right - u;
        dy = u_down  - u;
        tvVal = tvVal + sum(sqrt(dx(:).^2 + dy(:).^2));
    end
end