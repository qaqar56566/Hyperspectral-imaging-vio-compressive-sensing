%% TwIST reconstruction for hyperspectral CS (delta FP filters) + spatial TV (circular boundary)
% Requirements in MATLAB path:
%   - TwIST.m  (the TwIST code you pasted)
%   - conv2c.m + wraparound.m
%   - tv_norm2d_conv2c.m
%   - tv_denoise_chambolle_conv2c.m
%
% Saved files (you said you already saved):
%   - HS.mat                 variable: HS          (H x W x B)
%   - measurement_matrix.mat variable: A           (H x W x B)  % mask cube (0/1 or logical)
%   - idxMap.mat             variable: idxMap      (H x W)
%   - compressed_img         (preferably .mat). If only PNG exists, code tries to read it.clear; clc; close all;

clear; clc; close all;
%% -------------------- Load data --------------------
load('HS.mat','HS');                            % Ground truth cube: HxWxB
load('measurement_matrix.mat','A');             % Measurement mask/patterns: HxWxB
load('compressed_img.mat','compressed_img');    % compressed image: HxW
% idxMap 
if isfile('idxMap.mat')
    load('idxMap.mat','idxMap'); % #ok<NASGU>
end

hypercube_gt = single(HS);
compressed_img = single(compressed_img);
measureMatrix = single(A);

[imgHeight, imgWidth, numBands] = size(hypercube_gt);

figure; imshow(compressed_img, []); title('Measurement (raw)');


%% -------------------- Global max normalization  --------------------
% Normalize measurement to [0,1]-scale (global max abs)
compImg_maxAbs = max(abs(compressed_img(:)));
if compImg_maxAbs == 0
    error('Measurement is all zeros; cannot normalize.');
end
compImg_norm = compressed_img ./ compImg_maxAbs;

% Normalize mask/pattern cube to [0,1]-scale (global max abs)
mask_maxAbs = max(abs(measureMatrix(:)));
if mask_maxAbs == 0
    error('Mask cube is all zeros; cannot normalize.');
end
msureMatrix_norm = measureMatrix ./ mask_maxAbs;

figure; imshow(compImg_norm, []); title('Measurement (global max normalized)');

%% -------------------- Operator normalization C (recommended) --------------------
% For operator y = sum_b mask_norm(:,:,b) .* x(:,:,b),
% ||A||_2^2 = max_{m,n} sum_b mask_norm(m,n,b)^2

op_norm2_map = sum(msureMatrix_norm.^2, 3);               % HxW
operator_normC = sqrt(max(op_norm2_map(:)));       % scalar C

if operator_normC == 0
    error('Operator norm C is zero; check mask_norm.');
end

% Define scaled operators (use clear names)
forwardOp  = @(cube_est) HS_forward(cube_est, msureMatrix_norm) ./ operator_normC;
adjointOp  = @(meas_in)  HS_adjoint(meas_in,  msureMatrix_norm) ./ operator_normC;

measurement_scaled = compImg_norm ./ operator_normC;

%% -------------------- TV Psi / Phi (circular boundary) --------------------

tvInnerIters = 30;   % recommended >= 80 for stronger prox accuracy
psiTV = @(cube_in, tau_prox) tvdenoise_cube(cube_in, tau_prox, tvInnerIters);
phiTV = @(cube_in) TV_Phi_cube_tvdenoiseBC(cube_in);  % 与 tvdenoise 的边界一致（建议）


%% -------------------- TwIST parameters --------------------
% IMPORTANT: tau must match the scaled measurement/operator.
% Since we scaled measurement by meas_maxAbs and operator by operator_normC,
% tau in this coordinate is "tau_scaled". Start from a moderate value and sweep later.

% regularizationTau = 0.01;
regularizationTau = 0.003;

maxMainIters   = 200;
minMainIters   = 100;
toleranceStop  = 1e-8;

% TwIST internal parameter (lam1): use small value for severly ill-conditioned problems
lambdaMinEig = 1e-4;

% Initialization: x0 = A'*y, optionally with light TV
init_cube = adjointOp(measurement_scaled);
init_cube = psiTV(init_cube, 0.01 * regularizationTau);   % light TV pre-denoise


%% -------------------- Run TwIST --------------------


[recon_cube, recon_debias, objectiveVals, cpuTimes, debiasStartIter, mseCurve] = TwIST( ...
    measurement_scaled, forwardOp, regularizationTau, ...
    'AT', adjointOp, ...
    'lambda', lambdaMinEig, ...
    'Psi', psiTV, ...
    'Phi', phiTV, ...
    'Initialization', init_cube, ...
    'Monotone', 1, ...
    'Sparse', 0, ...
    'StopCriterion', 2, ...
    'ToleranceA', toleranceStop, ...
    'MaxiterA', maxMainIters, ...
    'MiniterA', minMainIters, ...
    'Verbose', 1);

%% -------------------- Per-band MSE evaluation --------------------
band_mse = zeros(numBands,1,'single');
for bandIdx = 1:numBands
    diffBand = recon_cube(:,:,bandIdx) - hypercube_gt(:,:,bandIdx);
    band_mse(bandIdx) = mean(diffBand(:).^2);
end
mean_mse = mean(band_mse);

fprintf('\nPer-band MSE (1..%d):\n', numBands);
disp(band_mse.');

fprintf('Average MSE across bands: %.6e\n', mean_mse);

figure;
plot(1:numBands, band_mse, 'o-');
xlabel('Band index'); ylabel('MSE');
title('Reconstruction MSE per band'); grid on;

% Show one band example
bandShow = min(10, numBands);
figure;
subplot(1,2,1); imshow(hypercube_gt(:,:,bandShow), []); title(sprintf('GT band %d', bandShow));
subplot(1,2,2); imshow(recon_cube(:,:,bandShow), []);   title(sprintf('TwIST+TV recon band %d', bandShow));

% Objective curve
figure;
plot(objectiveVals, '-');
xlabel('Iteration'); ylabel('Objective');
title('TwIST objective'); grid on;

%% -------------------- Save reconstruction results --------------------
save('HS_recon_TwIST_TV_norm.mat', ...
    'recon_cube', 'objectiveVals', 'cpuTimes', 'band_mse', 'mean_mse', ...
    'regularizationTau', 'tvInnerIters', 'lambdaMinEig', ...
    'operator_normC', 'compImg_maxAbs', 'mask_maxAbs', '-v7.3');

fprintf('Saved: HS_recon_TwIST_TV_norm.mat\n');

%% ==================== Local functions ====================

function meas_out = HS_forward(cube_in, mask_cube)
% Forward model (delta-mask): meas = sum_b mask(:,:,b).*cube(:,:,b)
    cube_in = single(cube_in);
    meas_out = sum(mask_cube .* cube_in, 3);
end

function cube_out = HS_adjoint(meas_in, mask_cube)
% Adjoint model: cube(:,:,b) = mask(:,:,b).*meas
    meas_in = single(meas_in);
    [imgHeight, imgWidth, numBands] = size(mask_cube);
    cube_out = zeros(imgHeight, imgWidth, numBands, 'single');
    for bandIdx = 1:numBands
        cube_out(:,:,bandIdx) = mask_cube(:,:,bandIdx) .* meas_in;
    end
end


function cube_out = tvdenoise_cube(cube_in, tau_prox, iters)
% TwIST prox: argmin_u 0.5||u-cube_in||^2 + tau_prox * TV(u)
% tvdenoise solves: min_u TV(u) + (lambda/2)||cube_in-u||^2
% mapping: lambda = 1/tau_prox

    cube_in = single(cube_in);

    tau_prox = max(single(tau_prox), single(1e-12));   % avoid divide-by-zero
    lambda_tv = 1 ./ tau_prox;

    % tvdenoise 可处理 2D 或 3D（3D按“多通道”vectorial TV）
    cube_out = single(tvdenoise(double(cube_in), double(lambda_tv), iters));
end

function tvVal = TV_Phi_cube_tvdenoiseBC(cube_in)
% Isotropic TV consistent with tvdenoise's finite differences
% (vectorial TV across 3rd dim is also acceptable, but here we sum per-band scalar TV)

    cube_in = single(cube_in);
    [H,W,B] = size(cube_in);

    tvVal = 0;
    for b = 1:B
        u = cube_in(:,:,b);

        % forward differences consistent with tvdenoise indexing
        u_right = u(:, [2:W W]);
        u_down  = u([2:H H], :);

        dx = u_right - u;
        dy = u_down  - u;

        tvVal = tvVal + sum( sqrt(dx(:).^2 + dy(:).^2) );
    end
end
