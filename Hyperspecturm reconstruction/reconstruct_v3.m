%% TwIST reconstruction for hyperspectral CS + spatial TV
% Requirements in MATLAB path:
%   - TwIST.m
%   - tvdenoise.m (e.g., tv_denoise_chambolle_conv2c.m)
clear; clc; close all;

%% -------------------- Load data --------------------
fprintf('Loading data...\n');
load('HS.mat', 'HS');                            % Ground truth cube: HxWxB
load('measurement_matrix.mat', 'A');             % Measurement mask/patterns: HxWxB
load('compressed_img.mat', 'compressed_img');    % Compressed image: HxW

if isfile('idxMap.mat')
    load('idxMap.mat', 'idxMap'); 
end

hypercube_gt = single(HS);
compressed_img = single(compressed_img);
measureMatrix = single(A);

[imgHeight, imgWidth, numBands] = size(hypercube_gt);

figure('Color', 'w'); 
imshow(compressed_img, []); 
title('Measurement (Compressed Image)');

%% -------------------- Operator Normalization C --------------------
% To ensure TwIST converges properly, the linear operator must be scaled 
% such that its maximum eigenvalue (norm) is <= 1.
% For operator y = sum_b A(:,:,b) .* x(:,:,b), 
% ||A||_2 = max_{m,n} sqrt( sum_b A(m,n,b)^2 )

op_norm2_map = sum(measureMatrix.^2, 3);           % H x W map
operator_normC = sqrt(max(op_norm2_map(:)));       % Scalar max

if operator_normC == 0
    error('Operator norm C is zero; check measurement matrix.');
end

fprintf('Operator Norm C: %.4f\n', operator_normC);

% Define scaled operators (Using implicit expansion for fast adjoint)
forwardOp  = @(cube_est) sum(measureMatrix .* cube_est, 3) ./ operator_normC;
adjointOp  = @(meas_in)  (measureMatrix .* meas_in) ./ operator_normC;

% Scale the measurement accordingly
measurement_scaled = compressed_img ./ operator_normC;

%% -------------------- TV Psi / Phi --------------------
tvInnerIters = 30;   % Recommended >= 30 for prox accuracy
psiTV = @(cube_in, tau_prox) tvdenoise_cube(cube_in, tau_prox, tvInnerIters);
phiTV = @(cube_in) TV_Phi_cube_tvdenoiseBC(cube_in); 

%% -------------------- TwIST parameters --------------------
% You can tune regularizationTau to balance sharpness vs noise
regularizationTau = 0.005;  
maxMainIters   = 200;
minMainIters   = 50;
toleranceStop  = 1e-3;
lambdaMinEig   = 1e-4; % For ill-conditioned problems

% Initialization: x0 = A'*y
init_cube = adjointOp(measurement_scaled);
init_cube = psiTV(init_cube, 0.01 * regularizationTau); % Light TV pre-denoise

%% -------------------- Run TwIST --------------------
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

% Bound the output to physical reflectance values [0, 1]
recon_cube = max(0, min(1, recon_cube));

%% -------------------- Evaluation & Visualization --------------------
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

figure('Color', 'w');
subplot(1,2,1); plot(1:numBands, band_mse, 'o-', 'LineWidth', 1.5);
xlabel('Band index'); ylabel('MSE'); title('Reconstruction MSE'); grid on;
subplot(1,2,2); plot(1:numBands, band_psnr, 'r^-', 'LineWidth', 1.5);
xlabel('Band index'); ylabel('PSNR (dB)'); title('Reconstruction PSNR'); grid on;

% Show one band example side-by-side
bandShow = min(15, numBands); % Change this to view a different band
figure('Color', 'w', 'Position', [100 100 900 400]);
subplot(1,3,1); imshow(hypercube_gt(:,:,bandShow), []); title(sprintf('Ground Truth (Band %d)', bandShow));
subplot(1,3,2); imshow(recon_cube(:,:,bandShow), []);   title(sprintf('Reconstruction (Band %d)', bandShow));
subplot(1,3,3); imshow(abs(recon_cube(:,:,bandShow) - hypercube_gt(:,:,bandShow)), [0 0.2]); colormap(gca, 'hot'); colorbar; title('Absolute Error');

% Objective curve
figure('Color', 'w');
plot(objectiveVals, 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Objective Value');
title('TwIST Convergence'); grid on;

%% -------------------- Save reconstruction results --------------------
save('HS_recon_TwIST.mat', ...
    'recon_cube', 'objectiveVals', 'cpuTimes', 'band_mse', 'band_psnr', ...
    'regularizationTau', 'tvInnerIters', 'operator_normC', '-v7.3');
fprintf('\nSaved: HS_recon_TwIST.mat\n');

%% ==================== Local functions ====================

function cube_out = tvdenoise_cube(cube_in, tau_prox, iters)
% Mapping TwIST prox to tvdenoise parameter lambda
    cube_in = single(cube_in);
    tau_prox = max(single(tau_prox), single(1e-12));   
    lambda_tv = 1 ./ tau_prox;
    
    % NOTE: Assumes 'tvdenoise' function exists in your MATLAB path.
    % If your function is named differently (e.g., tv_denoise_chambolle_conv2c), 
    % replace 'tvdenoise' below with that function name.
    cube_out = single(tvdenoise(double(cube_in), double(lambda_tv), iters));
end

function tvVal = TV_Phi_cube_tvdenoiseBC(cube_in)
% Isotropic TV calculation
    cube_in = single(cube_in);
    [H, W, B] = size(cube_in);
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