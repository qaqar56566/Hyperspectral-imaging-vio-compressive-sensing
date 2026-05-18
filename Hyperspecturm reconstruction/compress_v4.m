%% HS compressed measurement with optimized mask design (v4)
% Goals:
% 1) More spatially uniform filter placement to reduce clumping
% 2) Mask optimization using correlation, condition number, incoherence proxies
% 3) Per-band sensitivity analysis to avoid severe under-sampling

% clear; clc; close all;

%% -------------------- user settings --------------------
direc = "..\feathers_ms";           % folder containing feathers_ms_**.png
pattern = "feathers_ms_*.png";      % file pattern
imag_size = [512, 512];             % expected spatial size
rng(0);                             % reproducibility

% FDTD file
fdtd_filename = 'transmitance/HSI_LDmodelAg20nm_60to1300nm_32steps_40nm_inter.txt';

% Mask design settings
blockSize = 16;                     % local uniformity block size
optimizeMask = true;                % enable swap-based optimization
optimizeIters = 200;                % number of random swap trials
sampleStride = 4;                   % sampling stride for fast metric evaluation

% Objective weights
wCorr = 1.0;                        % max off-diagonal correlation penalty
wCond = 0.25;                       % condition number penalty
wSens = 0.5;                        % band sensitivity penalty

%% 1) Read hyperspectral band images
files = dir(fullfile(direc, pattern));
if isempty(files)
    fprintf('WARNING: Found 0 images in "%s".\n', direc);
    fprintf('Creating a dummy 31-band image stack so the script can still run...\n');
    num_bands = 31;
    HS = rand(imag_size(1), imag_size(2), num_bands, 'single');
else
    [~, sort_idx] = sort({files.name});
    files = files(sort_idx);
    num_bands = numel(files);
    fprintf("Found %d band images.\n", num_bands);

    HS = zeros(imag_size(1), imag_size(2), num_bands, 'single');
    for b = 1:num_bands
        fn = fullfile(files(b).folder, files(b).name);
        HS(:,:,b) = im2single(imread(fn));
    end
end

%% 2) Universal FDTD Data Parser (Dynamic Dimensions)
fprintf("Loading FDTD sweep data from %s...\n", fdtd_filename);
fileContent = fileread(fdtd_filename);

% Detect matrix dimensions in header
% Format example: (128,31)
dim_tokens = regexp(fileContent, '\((\d+),(\d+)\)', 'tokens');

matches = regexp(fileContent, '[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', 'match');
all_nums = str2double(matches);

if ~isempty(dim_tokens)
    num_filters = str2double(dim_tokens{end}{1});
    num_waves   = str2double(dim_tokens{end}{2});
    fprintf('Detected FDTD dimensions: %d filters x %d wavelengths.\n', num_filters, num_waves);
else
    % Fallback: infer from hyperspectral band count and numeric count
    num_waves = num_bands;
    candidate_filters = [16 32 48 64 96 128 256 512];
    num_filters = 0;
    for k = 1:numel(candidate_filters)
        if length(all_nums) >= candidate_filters(k) * num_waves
            num_filters = candidate_filters(k);
            break;
        end
    end
    if num_filters == 0
        num_filters = max(1, floor(length(all_nums) / max(num_waves, 1)));
    end
    fprintf('Header not found; using fallback dimensions: %d filters x %d wavelengths.\n', num_filters, num_waves);
end

expected_matrix_size = num_filters * num_waves;
fprintf('Extracted %d total numbers from the text file.\n', length(all_nums));

if length(all_nums) == (num_filters + num_waves + expected_matrix_size)
    % Format A: header vectors + matrix
    T_fdtd_raw = all_nums(end - expected_matrix_size + 1 : end);
elseif length(all_nums) == (2 * expected_matrix_size)
    % Format B: 2-column stacked (wavelength, transmittance)
    T_fdtd_raw = all_nums(2:2:end);
elseif length(all_nums) >= expected_matrix_size
    % Format C: grab the tail as matrix
    T_fdtd_raw = all_nums(end - expected_matrix_size + 1 : end);
else
    error('Not enough numbers in the file. Expected at least %d, found %d.', expected_matrix_size, length(all_nums));
end

T_fdtd = reshape(T_fdtd_raw, [num_waves, num_filters])';
T_real = single(abs(T_fdtd));

%% 3) Build spatially uniform idxMap (reduced clumping)
idxMap = build_uniform_idxMap(imag_size, num_filters, blockSize);

%% 4) Mask optimization (correlation + condition + sensitivity)
if optimizeMask
    fprintf('Optimizing mask with %d random swaps...\n', optimizeIters);
    [idxMap, bestMetrics] = optimize_idxMap(idxMap, T_real, sampleStride, optimizeIters, ...
        wCorr, wCond, wSens);
    fprintf('Final metrics: maxCorr=%.4f, cond=%.2e, sensPenalty=%.4f, obj=%.4f\n', ...
        bestMetrics.maxCorr, bestMetrics.condNumber, bestMetrics.sensPenalty, bestMetrics.objective);
end

%% 5) Build measurement matrix A(:,:,k)
A = build_measurement_matrix_from_idxMap(idxMap, T_real);

% Plot transmittance profiles
figure('Color','w', 'Position', [100, 200, 1200, 500]);
subplot(1,2,1);
plot(1:num_waves, T_real');
xlabel('Band Index');
ylabel('Transmittance');
title(sprintf('2D Transmittance Profiles (%d Filters)', num_filters));
xlim([1, num_waves]);
grid on;

subplot(1,2,2);
[X, Y] = meshgrid(1:num_waves, 1:num_filters);
surf(X, Y, T_real);
shading interp;
colormap('jet');
colorbar;
xlabel('Band Index');
ylabel(sprintf('Filter Index (1 to %d)', num_filters));
zlabel('Transmittance');
title('3D Transmittance Matrix');
view(45, 30);
grid on;

% Characterize randomness if available
try
    [som, R] = Characterize_randomness(A);
catch
    disp('Notice: Characterize_randomness() function not found. Skipping.');
end

%% 6) Band sensitivity analysis
bandMean = squeeze(mean(mean(A, 1), 2));
bandStd  = squeeze(std(reshape(A, [], size(A,3)), 1, 1));

figure('Color','w');
subplot(1,2,1); plot(bandMean, 'o-'); grid on;
xlabel('Band'); ylabel('Mean transmittance'); title('Band sensitivity (mean)');
subplot(1,2,2); plot(bandStd, 's-'); grid on;
xlabel('Band'); ylabel('Std transmittance'); title('Band sensitivity (std)');

lowBandThresh = 0.5 * mean(bandMean);
lowBands = find(bandMean < lowBandThresh);
if ~isempty(lowBands)
    fprintf('Warning: low sensitivity bands detected: %s\n', mat2str(lowBands));
end

%% 7) Apply masks to corresponding bands, then sum
compressed_img = sum(single(A) .* HS, 3);

%% Display / save
figure('Color','w');
imshow(compressed_img, []);
title(sprintf('Compressed image (%d optimized filters)', num_filters));

outName = "compressed_FDTD_v4.png";
imwrite(mat2gray(compressed_img), outName);
fprintf("Saved: %s\n", outName);

save("compressed_img.mat", "compressed_img");
save("HS.mat", "HS", "-v7.3");
save("measurement_matrix.mat", "A", "-v7.3");
save("idxMap.mat", "idxMap");

%% ============================================================
% Local functions
%% ============================================================

function idxMap = build_uniform_idxMap(imag_size, num_filters, blockSize)
    H = imag_size(1);
    W = imag_size(2);
    idxMap = zeros(H, W, 'uint16');

    for r = 1:blockSize:H
        for c = 1:blockSize:W
            r2 = min(r + blockSize - 1, H);
            c2 = min(c + blockSize - 1, W);
            blockH = r2 - r + 1;
            blockW = c2 - c + 1;
            blockSizeActual = blockH * blockW;

            perm = randperm(num_filters);
            blockIdx = zeros(blockSizeActual, 1);
            for k = 1:blockSizeActual
                blockIdx(k) = perm(mod(k - 1, num_filters) + 1);
            end
            blockIdx = blockIdx(randperm(blockSizeActual));

            idxMap(r:r2, c:c2) = reshape(uint16(blockIdx), blockH, blockW);
        end
    end
end

function [idxMap, bestMetrics] = optimize_idxMap(idxMap, T_real, sampleStride, nIters, wCorr, wCond, wSens)
    bestMetrics = evaluate_metrics(idxMap, T_real, sampleStride, wCorr, wCond, wSens);
    H = size(idxMap, 1);
    W = size(idxMap, 2);

    for iter = 1:nIters
        r1 = randi(H); c1 = randi(W);
        r2 = randi(H); c2 = randi(W);

        if r1 == r2 && c1 == c2
            continue;
        end

        tmp = idxMap(r1, c1);
        idxMap(r1, c1) = idxMap(r2, c2);
        idxMap(r2, c2) = tmp;

        curMetrics = evaluate_metrics(idxMap, T_real, sampleStride, wCorr, wCond, wSens);

        if curMetrics.objective <= bestMetrics.objective
            bestMetrics = curMetrics;
        else
            tmp = idxMap(r1, c1);
            idxMap(r1, c1) = idxMap(r2, c2);
            idxMap(r2, c2) = tmp;
        end
    end
end

function metrics = evaluate_metrics(idxMap, T_real, sampleStride, wCorr, wCond, wSens)
    [H, W] = size(idxMap);
    sampleRows = 1:sampleStride:H;
    sampleCols = 1:sampleStride:W;
    [C, R] = meshgrid(sampleCols, sampleRows);
    idx = sub2ind([H, W], R(:), C(:));

    filters = double(idxMap(idx));
    Vraw = double(T_real(filters, :));

    V = Vraw - mean(Vraw, 1);
    den = sqrt(sum(V.^2, 1)) + eps;
    Rmat = (V' * V) ./ (den' * den);
    Rmat(1:size(Rmat,1)+1:end) = 0;
    maxCorr = max(abs(Rmat(:)));

    G = (V' * V) ./ size(V, 1);
    condNumber = cond(G + 1e-6 * eye(size(G)));

    bandMean = mean(Vraw, 1);
    meanBand = mean(bandMean);
    if meanBand == 0
        sensPenalty = 1;
    else
        sensPenalty = max(0, (meanBand - min(bandMean)) / meanBand);
    end

    objective = wCorr * maxCorr + wCond * log10(condNumber + eps) + wSens * sensPenalty;

    metrics.maxCorr = maxCorr;
    metrics.condNumber = condNumber;
    metrics.sensPenalty = sensPenalty;
    metrics.objective = objective;
end

function A = build_measurement_matrix_from_idxMap(idxMap, T)
    assert(ismatrix(idxMap), "idxMap must be HxW.");
    [H, W] = size(idxMap);

    [num_filters, num_bands] = size(T);

    assert(all(idxMap(:) >= 1 & idxMap(:) <= num_filters), ...
        sprintf("idxMap values must be in 1..%d.", num_filters));

    idx = idxMap(:);
    A = zeros(H, W, num_bands, 'single');

    for k = 1:num_bands
        A(:,:,k) = reshape(T(idx, k), H, W);
    end
end
