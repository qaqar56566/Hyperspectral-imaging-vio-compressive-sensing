%% ============================================================
%  HS -> sRGB using CIE 1931 2° CMFs
%  1) Read CIE_xyz_1931_2deg.csv
%  2) Interpolate to wavelength_nm = (400:10:700)
%  3) Save projection table to .mat
%  4) Use hyperspectral cube (HxWx31) to generate pseudo-color sRGB image
%
%  Expected CSV format (typical):
%    Col1: wavelength (nm)
%    Col2: xbar
%    Col3: ybar
%    Col4: zbar
%
%  Requirements: base MATLAB (no extra toolbox)
% ============================================================

% clear; clc; close all;

%% -------------------- User settings --------------------
cieCsvFile = 'CIE_xyz_1931_2deg.csv';   % CMF file
wavelength_nm = (400:10:700)';         % 31 bands

% Hyperspectral cube file
hsVarCandidates = {'recon_cube','x_hat','HS','hypercube','hypercube_gt'}; % try these in order
load('HS_recon_TwIST_TV_norm.mat','recon_cube');


% RGB ground truth imge
rgb_gt = imread('feathers_ms\feathers_RGB.bmp');
figure;
imshow(rgb_gt)


% Output
cmfMatOut = 'cie1931_2deg_400_700_10nm.mat';
rgbOutPng = 'pseudoColor_sRGB.png';

% %% -------------------- 1) Read CIE CSV --------------------
% assert(isfile(cieCsvFile)==1, "Cannot find %s", cieCsvFile);
% 
% % Robust read (handles header lines automatically)
% cieTable = readtable(cieCsvFile);
% 
% % Identify columns (by position as default)
% % If your CSV has headers, you can also use: cieTable.Properties.VariableNames
% wl_full  = cieTable{:,1};
% xbar_full = cieTable{:,2};
% ybar_full = cieTable{:,3};
% zbar_full = cieTable{:,4};
% 
% wl_full  = double(wl_full(:));
% xbar_full = double(xbar_full(:));
% ybar_full = double(ybar_full(:));
% zbar_full = double(zbar_full(:));
% 
% % Basic sanity
% if any(diff(wl_full) <= 0)
%     % sort if needed
%     [wl_full, sortIdx] = sort(wl_full);
%     xbar_full = xbar_full(sortIdx);
%     ybar_full = ybar_full(sortIdx);
%     zbar_full = zbar_full(sortIdx);
% end
% 
% %% -------------------- 2) Interpolate to 400:10:700 --------------------
% % Use shape-preserving interpolation; clamp outside range if needed
% xbar_10 = interp1(wl_full, xbar_full, wavelength_nm, 'pchip', 'extrap');
% ybar_10 = interp1(wl_full, ybar_full, wavelength_nm, 'pchip', 'extrap');
% zbar_10 = interp1(wl_full, zbar_full, wavelength_nm, 'pchip', 'extrap');
% 
% % Replace any negative tiny values due to interpolation artifacts
% xbar_10 = max(xbar_10, 0);
% ybar_10 = max(ybar_10, 0);
% zbar_10 = max(zbar_10, 0);
% 
% %% -------------------- 3) Save projection table --------------------
% cieProjection.wavelength_nm = wavelength_nm;
% cieProjection.xbar = single(xbar_10);
% cieProjection.ybar = single(ybar_10);
% cieProjection.zbar = single(zbar_10);
% 
% save(cmfMatOut, '-struct', 'cieProjection');
% fprintf("Saved CIE projection table: %s\n", cmfMatOut);
% 
%% -------------------- 4) Load hyperspectral cube --------------------
hsCube = recon_cube;

hsCube = single(hsCube);
[H,W,B] = size(hsCube);
assert(B == numel(wavelength_nm), "HS cube bands (%d) must match wavelength list (%d).", B, numel(wavelength_nm));

% Ensure nonnegative (recommended for reflectance-like cube)
hsCube = max(hsCube, 0);

cieProjection = load("cie1931_2deg_400_700_10nm.mat");
%% -------------------- 5) Convert HS -> XYZ -> sRGB --------------------
rgb = hs_to_srgb_CIE1931(hsCube, cieProjection.wavelength_nm, cieProjection.xbar, cieProjection.ybar, cieProjection.zbar);

figure; imshow(rgb); 
title('Pseudo-color sRGB (CIE1931 2°)');
imwrite(rgb, rgbOutPng);
fprintf("Saved RGB image: %s\n", rgbOutPng);

figure;
subplot(1,2,1); imshow(rgb_gt); 
title("GT RGB");
subplot(1,2,2); imshow(rgb); 
title('Pseudo-color sRGB (CIE1931 2°)');

%% ============================================================
% Local functions
%% ============================================================

function rgb = hs_to_srgb_CIE1931(hypercube, wavelength_nm, xbar, ybar, zbar)
%HS_TO_SRGB_CIE1931 Convert HS cube to displayable sRGB using CIE 1931 2° CMFs.
% hypercube: HxWxB (single/double), B bands
% wavelength_nm: Bx1
% xbar,ybar,zbar: Bx1 at same wavelengths

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
    % Use robust percentile to avoid a few bright pixels dominating
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
%SRGB_GAMMA_ENCODE Convert linear RGB to sRGB (IEC 61966-2-1)

    rgb_lin = single(rgb_lin);
    a = single(0.055);
    thr = single(0.0031308);

    rgb = zeros(size(rgb_lin), 'single');

    idx = rgb_lin <= thr;
    rgb(idx)  = 12.92 * rgb_lin(idx);
    rgb(~idx) = (1+a) * (rgb_lin(~idx).^(1/2.4)) - a;

    rgb = min(max(rgb, 0), 1);
end
