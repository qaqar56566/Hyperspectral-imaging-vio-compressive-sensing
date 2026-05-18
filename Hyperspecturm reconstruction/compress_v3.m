%% Video-rate HS camera style compressed measurement (forward model)
clear; clc;
close all;

%% -------------------- user settings --------------------
direc = "..\feathers_ms";           % folder containing feathers_ms_**.png
pattern = "feathers_ms_*.png";      % file pattern
imag_size = [512, 512];             % expected spatial size
rng(0);                             % reproducibility

% Your new FDTD file name
% fdtd_filename = 'HSI_LDmodelAg20nm_500to2390nm_128steps_15nm_inter_sweep.txt'; 
% fdtd_filename = 'HSI_LDmodelAg20nm_60to2600nm_128steps_20nm_inter.txt'; 
% fdtd_filename = 'HSI_LDmodelAg20nm_60to1320nm_64steps_20nm_inter.txt'; 
% fdtd_filename = 'HSI_LDmodelAg20nm_60to680nm_32steps_20nm_inter.txt'; 
% fdtd_filename = 'HSI_LDmodelAg20nm_60to1300nm_32steps_40nm_inter.txt'; 
% fdtd_filename = 'HSI_LDmodelAg20nm_120to1320nm_32steps_40nm_inter.txt'; 
% fdtd_filename = 'HSI_31ch_32steps_sweep50to1009nm_Ag20nm.txt'; 
% fdtd_filename = 'transmitance/HSI_LDmodelAg20nm_40to660nm_32steps_20nm_inter.txt'; 
fdtd_filename = 'transmitance/HSI_LDmodelAg20nm_60to1300nm_32steps_40nm_inter.txt'; 
% fdtd_filename = 'transmitance/HSI_LDmodelAg20nm_60to1920nm_32steps_60nm_inter.txt'; 
% fdtd_filename = 'transmitance/HSI_31ch_64steps_sweep50to2000nm_Ag20nm.txt'; 
% --------------------------------------------------------

%% 1) Read hyperspectral band images (With Fail-Safe)
files = dir(fullfile(direc, pattern));
if isempty(files)
    fprintf('WARNING: Found 0 images in "%s".\n', direc);
    fprintf('Creating a dummy 31-band image stack so the script can still run...\n');
    num_bands = 31;
    HS = rand(imag_size(1), imag_size(2), num_bands, 'single'); % Dummy data
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

% --- DYNAMICALLY DETECT DIMENSIONS ---
% Find patterns like "(128,31)" in the text headers
dim_tokens = regexp(fileContent, '\((\d+),(\d+)\)', 'tokens');
if ~isempty(dim_tokens)
    % The last token is always the 2D matrix size header in Lumerical exports
    num_filters = str2double(dim_tokens{end}{1});
    num_waves   = str2double(dim_tokens{end}{2});
    fprintf('Dynamically detected FDTD dimensions: %d filters x %d wavelengths.\n', num_filters, num_waves);
else
    error('Could not detect matrix dimensions from the text file header.');
end

% Use Regular Expressions to extract ALL numbers, ignoring all text/headers
matches = regexp(fileContent, '[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', 'match');
all_nums = str2double(matches);

expected_matrix_size = num_filters * num_waves; 
fprintf('Extracted %d total numbers from the text file.\n', length(all_nums));

% Safely grab the exact matrix size from the very end of the file
if length(all_nums) >= expected_matrix_size
    T_fdtd_raw = all_nums(end - expected_matrix_size + 1 : end);
else
    error('Not enough numbers in the file. Expected at least %d, found %d.', expected_matrix_size, length(all_nums));
end

% Reshape into [wavelengths x filters] and transpose
T_fdtd = reshape(T_fdtd_raw, [num_waves, num_filters])'; 
T_real = single(abs(T_fdtd)); % Absolute value to fix negative Re(T) directions

%% 3) Randomly assign filter types over 512x512
% Changed to uint16 to support up to 65,535 filters without integer overflow
idxMap = randi(num_filters, imag_size(1), imag_size(2), 'uint16');


% %% 3) Assign filter types with improved spatial randomness (randperm)
% % Using randperm guarantees that each filter type is distributed uniformly 
% % across the entire sensor, preventing random "clumping".
% 
% Total_pixels = imag_size(1) * imag_size(2);
% P = randperm(Total_pixels); % Generate unique random permutation
% 
% % Use modulo to map the permutation [1 to Total] down to [0 to num_filters-1], 
% % then add 1 to get MATLAB 1-based indexing [1 to num_filters].
% idxMap_1D = mod(P, num_filters) + 1;
% 
% % Reshape the 1D array back into the 2D image dimensions and convert to uint16
% idxMap = reshape(idxMap_1D, imag_size(1), imag_size(2));
% idxMap = uint16(idxMap);

% (Optional) Print verification of perfectly uniform distribution
% min_usage = min(histcounts(idxMap(:), num_filters));
% max_usage = max(histcounts(idxMap(:), num_filters));
% fprintf('Filter usage balance: Min = %d pixels, Max = %d pixels.\n', min_usage, max_usage);

%% 4) Build transmittance patterns A(:,:,k)
A = build_measurement_matrix_from_idxMap(idxMap, T_real); 

% -------------------------------------------------------------------------
% Plot the FDTD transmittances to verify it loaded correctly (2D and 3D)
% -------------------------------------------------------------------------
figure('Color','w', 'Position', [100, 200, 1200, 500]); 

% --- 2D Line Plot ---
subplot(1,2,1);
plot(1:num_waves, T_real'); 
xlabel('Band Index');
ylabel('Transmittance');
title(sprintf('2D Transmittance Profiles (%d Filters)', num_filters));
xlim([1, num_waves]);
grid on;

% --- 3D Surface Plot ---
subplot(1,2,2);
[X, Y] = meshgrid(1:num_waves, 1:num_filters);
surf(X, Y, T_real);
shading interp;      % Smooth out the surface colors
colormap('jet');     % Classic photonics/heatmap colors
colorbar;            % Add color scale
xlabel('Band Index');
ylabel(sprintf('Filter Index (1 to %d)', num_filters)); % Dynamically labeled
zlabel('Transmittance');
title('3D Transmittance Matrix');
view(45, 30);        % Adjust the 3D viewing angle
grid on;
% -------------------------------------------------------------------------

% Characterize the randomness of measurement matrix (if function exists)
try
    [som, R] = Characterize_randomness(A);
catch
    disp('Notice: Characterize_randomness() function not found. Skipping.');
end

%% 5) Apply masks to corresponding bands, then sum
compressed_img = sum(single(A) .* HS, 3);

%% Display / save
figure('Color','w'); 
imshow(compressed_img, []); 
title(sprintf("Compressed image (%d FDTD Metasurface filters)", num_filters));

outName = "compressed_FDTD.png";
imwrite(mat2gray(compressed_img), outName);
fprintf("Saved: %s\n", outName);

% Save intermediate results for reconstruction
save("compressed_img.mat", "compressed_img");
save("HS.mat", "HS", "-v7.3"); 
save("measurement_matrix.mat", "A", "-v7.3");
save("idxMap.mat", "idxMap");

%% local functions
function A = build_measurement_matrix_from_idxMap(idxMap, T)
%BUILD_MEASUREMENT_MATRIX_FROM_IDXMAP
%   idxMap: HxW integer map in {1..num_filters} (filter type per pixel)
%   T     : num_filters x num_bands, T(i,k)=transmittance of filter i at band k
    
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