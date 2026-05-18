%% Video-rate HS camera style compressed measurement (forward model)
clear; clc;
close all;

%% -------------------- user settings --------------------
direc = "..\feathers_ms";           % folder containing feathers_ms_**.png
pattern = "feathers_ms_*.png";   % file pattern
imag_size = [512, 512];          % expected spatial size
rng(0);                          % reproducibility

% Your new FDTD file name
% fdtd_filename = 'HSI_31ch_64steps_sweep50to2000nm_Ag20nm.txt'; 
% fdtd_filename = 'HSI_31ch_64steps_sweep740to2000nm_Ag20nm2.txt'; 
% fdtd_filename = 'HSI_31ch_64steps_sweep500to2390nm_Ag20nm.txt'; 
% fdtd_filename = 'HSI_sub0_Ag20nm_500to2390nm_64steps_30nm_inter_sweep.txt'; 
fdtd_filename = 'HSI_LDmodelAg20nm_500to2390nm_128steps_15nm_inter_sweep.txt'; 

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

%% 2) Universal FDTD Data Parser
fprintf("Loading FDTD sweep data from %s...\n", fdtd_filename);
fileContent = fileread(fdtd_filename);

% Use Regular Expressions to extract ALL numbers, ignoring all text/headers
matches = regexp(fileContent, '[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', 'match');
all_nums = str2double(matches);

num_filters = 128;
num_waves = 31;
expected_matrix_size = num_filters * num_waves; % 64 * 31 = 1984

fprintf('Extracted %d total numbers from the text file.\n', length(all_nums));

% Automatically detect the export format based on number count
if length(all_nums) == (num_filters + num_waves + expected_matrix_size)
    % Format A: Standard 2D Lumerical Sweep (h_etch + lambda + matrix)
    T_fdtd_raw = all_nums(end - expected_matrix_size + 1 : end);
    
elseif length(all_nums) == (2 * expected_matrix_size)
    % Format B: 2-Column CSV format (Wavelength, Transmittance) stacked 64 times
    T_fdtd_raw = all_nums(2:2:end); % Extract only the 'Y' transmittance values
    
elseif length(all_nums) >= expected_matrix_size
    % Format C: Safely grab the last 1984 numbers as the matrix
    T_fdtd_raw = all_nums(end - expected_matrix_size + 1 : end);
    
else
    error('Not enough numbers in the file. Expected at least %d, found %d.', expected_matrix_size, length(all_nums));
end

% Reshape into [31 wavelengths x 64 filters] and transpose to [64 x 31]
T_fdtd = reshape(T_fdtd_raw, [num_waves, num_filters])'; 
T_real = single(abs(T_fdtd)); % Absolute value to fix negative Re(T) directions

%% 3) Randomly assign 64 filter types over 512x512
% idxMap(x,y) randomly selects from the 64 available FDTD filters
idxMap = randi(num_filters, imag_size(1), imag_size(2), 'uint8');

%% 4) Build transmittance patterns A(:,:,k)
% A will be size: [512, 512, 31]
A = build_measurement_matrix_from_idxMap(idxMap, T_real); 

% % Plot the FDTD transmittances to verify it loaded correctly
% figure('Color','w'); 
% plot(1:num_waves, T_real'); 
% xlabel('Band Index (1 to 31)');
% ylabel('Transmittance');
% title(sprintf('Metasurface Transmittances (%d Filters)', num_filters));
% grid on;

% -------------------------------------------------------------------------
% Plot the FDTD transmittances to verify it loaded correctly (2D and 3D)
% -------------------------------------------------------------------------
figure('Color','w', 'Position', [100, 200, 1200, 500]); 

% --- 2D Line Plot ---
subplot(1,2,1);
plot(1:num_waves, T_real'); 
xlabel('Band Index (1 to 31)');
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
xlabel('Band Index (1 to 31)');
ylabel('Filter Index (1 to 64)');
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
% compressed_img = sum_b A(:,:,b) .* HS(:,:,b)
compressed_img = sum(single(A) .* HS, 3);

%% Display / save
figure('Color','w'); 
imshow(compressed_img, []); 
title("Compressed image (FDTD Metasurface filters)");

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