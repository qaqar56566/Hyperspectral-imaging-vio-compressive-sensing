%% 1. Load the raw text
% filename = 'HSI_sweep400to2000nm_Ag20nm.txt';
filename = 'HSI_31ch_64steps_sweep740to2000nm_Ag20nm.txt'; 
fileContent = fileread(filename);

%% 2. Extract h_etch (101 points)
% Find the position after "h_etch(101,1)"
startIdx = strfind(fileContent, 'h_etch(101,1)') + length('h_etch(101,1)');
% Find the start of "lambda" to know where to stop
endIdx = strfind(fileContent, 'lambda(50,1)');
% Extract and convert
h_etch_str = fileContent(startIdx:endIdx-1);
h_etch = sscanf(h_etch_str, '%f');

%% 3. Extract lambda (50 points)
startIdx = endIdx + length('lambda(50,1)');
% Find the start of the data matrix
endIdx = strfind(fileContent, 'sweep:trans: Re(T) vs position(101,50)');
lambda_str = fileContent(startIdx:endIdx-1);
lambda = sscanf(lambda_str, '%f');

%% 4. Extract the Transmission Matrix (101x50)
startIdx = endIdx + length('sweep:trans: Re(T) vs position(101,50)');
data_str = fileContent(startIdx:end);
% Read all remaining numbers
all_data = sscanf(data_str, '%f');

% Robust check: In FDTD text exports, the matrix is usually 
% written row by row (H_etch index changes slowest).
% Number of elements must be 5050.
if length(all_data) == 5050
    T_matrix = reshape(all_data, [50, 101])'; 
else
    error('Data size mismatch. Expected 5050 elements, found %d.', length(all_data));
end

%% 5. Plot 3D Surface
figure('Color', 'w');

% Convert to Nanometers for standard photonics plotting
X_nm = h_etch * 1e9; 
Y_nm = lambda * 1e9;

% surf(X, Y, Z) - Note: X and Y vectors must match matrix dimensions
surf(Y_nm, X_nm, T_matrix);

shading interp;
colormap('jet');
colorbar;
xlabel('Wavelength (nm)');
ylabel('Etch Height (nm)');
zlabel('Re(Transmission)');
title('Metasurface Transmission Sweep (Ag 20nm)');
grid on;
view(45, 30); % Rotates to a nice 3D view