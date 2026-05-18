function [sigma_over_mu, R, mu, sigma] = Characterize_randomness(A, doPlot)
% Characterize spatial and spectral randomness of HS mask.
%
%   [sigma_over_mu, R, mu, sigma] = Characterize_randomness(A)
%   [sigma_over_mu, R, mu, sigma] = Characterize_randomness(A, doPlot)
%   [som, R, mu, sigma] = Characterize_randomness(A);       
%   [som, R] = Characterize_randomness(A, false);           
%
%   Inputs:
%     A      : H x W x B transmittance patterns (numeric or logical).
%              A(:,:,i) is the transmittance pattern at band i.
%     doPlot : (optional) true/false. Default = true.
%
%   Outputs:
%     sigma_over_mu : Bx1 vector, spatial randomness index sigma/mu per band.
%     R             : BxB matrix, spectral randomness correlation coefficients r_ij.
%     mu            : Bx1 vector, mean transmittance per band.
%     sigma         : Bx1 vector, std (population, 1/n) per band.
%
%   Notes:
%     - Uses population standard deviation std(v,1) (normalized by n) to match 1/n definition.
%     - Correlation is Pearson correlation computed over all pixels (2D flattened).
%     - Adds eps to avoid divide-by-zero.

    if nargin < 1
        error("Characterize_randomness:MissingInput", ...
              "Input A is required (HxWxB transmittance patterns).");
    end
    if nargin < 2 || isempty(doPlot)
        doPlot = true;
    end

    if ~(isnumeric(A) || islogical(A))
        error("Characterize_randomness:InvalidType", ...
              "A must be numeric or logical array of size HxWxB.");
    end

    if ndims(A) ~= 3
        error("Characterize_randomness:InvalidDims", ...
              "A must be a 3D array of size HxWxB.");
    end

    A = single(A);  % ensure numeric type for computation
    [H, W, B] = size(A);
    n = H * W;

    %% 1) Spatial randomness: sigma/mu for each band
    mu = zeros(B, 1, 'single');
    sigma = zeros(B, 1, 'single');
    sigma_over_mu = zeros(B, 1, 'single');

    for i = 1:B
        v = A(:,:,i);
        v = v(:);                    % vectorize
        mu(i) = mean(v);
        sigma(i) = std(v, 1);        % population std (1/n)
        sigma_over_mu(i) = sigma(i) / (mu(i) + eps('single'));
    end

    %% 2) Spectral randomness: correlation matrix r_ij
    % Flatten A -> X (n x B), each column corresponds to one band
    X = reshape(A, n, B);
    X = X - mean(X, 1);              % zero-mean each band

    den = sqrt(sum(X.^2, 1));
    den = den + eps('single');       % avoid divide-by-zero

    R = (X' * X) ./ (den' * den);    % B x B
    R(1:B+1:end) = 1;                % enforce diagonal exactly 1

    %% Optional visualization
    if doPlot
        % fprintf("Spatial randomness sigma/mu per band:\n");
        % disp(sigma_over_mu.');

        figure; plot(1:B, sigma_over_mu, "o-");
        xlabel("Band index");
        ylabel("\sigma/\mu");
        title("Spatial randomness per band");

        figure; imagesc(R); axis image; colorbar;
        title("Spectral randomness: correlation matrix r_{ij}");
        xlabel("Band j");
        ylabel("Band i");
    end
end
