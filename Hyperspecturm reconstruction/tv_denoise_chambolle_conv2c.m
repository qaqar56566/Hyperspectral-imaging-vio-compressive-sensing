function u = tv_denoise_chambolle_conv2c(f, lambda, nIter)
%TV_DENOISE_CHAMBOLLE_CONV2C ROF TV denoising with circular boundary.
%
%   u = argmin_u 0.5||u - f||_2^2 + lambda * TV(u)
%   TV(u) = sum sqrt( (Dh u)^2 + (Dv u)^2 )
%
% Dh and Dv implemented via conv2c with h=[0 1 -1] (circular wrap-around).
% The divergence operator is implemented as the negative adjoint of Dh/Dv,
% i.e., div(p) = -(Dh^T p1 + Dv^T p2), consistent with the same boundary.
%
% Requires: conv2c.m

    f = single(f);
    if lambda <= 0
        u = f;
        return;
    end
    if nargin < 3 || isempty(nIter)
        nIter = 50;
    end

    [H,W] = size(f);
    p1 = zeros(H,W,'single');
    p2 = zeros(H,W,'single');

    % Difference filters (as in your diffh/diffv)
    h  = single([0 1 -1]);     % Dh via conv2c(u,h)
    hv = h';                   % Dv via conv2c(u,h')

    % Adjoint filters for circular convolution:
    % If Dh(u) = conv2c(u, h), then Dh^T(p) = conv2c(p, rot90(h,2))
    % Likewise for Dv.
    hT  = rot90(h, 2);
    hvT = rot90(hv,2);

    tau = 0.25; % stable step for Chambolle dual updates (typical choice)

    for k = 1:nIter
        % div(p) = -(Dh^T p1 + Dv^T p2)
        divp = -(conv2c(p1, hT) + conv2c(p2, hvT));

        % primal variable corresponding to current dual field
        u = f - lambda * divp;

        % gradients of u (circular)
        ux = conv2c(u, h);
        uy = conv2c(u, hv);

        % dual update with projection onto unit ball (isotropic)
        % p <- (p + (tau/lambda)*grad(u)) / max(1, |p + ...|)
        % implemented using denom = 1 + (tau/lambda)*|grad(u)| (classic form)
        g = sqrt(ux.^2 + uy.^2);
        denom = 1 + (tau/lambda) * g;

        p1 = (p1 + (tau/lambda) * ux) ./ denom;
        p2 = (p2 + (tau/lambda) * uy) ./ denom;
    end

    % final u using final dual field
    divp = -(conv2c(p1, hT) + conv2c(p2, hvT));
    u = f - lambda * divp;
end
