function I = cond_mutual_information(x, y, z, alpha)
    % Estimates conditional mutual information I(X;Y|Z)
    % using the Hutter estimator with optional Dirichlet prior alpha.

    % Default alpha
    if nargin < 4
        % Number of unique combinations of (x, y, z)
        alpha = 1 / (numel(unique(x)) * numel(unique(y)) * numel(unique(z)));
    end

    % Unique values
    ux = unique(x);
    uy = unique(y);
    uz = unique(z);

    % Allocate count matrices
    N_xyz = zeros(length(ux), length(uy), length(uz));
    N_xz  = zeros(length(ux), length(uz));
    N_yz  = zeros(length(uy), length(uz));
    N_z   = zeros(1, length(uz));

    % Fill count matrices with alpha prior
    for i = 1:length(ux)
        for j = 1:length(uy)
            for k = 1:length(uz)
                mask = (x == ux(i)) & (y == uy(j)) & (z == uz(k));
                N_xyz(i,j,k) = alpha + sum(mask);
            end
        end
    end

    for i = 1:length(ux)
        for k = 1:length(uz)
            mask = (x == ux(i)) & (z == uz(k));
            N_xz(i,k) = alpha + sum(mask);
        end
    end

    for j = 1:length(uy)
        for k = 1:length(uz)
            mask = (y == uy(j)) & (z == uz(k));
            N_yz(j,k) = alpha + sum(mask);
        end
    end

    for k = 1:length(uz)
        mask = (z == uz(k));
        N_z(k) = alpha + sum(mask);
    end

    % Total samples
    n = sum(N_xyz(:));

    % Compute conditional mutual information
    I = 0;
    for i = 1:length(ux)
        for j = 1:length(uy)
            for k = 1:length(uz)
                n_ijk = N_xyz(i,j,k);
                n_ik  = N_xz(i,k);
                n_jk  = N_yz(j,k);
                n_k   = N_z(k);

                if n_ijk > 0
                    I = I + n_ijk * (psi(n_ijk + 1) - psi(n_ik + 1) ...
                                   - psi(n_jk + 1) + psi(n_k + 1));
                end
            end
        end
    end

    I = I / n;  % Normalize
end