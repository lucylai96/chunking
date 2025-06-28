function mi = estimate_ssa(S_prev, S_curr, A)
    % Estimate I(S_{t-1}; S_t, A) from discrete data

    % Joint variable Z = [S_t, A]
    Z = [S_curr, A];

    % Convert to single indices for joint distributions
    [~, ~, Sprev_ids] = unique(S_prev, 'rows');
    [~, ~, Z_ids]     = unique(Z, 'rows');
    [~, ~, Scurr_ids] = unique(S_curr, 'rows');
    [~, ~, A_ids]     = unique(A, 'rows');

    % Estimate joint distribution P(S_prev, Z)
    joint_counts = accumarray([Sprev_ids, Z_ids], 1);
    P_Sprev_Z = joint_counts / sum(joint_counts(:));

    % Estimate marginals
    P_Sprev = sum(P_Sprev_Z, 2);
    P_Z     = sum(P_Sprev_Z, 1);

    % Compute mutual information
    [i_idx, j_idx] = find(P_Sprev_Z > 0); % Only non-zero entries
    mi = 0;
    for k = 1:length(i_idx)
        i = i_idx(k);
        j = j_idx(k);
        p_joint = P_Sprev_Z(i, j);
        p_i = P_Sprev(i);
        p_j = P_Z(j);
        mi = mi + p_joint * log2(p_joint / (p_i * p_j));
    end
end