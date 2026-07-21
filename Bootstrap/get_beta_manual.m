function [Beta, eDF, GCV, RSS] = get_beta_manual(Y, X, z_list)
    [T, P] = size(X);
    L = length(z_list);
    if P > T
        [U_t, Sigma_sq_matrix, ~] = svd((X * X') / T);
        Sigma_sq  = diag(Sigma_sq_matrix);
        Sigma_inv = (Sigma_sq * T).^(-1/2);
        V = X' * U_t * diag(Sigma_inv);
    else
        [V, Sigma_sq_matrix, ~] = svd((X' * X) / T);
        Sigma_sq = diag(Sigma_sq_matrix);
    end
    signal_projected = V' * ((X' * Y) / T);
    Beta = nan(P, L); eDF = nan(1, L); GCV = nan(1, L); RSS = nan(1, L);
    tol = max(T, P) * eps(max(Sigma_sq));
    for l = 1:L
        z = z_list(l);
        if z == 0
            shrinkage_filter            = zeros(size(Sigma_sq));
            valid_idx                   = Sigma_sq > tol;
            shrinkage_filter(valid_idx) = 1 ./ Sigma_sq(valid_idx);
        else
            shrinkage_filter = 1 ./ (Sigma_sq + z);
        end
        Beta(:, l) = V * (shrinkage_filter .* signal_projected);
        if z == 0
            eDF(l) = sum(valid_idx);
        else
            eDF(l) = sum(Sigma_sq ./ (Sigma_sq + z));
        end
        Y_hat  = X * Beta(:, l);
        RSS(l) = sum((Y - Y_hat).^2);
        denom  = (1 - (eDF(l) / T))^2;
        GCV(l) = (RSS(l) / T) / denom;
    end
end