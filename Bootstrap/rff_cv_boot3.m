function [] = rff_cv_boot3(X, Y, dates, zlist, gamma, vol_stdize, trnwin, nval, demeanX, demeanY, isim, output_dir, iboot)
% nval : number of validation steps (e.g. 5 means predict t-5, t-4, t-3, t-2, t-1)
% --- Grid for P ---
maxP  = 12000;
Plist = [maxP];
nP    = length(Plist);
nL    = length(zlist);
% --- Load and Prepare Data ---
if vol_stdize == 1
    X     = volstdbwd(X, []);
    Y2    = movmean(Y.^2, [11 0]);
    Y     = Y(37:end) ./ sqrt(Y2(36:end-1));
    X     = X(37:end, :);
    dates = dates(37:end, :);
end
T = length(Y);
X = X';
Y = Y';
d = size(X, 1);
% --- Output Space ---
Yprd  = nan(T, nP);
Bnrm  = nan(T, nP);
zSel  = nan(T, nP);
CVerr = nan(T, nP, nL);
% --- Weight Mat ---
rng(iboot + isim*1000);
W = randn(maxP, d);
% --- Recursive Estimation -------------------------------------------------
for p = 1:nP
    currP = floor(Plist(p) / 2);
    wtmp  = W(1:currP, :);
    Z     = [cos(gamma * wtmp * X); sin(gamma * wtmp * X)];

    for t = trnwin+1:T
        % Original sequential window for Y (and Z when iboot == 0)
        seq_idx = (t - trnwin) : (t - 1);

        % Draw block for Z
        if iboot == 0
            rnd_idx = seq_idx;
        else
            blk_start = randi(T - trnwin);  % ensure t_boot = blk_start+trnwin <= T
            rnd_idx   = blk_start : blk_start + trnwin - 1;
        end

        % ── Step 1: nval-fold CV to select z ──────────────────────────────
        fold_err = zeros(nval, nL);
        for k = 1:nval
            n_trn_cv   = trnwin - nval + k - 1;
            trn_cv     = rnd_idx(1 : n_trn_cv);
            seq_trn_cv = seq_idx(1 : n_trn_cv);
            val_cv     = rnd_idx(n_trn_cv + 1);
            seq_val_cv = seq_idx(n_trn_cv + 1);

            Ycv  = Y(seq_trn_cv);
            Zcv  = Z(:, trn_cv);
            Zval = Z(:, val_cv);

            % Demean
            Ymn_cv  = (demeanY == 1) * nanmean(Ycv);
            Zmn_cv  = (demeanX == 1) * nanmean(Zcv, 2);
            Ycv_dm  = Ycv  - Ymn_cv;
            Zcv_dm  = Zcv  - Zmn_cv;
            Zval_dm = Zval - Zmn_cv;

            % Standardize
            Zstd_cv  = nanstd(Zcv_dm, [], 2) + 1e-6;
            Zcv_dm   = Zcv_dm  ./ Zstd_cv;
            Zval_dm  = Zval_dm ./ Zstd_cv;

            % Fit all z
            [Beta_cv, ~, ~, ~] = get_beta_manual(Ycv_dm', Zcv_dm', zlist);

            % Squared error for each z at this fold
            Yhat_val       = Beta_cv' * Zval_dm + Ymn_cv;
            fold_err(k, :) = (Y(seq_val_cv) - Yhat_val').^2;
        end

        % Average MSE across folds, pick best z
        mean_cv_err       = mean(fold_err, 1);
        CVerr(t, p, :)    = mean_cv_err;
        [~, z_best]       = min(mean_cv_err);
        zSel(t, p)        = z_best;

        % ── Step 2: Full block → predict t ────────────────────────────────
        Ytrn = Y(seq_idx);
        Ztrn = Z(:, rnd_idx);

        % Test point: observation right after the random block (or real t if iboot==0)
        if iboot == 0
            t_tst = t;
        else
            t_tst = blk_start + trnwin;
        end
        Ztst = Z(:, t_tst);

        % Demean
        Ymn  = (demeanY == 1) * nanmean(Ytrn);
        Zmn  = (demeanX == 1) * nanmean(Ztrn, 2);
        Ytrn = Ytrn - Ymn;
        Ztrn = Ztrn - Zmn;
        Ztst = Ztst - Zmn;

        % Standardize
        Zstd = nanstd(Ztrn, [], 2) + 1e-6;
        Ztrn = Ztrn ./ Zstd;
        Ztst = Ztst ./ Zstd;

        % Fit selected z only
        [Beta_full, ~, ~, ~] = get_beta_manual(Ytrn', Ztrn', zlist(z_best));
        Yprd(t, p) = Beta_full' * Ztst + Ymn;
        Bnrm(t, p) = sum(Beta_full.^2);
    end
end
% --- Save ------------------------------------------------------------------
save_file = fullfile(output_dir, sprintf('Sim%d.mat', isim));
save(save_file, 'Yprd', 'Bnrm', 'zSel', 'CVerr', 'dates', 'zlist', 'nval');
end