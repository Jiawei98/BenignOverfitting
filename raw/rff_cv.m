function [] = rff_cv(X, Y, dates, zlist, gamma, vol_stdize, trnwin, nval, demeanX, demeanY, isim, output_dir)
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
eDF   = nan(T, nP);
GCV   = nan(T, nP);
RSS   = nan(T, nP);
% --- Weight Mat ---
rng(isim);
W = randn(maxP, d);
% --- Recursive Estimation -------------------------------------------------
for p = 1:nP
    currP = floor(Plist(p) / 2);
    wtmp  = W(1:currP, :);
    Z     = [cos(gamma * wtmp * X); sin(gamma * wtmp * X)];

    for t = trnwin+1:T
        % ── Step 1: nval-fold walk-forward CV to select z ─────────────────
        % Fold k: train on (t-trnwin) to (t-nval+k-2), predict (t-nval+k-1)
        % k=1 → train trnwin-nval obs, predict t-nval
        % k=nval → train trnwin-1 obs,  predict t-1
        fold_err = zeros(nval, nL);
        for k = 1:nval
            val_t   = t - nval + k - 1;
            trn_loc = (t - trnwin) : (val_t - 1);
            Ycv  = Y(trn_loc);
            Zcv  = Z(:, trn_loc);
            Zval = Z(:, val_t);

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
            fold_err(k, :) = (Y(val_t) - Yhat_val').^2;
        end

        % Average MSE across folds, pick best z
        mean_cv_err       = mean(fold_err, 1);
        CVerr(t, p, :)    = mean_cv_err;
        [~, z_best]       = min(mean_cv_err);
        zSel(t, p)        = z_best;

        % ── Step 2: Full training window → predict t ──────────────────────
        trn_loc = (t - trnwin) : (t - 1);
        Ytrn    = Y(trn_loc);
        Ztrn    = Z(:, trn_loc);
        Ztst    = Z(:, t);

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
        [Beta_full, eDFtp, GCVtp, RSStp] = get_beta_manual(Ytrn', Ztrn', zlist(z_best));
        Yprd(t, p) = Beta_full' * Ztst + Ymn;
        Bnrm(t, p) = sum(Beta_full.^2);
        eDF(t, p)  = eDFtp;
        GCV(t, p)  = GCVtp;
        RSS(t, p)  = RSStp;
    end
end
% --- Save ------------------------------------------------------------------
save_file = fullfile(output_dir, sprintf('Sim%d.mat', isim));
save(save_file, 'Yprd', 'Bnrm', 'zSel', 'CVerr', 'eDF', 'GCV', 'RSS', 'dates', 'zlist', 'nval');
end