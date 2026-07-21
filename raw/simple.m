function [] = simple(X, Y, dates, vol_stdize, trnwin, demeanX, demeanY, output_dir)
% simple.m — OLS prediction using raw X features (no RFF transformation)
% No isim parameter needed — no randomness beyond bootstrap

% --- Grid for Lambda Values ---
zlist = [0, logspace(0, 4, 5), 1e9];
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

% --- Output Space (no P dimension — using raw X directly) ----------------
nX   = size(X, 1);   % number of original features
Yprd = nan(T, nL);
Bnrm = nan(T, nL);
eDF  = nan(T, nL);
GCV  = nan(T, nL);
RSS  = nan(T, nL);

% --- Recursive Estimation ------------------------------------------------
for t = trnwin+1:T

    % ── Y: always standard rolling window ─────────────────────────────────
    Ytrn_loc = (t-trnwin):(t-1);
    Ytrn     = Y(Ytrn_loc);
    Xtrn = X(:, Ytrn_loc);
    Xtst = X(:, t);

    % Demean
    Ymn  = (demeanY == 1) * nanmean(Ytrn);
    Xmn  = (demeanX == 1) * nanmean(Xtrn, 2);
    Ytrn = Ytrn - Ymn;
    Xtrn = Xtrn - Xmn;
    Xtst = Xtst - Xmn;

    % Standardize features
    Xstd = nanstd(Xtrn, [], 2) + 1e-6;
    Xtrn = Xtrn ./ Xstd;
    Xtst = Xtst ./ Xstd;

    % Training
    [Betatp, eDFtp, GCVtp, RSStp] = get_beta_manual(Ytrn', Xtrn', zlist);

    % Record Results
    Yprd(t, :) = Betatp' * Xtst + Ymn;
    Bnrm(t, :) = sum(Betatp.^2);
    eDF(t, :)  = eDFtp;
    GCV(t, :)  = GCVtp;
    RSS(t, :)  = RSStp;
end

% --- Finalization & Saving -----------------------------------------------
save_file = fullfile(output_dir,'result.mat');
save(save_file);   % save everything for raw data run
end