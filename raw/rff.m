function [] = rff(X, Y, dates, zlist, gamma, vol_stdize, trnwin, demeanX, demeanY, isim, output_dir)
%fprintf('--- Starting rff seed: %d ---\n', iSim);
% --- Grid for P (Number of Features) ---
maxP        = 12000;
Plist       = round(logspace(log10(2), log10(maxP), 100));
nP          = length(Plist);
nL          = length(zlist);

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
Yprd = nan(T, nP, nL);
Bnrm = nan(T, nP, nL);
eDF  = nan(T, nP, nL);
GCV  = nan(T, nP, nL);
RSS  = nan(T, nP, nL);

% --- Weight Mat ---
rng(isim);
W = randn(maxP, d);

% --- Recursive Estimation -------------------------------------------------
for p = 1:nP
    currP = floor(Plist(p)/2);
    wtmp  = W(1:currP, :);
    Z     = [cos(gamma*wtmp*X); sin(gamma*wtmp*X)];

    for t = trnwin+1:T

        % ── Y training window: always the standard rolling window ──────────
        Ytrn_loc = (t-trnwin):t-1;
        Ytrn     = Y(Ytrn_loc);
        Ztrn     = Z(:, Ytrn_loc);
        Ztst     = Z(:, t);
        % Demean
        Ymn  = (demeanY == 1) * nanmean(Ytrn);
        Zmn  = (demeanX == 1) * nanmean(Ztrn, 2);

        Ytrn = Ytrn - Ymn;
        Ztrn = Ztrn - Zmn;
        Ztst = Ztst - Zmn;

        % Standardize features
        Zstd = nanstd(Ztrn, [], 2) + 1e-6;
        Ztrn = Ztrn ./ Zstd;
        Ztst = Ztst ./ Zstd;

        % Training
        [Betatp, eDFtp, GCVtp, RSStp] = get_beta_manual(Ytrn', Ztrn', zlist);

        % Record Results
        Yprd(t, p, :) = Betatp' * Ztst + Ymn;
        Bnrm(t, p, :) = sum(Betatp.^2);
        eDF(t, p, :)  = eDFtp;
        GCV(t, p, :)  = GCVtp;
        RSS(t, p, :)  = RSStp;
    end
end

% --- Finalization & Saving ------------------------------------------------
save_file = fullfile(output_dir, sprintf('Sim%d.mat', isim));
if isim == 1
    save(save_file);
else
    save(save_file, 'Yprd', 'Bnrm', 'eDF', 'GCV', 'RSS');
end
end
