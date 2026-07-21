%% test_boot_cv.m
%
% Builds ONE consolidated table for the RFF cross-validation BOOTSTRAP model.
% Direct analogue of table_rff_cv.m, with an outer loop over bootstrap draws.
%
% Each row = one setting (iboot x window x demean combo x sub-period).
%   iboot = 0          -> real-data / no-resampling pass (the point estimate)
%   iboot = 1..num_boots -> bootstrap replicates (null / sampling distribution)
%
% Columns: iboot, Window, DemeanX, DemeanY, Period_beg, Period_end,
%          Sharpe_Ratio, OOS_R2, Corr_HistAvg,
%          Alpha1, Alpha1_tstat,
%          Alpha2, Alpha2_tstat
%
% Alpha1 : RFF-CV portfolio regressed on Market
% Alpha2 : RFF-CV portfolio regressed on [Market, HistAvg]
%
% Predictions are read from the layout written by run_rff_cv_boot3_par.m:
%   raw_data/rff_cv_boot3/<param_str>/boot<iboot>/Sim<isim>.mat
%   param_str = gam%d_vol%d_win%d_nval%d_dX%d_dY%d        (NOTE: no "_CV" suffix)

clear; clc;

% ===========================================================================
%  PARAMETERS  (mirror run_rff_cv_boot3_par.m and table_rff_cv.m)
% ===========================================================================
vol_stdize = 1;
gamma      = 2;
num_sims   = 100;
num_boots  = 100;          % iboot runs 0 : num_boots (0 = point estimate)
windows    = [12,60,120];
nval       = 1;            % must match what was used in rff_cv_boot3.m

demean_settings = [1, 1];

freq    = 12;
tc_rate = 0.0025;

subbeg = [1930, 1975, 1930];
subend = [1974, 2020, 2020];
n_sp   = length(subbeg);

nW = length(windows);
nM = size(demean_settings, 1);

% ===========================================================================
%  LOAD DATA
% ===========================================================================
fprintf('Loading GYdata.mat ...\n');
load('GYdata.mat');

if vol_stdize == 1
    Y2        = movmean(Y.^2, [11 0]);
    Y_std     = Y(37:end) ./ sqrt(Y2(36:end-1));
    dates_std = dates(37:end, :);
else
    Y_std     = Y;
    dates_std = dates;
end
T_full = length(Y_std);
fprintf('  T after vol-stdize = %d observations\n', T_full);

if isnumeric(dates_std)
    years_full = floor(dates_std(:, 1) / 100);
else
    years_full = str2double(cellstr(num2str(dates_std(:, 1:4))));
end

% ===========================================================================
%  ROW ACCUMULATORS
% ===========================================================================
tbl_Boot  = [];
tbl_Win   = [];
tbl_DemX  = [];
tbl_DemY  = [];
tbl_Pbeg  = [];
tbl_Pend  = [];
tbl_SR    = [];
tbl_R2    = [];
tbl_Corr  = [];
tbl_A1    = [];  tbl_tA1 = [];
tbl_A2    = [];  tbl_tA2 = [];

% Benchmark rows (Market + HistAvg). These are invariant to iboot, so they
% are accumulated once per (window, demean, sub-period).
bm_rows = {};

% ===========================================================================
%  MAIN LOOP
% ===========================================================================
SEP      = repmat('-', 1, 60);
t_global = tic;

for w = 1:nW
    trnwin     = windows(w);
    Y_bar_full = rolling_mean(Y_std, trnwin);

    for m = 1:nM
        demeanX = demean_settings(m, 1);
        demeanY = demean_settings(m, 2);

        % Folder convention written by run_rff_cv_boot3_par.m (no "_CV" suffix)
        boot_param_str = sprintf('gam%d_vol%d_win%d_nval%d_dX%d_dY%d', ...
                                 gamma, vol_stdize, trnwin, nval, demeanX, demeanY);

        fprintf('\n%s\n  trnwin = %3d  |  demeanX = %d  |  demeanY = %d\n%s\n', ...
                SEP, trnwin, demeanX, demeanY, SEP);

        % ── Benchmark rows (computed once; independent of iboot) ─────────────
        for sp = 1:n_sp
            beg_yr = subbeg(sp);
            end_yr = subend(sp);

            full_eval = (trnwin + 1) : T_full;
            sp_mask   = years_full(full_eval) >= beg_yr & ...
                        years_full(full_eval) <= end_yr;
            eval_idx  = full_eval(sp_mask);
            if isempty(eval_idx), continue; end

            Y_eval    = Y_std(eval_idx);
            Ybar_eval = Y_bar_full(eval_idx);

            port_mkt  = Y_eval;
            port_hist = tc_ret(Ybar_eval, Y_eval, tc_rate);

            bm_sr_mkt = SR_calc(port_mkt,  freq);
            bm_sr_hst = SR_calc(port_hist, freq);
            [bm_a1_hst, bm_ta1_hst] = reg_alpha(port_hist, port_mkt, 4);

            bm_rows{end+1} = {'Market',  trnwin, demeanX, demeanY, beg_yr, end_yr, ...
                              NaN, bm_sr_mkt, NaN, NaN, NaN, NaN, NaN}; %#ok<SAGROW>
            bm_rows{end+1} = {'HistAvg', trnwin, demeanX, demeanY, beg_yr, end_yr, ...
                              0, bm_sr_hst, NaN, bm_a1_hst, bm_ta1_hst, NaN, NaN}; %#ok<SAGROW>
        end

        % ── Bootstrap loop ───────────────────────────────────────────────────
        for iboot = 0:num_boots

            cv_dir = fullfile('raw_data', 'rff_cv_boot3', boot_param_str, ...
                              sprintf('boot%d', iboot));

            % Load and average RFF-CV predictions across sims for this boot.
            %   rff_cv_boot3.m saves Yprd as (T x nP) with nP=1 (only P=12000).
            Yall_cv  = nan(T_full, num_sims);
            n_loaded = 0;
            for isim = 1:num_sims
                cv_path = fullfile(cv_dir, sprintf('Sim%d.mat', isim));
                if exist(cv_path, 'file')
                    Ctmp = load(cv_path, 'Yprd');
                    Yall_cv(:, isim) = Ctmp.Yprd(:, 1);
                    n_loaded = n_loaded + 1;
                end
            end

            if n_loaded == 0
                warning('No RFF-CV boot files found in: %s', cv_dir);
                continue;
            end
            Yprd_cv_full = nanmean(Yall_cv, 2);   % (T_full x 1)

            fprintf('  boot %3d/%d | sims loaded: %3d / %d\n', ...
                    iboot, num_boots, n_loaded, num_sims);

            % ── Sub-period loop ───────────────────────────────────────────────
            for sp = 1:n_sp
                beg_yr = subbeg(sp);
                end_yr = subend(sp);

                full_eval = (trnwin + 1) : T_full;
                sp_mask   = years_full(full_eval) >= beg_yr & ...
                            years_full(full_eval) <= end_yr;
                eval_idx  = full_eval(sp_mask);
                if isempty(eval_idx), continue; end

                Y_eval    = Y_std(eval_idx);
                Ybar_eval = Y_bar_full(eval_idx);
                Yprd_cv   = Yprd_cv_full(eval_idx);          % (n_eval x 1)

                % Benchmark / reference portfolios
                port_mkt  = Y_eval;
                port_hist = tc_ret(Ybar_eval, Y_eval, tc_rate);

                % ── RFF-CV portfolio stats ────────────────────────────────────
                pret_cv = tc_ret(Yprd_cv, Y_eval, tc_rate);
                v       = ~isnan(pret_cv);

                if sum(v) < 10
                    warning('  Too few valid obs (boot %d, [%d-%d]), skipping.', ...
                            iboot, beg_yr, end_yr);
                    continue;
                end

                pret  = pret_cv(v);
                mr    = port_mkt(v);
                hr    = port_hist(v);
                Yv    = Y_eval(v);
                Yb    = Ybar_eval(v);
                Ypv   = Yprd_cv(v);

                sr   = SR_calc(pret, freq);
                r2   = oos_r2(Yv, Ypv, Yb);
                cc   = corrcoef(pret, hr);
                corr_hist = cc(1, 2);

                [a1, ta1] = reg_alpha(pret, mr,       4);
                [a2, ta2] = reg_alpha(pret, [mr, hr], 4);

                % Accumulate row
                tbl_Boot(end+1) = iboot;   %#ok<*AGROW>
                tbl_Win(end+1)  = trnwin;
                tbl_DemX(end+1) = demeanX;
                tbl_DemY(end+1) = demeanY;
                tbl_Pbeg(end+1) = beg_yr;
                tbl_Pend(end+1) = end_yr;
                tbl_SR(end+1)   = sr;
                tbl_R2(end+1)   = r2;
                tbl_Corr(end+1) = corr_hist;
                tbl_A1(end+1)   = a1;   tbl_tA1(end+1) = ta1;
                tbl_A2(end+1)   = a2;   tbl_tA2(end+1) = ta2;

            end  % sub-period
        end  % iboot
    end  % demean setting
end  % window

fprintf('\nAll computations done in %.2f min.\n', toc(t_global) / 60);

% ===========================================================================
%  SAVE TABLES
% ===========================================================================
fprintf('\nWriting tables ...\n');
output_dir = fullfile('tables', 'cv_boot3');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% ── Main consolidated table (one row per setting x bootstrap draw) ─────────
hdr = {'iboot', 'Window', 'DemeanX', 'DemeanY', 'Period_beg', 'Period_end', ...
       'Sharpe_Ratio', 'OOS_R2', 'Corr_HistAvg', ...
       'Alpha1', 'Alpha1_tstat', ...
       'Alpha2', 'Alpha2_tstat'};

n_rows = length(tbl_Win);
data   = [num2cell(tbl_Boot(:)), num2cell(tbl_Win(:)),  num2cell(tbl_DemX(:)), ...
          num2cell(tbl_DemY(:)), num2cell(tbl_Pbeg(:)), num2cell(tbl_Pend(:)), ...
          num2cell(tbl_SR(:)),   num2cell(tbl_R2(:)),   num2cell(tbl_Corr(:)), ...
          num2cell(tbl_A1(:)),   num2cell(tbl_tA1(:)), ...
          num2cell(tbl_A2(:)),   num2cell(tbl_tA2(:))];

main_file = fullfile(output_dir, 'RFF_CV_Boot3_Results.xlsx');
writecell([hdr; data], main_file);
fprintf('  Saved: %s  (%d rows)\n', main_file, n_rows);

% ── Benchmark table (invariant to iboot) ───────────────────────────────────
bm_hdr = {'Benchmark', 'Window', 'DemeanX', 'DemeanY', 'Period_beg', 'Period_end', ...
          'OOS_R2', 'Sharpe_Ratio', 'Corr_HistAvg', ...
          'Alpha1', 'Alpha1_tstat', 'Alpha2', 'Alpha2_tstat'};

bm_file = fullfile(output_dir, 'Benchmarks.xlsx');
writecell([bm_hdr; vertcat(bm_rows{:})], bm_file);
fprintf('  Saved: %s\n', bm_file);

fprintf('\n%s\nAll tables written to "%s/"\n%s\n', ...
        repmat('=',1,60), output_dir, repmat('=',1,60));


% ===========================================================================
%  LOCAL HELPER FUNCTIONS  (identical to table_rff_cv.m)
% ===========================================================================

function ret = tc_ret(w, Y, tc_rate)
    w    = w(:);  Y = Y(:);
    wlag = [NaN; w(1:end-1)];
    cost = tc_rate .* abs(w - wlag);
    ret  = (1 - cost) .* (1 + w .* Y) - 1;
end

function Y_bar = rolling_mean(Y_actual, trnwin)
    T_len = length(Y_actual);
    Y_bar = nan(T_len, 1);
    for t = trnwin + 1 : T_len
        Y_bar(t) = mean(Y_actual(t - trnwin : t - 1), 'omitnan');
    end
end

function r2 = oos_r2(Y, Yprd, Ybar)
    r2 = 1 - sum((Y - Yprd).^2) / sum((Y - Ybar).^2);
end

function sr = SR_calc(ret, freq)
    sr = mean(ret, 'omitnan') / std(ret, 'omitnan') * sqrt(freq);
end

function S = nw_meat(X, e, L)
    Xe = X .* e;
    S  = Xe' * Xe;
    for l = 1:L
        Gl = Xe(l+1:end,:)' * Xe(1:end-l,:);
        wl = 1 - l / (L + 1);
        S  = S + wl * (Gl + Gl');
    end
end

function [alpha, t_alpha] = reg_alpha(y, X_reg, nlags)
    valid   = ~isnan(y) & all(~isnan(X_reg), 2);
    y       = y(valid);
    X_reg   = X_reg(valid, :);
    T       = length(y);
    Xc      = [ones(T,1), X_reg];
    b       = (Xc'*Xc) \ (Xc'*y);
    e       = y - Xc*b;
    S       = nw_meat(Xc, e, nlags);
    XtX_i   = (Xc'*Xc) \ eye(size(Xc,2));
    V       = XtX_i * S * XtX_i;
    alpha   = b(1);
    t_alpha = b(1) / sqrt(max(V(1,1), 0));
end
