%% table_rff_cv.m
%
% Builds ONE consolidated table for the RFF cross-validation model.
% Each row = one setting (window × demean combo × sub-period).
% Columns: Window, DemeanX, DemeanY, Period_beg, Period_end,
%          Sharpe_Ratio, OOS_R2, Corr_HistAvg,
%          Alpha1, Alpha1_tstat,
%          Alpha2, Alpha2_tstat,
%          Alpha3, Alpha3_tstat
%
% Alpha1 : RFF-CV portfolio regressed on Market
% Alpha2 : RFF-CV portfolio regressed on [Market, HistAvg]
% Alpha3 : RFF-CV portfolio regressed on [Market, HistAvg, Simple_z1000]
%          where Simple_z1000 always comes from the dX=1, dY=1 fixed model
%          (same reference convention as table_TC__1_.m)

clear; clc;

% ===========================================================================
%  PARAMETERS  (mirror run_rff_cv.m and table_TC__1_.m)
% ===========================================================================
vol_stdize = 1;
gamma      = 2;
num_sims   = 100;
windows    = [12, 60, 120];
nval       = 1;            % must match what was used in rff_cv.m

demean_settings = [1, 1;   % setting 1
                   0, 0;   % setting 2
                   1, 0];  % setting 3

% Fixed demean reference for the Simple z=1000 alpha3 regressor
ref_demX = 1;
ref_demY = 1;

zlist_simple = [0, logspace(0, 4, 5), 1e9];   % used by the simple/fixed model
z1000_idx    = find(zlist_simple == 1000, 1);

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
tbl_A3    = [];  tbl_tA3 = [];

% Also collect benchmark rows (Market + HistAvg) for reference
bm_rows = {};

% ===========================================================================
%  MAIN LOOP
% ===========================================================================
SEP      = repmat('-', 1, 60);
t_global = tic;

for w = 1:nW
    trnwin     = windows(w);
    Y_bar_full = rolling_mean(Y_std, trnwin);

    % ── Load Simple z=1000 from the FIXED reference setting (alpha3 regressor)
    ref_str  = sprintf('vol%d_win%d_dX%d_dY%d', vol_stdize, trnwin, ref_demX, ref_demY);
    ref_path = fullfile('raw_data', 'simple_fix', ref_str, 'result.mat');
    if ~exist(ref_path, 'file')
        warning('Missing reference Simple file: %s', ref_path);
        Yprd_ref_full = nan(T_full, length(zlist_simple));
    else
        Rtmp          = load(ref_path, 'Yprd');
        Yprd_ref_full = Rtmp.Yprd;     % (T_full x nL_simple)
    end
    fprintf('  Reference Simple (dX=%d,dY=%d) loaded for window=%d\n', ...
            ref_demX, ref_demY, trnwin);

    for m = 1:nM
        demeanX = demean_settings(m, 1);
        demeanY = demean_settings(m, 2);

        fprintf('\n%s\n  trnwin = %3d  |  demeanX = %d  |  demeanY = %d\n%s\n', ...
                SEP, trnwin, demeanX, demeanY, SEP);

        % ── Load and average RFF-CV predictions across sims ──────────────────
        %   rff_cv.m saves Yprd as (T x nP) with nP=1 (only P=12000).
        %   We extract column 1 from each sim and average.
        cv_str  = sprintf('gam%d_vol%d_win%d_nval%d_dX%d_dY%d_CV', ...
                          gamma, vol_stdize, trnwin, nval, demeanX, demeanY);
        cv_dir  = fullfile('raw_data', 'rff_cv', cv_str);

        Yall_cv  = nan(T_full, num_sims);
        n_loaded = 0;
        for isim = 1:num_sims
            cv_path = fullfile(cv_dir, sprintf('Sim%d.mat', isim));
            if exist(cv_path, 'file')
                Ctmp = load(cv_path, 'Yprd');
                % Yprd is (T x nP); nP=1 here, take column 1
                Yall_cv(:, isim) = Ctmp.Yprd(:, 1);
                n_loaded = n_loaded + 1;
            end
        end

        if n_loaded == 0
            warning('No RFF-CV files found in: %s', cv_dir);
            Yprd_cv_full = nan(T_full, 1);
        else
            Yprd_cv_full = nanmean(Yall_cv, 2);   % (T_full x 1)
        end
        fprintf('  RFF-CV sims loaded: %d / %d\n', n_loaded, num_sims);

        % ── Sub-period loop ───────────────────────────────────────────────────
        for sp = 1:n_sp
            beg_yr = subbeg(sp);
            end_yr = subend(sp);

            full_eval = (trnwin + 1) : T_full;
            sp_mask   = years_full(full_eval) >= beg_yr & ...
                        years_full(full_eval) <= end_yr;
            eval_idx  = full_eval(sp_mask);

            if isempty(eval_idx)
                continue;
            end
            fprintf('  [%d-%d]  n_eval = %d\n', beg_yr, end_yr, length(eval_idx));

            Y_eval    = Y_std(eval_idx);
            Ybar_eval = Y_bar_full(eval_idx);
            Yprd_cv   = Yprd_cv_full(eval_idx);          % (n_eval x 1)

            % Benchmark portfolios
            port_mkt  = Y_eval;
            port_hist = tc_ret(Ybar_eval, Y_eval, tc_rate);

            % Simple z=1000 portfolio from fixed reference — alpha3 regressor
            ps_z1000_ref      = Yprd_ref_full(eval_idx, z1000_idx);
            simple_port_z1000 = tc_ret(ps_z1000_ref, Y_eval, tc_rate);

            % ── Benchmark stats (Market & HistAvg) ──────────────────────────
            bm_sr_mkt = SR_calc(port_mkt,  freq);
            bm_sr_hst = SR_calc(port_hist, freq);
            [bm_a1_hst, bm_ta1_hst] = reg_alpha(port_hist, port_mkt, 4);
            bm_rows{end+1} = {'Market',  trnwin, demeanX, demeanY, beg_yr, end_yr, ...
                              NaN,  bm_sr_mkt, NaN,  NaN, NaN, NaN, NaN, NaN, NaN}; %#ok<SAGROW>
            bm_rows{end+1} = {'HistAvg', trnwin, demeanX, demeanY, beg_yr, end_yr, ...
                              0, bm_sr_hst, NaN, bm_a1_hst, bm_ta1_hst, NaN, NaN, NaN, NaN}; %#ok<SAGROW>

            % ── RFF-CV portfolio stats ────────────────────────────────────────
            pret_cv = tc_ret(Yprd_cv, Y_eval, tc_rate);
            v       = ~isnan(pret_cv);

            if sum(v) < 10
                warning('  Too few valid obs [%d-%d], skipping.', beg_yr, end_yr);
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

            % Alpha3 — add Simple z=1000 as third regressor
            vs3 = v & ~isnan(simple_port_z1000);
            if sum(vs3) > 10
                [a3, ta3] = reg_alpha(pret_cv(vs3), ...
                    [port_mkt(vs3), port_hist(vs3), simple_port_z1000(vs3)], 4);
            else
                a3 = NaN;  ta3 = NaN;
            end

            % Accumulate row
            tbl_Win(end+1)  = trnwin;  %#ok<*AGROW>
            tbl_DemX(end+1) = demeanX;
            tbl_DemY(end+1) = demeanY;
            tbl_Pbeg(end+1) = beg_yr;
            tbl_Pend(end+1) = end_yr;
            tbl_SR(end+1)   = sr;
            tbl_R2(end+1)   = r2;
            tbl_Corr(end+1) = corr_hist;
            tbl_A1(end+1)   = a1;   tbl_tA1(end+1) = ta1;
            tbl_A2(end+1)   = a2;   tbl_tA2(end+1) = ta2;
            tbl_A3(end+1)   = a3;   tbl_tA3(end+1) = ta3;

        end  % sub-period
    end  % demean setting
end  % window

fprintf('\nAll computations done in %.2f min.\n', toc(t_global) / 60);

% ===========================================================================
%  SAVE TABLES
% ===========================================================================
fprintf('\nWriting tables ...\n');
output_dir = fullfile('tables', 'cv');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% ── Main consolidated table (one row per setting) ──────────────────────────
hdr = {'Window', 'DemeanX', 'DemeanY', 'Period_beg', 'Period_end', ...
       'Sharpe_Ratio', 'OOS_R2', 'Corr_HistAvg', ...
       'Alpha1', 'Alpha1_tstat', ...
       'Alpha2', 'Alpha2_tstat', ...
       'Alpha3', 'Alpha3_tstat'};

n_rows = length(tbl_Win);
data   = [num2cell(tbl_Win(:)),  num2cell(tbl_DemX(:)), num2cell(tbl_DemY(:)), ...
          num2cell(tbl_Pbeg(:)), num2cell(tbl_Pend(:)), ...
          num2cell(tbl_SR(:)),   num2cell(tbl_R2(:)),   num2cell(tbl_Corr(:)), ...
          num2cell(tbl_A1(:)),   num2cell(tbl_tA1(:)), ...
          num2cell(tbl_A2(:)),   num2cell(tbl_tA2(:)), ...
          num2cell(tbl_A3(:)),   num2cell(tbl_tA3(:))];

main_file = fullfile(output_dir, 'RFF_CV_Results.xlsx');
writecell([hdr; data], main_file);
fprintf('  Saved: %s  (%d rows)\n', main_file, n_rows);

% ── Benchmark table ─────────────────────────────────────────────────────────
bm_hdr = {'Benchmark', 'Window', 'DemeanX', 'DemeanY', 'Period_beg', 'Period_end', ...
          'OOS_R2', 'Sharpe_Ratio', 'Corr_HistAvg', ...
          'Alpha1', 'Alpha1_tstat', 'Alpha2', 'Alpha2_tstat', ...
          'Alpha3', 'Alpha3_tstat'};

bm_file = fullfile(output_dir, 'Benchmarks.xlsx');
writecell([bm_hdr; vertcat(bm_rows{:})], bm_file);
fprintf('  Saved: %s\n', bm_file);

fprintf('\n%s\nAll tables written to "%s/"\n%s\n', ...
        repmat('=',1,60), output_dir, repmat('=',1,60));


% ===========================================================================
%  LOCAL HELPER FUNCTIONS
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