% table_fix.m
%
% Three demean settings:
%   Setting 1: demeanX=1, demeanY=1
%   Setting 2: demeanX=0, demeanY=0
%   Setting 3: demeanX=1, demeanY=0
%
% Alpha3 always uses the Simple model at z=1000 from the FIXED reference
% demean setting (ref_demX=1, ref_demY=1), regardless of the current loop's
% demeanX/demeanY.

clear; clc;

% ===========================================================================
%  PARAMETERS
% ===========================================================================
vol_stdize    = 1;
windows       = [12, 60, 120];

% Each row = [demeanX, demeanY]
demean_settings = [1, 1;   % setting 1
                   0, 0;   % setting 2
                   1, 0];  % setting 3

% Fixed demean setting for the Simple z=1000 regressor in alpha3
ref_demX = 1;
ref_demY = 1;

zlist         = [0, logspace(0, 4, 5), 1e9];
nL            = length(zlist);

gamma         = 2;
num_sims      = 100;
maxP          = 12000;
Plist         = round(logspace(log10(2), log10(maxP), 100));
nP_grid       = length(Plist);
p_idx         = nP_grid;

freq          = 12;
tc_rate       = 0.0025;
nW            = length(windows);
nM            = size(demean_settings, 1);

subbeg  = [1930, 1975, 1930];
subend  = [1974, 2020, 2020];
n_sp    = length(subbeg);

z1000_idx = find(zlist == 1000, 1);

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
%  COLUMN LABELS
% ===========================================================================
z_labels = cell(1, nL);
for l = 1:nL
    z = zlist(l);
    if     z == 0,   z_labels{l} = 'z_0';
    elseif z >= 1e8, z_labels{l} = 'z_Inf';
    else,            z_labels{l} = sprintf('z_%g', z);
    end
end

% ===========================================================================
%  ROW ACCUMULATORS
% ===========================================================================
row_Model = {};
row_Win   = []; row_DemX = []; row_DemY = []; row_Pbeg = []; row_Pend = [];

R2_rows   = zeros(0, nL);
SR_rows   = zeros(0, nL);
Ta1_rows  = zeros(0, nL);  Ta2_rows = zeros(0, nL);  Ta3_rows = zeros(0, nL);
A1_rows   = zeros(0, nL);  A2_rows  = zeros(0, nL);  A3_rows  = zeros(0, nL);
Corr_rows = zeros(0, nL);

bm_Win = []; bm_DemX = []; bm_DemY = []; bm_Pbeg = []; bm_Pend = [];
bm_MktSR = []; bm_MktR2 = [];
bm_HstSR = []; bm_HstR2 = []; bm_HstTa1 = [];

% ===========================================================================
%  MAIN LOOP
% ===========================================================================
SEP      = repmat('-', 1, 60);
t_global = tic;

for w = 1:nW
    trnwin     = windows(w);
    Y_bar_full = rolling_mean(Y_std, trnwin);

    % -- Load fixed Simple predictions (dX=1, dY=1) for alpha3 regressor ----
    % Alpha3 always controls for the Simple z=1000 portfolio from Setting I,
    % regardless of which demean setting is currently being evaluated.
    ps_str_fixed  = sprintf('vol%d_win%d_dX1_dY1', vol_stdize, trnwin);
    s_path_fixed  = fullfile('raw_data', 'simple_fix', ps_str_fixed, 'result.mat');
    if ~exist(s_path_fixed, 'file')
        warning('Missing fixed Simple (dX=1,dY=1) file: %s', s_path_fixed);
        Yprd_s_fixed_full = nan(T_full, nL);
    else
        Stmp_fixed        = load(s_path_fixed, 'Yprd');
        Yprd_s_fixed_full = Stmp_fixed.Yprd;   % (T_full x nL)
    end

    % ── Load Simple z=1000 from FIXED reference setting (once per window) ──
    % Used as the regressor in alpha3 for ALL demean loop iterations.
    ref_str       = sprintf('vol%d_win%d_dX%d_dY%d', ...
                            vol_stdize, trnwin, ref_demX, ref_demY);
    ref_path      = fullfile('raw_data', 'simple_fix', ref_str, 'result.mat');
    if ~exist(ref_path, 'file')
        warning('Missing reference Simple file: %s', ref_path);
        Yprd_ref_full = nan(T_full, nL);
    else
        Rtmp2         = load(ref_path, 'Yprd');
        Yprd_ref_full = Rtmp2.Yprd;              % (T_full x nL)
    end
    fprintf('  Reference Simple (dX=%d,dY=%d) loaded for window=%d\n', ...
            ref_demX, ref_demY, trnwin);

    for m = 1:nM
        demeanX = demean_settings(m, 1);
        demeanY = demean_settings(m, 2);

        fprintf('\n%s\n  trnwin = %3d  |  demeanX = %d  |  demeanY = %d\n%s\n', ...
                SEP, trnwin, demeanX, demeanY, SEP);

        % -- Load Simple predictions (current demean setting) ---------------
        ps_str = sprintf('vol%d_win%d_dX%d_dY%d', ...
                         vol_stdize, trnwin, demeanX, demeanY);
        s_path = fullfile('raw_data', 'simple_fix', ps_str, 'result.mat');
        if ~exist(s_path, 'file')
            warning('Missing Simple file: %s', s_path);
            Yprd_s_full = nan(T_full, nL);
        else
            Stmp        = load(s_path, 'Yprd');
            Yprd_s_full = Stmp.Yprd;
        end

        % -- Load & average RFF predictions, P=12000 ------------------------
        pr_str  = sprintf('gam%d_vol%d_win%d_dX%d_dY%d', ...
                          gamma, vol_stdize, trnwin, demeanX, demeanY);
        rff_dir = fullfile('raw_data', 'rff_fix', pr_str);

        Yall_r   = nan(T_full, nL, num_sims);
        n_loaded = 0;
        for isim = 1:num_sims
            rff_path = fullfile(rff_dir, sprintf('Sim%d.mat', isim));
            if exist(rff_path, 'file')
                Rtmp = load(rff_path, 'Yprd');
                Yall_r(:, :, isim) = squeeze(Rtmp.Yprd(:, p_idx, :));
                n_loaded = n_loaded + 1;
            end
        end
        if n_loaded == 0
            warning('No RFF files found: %s', rff_dir);
            Yprd_r_full = nan(T_full, nL);
        else
            Yprd_r_full = nanmean(Yall_r, 3);
        end
        fprintf('  RFF sims loaded: %d / %d\n', n_loaded, num_sims);

        % -- Sub-period loop ------------------------------------------------
        for sp = 1:n_sp
            beg_yr = subbeg(sp);
            end_yr = subend(sp);

            full_eval = (trnwin + 1) : T_full;
            sp_mask   = years_full(full_eval) >= beg_yr & ...
                        years_full(full_eval) <= end_yr;
            eval_idx  = full_eval(sp_mask);

            if isempty(eval_idx), continue; end
            fprintf('  [%d-%d]  n_eval = %d\n', beg_yr, end_yr, length(eval_idx));

            Y_eval    = Y_std(eval_idx);
            Ybar_eval = Y_bar_full(eval_idx);
            Yprd_s    = Yprd_s_full(eval_idx, :);
            Yprd_r    = Yprd_r_full(eval_idx, :);

            % Benchmark portfolios with TC
            port_mkt  = Y_eval;
            port_hist = tc_ret(Ybar_eval, Y_eval, tc_rate);

            % Simple z=1000 portfolio — always from dX=1,dY=1 Setting I (alpha3 regressor)
            ps_z1000_fixed    = Yprd_s_fixed_full(eval_idx, z1000_idx);
            simple_port_z1000 = tc_ret(ps_z1000_fixed, Y_eval, tc_rate);

            % Benchmark stats
            [~, bm_hst_ta1] = reg_alpha(port_hist, port_mkt, 4);

            bm_Win(end+1)    = trnwin; %#ok<*AGROW>
            bm_DemX(end+1)   = demeanX;
            bm_DemY(end+1)   = demeanY;
            bm_Pbeg(end+1)   = beg_yr;
            bm_Pend(end+1)   = end_yr;
            bm_MktSR(end+1)  = SR_calc(port_mkt,  freq);
            bm_MktR2(end+1)  = NaN;
            bm_HstSR(end+1)  = SR_calc(port_hist, freq);
            bm_HstR2(end+1)  = 0;
            bm_HstTa1(end+1) = bm_hst_ta1;

            % Per-z results
            r2_s  = nan(1,nL); r2_r  = nan(1,nL);
            sr_s  = nan(1,nL); sr_r  = nan(1,nL);
            ta1_s = nan(1,nL); ta1_r = nan(1,nL);
            ta2_s = nan(1,nL); ta2_r = nan(1,nL);
            ta3_r = nan(1,nL);
            a1_s  = nan(1,nL); a1_r  = nan(1,nL);
            a2_s  = nan(1,nL); a2_r  = nan(1,nL);
            a3_r  = nan(1,nL);
            cr_s  = nan(1,nL); cr_r  = nan(1,nL);

            for l = 1:nL

                % -- Simple --------------------------------------------------
                ps     = Yprd_s(:, l);
                pret_s = tc_ret(ps, Y_eval, tc_rate);
                vs     = ~isnan(pret_s);
                if sum(vs) > 10
                    pret = pret_s(vs);
                    mr   = port_mkt(vs);
                    hr   = port_hist(vs);
                    r2_s(l)  = oos_r2(Y_eval(vs), ps(vs), Ybar_eval(vs));
                    sr_s(l)  = SR_calc(pret, freq);
                    [a1_s(l), ta1_s(l)] = reg_alpha(pret, mr,       4);
                    [a2_s(l), ta2_s(l)] = reg_alpha(pret, [mr, hr], 4);
                    cc = corrcoef(pret, hr);
                    cr_s(l) = cc(1, 2);
                end

                % -- RFF -----------------------------------------------------
                pr     = Yprd_r(:, l);
                pret_r = tc_ret(pr, Y_eval, tc_rate);
                vr     = ~isnan(pret_r);
                if sum(vr) > 10
                    pret = pret_r(vr);
                    mr   = port_mkt(vr);
                    hr   = port_hist(vr);
                    r2_r(l)  = oos_r2(Y_eval(vr), pr(vr), Ybar_eval(vr));
                    sr_r(l)  = SR_calc(pret, freq);
                    [a1_r(l), ta1_r(l)] = reg_alpha(pret, mr,       4);
                    [a2_r(l), ta2_r(l)] = reg_alpha(pret, [mr, hr], 4);
                    cc = corrcoef(pret, hr);
                    cr_r(l) = cc(1, 2);

                    % Alpha3: RFF only — fixed-reference Simple z=1000
                    vs3 = vr & ~isnan(simple_port_z1000);
                    if sum(vs3) > 10
                        [a3_r(l), ta3_r(l)] = reg_alpha( ...
                            pret_r(vs3), ...
                            [port_mkt(vs3), port_hist(vs3), simple_port_z1000(vs3)], 4);
                    end
                end

            end  % z

            % Append one row per model
            for mdl_id = 1:2
                if mdl_id == 1
                    tag   = 'Simple';
                    r2_v  = r2_s;  sr_v  = sr_s;
                    ta1_v = ta1_s; ta2_v = ta2_s; ta3_v = nan(1,nL);
                    a1_v  = a1_s;  a2_v  = a2_s;  a3_v  = nan(1,nL);
                    cr_v  = cr_s;
                else
                    tag   = 'RFF';
                    r2_v  = r2_r;  sr_v  = sr_r;
                    ta1_v = ta1_r; ta2_v = ta2_r; ta3_v = ta3_r;
                    a1_v  = a1_r;  a2_v  = a2_r;  a3_v  = a3_r;
                    cr_v  = cr_r;
                end

                row_Model{end+1} = tag;
                row_Win(end+1)   = trnwin;
                row_DemX(end+1)  = demeanX;
                row_DemY(end+1)  = demeanY;
                row_Pbeg(end+1)  = beg_yr;
                row_Pend(end+1)  = end_yr;

                R2_rows   = [R2_rows;   r2_v];
                SR_rows   = [SR_rows;   sr_v];
                Ta1_rows  = [Ta1_rows;  ta1_v];
                Ta2_rows  = [Ta2_rows;  ta2_v];
                Ta3_rows  = [Ta3_rows;  ta3_v];
                A1_rows   = [A1_rows;   a1_v];
                A2_rows   = [A2_rows;   a2_v];
                A3_rows   = [A3_rows;   a3_v];
                Corr_rows = [Corr_rows; cr_v];
            end

        end  % sub-period
    end  % demean setting
end  % window

fprintf('\n\nAll computations done in %.2f min.\n', toc(t_global) / 60);

% ===========================================================================
%  SAVE TABLES
% ===========================================================================
fprintf('\nWriting tables ...\n');
output_dir = fullfile('tables', 'fix');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

prefix_hdr  = {'Model','Window','DemeanX','DemeanY','Period_beg','Period_end'};
prefix_data = [row_Model(:), num2cell(row_Win(:)), ...
               num2cell(row_DemX(:)), num2cell(row_DemY(:)), ...
               num2cell(row_Pbeg(:)), num2cell(row_Pend(:))];

save_table(R2_rows,   prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'OOS_R2.xlsx'));
save_table(SR_rows,   prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Sharpe_Ratio.xlsx'));
save_table(Ta1_rows,  prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Alpha1_tstat.xlsx'));
save_table(Ta2_rows,  prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Alpha2_tstat.xlsx'));
save_table(Ta3_rows,  prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Alpha3_tstat.xlsx'));
save_table(A1_rows,   prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Alpha1_estimate.xlsx'));
save_table(A2_rows,   prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Alpha2_estimate.xlsx'));
save_table(A3_rows,   prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Alpha3_estimate.xlsx'));
save_table(Corr_rows, prefix_data, prefix_hdr, z_labels, fullfile(output_dir,'Corr_HistAvg.xlsx'));

% Benchmark table
bm_hdr = {'Benchmark','Window','DemeanX','DemeanY','Period_beg','Period_end', ...
          'OOS_R2','Sharpe_Ratio','Alpha1_tstat'};
n_bm   = length(bm_Win);
bm_rows_mkt = [repmat({'Market'},  n_bm, 1), ...
               num2cell(bm_Win(:)), num2cell(bm_DemX(:)), num2cell(bm_DemY(:)), ...
               num2cell(bm_Pbeg(:)), num2cell(bm_Pend(:)), ...
               num2cell(bm_MktR2(:)), num2cell(bm_MktSR(:)), repmat({NaN},n_bm,1)];
bm_rows_hst = [repmat({'HistAvg'}, n_bm, 1), ...
               num2cell(bm_Win(:)), num2cell(bm_DemX(:)), num2cell(bm_DemY(:)), ...
               num2cell(bm_Pbeg(:)), num2cell(bm_Pend(:)), ...
               num2cell(bm_HstR2(:)), num2cell(bm_HstSR(:)), num2cell(bm_HstTa1(:))];
bm_all = cell(2*n_bm, length(bm_hdr));
bm_all(1:2:end,:) = bm_rows_mkt;
bm_all(2:2:end,:) = bm_rows_hst;
writecell([bm_hdr; bm_all], fullfile(output_dir,'Benchmarks.xlsx'));
fprintf('  Saved: Benchmarks.xlsx\n');

fprintf('\n%s\nAll tables written to "%s/"\n%s\n', SEP, output_dir, SEP);


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
    valid  = ~isnan(y) & all(~isnan(X_reg), 2);
    y      = y(valid);
    X_reg  = X_reg(valid, :);
    T      = length(y);
    Xc     = [ones(T,1), X_reg];
    b      = (Xc'*Xc) \ (Xc'*y);
    e      = y - Xc*b;
    S      = nw_meat(Xc, e, nlags);
    XtX_i  = (Xc'*Xc) \ eye(size(Xc,2));
    V      = XtX_i * S * XtX_i;
    alpha   = b(1);
    t_alpha = b(1) / sqrt(max(V(1,1), 0));
end

function save_table(mat, prefix_data, prefix_hdr, z_labels, fname)
    hdr  = [prefix_hdr(:)', z_labels(:)'];
    data = [prefix_data, num2cell(mat)];
    writecell([hdr; data], fname);
    fprintf('  Saved: %s\n', fname);
end
