%% table_boot3_cv.m
%
% Bootstrap summary statistics for the RFF-CV model, read DIRECTLY from the
% consolidated results written by test_boot3_cv.m (no raw Sim reloading).
%
% Uses the FULL-SAMPLE sub-period (1930-2020) and demean = [dX1 dY1].
%
% For each window {12,60,120} we summarize the bootstrap sampling
% distribution of four statistics:
%     Sharpe_Ratio, OOS_R2, Alpha1_tstat (tA1), Alpha2_tstat (tA2)
%
% boot0 (iboot==0) is the point estimate; boots 1..N are the null draws.
%
% Output:
%   * One summary table per window  -> printed + saved as its own sheet.
%   * A final pooled row giving the boot0 percentile rank of tA1/tA2 against
%     the bootstrap draws POOLED across ALL windows (unconditional on window).

clear; clc;

% ===========================================================================
%  PARAMETERS
% ===========================================================================
results_file = fullfile('tables', 'cv_boot3', 'RFF_CV_Boot3_Results.xlsx');
windows      = [12, 60, 120];
demeanX      = 1;
demeanY      = 1;
sp_beg       = 1930;     % full-sample sub-period
sp_end       = 2020;

out_dir  = fullfile('tables', 'cv_boot3');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

SEP = repmat('=', 1, 70);

% ===========================================================================
%  LOAD CONSOLIDATED RESULTS  (from test_boot3_cv.m)
% ===========================================================================
fprintf('Reading %s ...\n', results_file);
R = readtable(results_file);

% Restrict to full-sample sub-period and the chosen demean setting.
keep = R.Period_beg == sp_beg & R.Period_end == sp_end & ...
       R.DemeanX   == demeanX & R.DemeanY   == demeanY;
R = R(keep, :);

% ===========================================================================
%  POOLED (cross-window) bootstrap draws for every statistic
% ===========================================================================
boot_mask = R.iboot >= 1;

% ===========================================================================
%  TABLE DEFINITION
% ===========================================================================
stat_names = {'Sharpe_Ratio', 'OOS_R2', 'Alpha1_tstat', 'Alpha2_tstat'};
col_labels = {'SR', 'OOS_R2', 'tAlpha1', 'tAlpha2'};

% Pooled (cross-window) bootstrap draws for every statistic.
pool = cell(1, numel(stat_names));
for s = 1 : numel(stat_names)
    vp = R.(stat_names{s})(boot_mask);
    pool{s} = vp(~isnan(vp));
end

row_labels = {'Original (boot0)'; 'boot0 pct rank'; 'Mean'; 'Median'; ...
              'Std'; '5th pct'; '95th pct'; 'boot0 pct rank (pooled)'};

pct_row          = false(numel(row_labels), 1);
pct_row([2 8])   = true;            % percentile-rank rows (print as %.1f)

% ===========================================================================
%  PER-WINDOW SUMMARIES
% ===========================================================================
for w = 1 : numel(windows)
    trnwin = windows(w);
    Rw     = R(R.Window == trnwin, :);

    is0 = Rw.iboot == 0;            % point estimate
    isB = Rw.iboot >= 1;            % bootstrap replicates

    nS  = numel(stat_names);
    blk = nan(numel(row_labels), nS);

    for s = 1 : nS
        v0 = Rw.(stat_names{s})(is0);
        vb = Rw.(stat_names{s})(isB);  vb = vb(~isnan(vb));
        if isempty(v0), v0 = NaN; else, v0 = v0(1); end

        blk(1, s) = v0;                       % Original (boot0)
        blk(2, s) = mean(vb <= v0) * 100;     % boot0 pct (this window)
        blk(3, s) = mean(vb);                 % Mean
        blk(4, s) = median(vb);               % Median
        blk(5, s) = std(vb);                  % Std
        blk(6, s) = prctile(vb,  5);          % 5th pct
        blk(7, s) = prctile(vb, 95);          % 95th pct
        if ~isnan(v0)
            blk(8, s) = mean(pool{s} <= v0) * 100;   % boot0 pct (pooled, all windows)
        end
    end

    % ── Console print ─────────────────────────────────────────────────────
    fprintf('\n%s\nwindow = %d   (sub-period %d-%d, dX%d dY%d)\n%s\n', ...
            SEP, trnwin, sp_beg, sp_end, demeanX, demeanY, SEP);
    fprintf('%-26s  %10s  %10s  %10s  %10s\n', '', col_labels{:});
    for r = 1 : numel(row_labels)
        fprintf('%-26s', row_labels{r});
        for s = 1 : nS
            if pct_row(r)
                fprintf('  %10.1f', blk(r, s));
            else
                fprintf('  %10.4f', blk(r, s));
            end
        end
        fprintf('\n');
    end

    % ── Save as a per-window CSV ─────────────────────────────────────────
    Tw = cell2table([row_labels, num2cell(blk)], ...
                    'VariableNames', [{'Statistic'}, col_labels]);
    csv_file = fullfile(out_dir, sprintf('Boot_Summary_win%d.csv', trnwin));
    writetable(Tw, csv_file);
    fprintf('  saved: %s\n', csv_file);
end

fprintf('\n%s\nSaved 3 per-window CSV tables to: %s/\n%s\n', SEP, out_dir, SEP);