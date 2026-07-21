%% z_frequency_table.m
%
% For demeanX=1, demeanY=1:
%   1. Load zSel from every sim file → (T x num_sims)
%   2. Pool ALL z values (across all sims and all time points)
%   3. Bin pooled z into log10 buckets → frequency table
%   4. Report mean then median of pooled z
%   5. Load eDF from every sim file → mean then median of pooled eDF
%   6. Report single percentage of all raw z > 10^7 (no aggregation at all)

clear; clc;

% ===========================================================================
%  PARAMETERS
% ===========================================================================
gamma      = 2;
vol_stdize = 1;
num_sims   = 100;
nval       = 1;
windows    = [12, 60, 120];
zlist      = logspace(-9, 9, 100);
demeanX    = 1;
demeanY    = 1;

edges = -9 : 1 : 9;
bin_labels = cell(length(edges)-1, 1);
for b = 1 : length(edges)-1
    bin_labels{b} = sprintf('[%3d, %3d)', edges(b), edges(b+1));
end
nBins = length(bin_labels);

% ===========================================================================
%  LOAD DATA (for T)
% ===========================================================================
load('GYdata.mat');
if vol_stdize == 1
    Y2 = movmean(Y.^2, [11 0]);
    Y  = Y(37:end) ./ sqrt(Y2(36:end-1));
end
T = length(Y);

% ===========================================================================
%  LOOP OVER WINDOWS
% ===========================================================================
bin_counts_all = zeros(nBins, length(windows));
total_sel      = zeros(1,     length(windows));
mean_z         = zeros(1,     length(windows));
median_z       = zeros(1,     length(windows));
mean_edf       = zeros(1,     length(windows));
median_edf     = zeros(1,     length(windows));
pct_above_1e7  = zeros(1,     length(windows));

for w = 1 : length(windows)
    trnwin    = windows(w);
    param_str = sprintf('gam%d_vol%d_win%d_nval%d_dX%d_dY%d_CV', ...
                        gamma, vol_stdize, trnwin, nval, demeanX, demeanY);
    sim_dir   = fullfile('raw_data', 'rff_cv', param_str);

    % ── Step 1: collect z and eDF values across sims → (T x num_sims) ──────
    zval_all = nan(T, num_sims);
    edf_all  = nan(T, num_sims);
    n_loaded = 0;

    for isim = 1 : num_sims
        sim_file = fullfile(sim_dir, sprintf('Sim%d.mat', isim));
        if ~exist(sim_file, 'file'), continue; end

        S    = load(sim_file, 'zSel', 'eDF');
        zidx = S.zSel(:, end);

        valid        = ~isnan(zidx);
        zvals        = nan(T, 1);
        zvals(valid) = zlist(zidx(valid));

        zval_all(:, isim) = zvals;
        edf_all(:, isim)  = S.eDF(:, end);
        n_loaded = n_loaded + 1;
    end

    fprintf('win=%3d  sims loaded: %d/%d\n', trnwin, n_loaded, num_sims);

    % ── Step 2: pool ALL z and eDF values (no aggregation across sims) ──────
    zval_pooled = zval_all(:);
    zval_pooled = zval_pooled(~isnan(zval_pooled));

    edf_pooled  = edf_all(:);
    edf_pooled  = edf_pooled(~isnan(edf_pooled));

    if isempty(zval_pooled)
        warning('No valid z for win=%d', trnwin);
        continue;
    end

    % ── Step 3: mean and median of pooled z ─────────────────────────────────
    mean_z(w)   = mean(zval_pooled);
    median_z(w) = median(zval_pooled);

    % ── Step 4: mean and median of pooled eDF ───────────────────────────────
    mean_edf(w)   = mean(edf_pooled);
    median_edf(w) = median(edf_pooled);

    % ── Step 5: single pct of ALL raw z > 10^7 (no aggregation at all) ──────
    pct_above_1e7(w) = mean(zval_pooled > 1e7) * 100;

    % ── Step 6: bin pooled z on log10 scale ─────────────────────────────────
    bc = histcounts(log10(zval_pooled), edges);
    bin_counts_all(:, w) = bc(:);
    total_sel(w)         = sum(bc);
end

% ── Percentages ─────────────────────────────────────────────────────────────
pct_all = zeros(nBins, length(windows));
for w = 1 : length(windows)
    if total_sel(w) > 0
        pct_all(:, w) = bin_counts_all(:, w) / total_sel(w) * 100;
    end
end

% ── Print table ──────────────────────────────────────────────────────────────
col_w = 24;
sep   = repmat('=', 1, 20 + col_w * length(windows));

fprintf('\n%s\n', sep);
fprintf('  TABLE: Both Demeaned (demeanX=1, demeanY=1)\n');
fprintf('  (z pooled across all %d sims and all time points, then binned)\n', num_sims);
fprintf('%s\n', sep);

fprintf('  %-16s', 'log10(z) bin');
for w = 1 : length(windows)
    fprintf('  %-22s', sprintf('Window = %d', windows(w)));
end
fprintf('\n');

fprintf('  %-16s', '');
for w = 1 : length(windows)
    fprintf('  %-10s  %-10s', 'Count', 'Pct (%)');
end
fprintf('\n');

fprintf('  %s\n', repmat('-', 1, 16 + col_w * length(windows)));

for b = 1 : nBins
    fprintf('  %-16s', bin_labels{b});
    for w = 1 : length(windows)
        fprintf('  %-10d  %-10.2f', bin_counts_all(b,w), pct_all(b,w));
    end
    fprintf('\n');
end

fprintf('  %s\n', repmat('-', 1, 16 + col_w * length(windows)));
fprintf('  %-16s', 'TOTAL');
for w = 1 : length(windows)
    fprintf('  %-10d  %-10.2f', total_sel(w), 100.0);
end
fprintf('\n');

% ── Summary statistics ───────────────────────────────────────────────────────
fprintf('  %s\n', repmat('-', 1, 16 + col_w * length(windows)));

fprintf('  %-16s', 'Mean z');
for w = 1 : length(windows)
    fprintf('  %-10.4e  %-10s', mean_z(w), '');
end
fprintf('\n');

fprintf('  %-16s', 'Median z');
for w = 1 : length(windows)
    fprintf('  %-10.4e  %-10s', median_z(w), '');
end
fprintf('\n');

fprintf('  %s\n', repmat('-', 1, 16 + col_w * length(windows)));

fprintf('  %-16s', 'Mean eDF');
for w = 1 : length(windows)
    fprintf('  %-10.4f  %-10s', mean_edf(w), '');
end
fprintf('\n');

fprintf('  %-16s', 'Median eDF');
for w = 1 : length(windows)
    fprintf('  %-10.4f  %-10s', median_edf(w), '');
end
fprintf('\n');

% ── Single overall pct of raw z > 10^7 ───────────────────────────────────────
fprintf('  %s\n', repmat('-', 1, 16 + col_w * length(windows)));
fprintf('  %-16s', 'Pct z > 1e7 (%%)');
for w = 1 : length(windows)
    fprintf('  %-10.2f  %-10s', pct_above_1e7(w), '');
end
fprintf('\n%s\n', sep);